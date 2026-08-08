"""Graph builder: events -> typed provenance graph (+ splits + leakage audit).

This module implements, in Python, exactly what the future KQL stage will
do with ``make-graph``-style joins:

* nodes  = events, with the feature matrix defined in ``schema.py``;
* edges  = joins over shared entity columns within timespans
  (process lineage, sessions, host/user sequences, network correlation,
  config -> spawned-process triggers);
* splits = temporal train/val/test with an edge-dead gap zone between them
  so no label information can leak across time.

A single-feature leakage audit runs on the TRAIN split and hard-fails if any
feature separates the classes too well — cheating is caught automatically.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData

from .config import Config
from .schema import (
    ETYPE_ID, FEATURE_DIM, FEATURE_OFFSETS, RELATIONS, dt_bucket,
    EDGE_FEATURE_DIM,
)
from .synthetic import EventTable, generate_dataset

ET = ETYPE_ID

# bump whenever generator/edge semantics change, so cached graphs built by
# an older version are never silently reused
DATASET_VERSION = 6


# --------------------------------------------------------------------------- #
# small vectorised helpers                                                    #
# --------------------------------------------------------------------------- #
def _onehot(v: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((len(v), k), dtype=np.float32)
    valid = v >= 0
    if valid.any():
        out[valid, v[valid].astype(np.int64)] = 1.0
    return out


def _group_rank(sorted_keys: np.ndarray) -> np.ndarray:
    """0-based rank within each run of equal (already sorted) keys."""
    idx = np.arange(len(sorted_keys))
    change = np.flatnonzero(np.diff(sorted_keys) != 0) + 1
    base = np.zeros(len(sorted_keys), dtype=np.int64)
    base[change] = change
    np.maximum.accumulate(base, out=base)
    return idx - base


def _cap_outdegree(src, dst, dt, extra: List[np.ndarray], k: int):
    """Keep at most k outgoing edges per src, preferring the smallest dt."""
    if len(src) == 0:
        return src, dst, dt, extra
    order = np.lexsort((dt, src))
    s = src[order]
    keep = _group_rank(s) < k
    order = order[keep]
    return src[order], dst[order], dt[order], [e[order] for e in extra]


# --------------------------------------------------------------------------- #
# node features                                                               #
# --------------------------------------------------------------------------- #
def build_node_features(tab: EventTable) -> np.ndarray:
    c = tab.cols
    N = len(tab)
    X = np.zeros((N, FEATURE_DIM), dtype=np.float32)

    def put(name: str, block: np.ndarray) -> None:
        s, e = FEATURE_OFFSETS[name]
        assert block.shape == (N, e - s), (name, block.shape, (N, e - s))
        X[:, s:e] = block

    etype = c["etype"].astype(np.int64)
    is_signin = etype == ET["SignIn"]
    is_proc = etype == ET["ProcessCreate"]
    is_net = etype == ET["NetworkConnection"]
    is_file = etype == ET["FileActivity"]
    is_conf = etype == ET["SystemConfig"]
    is_dns = etype == ET["DnsQuery"]

    put("etype_onehot", _onehot(etype, 6))

    hour = (c["ts"] % 1440.0) / 60.0
    put("hour_sin_cos", np.stack([np.sin(2 * np.pi * hour / 24.0),
                                  np.cos(2 * np.pi * hour / 24.0)], axis=1))
    dow = (c["ts"] // 1440.0) % 7.0
    put("dow_sin_cos", np.stack([np.sin(2 * np.pi * dow / 7.0),
                                 np.cos(2 * np.pi * dow / 7.0)], axis=1))

    host = c["host"].astype(np.int64)
    user = c["user"].astype(np.int64)
    put("host_role_onehot", _onehot(tab.host_role[host], 6))
    put("os_onehot", _onehot(tab.host_os[host], 2))
    put("user_role_onehot", _onehot(np.where(user >= 0, tab.user_role[np.clip(user, 0, None)], -1), 3))

    auth = np.stack([c["auth_success"], c["mfa"], c["remote"], c["privileged"]],
                    axis=1).astype(np.float32) * is_signin[:, None]
    put("auth_flags", auth)
    put("integrity_onehot", _onehot(c["integrity"].astype(np.int64), 3))
    put("net_flags", np.stack([c["outbound"], (c["dst_host"] >= 0).astype(np.int64)],
                              axis=1).astype(np.float32) * is_net[:, None])
    put("protocol_onehot", _onehot(np.where(is_net, c["protocol"], -1).astype(np.int64), 3))

    port = c["port"]
    pclass = np.full(N, 0, dtype=np.int64)                    # na
    pclass[is_net & (port >= 0) & (port < 1024)] = 1
    pclass[is_net & (port >= 1024) & (port < 49152)] = 2
    pclass[is_net & (port >= 49152)] = 3
    put("port_class_onehot", _onehot(pclass, 4))

    bytesf = np.where((is_net | is_file) & (c["bytes"] > 0),
                      np.log1p(c["bytes"]) / 20.0, 0.0).astype(np.float32)
    put("log_bytes", bytesf[:, None])

    put("process_cat_onehot", _onehot(np.where(is_proc, c["process_cat"], -1).astype(np.int64), 14))
    put("file_cat_onehot", _onehot(np.where(is_file, c["file_cat"], -1).astype(np.int64), 9))
    put("file_action_onehot", _onehot(np.where(is_file, c["file_action"], -1).astype(np.int64), 4))
    put("config_kind_onehot", _onehot(np.where(is_conf, c["config_kind"], -1).astype(np.int64), 5))
    put("dns_feats", np.stack([c["dns_rarity"], np.log1p(c["dns_len"]) / 4.0], axis=1)
        * is_dns[:, None].astype(np.float32))

    ent4 = np.clip(c["cmd_entropy"].astype(np.int64), 0, 3)
    cmdline = np.concatenate([
        _onehot(np.where(is_proc, ent4, -1), 4),
        (c["encoded_cmd"] * is_proc).astype(np.float32)[:, None],
        (np.log1p(c["argc"]) / 3.0 * is_proc).astype(np.float32)[:, None],
    ], axis=1)
    put("cmdline_feats", cmdline)
    return X


# --------------------------------------------------------------------------- #
# edges                                                                       #
# --------------------------------------------------------------------------- #
def build_edges(tab: EventTable, gcfg) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Returns rel -> (src, dst, dt_min, bytes). Mirrors KQL make-graph joins."""
    c = tab.cols
    ts, host, user, etype = c["ts"], c["host"], c["user"], c["etype"]
    N = len(tab)
    edges: Dict[str, Tuple] = {}

    def finish(rel, src, dst, extra_bytes=None):
        src = np.asarray(src, dtype=np.int64); dst = np.asarray(dst, dtype=np.int64)
        ok = (src >= 0) & (dst >= 0) & (src != dst) & (ts[dst] >= ts[src])
        src, dst = src[ok], dst[ok]
        dt = np.clip(ts[dst] - ts[src], 0.0, None)
        b = extra_bytes[ok] if extra_bytes is not None else np.zeros(len(src))
        src, dst, dt, (b,) = _cap_outdegree(src, dst, dt, [b], gcfg.max_out_degree_per_rel)
        edges[rel] = (src, dst, dt, b)

    # PROCESS_CHILD / SESSION_ACTION come straight from lineage columns
    v = c["parent"] >= 0
    finish("PROCESS_CHILD", c["parent"][v], np.where(v)[0])
    v = c["signin"] >= 0
    finish("SESSION_ACTION", c["signin"][v], np.where(v)[0])

    # HOST_SEQUENCE / USER_SEQUENCE: consecutive events within windows
    def seq_edges(key: np.ndarray, window: float, valid: np.ndarray = None):
        if valid is not None:
            idx = np.where(valid)[0]
        else:
            idx = np.arange(N)
        o = np.lexsort((ts[idx], key[idx]))
        gid = idx[o]
        ks = key[gid]; tss = ts[gid]
        ch = np.flatnonzero(np.diff(ks) != 0) + 1
        pairs_src, pairs_dst = [], []
        for b in np.split(np.arange(len(gid)), ch):
            if len(b) < 2:
                continue
            d = tss[b[1:]] - tss[b[:-1]]
            m = d <= window
            pairs_src.append(gid[b[:-1]][m]); pairs_dst.append(gid[b[1:]][m])
        if pairs_src:
            return np.concatenate(pairs_src), np.concatenate(pairs_dst)
        return np.empty(0, np.int64), np.empty(0, np.int64)

    s, d = seq_edges(host, gcfg.host_seq_window_min)
    finish("HOST_SEQUENCE", s, d)
    s, d = seq_edges(user, gcfg.user_seq_window_min, user >= 0)
    finish("USER_SEQUENCE", s, d)

    # per-host event blocks for correlation joins
    host_order = np.lexsort((ts, host))
    hs = host[host_order]; tss = ts[host_order]
    ch = np.flatnonzero(np.diff(hs) != 0) + 1
    host_blocks: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for b in np.split(np.arange(N), ch):
        host_blocks[int(hs[b[0]])] = (host_order[b], tss[b])

    # NET_TRIGGER: net event -> first events on dst host within window
    net_idx = np.where((etype == ET["NetworkConnection"]) & (c["dst_host"] >= 0))[0]
    src_l: List[np.ndarray] = []; dst_l: List[np.ndarray] = []; by_l: List[np.ndarray] = []
    w = gcfg.net_trigger_window_min; k = gcfg.net_trigger_max_links
    for e in net_idx:
        h = int(c["dst_host"][e])
        blk = host_blocks.get(h)
        if blk is None:
            continue
        ev_idx, ev_ts = blk
        lo = np.searchsorted(ev_ts, ts[e], side="right")
        hi = np.searchsorted(ev_ts, ts[e] + w, side="right")
        take = ev_idx[lo:min(lo + k, hi)]
        if len(take):
            src_l.append(np.full(len(take), e)); dst_l.append(take)
            by_l.append(np.full(len(take), c["bytes"][e]))
    finish("NET_TRIGGER",
           np.concatenate(src_l) if src_l else [],
           np.concatenate(dst_l) if dst_l else [],
           np.concatenate(by_l) if by_l else None)

    # CONFIG_TRIGGER: SystemConfig -> ProcessCreate spawned later on same host
    conf_idx = np.where(etype == ET["SystemConfig"])[0]
    proc_by_host: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    proc_idx = np.where(etype == ET["ProcessCreate"])[0]
    if len(proc_idx):
        po = np.lexsort((ts[proc_idx], host[proc_idx]))
        pg = proc_idx[po]
        ph = host[pg]; pt = ts[pg]
        pch = np.flatnonzero(np.diff(ph) != 0) + 1
        for b in np.split(np.arange(len(pg)), pch):
            proc_by_host[int(ph[b[0]])] = (pg[b], pt[b])
    lo_d, hi_d = gcfg.config_trigger_min_min, gcfg.config_trigger_max_h * 60.0
    src_l, dst_l = [], []
    for e in conf_idx:
        blk = proc_by_host.get(int(host[e]))
        if blk is None:
            continue
        pidx, pts = blk
        lo = np.searchsorted(pts, ts[e] + lo_d, side="left")
        hi = np.searchsorted(pts, ts[e] + hi_d, side="right")
        take = pidx[lo:min(lo + gcfg.config_trigger_max_links, hi)]
        if len(take):
            src_l.append(np.full(len(take), e)); dst_l.append(take)
    finish("CONFIG_TRIGGER",
           np.concatenate(src_l) if src_l else [],
           np.concatenate(dst_l) if dst_l else [])
    return edges


def make_edge_features(dt_min: np.ndarray, bytes_: np.ndarray) -> np.ndarray:
    from .schema import EDGE_DT_BOUNDARIES_MIN
    E = len(dt_min)
    ef = np.zeros((E, EDGE_FEATURE_DIM), dtype=np.float32)
    ef[:, 0] = np.log1p(dt_min) / 8.0
    buckets = np.searchsorted(np.asarray(EDGE_DT_BOUNDARIES_MIN),
                              np.clip(dt_min, 0.0, 1e6), side="right")
    ef[np.arange(E), 1 + buckets] = 1.0
    ef[:, 7] = np.log1p(bytes_) / 20.0
    return ef


# --------------------------------------------------------------------------- #
# temporal split                                                              #
# --------------------------------------------------------------------------- #
def temporal_split(ts: np.ndarray, gcfg) -> np.ndarray:
    """0=train, 1=val, 2=test, -1=gap. Gaps are an edge-dead firewall zone."""
    T = ts.max()
    b1 = gcfg.split_train_frac * T
    b2 = (gcfg.split_train_frac + gcfg.split_val_frac) * T
    g = gcfg.split_gap_hours * 60.0
    sid = np.full(len(ts), -1, dtype=np.int64)
    sid[ts < b1] = 0
    sid[(ts >= b1 + g) & (ts < b2)] = 1
    sid[ts >= b2 + g] = 2
    return sid


# --------------------------------------------------------------------------- #
# leakage audit                                                               #
# --------------------------------------------------------------------------- #
def leakage_audit(X: np.ndarray, y: np.ndarray, split_id: np.ndarray,
                  threshold: float) -> List[Tuple[str, float]]:
    """Single-feature separability audit on the TRAIN split only.

    If any one feature almost perfectly separates the classes, the generator
    is cheating and the run is aborted. Returns sorted (feature, auc) list.
    """
    from sklearn.metrics import roc_auc_score
    tr = split_id == 0
    Xt, yt = X[tr], y[tr]
    if yt.sum() == 0 or yt.sum() == len(yt):
        raise RuntimeError("leakage audit: train split has a single class")
    scores = []
    from .schema import FEATURE_BLOCKS, FEATURE_OFFSETS
    for j in range(Xt.shape[1]):
        col = Xt[:, j]
        if col.min() == col.max():
            continue
        auc = roc_auc_score(yt, col)
        auc = max(auc, 1.0 - auc)
        scores.append((j, auc))
    scores.sort(key=lambda t: -t[1])
    named = []
    for j, auc in scores:
        blk = next(b for b in FEATURE_BLOCKS
                   if FEATURE_OFFSETS[b.name][0] <= j < FEATURE_OFFSETS[b.name][1])
        named.append((f"{blk.name}[{j - FEATURE_OFFSETS[blk.name][0]}]", auc))
    worst = named[0][1] if named else 0.0
    if worst >= threshold:
        raise RuntimeError(
            f"LEAKAGE AUDIT FAILED: single feature '{named[0][0]}' reaches "
            f"AUC {worst:.3f} >= {threshold}. The data generator is leaking "
            f"labels through features — fix the generator, do not train on this.")
    return named


# --------------------------------------------------------------------------- #
# dataset bundle                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class DatasetBundle:
    data: HeteroData
    X_mean: np.ndarray
    X_std: np.ndarray
    split_counts: Dict[str, Dict[str, int]]
    leakage_top: List[Tuple[str, float]]
    edge_stats: Dict[str, int]


def _cache_path(cfg: Config) -> str:
    import dataclasses
    d = dataclasses.asdict(cfg.data)
    d.pop("regenerate", None)              # not part of the dataset identity
    payload = json.dumps({"v": DATASET_VERSION, "data": d,
                          "graph": dataclasses.asdict(cfg.graph)}, sort_keys=True)
    h = hashlib.md5(payload.encode()).hexdigest()[:12]
    return os.path.join(cfg.data.data_dir, f"graph_{h}.pt")


def prepare_dataset(cfg: Config, log=print) -> DatasetBundle:
    path = _cache_path(cfg)
    if os.path.exists(path) and not cfg.data.regenerate:
        log(f"[data] loading cached dataset from {path}")
        obj = torch.load(path, weights_only=False)
        log(_summary_lines(obj))
        return obj

    t0 = time.time()
    log("[data] generating synthetic Sentinel-like event stream ...")
    tab = generate_dataset(cfg)
    N = len(tab)
    y = tab.cols["y"].astype(np.int64)
    log(f"[data]   events={N:,}  hosts={tab.n_hosts}  users={tab.n_users}  "
        f"days={tab.days}  positive events={int(y.sum()):,} ({100*y.mean():.2f}%)  "
        f"[{time.time()-t0:.1f}s]")

    log("[data] building node features ...")
    X = build_node_features(tab)

    log("[data] building provenance edges (KQL make-graph equivalent) ...")
    edges = build_edges(tab, cfg.graph)

    log("[data] temporal split ...")
    split_id = temporal_split(tab.cols["ts"], cfg.graph)

    log("[data] leakage audit (single-feature AUC on train split) ...")
    top = leakage_audit(X, y, split_id, cfg.graph.leakage_auc_threshold)
    log("[data]   top-5 single-feature AUCs (must stay < "
        f"{cfg.graph.leakage_auc_threshold}):")
    for name, auc in top[:5]:
        log(f"[data]     {name:<32} AUC={auc:.3f}")

    log("[data] assembling HeteroData ...")
    data = HeteroData()
    data["event"].x = torch.from_numpy(X)
    data["event"].y = torch.from_numpy(y)
    data["event"].ts = torch.from_numpy(tab.cols["ts"].astype(np.float64))
    data["event"].split = torch.from_numpy(split_id)

    edge_stats = {}
    for rel in RELATIONS:
        src, dst, dt, by = edges[rel]
        keep = (split_id[src] == split_id[dst]) & (split_id[src] >= 0)
        src, dst, dt, by = src[keep], dst[keep], dt[keep], by[keep]
        data["event", rel, "event"].edge_index = torch.from_numpy(
            np.stack([src, dst]).astype(np.int64))
        data["event", rel, "event"].edge_attr = torch.from_numpy(
            make_edge_features(dt, by))
        edge_stats[rel] = int(len(src))
        log(f"[data]   {rel:<16} edges={len(src):>9,}")

    # feature standardisation stats from TRAIN only
    tr = split_id == 0
    mean = X[tr].mean(axis=0)
    std = np.clip(X[tr].std(axis=0), 1e-2, None)

    counts = {}
    for name, s in (("train", 0), ("val", 1), ("test", 2)):
        m = split_id == s
        counts[name] = {"nodes": int(m.sum()), "positive": int(y[m].sum()),
                        "pos_rate": float(y[m].mean())}
        log(f"[data]   split {name:<5}: nodes={counts[name]['nodes']:>8,}  "
            f"positive={counts[name]['positive']:>6,}  "
            f"pos_rate={100*counts[name]['pos_rate']:.2f}%")
    for name in ("train", "val", "test"):
        if counts[name]["positive"] == 0:
            raise RuntimeError(f"split '{name}' contains no positive events — "
                               "increase n_incidents or the timeline length")

    bundle = DatasetBundle(data=data, X_mean=mean, X_std=std,
                           split_counts=counts, leakage_top=top,
                           edge_stats=edge_stats)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(bundle, path)
    log(f"[data] cached dataset -> {path}  (total {time.time()-t0:.1f}s)")
    return bundle


def _summary_lines(obj: DatasetBundle) -> str:
    d = obj.data
    lines = [f"[data]   nodes={d['event'].num_nodes:,}"]
    for rel in RELATIONS:
        lines.append(f"[data]   {rel:<16} edges={obj.edge_stats.get(rel, 0):>9,}")
    for name in ("train", "val", "test"):
        cc = obj.split_counts[name]
        lines.append(f"[data]   split {name:<5}: nodes={cc['nodes']:>8,}  "
                     f"positive={cc['positive']:>6,}  pos_rate={100*cc['pos_rate']:.2f}%")
    return "\n".join(lines)
