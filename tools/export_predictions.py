#!/usr/bin/env python
"""Export per-event test predictions + incident metadata for the viz server.

For a trained run directory this re-runs the SAME evaluation used at training
time (val-tuned threshold, never fitted on test), joins each test event with
the incident metadata of the exact generated world, and writes a
self-contained ``viz_data.json`` next to the checkpoint. That file is what
``tools/serve_viz.py`` renders on localhost — it answers "which events flared
up, and which ground-truth incidents did we catch".

Everything here is analysis-only: nothing written feeds back into training.

Usage:  python tools/export_predictions.py runs/<run_dir> [--device cpu]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

# torch + the model/eval modules are imported inside main() so that the pure
# helpers below (incident-metadata reconstruction, event descriptions) stay
# importable — and unit-testable — without torch installed. Config only needs
# yaml + schema, so it is safe at import time.
from eprgat.config import Config
from eprgat.schema import (
    CONFIG_KINDS, EVENT_TYPES, FILE_ACTIONS, FILE_CATS, HOST_ROLES,
    PROCESS_CATS, RELATIONS, REL_ID, USER_ROLES,
)
from eprgat.synthetic import SentinelEventGenerator, generate_dataset

TPL_NAMES = ["T1_phishing", "T2_exploit_web", "T3_valid_account",
             "T4_beacon_persist", "T5_service_lateral"]
_HOST_ABBR = ["ws", "srv", "dc", "fs", "web", "db"]


# --------------------------------------------------------------------------- #
# incident metadata reconstruction (mirrors tools/analyze_errors.py)          #
# --------------------------------------------------------------------------- #
class _RngProxy:
    """Delegates to the real Generator but records the template draw — the
    first ``choice(5, p=[5 probs])`` call inside ``_attack_events``."""

    def __init__(self, rng, state):
        self._rng = rng
        self._state = state

    def choice(self, a, *args, **kw):
        res = self._rng.choice(a, *args, **kw)
        if ("tpl" not in self._state and np.ndim(a) == 0 and int(a) == 5
                and len(kw.get("p", [])) == 5):
            self._state["tpl"] = int(res)
        return res

    def __getattr__(self, name):
        return getattr(self._rng, name)


def build_incident_metadata(cfg: Config):
    """Regenerate the exact world of the run and capture, per incident, which
    template produced it. Returns (EventTable, {incident_id: template})."""
    captured = {}
    orig_ae = SentinelEventGenerator._attack_events

    def patched(self, iid, split_target):
        state = {}
        real = self.rng
        self.rng = _RngProxy(real, state)
        try:
            orig_ae(self, iid, split_target)
        finally:
            self.rng = real
        captured[iid] = state.get("tpl", -1)

    SentinelEventGenerator._attack_events = patched
    try:
        tab = generate_dataset(cfg)
    finally:
        SentinelEventGenerator._attack_events = orig_ae
    return tab, captured


# --------------------------------------------------------------------------- #
# human-readable one-line description of an event (no unique identifiers go to
# the MODEL, but an analyst inspecting a flare may see the coarse world)      #
# --------------------------------------------------------------------------- #
def _fmt_bytes(b: float) -> str:
    if b <= 0:
        return ""
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if b >= div:
            return f"{b/div:.1f}{unit}"
    return f"{int(b)}B"


def describe_event(i: int, c, tab) -> str:
    et = int(c["etype"][i])
    hrole = _HOST_ABBR[int(tab.host_role[int(c["host"][i])])]
    host = f"H{int(c['host'][i])}"
    u = int(c["user"][i])
    user = f"U{u}" if u >= 0 else "-"
    if et == EVENT_TYPES.index("SignIn"):
        tags = []
        if c["remote"][i]:
            tags.append("remote")
        if not c["auth_success"][i]:
            tags.append("FAILED")
        if c["mfa"][i]:
            tags.append("mfa")
        if c["privileged"][i]:
            tags.append("privileged")
        return f"sign-in {user}→{hrole} {host} ({', '.join(tags) or 'local'})"
    if et == EVENT_TYPES.index("ProcessCreate"):
        cat = PROCESS_CATS[int(c["process_cat"][i])] if c["process_cat"][i] >= 0 else "?"
        tags = []
        if c["encoded_cmd"][i]:
            tags.append("encoded")
        if c["privileged"][i]:
            tags.append("elevated")
        if int(c["integrity"][i]) == 2:
            tags.append("high-integrity")
        tail = f" [{', '.join(tags)}]" if tags else ""
        return f"process {cat} on {hrole} {host} by {user}{tail}"
    if et == EVENT_TYPES.index("NetworkConnection"):
        dst = int(c["dst_host"][i])
        dst_s = f"H{dst}({_HOST_ABBR[int(tab.host_role[dst])]})" if dst >= 0 else "external"
        port = int(c["port"][i])
        b = _fmt_bytes(float(c["bytes"][i]))
        proto = ("udp" if int(c["protocol"][i]) == 1 else
                 "icmp" if int(c["protocol"][i]) == 2 else "tcp")
        return f"net {host}→{dst_s} :{port}/{proto} {b}"
    if et == EVENT_TYPES.index("FileActivity"):
        cat = FILE_CATS[int(c["file_cat"][i])] if c["file_cat"][i] >= 0 else "?"
        act = FILE_ACTIONS[int(c["file_action"][i])]
        b = _fmt_bytes(float(c["bytes"][i]))
        return f"file {act} {cat} on {host} {b}".strip()
    if et == EVENT_TYPES.index("SystemConfig"):
        kind = CONFIG_KINDS[int(c["config_kind"][i])] if c["config_kind"][i] >= 0 else "?"
        return f"config change {kind} on {host}"
    # DnsQuery
    return f"dns query on {host} (rarity {float(c['dns_rarity'][i]):.2f})"


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export per-event test predictions for the localhost viz.")
    ap.add_argument("run", help="path to a runs/<...> directory")
    ap.add_argument("--out", default=None,
                    help="output path (default <run>/viz_data.json)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    # heavy deps, imported here so the pure helpers stay torch-free
    import torch
    from eprgat.evaluate import _eval_with, _load_model
    from eprgat.graph import prepare_dataset
    from eprgat.metrics import compute_metrics, search_threshold

    run = args.run.rstrip("/")
    cfg = Config.load(os.path.join(run, "config.yaml"))
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[viz] run={run}  device={device}")

    # ------------------------------------------------- world + incident meta
    print("[viz] reconstructing incident metadata for the run's exact world ...")
    tab, captured_tpl = build_incident_metadata(cfg)
    c = tab.cols
    n = len(tab)
    y_all = c["y"].astype(np.int64)
    inc_all = c["incident"]

    # ------------------------------------------------- dataset + predictions
    print("[viz] loading dataset (cached when possible) ...")
    bundle = prepare_dataset(cfg, log=lambda m: print("       " + m))
    assert bundle.data["event"].x.shape[0] == n, \
        "regenerated world does not match the cached graph — use a run whose " \
        "data.seed/config produced data/graph_*.pt"
    split = bundle.data["event"].split.numpy()

    print("[viz] loading model + re-running val/test evaluation ...")
    model, _ = _load_model(cfg, bundle, os.path.join(run, "best.pt"), device)

    vy, vp, _ = _eval_with(model, bundle, cfg, 1, device)
    thr = search_threshold(vy, vp, policy=cfg.train.threshold_policy,
                           min_recall=cfg.train.threshold_min_recall)
    # the recall-first operating point too (threshold also from val only)
    thr_recall = search_threshold(vy, vp, policy="recall",
                                  min_recall=cfg.train.threshold_min_recall)
    print(f"[viz] val-tuned thresholds: {cfg.train.threshold_policy}={thr:.4f}  "
          f"recall>={cfg.train.threshold_min_recall:.2f} thr={thr_recall:.4f}")

    ty, tp, rel_mass = _eval_with(model, bundle, cfg, 2, device)
    test_m = compute_metrics(ty, tp, thr)
    print(f"[viz] test @{thr:.3f}: F1={test_m.f1:.4f} P={test_m.precision:.4f} "
          f"R={test_m.recall:.4f}  (AUPRC={test_m.auprc:.4f}, "
          f"baseline AP={test_m.baseline_ap:.4f})")

    te_idx = np.where(split == 2)[0]
    assert len(tp) == len(te_idx)

    # ------------------------------------------------- per-event enrichment
    tpl_of_event = np.full(n, -1, dtype=np.int64)
    for iid, tpl in captured_tpl.items():
        tpl_of_event[(inc_all == iid) & (y_all == 1)] = tpl

    # position within the incident chain (table is sorted by ts)
    pos_frac = np.full(n, -1.0)
    for iid in np.unique(inc_all[y_all == 1]):
        m = np.where((inc_all == iid) & (y_all == 1))[0]
        pos_frac[m] = np.arange(len(m)) / max(1, len(m) - 1)

    g_ts = c["ts"]
    events = {
        "id": te_idx.tolist(),
        "ts": np.round(g_ts[te_idx], 1).tolist(),
        "et": c["etype"][te_idx].astype(int).tolist(),
        "h": c["host"][te_idx].astype(int).tolist(),
        "hr": tab.host_role[c["host"][te_idx]].astype(int).tolist(),
        "u": np.where(c["user"][te_idx] >= 0, c["user"][te_idx], -1).astype(int).tolist(),
        "ur": np.where(c["user"][te_idx] >= 0,
                       tab.user_role[np.clip(c["user"][te_idx], 0, None)], -1)
              .astype(int).tolist(),
        "y": y_all[te_idx].astype(int).tolist(),
        "p": np.round(tp.astype(np.float64), 5).tolist(),
        "inc": inc_all[te_idx].astype(int).tolist(),
        "tpl": tpl_of_event[te_idx].astype(int).tolist(),
        "pos": np.round(pos_frac[te_idx], 3).astype(np.float64).tolist(),
    }
    print("[viz] writing per-event descriptions ...")
    events["d"] = [describe_event(int(i), c, tab) for i in te_idx]

    # ------------------------------------------------- fusion scores (optional)
    # if tools/dump_scores.py has persisted channel scores + fitted combiners for
    # this run, embed the per-event fused scores so the dashboard can switch
    # between the RGAT-only score and each fusion operating point.
    fusion_models = []
    scores_npz = os.path.join(run, "scores.npz")
    if os.path.exists(scores_npz):
        sc = np.load(scores_npz)
        recipes = [
            ("balanced", "fuse_balanced",
             "event×entity fusion · balanced (agg120+seq, GBM)"),
            ("maxeff", "fuse_maxeff",
             "event×entity fusion · max-efficiency (agg120+agg480+seq, GBM)"),
        ]
        for key, arr, label in recipes:
            if f"{arr}_te" not in sc.files:
                continue
            fte = sc[f"{arr}_te"]
            if len(fte) != len(te_idx):
                print(f"[viz] WARNING: scores.npz {arr} length mismatch — skipping")
                continue
            events[f"s_{key}"] = np.round(fte.astype(np.float64), 5).tolist()
            fusion_models.append({
                "key": key, "label": label,
                "threshold": float(sc[f"{arr}_thr"]) if f"{arr}_thr" in sc.files else 0.5,
                "recall_threshold": float(sc[f"{arr}_rthr"]) if f"{arr}_rthr" in sc.files else None,
            })
        if fusion_models:
            print(f"[viz] merged fusion score channels: {[m['key'] for m in fusion_models]}")
    else:
        print("[viz] no scores.npz — RGAT-only (run tools/dump_scores.py for fusion)")

    # ------------------------------------------------- per-incident summary
    te_set = set(te_idx.tolist())
    incidents = []
    for iid in sorted(set(inc_all[te_idx][y_all[te_idx] == 1].tolist())):
        if iid < 0:
            continue
        m = np.where((inc_all == iid) & (y_all == 1))[0]
        tpl = int(captured_tpl.get(iid, -1))
        incidents.append({
            "id": int(iid),
            "tpl": tpl,
            "tname": TPL_NAMES[tpl] if 0 <= tpl < len(TPL_NAMES) else "?",
            "n": int(len(m)),
            "t0": float(g_ts[m[0]]), "t1": float(g_ts[m[-1]]),
            "hosts": int(len(np.unique(c["host"][m]))),
            "users": int(len(np.unique(c["user"][m][c["user"][m] >= 0]))),
        })

    # ------------------------------------------------- intra-incident edges
    # provenance links whose BOTH endpoints are positive test events of the
    # SAME incident — exactly the chains an analyst traces through a flare
    edges = []
    for rel in RELATIONS:
        ei = bundle.data["event", rel, "event"].edge_index
        if ei is None or ei.numel() == 0:
            continue
        s = ei[0].numpy(); d = ei[1].numpy()
        keep = (split[s] == 2) & (split[d] == 2) \
            & (y_all[s] == 1) & (y_all[d] == 1) \
            & (inc_all[s] == inc_all[d]) & (inc_all[s] >= 0)
        rid = REL_ID[rel]
        edges.extend([int(a), int(b), rid]
                     for a, b in zip(s[keep].tolist(), d[keep].tolist()))

    # ------------------------------------------------- assemble + write
    report_path = os.path.join(run, "eval_report.json")
    eval_report = {}
    if os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as f:
            eval_report = json.load(f)

    rel_imp = rel_mass / max(1e-12, rel_mass.sum())
    out = {
        "meta": {
            "run": os.path.basename(run),
            "run_path": run,
            "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "split": "test",
            "n_events": int(len(te_idx)),
            "n_pos": int(int(y_all[te_idx].sum())),
            "threshold": float(thr),
            "threshold_policy": cfg.train.threshold_policy,
            "threshold_recall": float(thr_recall),
            "min_recall": float(cfg.train.threshold_min_recall),
            "model": {"layers": cfg.model.num_layers,
                      "hidden": cfg.model.hidden_dim,
                      "heads": cfg.model.num_heads,
                      "readout": getattr(cfg.model, "readout", "fusion")},
            "timeline_days": cfg.data.days,
            "relation_attention": {r: float(w) for r, w in zip(RELATIONS, rel_imp)},
            "eval_report": eval_report,
            "test_metrics_at_export": test_m.to_dict(),
            "fusion_models": fusion_models,
        },
        "vocab": {
            "etypes": EVENT_TYPES,
            "host_roles": HOST_ROLES,
            "user_roles": USER_ROLES,
            "relations": RELATIONS,
            "templates": TPL_NAMES,
        },
        "events": events,
        "incidents": incidents,
        "edges": edges,
    }

    out_path = args.out or os.path.join(run, "viz_data.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[viz] wrote {out_path} ({size_mb:.1f} MB): "
          f"{len(te_idx):,} test events, {len(incidents)} test incidents, "
          f"{len(edges):,} intra-incident provenance edges")
    print(f"[viz] next:  python tools/serve_viz.py   ->  "
          f"http://localhost:8123  (pick '{os.path.basename(run)}')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
