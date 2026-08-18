#!/usr/bin/env python
"""Entity-holdout go/no-go probe for the event+entity hybrid.

Hypothesis under test
---------------------
The per-event RGAT's residual misses (beacon / valid-account abuse) live at the
ENTITY level: an entity's *behaviour over a window* (cadence, fan-out, event
mix, burstiness) carries compromise signal that a single event cannot see. This
probe asks the decisive question BEFORE we build the hybrid:

    Can a classifier trained on entity behavioural features recognise a
    compromised entity, and does it still work on entities it has NEVER
    seen in training?

If out-of-fold AUPRC on held-out entities is far above the positive-rate
baseline, the entity signal is real and *generalises* (behaviour, not identity)
-> GO for the hybrid. If it collapses on unseen entities while staying high on
seen ones, the signal was entity-identity memorisation -> the naive
entity-as-nodes model would leak, and we must rethink.

Deliberately analysis-only: nothing here feeds back into the RGAT.

Usage:  python tools/entity_probe.py runs/<run_dir> [--window 120]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from eprgat.config import Config
from eprgat.synthetic import generate_dataset


# --------------------------------------------------------------------------- #
# entity-window feature builder                                               #
# --------------------------------------------------------------------------- #
def _distinct_per_group(group_idx: np.ndarray, value: np.ndarray,
                        n_groups: int) -> np.ndarray:
    """count of distinct `value` entries within each group."""
    if len(value) == 0:
        return np.zeros(n_groups, dtype=np.int64)
    pairs = np.unique(np.stack([group_idx, value], axis=1), axis=0)
    return np.bincount(pairs[:, 0], minlength=n_groups)


def build_entity_windows(tab, entity: str, window_min: float):
    """One row per (entity, window) with behavioural features + label.

    entity: 'host' or 'user'. Returns (X, y, ent_id, win_id, feat_names).
    """
    c = tab.cols
    ts = c["ts"]; n = len(tab)
    ent = c["host"] if entity == "host" else c["user"]
    peer = c["user"] if entity == "host" else c["host"]
    keep = ent >= 0
    ts, ent, peer = ts[keep], ent[keep], peer[keep]
    yv = c["y"][keep].astype(np.int64)
    etype = c["etype"][keep]; dst_host = c["dst_host"][keep]
    host = c["host"][keep]; user = c["user"][keep]
    outbound = c["outbound"][keep]; nbytes = c["bytes"][keep]
    remote = c["remote"][keep]; succ = c["auth_success"][keep]
    privileged = c["privileged"][keep]; integrity = c["integrity"][keep]
    encoded = c["encoded_cmd"][keep]; pcat = c["process_cat"][keep]
    fcat = c["file_cat"][keep]; faction = c["file_action"][keep]
    ckind = c["config_kind"][keep]; drar = c["dns_rarity"][keep]

    T = ts.max() + 1.0
    n_win = int(np.ceil(T / window_min))
    widx = np.clip((ts // window_min).astype(np.int64), 0, n_win - 1)
    key = ent * n_win + widx
    ukey, inv = np.unique(key, return_inverse=True)
    G = len(ukey)
    ent_id = ukey // n_win
    win_id = ukey % n_win

    def cnt(mask: np.ndarray) -> np.ndarray:
        return np.bincount(inv, weights=mask.astype(np.float64), minlength=G)

    label = (cnt(yv) > 0).astype(np.int64)
    feats = {}
    feats["n_events"] = cnt(np.ones(len(ts)))
    # window span actually covered by this entity's events (max ts - min ts)
    tmax = np.full(G, -np.inf); tmin = np.full(G, np.inf)
    np.maximum.at(tmax, inv, ts); np.minimum.at(tmin, inv, ts)
    feats["span_min"] = tmax - tmin
    feats["events_per_min"] = feats["n_events"] / np.maximum(1.0, feats["span_min"])

    ET = {"SignIn": 0, "ProcessCreate": 1, "NetworkConnection": 2,
          "FileActivity": 3, "SystemConfig": 4, "DnsQuery": 5}
    for name, eid in ET.items():
        feats[f"n_{name}"] = cnt(etype == eid)
        feats[f"f_{name}"] = feats[f"n_{name}"] / np.maximum(1, feats["n_events"])

    # network behaviour
    is_net = etype == ET["NetworkConnection"]
    feats["n_outbound"] = cnt(is_net & (outbound == 1))
    feats["n_internal"] = cnt(is_net & (dst_host >= 0))
    lb = np.where(is_net & (nbytes > 0), np.log1p(nbytes), 0.0)
    feats["net_bytes_log"] = np.bincount(inv, weights=lb, minlength=G)
    feats["n_distinct_dst_hosts"] = _distinct_per_group(
        inv[is_net & (dst_host >= 0)], dst_host[is_net & (dst_host >= 0)], G)
    feats["n_distinct_peer"] = _distinct_per_group(inv, peer, G)

    # sign-in behaviour
    is_si = etype == ET["SignIn"]
    feats["n_remote_signin"] = cnt(is_si & (remote == 1))
    feats["n_failed_signin"] = cnt(is_si & (succ == 0))
    feats["n_priv_signin"] = cnt(is_si & (privileged == 1))

    # process behaviour
    is_pr = etype == ET["ProcessCreate"]
    feats["n_shell"] = cnt(is_pr & (pcat == 3))
    feats["n_script"] = cnt(is_pr & (pcat == 4))
    feats["n_remote_access"] = cnt(is_pr & (pcat == 12))
    feats["n_encoded_cmd"] = cnt(is_pr & (encoded == 1))
    feats["n_high_integrity"] = cnt(is_pr & (integrity == 2))
    feats["n_priv_proc"] = cnt(is_pr & (privileged == 1))

    # file behaviour
    is_fa = etype == ET["FileActivity"]
    feats["n_credstore_read"] = cnt(is_fa & (fcat == 5) & (faction == 0))
    feats["n_archive_write"] = cnt(is_fa & (fcat == 4) & (faction == 1))
    feats["n_temp_write"] = cnt(is_fa & (fcat == 6) & (faction == 1))
    feats["n_executable"] = cnt(is_fa & (fcat == 1))

    # config / dns
    feats["n_config"] = cnt(etype == ET["SystemConfig"])
    feats["n_config_service"] = cnt((etype == ET["SystemConfig"]) & (ckind == 0))
    is_dns = etype == ET["DnsQuery"]
    feats["n_dns"] = cnt(is_dns)
    feats["n_rare_dns"] = cnt(is_dns & (drar > 0.6))

    # beacon periodicity: regularity of outbound inter-arrival times AND of the
    # outbound payload sizes (a real beacon is clock-like in both; benign bulk
    # transfers are not)
    cv = np.zeros(G); nout = np.zeros(G); regfrac = np.zeros(G); bytecv = np.zeros(G)
    order = np.lexsort((ts, inv))
    g_sorted = inv[order]; t_sorted = ts[order]
    ob_sorted = (outbound == 1)[order]; obb_sorted = nbytes[order]
    # group boundaries
    ch = np.flatnonzero(np.diff(g_sorted) != 0) + 1
    for b in np.split(np.arange(len(g_sorted)), ch):
        if len(b) == 0:
            continue
        g = g_sorted[b[0]]
        sel = ob_sorted[b]
        obt = t_sorted[b][sel]; obb = obb_sorted[b][sel]
        nout[g] = len(obt)
        if len(obt) >= 3:
            iv = np.diff(obt)
            if iv.mean() > 1e-6:
                cv[g] = iv.std() / iv.mean()      # low cv = clock-like beacon
                med = np.median(iv)
                if med > 1e-6:                    # fraction of near-regular gaps
                    regfrac[g] = float(((iv >= 0.5 * med) & (iv <= 2.0 * med)).mean())
            lb = np.log1p(obb[obb > 0])
            if len(lb) >= 3 and lb.mean() > 1e-6:
                bytecv[g] = lb.std() / lb.mean()  # low = uniform C2 payload size
    feats["outbound_count"] = nout
    feats["outbound_interval_cv"] = cv
    feats["regular_interval_frac"] = regfrac
    feats["outbound_byte_cv"] = bytecv
    feats["periodic_beacon"] = ((nout >= 3) & (cv < 0.35)).astype(np.float64)

    # coarse entity role context (shared vocabulary, NOT identity) — a beacon on
    # a regular user's workstation reads differently than a service account's job
    role = tab.host_role[ent_id] if entity == "host" else tab.user_role[ent_id]
    feats["entity_role"] = role.astype(np.float32)

    names = list(feats.keys())
    X = np.stack([feats[k] for k in names], axis=1).astype(np.float32)
    return X, label, ent_id, win_id, names


# --------------------------------------------------------------------------- #
# entity-holdout out-of-fold evaluation                                       #
# --------------------------------------------------------------------------- #
def entity_holdout_oof(X, label, ent_id, seed=0):
    """2-fold split over ENTITIES: every entity-window is scored by a model
    that never saw that entity. Returns out-of-fold probabilities."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score

    ents = np.unique(ent_id)
    ent_ever_pos = np.array([label[ent_id == e].max() for e in ents])
    rng = np.random.default_rng(seed)
    # stratify: keep the positive-entity ratio in both folds
    pos_e = ents[ent_ever_pos == 1].copy()
    neg_e = ents[ent_ever_pos == 0].copy()
    rng.shuffle(pos_e); rng.shuffle(neg_e)
    fold_of_ent = {}
    for i, e in enumerate(pos_e):
        fold_of_ent[e] = i % 2
    for i, e in enumerate(neg_e):
        fold_of_ent[e] = i % 2
    fold = np.array([fold_of_ent[e] for e in ent_id])

    oof = np.zeros(len(label))
    for k in (0, 1):
        tr = fold != k; te = fold == k
        if label[tr].sum() == 0 or label[te].sum() == 0:
            continue
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=4,
            l2_regularization=1.0, min_samples_leaf=20, random_state=seed)
        clf.fit(X[tr], label[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    ap = average_precision_score(label, oof)
    return oof, ap


def main() -> int:
    ap = argparse.ArgumentParser(description="Entity-holdout go/no-go probe.")
    ap.add_argument("run", help="runs/<dir> whose config defines the world")
    ap.add_argument("--window", type=float, default=120.0,
                    help="entity window length in minutes")
    ap.add_argument("--misses", default=None,
                    help="viz_data.json: connect entity signal to RGAT misses")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = Config.load(os.path.join(args.run, "config.yaml"))
    print(f"[probe] regenerating world for {args.run} (window={args.window:.0f}m) ...")
    tab = generate_dataset(cfg)
    y_all = tab.cols["y"].astype(np.int64)
    print(f"[probe] events={len(tab):,}  positives={int(y_all.sum()):,}  "
          f"hosts={tab.n_hosts} users={tab.n_users}")

    results = {}
    for entity in ("host", "user"):
        X, lab, ent_id, win_id, names = build_entity_windows(
            tab, entity, args.window)
        base = lab.mean()
        oof, ap_score = entity_holdout_oof(X, lab, ent_id, seed=args.seed)
        n_win = len(lab); n_pos = int(lab.sum())
        n_ent = len(np.unique(ent_id))
        n_pos_ent = int(np.array([lab[ent_id == e].max()
                                  for e in np.unique(ent_id)]).sum())
        lift = ap_score / base if base > 0 else float("nan")
        results[entity] = (X, lab, ent_id, win_id, oof, ap_score, base)
        print(f"\n[{entity}] entity-windows={n_win:,}  positive={n_pos:,} "
              f"({100*base:.2f}%)  entities={n_ent} (ever-compromised={n_pos_ent})")
        print(f"[{entity}] OUT-OF-FOLD AUPRC on UNSEEN entities = {ap_score:.4f}   "
              f"baseline AP = {base:.4f}   lift x{lift:.1f}")

    # ------------------------------------------------- rescue of RGAT misses
    if args.misses:
        import json
        viz = json.load(open(args.misses))
        ev = viz["events"]; meta = viz["meta"]
        thr = meta["threshold_recall"]
        n_win = int(np.ceil((tab.cols["ts"].max() + 1.0) / args.window))
        print(f"\n[rescue] RGAT recall-thr={thr:.3f}; mapping missed positives to "
              f"entity windows ...")
        hostX, hostlab, hostent, hostwin, hostoof, hostap, _ = results["host"]
        userX, userlab, userent, userwin, useroof, userap, _ = results["user"]
        # lookup: (entity, window) -> oof score
        hkey = hostent * n_win + hostwin
        ukey = userent * n_win + userwin
        hmap = dict(zip(hkey.tolist(), hostoof.tolist()))
        umap = dict(zip(ukey.tolist(), useroof.tolist()))

        pos = [i for i in range(len(ev["p"])) if ev["y"][i] == 1]
        missed = [i for i in pos if ev["p"][i] < thr]
        scores = []
        for i in missed:
            w = int(ev["ts"][i] // args.window)
            hs = hmap.get(ev["h"][i] * n_win + w, 0.0)
            us = umap.get(ev["u"][i] * n_win + w, 0.0) if ev["u"][i] >= 0 else 0.0
            scores.append(max(hs, us))
        scores = np.array(scores)
        # caught-vs-missed: do missed events sit in WEAKER entity windows?
        caught = [i for i in pos if ev["p"][i] >= thr]
        def ent_score(i):
            w = int(ev["ts"][i] // args.window)
            hs = hmap.get(ev["h"][i] * n_win + w, 0.0)
            us = umap.get(ev["u"][i] * n_win + w, 0.0) if ev["u"][i] >= 0 else 0.0
            return max(hs, us)
        c_scores = np.array([ent_score(i) for i in caught])
        print(f"[rescue] entity score of CAUGHT-pos windows: median={np.median(c_scores):.3f}")
        print(f"[rescue] entity score of MISSED-pos windows: median={np.median(scores):.3f}")

        # rescue-vs-cost sweep over entity operating points
        all_lab = np.concatenate([hostlab, userlab])
        all_oof = np.concatenate([hostoof, useroof])
        widx_all = np.clip((tab.cols["ts"] // args.window).astype(np.int64), 0, n_win - 1)
        hk = tab.cols["host"] * n_win + widx_all
        uk = tab.cols["user"] * n_win + widx_all
        n_tab = len(tab)
        uu = tab.cols["user"] >= 0
        print(f"[rescue] fixed entity thresholds -> missed rescued vs flagged cost:")
        print(f"[rescue] {'thr':>6} {'missed rescued':>15} {'events flagged':>16} "
              f"{'entity prec':>12}")
        for t in (0.3, 0.5, 0.7, 0.9):
            rescued = int((scores >= t).sum())
            fm = np.isin(hk, hkey[hostoof >= t]) | (uu & np.isin(uk, ukey[useroof >= t]))
            m = all_oof >= t
            entprec = float(all_lab[m].mean()) if m.any() else 0.0
            print(f"[rescue] {t:>6.2f} {rescued:>10}/{len(missed)} "
                  f"{int(fm.sum()):>11,} ({100*fm.mean():.2f}%) {entprec:>11.3f}")

    print("\n[probe] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
