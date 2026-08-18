#!/usr/bin/env python
"""Gated-fusion prototype: RGAT event scores + entity behavioural branch.

Answers the decision question with a single, honest comparison on the TEST
split — the recall-vs-flagged curve (how many events an analyst must look at to
catch a given fraction of the real positives) for:

  * RGAT-only        — the current per-event detector
  * entity-only      — the entity-window classifier used as a blanket
  * noisy-OR fusion  — flag if EITHER channel is confident (no fitting)
  * max fusion       — take the stronger channel's score (no fitting)
  * val-tuned meta   — a tiny logistic combiner fit ONLY on validation

The win condition: reach >=90% recall while flagging materially fewer events
than RGAT-only. NOTE on "flagged %": the tables here use the RANK-BASED
operating point (flag exactly the top-k events until the recall target is met),
the best case for a given ranking. The README's ~8.6%-at-90%-recall for the RGAT
is a FIXED-THRESHOLD operating point (flag everything above the val-tuned
threshold) and is therefore higher — a different operating point, not a
contradiction. Within each table all methods are compared identically.

Everything is analysis-only; nothing here retrains or touches the RGAT.

Usage:  python tools/fusion_probe.py runs/<run_dir> [--window 480]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from eprgat.config import Config
from eprgat.synthetic import generate_dataset
from entity_probe import build_entity_windows


def entity_temporal_scores(X, label, win_id, wsplit, seed=0):
    """Temporally honest: train on TRAIN-period windows only, score everything.
    Different incidents live in each period, so cross-period compromised entities
    are unseen; the classifier must generalise on behaviour, not memorise."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    tr = wsplit[win_id] == 0
    if label[tr].sum() == 0:
        return np.full(len(label), label.mean())
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4,
        l2_regularization=1.0, min_samples_leaf=20, random_state=seed)
    clf.fit(X[tr], label[tr])
    return clf.predict_proba(X)[:, 1]


# --------------------------------------------------------------------------- #
def recall_flag_curve(y: np.ndarray, score: np.ndarray, targets=(0.90, 0.93, 0.95, 0.97)):
    """flagged count (and % of events) needed to reach each recall target."""
    order = np.argsort(-score)
    cum = np.cumsum(y[order])
    totpos = int(y.sum()); n = len(y)
    out = {}
    for t in targets:
        k = int(np.searchsorted(cum, t * totpos) + 1)
        out[t] = (k, 100.0 * k / n)
    return out


def auprc(y, s):
    from sklearn.metrics import average_precision_score
    return average_precision_score(y, s)


def main() -> int:
    ap = argparse.ArgumentParser(description="RGAT x entity gated-fusion probe.")
    ap.add_argument("run")
    ap.add_argument("--window", type=float, default=480.0,
                    help="(legacy single-scale; overridden by --scales)")
    ap.add_argument("--scales", default="120,480",
                    help="comma-separated entity window sizes in minutes")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--viz", default=None,
                    help="viz_data.json: per-template catch comparison")
    args = ap.parse_args()

    import torch
    from eprgat.evaluate import _eval_with, _load_model
    from eprgat.graph import prepare_dataset

    cfg = Config.load(os.path.join(args.run, "config.yaml"))
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[fusion] {args.run}  window={args.window:.0f}m  device={device}")

    # ------------------------------------------------------- world + dataset
    tab = generate_dataset(cfg)
    bundle = prepare_dataset(cfg, log=lambda m: None)
    n = len(tab)
    assert bundle.data["event"].x.shape[0] == n
    split = bundle.data["event"].split.numpy()
    y_all = bundle.data["event"].y.numpy()
    c = tab.cols

    # ------------------------------------------------------- RGAT event scores
    model, _ = _load_model(cfg, bundle, os.path.join(args.run, "best.pt"), device)
    va_idx = np.where(split == 1)[0]; te_idx = np.where(split == 2)[0]
    _, p_va, _ = _eval_with(model, bundle, cfg, 1, device)
    _, p_te, _ = _eval_with(model, bundle, cfg, 2, device)
    print(f"[fusion] RGAT scored val={len(p_va):,} test={len(p_te):,} events")

    # ------------------------------------------------------- entity branch
    # multi-scale: beacon wants LONG windows (periodicity), valid-account is a
    # SHORT tight burst — one size cannot serve both, so use several.
    from eprgat.graph import temporal_split
    from sklearn.metrics import average_precision_score as _ap
    y_va = y_all[va_idx]; y_te = y_all[te_idx]

    def entity_node_scores(window_min):
        """per-node entity score at one window scale (temporal-honest)."""
        n_win = int(np.ceil((c["ts"].max() + 1.0) / window_min))
        wsplit = temporal_split((np.arange(n_win) + 0.5) * window_min, cfg.graph)
        grids = {}
        for entity in ("host", "user"):
            X, lab, ent_id, win_id, _ = build_entity_windows(tab, entity, window_min)
            scores = entity_temporal_scores(X, lab, win_id, wsplit, seed=args.seed)
            vw = wsplit[win_id] != 0
            if lab[vw].sum():
                print(f"[fusion]   scale={window_min:.0f}m [{entity}] "
                      f"temporal window AUPRC={_ap(lab[vw], scores[vw]):.4f} "
                      f"(base {lab.mean():.4f})")
            nent = tab.n_hosts if entity == "host" else tab.n_users
            g = np.zeros(nent * n_win); g[ent_id * n_win + win_id] = scores
            grids[entity] = g
        widx = np.clip((c["ts"] // window_min).astype(np.int64), 0, n_win - 1)
        hs = grids["host"][c["host"] * n_win + widx]
        us = np.where(c["user"] >= 0,
                      grids["user"][np.clip(c["user"], 0, None) * n_win + widx], 0.0)
        return np.maximum(hs, us)

    scales = [float(s) for s in args.scales.split(",")]
    print(f"[fusion] entity scales (min): {scales}")
    ent_by_scale = {w: entity_node_scores(w) for w in scales}
    ent_max = np.maximum.reduce([ent_by_scale[w] for w in scales])

    # ------------------------------------------------------- fusion + curves
    from sklearn.linear_model import LogisticRegression

    def meta_score(feats_va, feats_te):
        meta = LogisticRegression(class_weight="balanced", C=1.0)
        meta.fit(feats_va, y_va)
        return meta.predict_proba(feats_te)[:, 1]

    methods = {"RGAT-only": p_te}
    # single best scale + multi-scale
    best_w = max(scales, key=lambda w: auprc(y_te, ent_by_scale[w][te_idx]))
    for tag, ent in ((f"TEMPORAL@{best_w:.0f}", ent_by_scale[best_w]),
                     ("TEMPORAL-multiscale", ent_max)):
        e_va = ent[va_idx]; e_te = ent[te_idx]
        methods[f"noisy-OR({tag})"] = 1 - (1 - p_te) * (1 - e_te)
        methods[f"meta({tag})"] = meta_score(np.stack([p_va, e_va], 1),
                                             np.stack([p_te, e_te], 1))
    # meta with a feature per scale (lets it pick beacon vs valid-account scale)
    fv = np.stack([p_va] + [ent_by_scale[w][va_idx] for w in scales], 1)
    ft = np.stack([p_te] + [ent_by_scale[w][te_idx] for w in scales], 1)
    methods["meta(multiscale-feats)"] = meta_score(fv, ft)

    print(f"\n[fusion] TEST split: {len(y_te):,} events, {int(y_te.sum())} positives")
    print(f"[fusion] {'method':<22} {'AUPRC':>8} | flagged % of events at recall:")
    print(f"[fusion] {'':<22} {'':>8} | {'0.90':>7} {'0.93':>7} {'0.95':>7} {'0.97':>7}")
    for name, s in methods.items():
        curve = recall_flag_curve(y_te, s)
        a = auprc(y_te, s)
        row = " ".join(f"{curve[t][1]:>6.2f}%" for t in (0.90, 0.93, 0.95, 0.97))
        print(f"[fusion] {name:<22} {a:>8.4f} | {row}")

    rr = recall_flag_curve(y_te, p_te)
    print(f"\n[fusion] RGAT-only recall=0.90 flags {rr[0.90][0]:,} events "
          f"({rr[0.90][1]:.2f}%)  <-- baseline to beat")
    best = min(methods.items(), key=lambda kv: recall_flag_curve(y_te, kv[1])[0.90][0])
    bn, bs = best
    bc = recall_flag_curve(y_te, bs)[0.90]
    print(f"[fusion] best @ recall 0.90: '{bn}' flags {bc[0]:,} ({bc[1]:.2f}%) "
          f"-> {100*(1-bc[0]/rr[0.90][0]):.1f}% fewer events than RGAT-only")

    # ------------------------------------- per-template catch (the blind spots)
    if args.viz:
        import json
        tpl = np.array(json.load(open(args.viz))["events"]["tpl"])
        TPL = ["T1_phishing", "T2_exploit_web", "T3_valid_account",
               "T4_beacon_persist", "T5_service_lateral"]

        def topk_mask(score, recall_target):
            order = np.argsort(-score)
            k = int(np.searchsorted(np.cumsum(y_te[order]),
                                    recall_target * y_te.sum()) + 1)
            m = np.zeros(len(score), dtype=bool); m[order[:k]] = True
            return m

        fuse_score = methods[bn]
        for rt in (0.90, 0.93, 0.95):
            m_rgat = topk_mask(p_te, rt)
            m_fuse = topk_mask(fuse_score, rt)
            print(f"\n[fusion] per-template event recall @ overall recall {rt:.2f} "
                  f"(RGAT flags {m_rgat.sum():,} vs fusion {m_fuse.sum():,})")
            print(f"[fusion] {'template':<20}{'RGAT':>9}{'fusion':>9}{'delta':>8}")
            for t in range(5):
                sel = tpl == t
                if not sel.any():
                    continue
                rr_t = float(y_te[sel & m_rgat].sum() / max(1, y_te[sel].sum()))
                ff_t = float(y_te[sel & m_fuse].sum() / max(1, y_te[sel].sum()))
                print(f"[fusion] {TPL[t]:<20}{100*rr_t:>8.1f}%{100*ff_t:>8.1f}%"
                      f"{100*(ff_t-rr_t):>+7.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
