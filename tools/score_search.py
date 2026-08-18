#!/usr/bin/env python
"""Combiner search over the persisted channel scores (tools/dump_scores.py).

Tries many fusions of the RGAT event score with the aggregate/sequence entity
channels — cheap, because the expensive scoring already happened — and reports
the recall-vs-flagged curve plus the per-template catch. The aim: reach >=90%
recall at minimal flagged cost WITHOUT dropping valid-account (T3).

Run:  python tools/score_search.py runs/<run_dir>/scores.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

TPL = ["T1_phishing", "T2_exploit_web", "T3_valid_account",
       "T4_beacon_persist", "T5_service_lateral"]


def auprc(y, s):
    from sklearn.metrics import average_precision_score
    return average_precision_score(y, s)


def flagged_at(y, score, targets=(0.90, 0.93, 0.95, 0.97)):
    order = np.argsort(-score)
    cum = np.cumsum(y[order]); totpos = int(y.sum()); n = len(y)
    return {t: 100.0 * (int(np.searchsorted(cum, t * totpos) + 1)) / n
            for t in targets}


def topk_mask(y, score, rt):
    order = np.argsort(-score)
    k = int(np.searchsorted(np.cumsum(y[order]), rt * y.sum()) + 1)
    m = np.zeros(len(score), bool); m[order[:k]] = True
    return m


def template_recall(y_te, tpl, mask):
    out = {}
    for t in range(5):
        sel = tpl == t
        if sel.any():
            out[t] = 100.0 * y_te[sel & mask].sum() / max(1, y_te[sel].sum())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--recall", type=float, default=0.93,
                    help="recall level for the per-template table")
    args = ap.parse_args()

    d = np.load(args.npz)
    y_va, y_te = d["y_va"], d["y_te"]; tpl = d["tpl_te"]
    p_va, p_te = d["p_va"], d["p_te"]
    chan = sorted({k[:-3] for k in d.files
                   if k.endswith("_te") and k[:-3] not in ("p", "y", "tpl")})
    print(f"[search] channels: {chan}")
    print(f"[search] test events={len(y_te):,} positives={int(y_te.sum())}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier

    def fit_combo(feats_va, feats_te, kind):
        if kind == "log":
            m = LogisticRegression(class_weight="balanced", C=1.0)
        else:
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                               max_depth=3, min_samples_leaf=25,
                                               class_weight="balanced",
                                               random_state=0)
        m.fit(feats_va, y_va)
        return m.predict_proba(feats_te)[:, 1]

    methods = {"RGAT-only": p_te}
    for c in chan:
        cva, cte = d[f"{c}_va"], d[f"{c}_te"]
        methods[f"noisyOR(p+{c})"] = 1 - (1 - p_te) * (1 - cte)

    # every subset of the base channels (drop the redundant aggmax for subsets)
    base = [c for c in chan if c != "aggmax"]
    from itertools import combinations
    for r in range(1, len(base) + 1):
        for sub in combinations(base, r):
            name = "+".join(sub)
            fv = np.stack([p_va] + [d[f"{c}_va"] for c in sub], 1)
            ft = np.stack([p_te] + [d[f"{c}_te"] for c in sub], 1)
            for kind in ("log", "gbm"):
                methods[f"{kind}(p,{name})"] = fit_combo(fv, ft, kind)

    # ---------------- overall ranking table
    rows = []
    for name, s in methods.items():
        fa = flagged_at(y_te, s); a = auprc(y_te, s)
        rows.append((name, a, fa))
    rows.sort(key=lambda r: -r[1])
    print(f"\n[search] {'method':<24} {'AUPRC':>8} | flagged% @R=")
    print(f"[search] {'':<24} {'':>8} | {'0.90':>6} {'0.93':>6} {'0.95':>6} {'0.97':>6}")
    for name, a, fa in rows:
        print(f"[search] {name:<24} {a:>8.4f} | " +
              " ".join(f"{fa[t]:>5.2f}%" for t in (0.90, 0.93, 0.95, 0.97)))

    # ---------------- per-template at the chosen recall, curated frontier view
    # metrics per method at the chosen recall
    stats = {}
    for name, s in methods.items():
        mk = topk_mask(y_te, s, args.recall)
        tr = template_recall(y_te, tpl, mk)
        stats[name] = {"flag": 100.0 * mk.mean(),
                       "va": tr.get(2, float("nan")),   # T3 valid_account
                       "be": tr.get(3, float("nan")),   # T4 beacon
                       "auprc": auprc(y_te, s)}
    min_flag = min(v["flag"] for v in stats.values())
    by_auprc = sorted(stats, key=lambda k: -stats[k]["auprc"])
    # efficient methods that preserve valid-account best
    eff = [k for k in stats if stats[k]["flag"] <= max(2.0, 1.5 * min_flag)]
    by_va = sorted(eff, key=lambda k: -stats[k]["va"])
    keys = ["RGAT-only"] + by_auprc[:2] + [k for k in by_va[:2] if k not in by_auprc[:2]]
    keys = list(dict.fromkeys(keys))[:6]

    print(f"\n[search] per-template event recall @ overall recall {args.recall:.2f}")
    print(f"[search] (curated: best-AUPRC + best valid-account-preserving efficient)")
    hdr = f"[search] {'template':<18}" + "".join(f"{k[:20]:>22}" for k in keys)
    print(hdr)
    masks = {k: topk_mask(y_te, methods[k], args.recall) for k in keys}
    for t in range(5):
        row = f"[search] {TPL[t]:<18}"
        for k in keys:
            tr = template_recall(y_te, tpl, masks[k])
            row += f"{tr.get(t, float('nan')):>21.1f}%"
        print(row)
    print("[search] " + "-" * (18 + 22 * len(keys)))
    row = f"[search] {'flagged %':<18}"
    for k in keys:
        row += f"{stats[k]['flag']:>20.2f}%"
    print(row)
    row = f"[search] {'AUPRC':<18}"
    for k in keys:
        row += f"{stats[k]['auprc']:>21.4f}"
    print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
