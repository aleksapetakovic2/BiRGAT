"""Leak diagnostic: where does the per-event class signal actually live?

Prints, on the TRAIN split:
* top single-feature AUCs (named);
* per-class mean/p25/p75 for the worst features;
* hour-of-day occupancy for positives vs negatives;
* event-type mix per class.
Run: python tools/diagnose_leak.py --config configs/full.yaml
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import numpy as np
from sklearn.metrics import roc_auc_score

from eprgat.config import Config
from eprgat.graph import prepare_dataset
from eprgat.schema import FEATURE_BLOCKS, FEATURE_OFFSETS


def main(config_path: str) -> None:
    cfg = Config.load(config_path)
    bundle = prepare_dataset(cfg, log=lambda m: None)
    X = bundle.data["event"].x.numpy()
    y = bundle.data["event"].y.numpy()
    ts = bundle.data["event"].ts.numpy()
    split = bundle.data["event"].split.numpy()
    tr = split == 0
    X, y, ts = X[tr], y[tr], ts[tr]
    pos, neg = y == 1, y == 0

    print(f"train split: n={len(y):,}  pos={int(y.sum()):,} ({100*y.mean():.2f}%)")

    # ------------------------------------------------------------------ #
    scores = []
    for j in range(X.shape[1]):
        col = X[:, j]
        if col.min() == col.max():
            continue
        auc = roc_auc_score(y, col)
        auc = max(auc, 1.0 - auc)
        scores.append((j, auc))
    scores.sort(key=lambda t: -t[1])

    def feat_name(j):
        blk = next(b for b in FEATURE_BLOCKS
                   if FEATURE_OFFSETS[b.name][0] <= j < FEATURE_OFFSETS[b.name][1])
        return f"{blk.name}[{j - FEATURE_OFFSETS[blk.name][0]}]"

    print("\ntop-15 single-feature AUCs:")
    for j, auc in scores[:15]:
        p = X[pos, j]; q = X[neg, j]
        print(f"  {feat_name(j):<28} AUC={auc:.3f}   "
              f"pos mean={p.mean():.3f} [{np.percentile(p,25):.3f},{np.percentile(p,75):.3f}]   "
              f"neg mean={q.mean():.3f} [{np.percentile(q,25):.3f},{np.percentile(q,75):.3f}]")

    # ------------------------------------------------------------------ #
    hour = (ts % 1440.0) / 60.0
    print("\nhour-of-day occupancy (share of events in each 4h block):")
    bins = np.arange(0, 28, 4)
    hp, _ = np.histogram(hour[pos], bins=bins)
    hn, _ = np.histogram(hour[neg], bins=bins)
    hp = hp / max(1, hp.sum()); hn = hn / max(1, hn.sum())
    for i in range(len(hp)):
        print(f"  {bins[i]:>2}:00-{bins[i+1]:>2}:00   pos {100*hp[i]:5.2f}%   "
              f"neg {100*hn[i]:5.2f}%")

    # ------------------------------------------------------------------ #
    etype_off = FEATURE_OFFSETS["etype_onehot"]
    names = ["SignIn", "ProcessCreate", "NetworkConnection", "FileActivity",
             "SystemConfig", "DnsQuery"]
    print("\nevent-type mix:")
    for k, nm in enumerate(names):
        col = X[:, etype_off[0] + k]
        print(f"  {nm:<18} pos {100*col[pos].mean():6.2f}%   "
              f"neg {100*col[neg].mean():6.2f}%")

    # ------------------------------------------------------------------ #
    # per-block joint AUC via a tiny logistic probe (feature blocks)
    print("\nper-block logistic-probe AUC (joint signal inside one block):")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    for b in FEATURE_BLOCKS:
        s, e = FEATURE_OFFSETS[b.name]
        if e - s == 0:
            continue
        xb = X[:, s:e]
        if np.allclose(xb, 0):
            continue
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=400, C=1.0))
        m.fit(xb, y)
        p = m.predict_proba(xb)[:, 1]
        a = roc_auc_score(y, p)
        print(f"  {b.name:<24} AUC={max(a, 1-a):.3f}")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv
         else "configs/full.yaml")
