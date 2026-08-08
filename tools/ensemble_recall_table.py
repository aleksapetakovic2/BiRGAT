#!/usr/bin/env python
"""Operating-point table for an RGAT ensemble.

Averages the members' probabilities (thresholds tuned on VALIDATION only,
applied to test) and shows what each recall floor costs — the deployment
decision view. Usage:

    python tools/ensemble_recall_table.py runs/<a> runs/<b> [...]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve

from eprgat.config import Config
from eprgat.evaluate import _eval_with, _load_model
from eprgat.graph import prepare_dataset


def main():
    run_dirs = sys.argv[1:]
    if len(run_dirs) < 1:
        raise SystemExit("usage: ensemble_recall_table.py runs/<a> [...]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config.load(os.path.join(run_dirs[0], "config.yaml"))
    bundle = prepare_dataset(cfg, log=lambda m: None)

    vy = ty = None
    val_probs, test_probs = [], []
    for rd in run_dirs:
        c = Config.load(os.path.join(rd, "config.yaml"))
        model, _ = _load_model(c, bundle, os.path.join(rd, "best.pt"), device)
        y1, p1, _ = _eval_with(model, bundle, c, 1, device)
        y2, p2, _ = _eval_with(model, bundle, c, 2, device)
        if vy is None:
            vy, ty = y1, y2
        val_probs.append(p1)
        test_probs.append(p2)
    vp = np.mean(val_probs, axis=0)
    tp = np.mean(test_probs, axis=0)
    n_pos_test = int(ty.sum())

    # thresholds from VAL for a set of recall floors
    prec_v, rec_v, thr_v = precision_recall_curve(vy, vp)
    prec_v, rec_v = prec_v[:-1], rec_v[:-1]   # drop artificial last point

    print(f"ensemble of {len(run_dirs)} models | test events={len(ty):,} "
          f"positives={n_pos_test}")
    print(f"{'policy':<22}{'thr':>7}{'recall':>8}{'prec':>8}{'TP':>6}{'FN':>6}"
          f"{'FP':>8}{'flagged %':>11}")

    # F1-optimal on val
    f1 = np.where(prec_v + rec_v > 0,
                  2 * prec_v * rec_v / np.maximum(prec_v + rec_v, 1e-12), 0)
    i = int(np.argmax(f1))
    rows = [("F1-optimal", float(thr_v[i]))]
    for floor in (0.90, 0.93, 0.95, 0.97):
        ok = np.where(rec_v >= floor)[0]
        rows.append((f"recall >= {floor:.0%}",
                     float(thr_v[int(ok[-1])]) if len(ok) else float(thr_v[0])))
    for name, thr in rows:
        pred = (tp >= thr).astype(int)
        tpi = int((pred & (ty == 1)).sum())
        fpi = int((pred & (ty == 0)).sum())
        fni = n_pos_test - tpi
        rec = tpi / n_pos_test
        prec = tpi / max(1, tpi + fpi)
        print(f"{name:<22}{thr:>7.3f}{rec:>8.3f}{prec:>8.3f}{tpi:>6}{fni:>6}"
              f"{fpi:>8,}{100*(tpi+fpi)/len(ty):>10.2f}%")


if __name__ == "__main__":
    main()
