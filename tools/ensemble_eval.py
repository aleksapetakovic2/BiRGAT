#!/usr/bin/env python
"""Ensemble evaluation: average the predicted probabilities of N trained RGAT
runs that share the SAME dataset (they may differ in train.seed / init).

Ensembling independently trained RGATs is a standard, honest way to reduce
model-variance: no labels, features, or splits are touched — only the scores
of several fully-trained bidirectional RGATs are averaged. The threshold is
re-tuned on the ENSEMBLE's validation scores and applied to test.

Usage:
    python tools/ensemble_eval.py runs/<a> runs/<b> [runs/<c> ...]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import torch

from eprgat.config import Config
from eprgat.evaluate import _eval_with, _load_model, rewire_graph
from eprgat.graph import prepare_dataset
from eprgat.metrics import compute_metrics, format_metrics, search_threshold


def main():
    run_dirs = sys.argv[1:]
    if len(run_dirs) < 2:
        raise SystemExit("usage: ensemble_eval.py runs/<a> runs/<b> [...]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = Config.load(os.path.join(run_dirs[0], "config.yaml"))
    for rd in run_dirs[1:]:
        c = Config.load(os.path.join(rd, "config.yaml"))
        if c.data != cfg.data or c.graph != cfg.graph:
            raise SystemExit(f"{rd}: dataset/graph config differs from "
                             f"{run_dirs[0]} — ensembles must share data")
    bundle = prepare_dataset(cfg, log=lambda m: None)

    models, cfgs = [], []
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
        models.append(model)
        cfgs.append(c)
        print(f"[ens ] scored {rd}")

    vp = np.mean(val_probs, axis=0)
    tp = np.mean(test_probs, axis=0)

    print("\n" + "=" * 66)
    print(f"ENSEMBLE of {len(run_dirs)} models")
    for policy in ("f1", "recall"):
        thr = search_threshold(vy, vp, policy=policy,
                               min_recall=cfg.train.threshold_min_recall)
        print(format_metrics(compute_metrics(ty, tp, thr),
                             f"TEST ensemble, operating point: {policy}"))

    # rewire ablation on the ensemble (same permutation for every member)
    rewired = rewire_graph(bundle, seed=cfg.data.seed + 1)
    rp = []
    for model, c in zip(models, cfgs):
        _, p, _ = _eval_with(model, rewired, c, 2, device)
        rp.append(p)
    rp = np.mean(rp, axis=0)
    thr = search_threshold(vy, vp, policy="f1",
                           min_recall=cfg.train.threshold_min_recall)
    print(format_metrics(compute_metrics(ty, rp, thr),
                         "TEST ensemble, edges REWIRED"))
    print("=" * 66)


if __name__ == "__main__":
    main()
