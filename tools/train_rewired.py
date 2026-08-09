#!/usr/bin/env python
"""Retraining control for the edge-rewire ablation.

The standard rewire ablation (src/eprgat/evaluate.py) evaluates the TRAINED
model on a destroyed graph. That shows the model's *dependence* on
connectivity, but conflates two explanations:

  (a) the topology carries the signal, vs
  (b) the model merely fitted training-time structure and breaks under any
      structural distribution shift.

The control that separates them trains from scratch on the destroyed graph,
with everything else identical (same config, same seeds). All rewired
connectivity is label-independent by construction, so a rewired-trained model
can only use node features + noise edges. If the intact graph's gain came
from topology, the rewired-trained model should fall back to feature-level
performance (around the features-only MLP baseline).

Rewiring here is stricter than evaluate.rewire_graph:

* destinations are permuted **within each train/val/test split** — no
  cross-split edges are introduced (the eval-time rewire can create them);
* per-source out-degree and the in-degree multiset are preserved exactly
  (destination permutation, not uniform resampling);
* edge features are recomputed for the new pairs (log dt + dt bucket from
  the actual new timestamps; byte volume stays with the source event), so
  the destroyed condition does not also carry inconsistent attributes.

Usage (from the project root):

    python tools/train_rewired.py --config runs/<canonical_run>/config.yaml
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import numpy as np
import torch

from eprgat.cli import (RunLogger, make_run_dir, resolve_device,
                        seed_everything, training_seed)
from eprgat.config import Config, apply_cli_overrides
from eprgat.evaluate import evaluate_run
from eprgat.graph import make_edge_features, prepare_dataset
from eprgat.sampling import BalancedGraphSamplers
from eprgat.schema import RELATIONS
from eprgat.trainer import Trainer


def rewire_within_split(data, split_id: np.ndarray, seed: int) -> int:
    """Destroy topology in place: per relation, per split, permute edge
    destinations. Returns the number of self-loop edges dropped."""
    rng = np.random.default_rng(seed)
    ts = data["event"].ts.numpy()
    dropped_total = 0
    for rel in RELATIONS:
        store = data["event", rel, "event"]
        ei = store.edge_index.numpy()
        E = ei.shape[1]
        if E == 0:
            continue
        src, dst = ei[0].copy(), ei[1].copy()
        attr = store.edge_attr.numpy()
        assert (split_id[src] >= 0).all(), f"{rel}: edges in gap zone"
        assert (split_id[src] == split_id[dst]).all(), f"{rel}: cross-split edge"

        sp = split_id[src]                     # invariant under dst permutation
        new_dst = dst.copy()
        keep = np.ones(E, dtype=bool)
        for s in (0, 1, 2):
            g = np.where(sp == s)[0]
            if len(g) == 0:
                continue
            vals = dst[g]
            nd = rng.permutation(vals)         # preserves the dst multiset
            s_src = src[g]
            for _ in range(50):                # repair self-loops in place
                bad = s_src == nd
                if not bad.any():
                    break
                nd[bad] = vals[rng.integers(0, len(vals), size=int(bad.sum()))]
            bad = s_src == nd                  # drop any leftovers
            if bad.any():
                dropped_total += int(bad.sum())
                keep[g[bad]] = False
                g = g[~bad]
                nd = nd[~bad]
            new_dst[g] = nd

        src, dst = src[keep], new_dst[keep]
        attr = attr[keep]
        # recompute edge features for the new pairs: |dt| + bucket from the
        # actual timestamps; bytes column is inverted from the stored log
        dt_min = np.abs(ts[dst] - ts[src])
        bytes_ = np.expm1(attr[:, 7] * 20.0)
        store.edge_index = torch.from_numpy(
            np.stack([src, dst]).astype(np.int64))
        store.edge_attr = torch.from_numpy(make_edge_features(dt_min, bytes_))
    return dropped_total


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train from scratch on the rewired graph (ablation control).")
    ap.add_argument("--config", required=True,
                    help="config of the canonical run to replicate, e.g. "
                         "runs/20260807_155539_full_rgat/config.yaml")
    ap.add_argument("--rewire-seed", type=int, default=None,
                    help="default: data.seed + 1")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="GROUP.KEY=VALUE",
                    help="override any config value, e.g. --set train.epochs=30")
    args = ap.parse_args()

    overrides = {}
    for item in args.overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects group.key=value, got '{item}'")
        k, v = item.split("=", 1)
        overrides[k.strip()] = v.strip()
    cfg = apply_cli_overrides(Config.load(args.config), overrides)
    cfg.train.run_name = "rewire_retrain"
    rewire_seed = (args.rewire_seed if args.rewire_seed is not None
                   else cfg.data.seed + 1)

    run_dir = make_run_dir(cfg)
    log = RunLogger(run_dir)
    log("=" * 78)
    log("REWIRE RETRAIN CONTROL  |  same config + seeds, topology destroyed")
    log("=" * 78)
    cfg.save(os.path.join(run_dir, "config.yaml"))
    log(f"[ctrl] replicating config: {args.config}")
    seed_everything(training_seed(cfg))
    device = resolve_device(cfg, log)

    bundle = prepare_dataset(cfg, log)
    split_id = bundle.data["event"].split.numpy()
    y = bundle.data["event"].y.numpy()

    log(f"[ctrl] rewiring destinations within splits (seed {rewire_seed}); "
        "degrees preserved, edge dt recomputed, no cross-split edges")
    dropped = rewire_within_split(bundle.data, split_id, rewire_seed)
    log(f"[ctrl] rewired; self-loop edges dropped: {dropped}")

    samplers = BalancedGraphSamplers(
        bundle.data, split_id, y, cfg.sampling.fanouts, cfg.sampling.batch_seeds,
        cfg.sampling.pos_seed_frac, cfg.sampling.eval_batch_seeds,
        cfg.sampling.eval_fanout_mult, cfg.sampling.num_workers,
        seed=training_seed(cfg), max_frontier=cfg.sampling.max_frontier,
        min_steps_per_epoch=cfg.sampling.min_steps_per_epoch,
        reverse_edges=cfg.sampling.reverse_edges)
    trainer = Trainer(cfg, bundle, samplers, run_dir, device, log)
    trainer.train()

    # threshold tuned on the REWIRED val, applied to the REWIRED test: this
    # model sees the same (destroyed) condition at train and eval time, so
    # its score cannot be blamed on distribution shift
    evaluate_run(cfg, bundle, run_dir, device, log, do_rewire=False)

    log("[ctrl] done. Arms to compare: intact-trained RGAT (canonical run), "
        "rewired-trained RGAT (this run), features-only MLP baseline.")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
