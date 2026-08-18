#!/usr/bin/env python
"""Compute + persist every detection channel's val/test scores once, so that
combiner search (tools/score_search.py) can try many fusions WITHOUT re-running
the expensive RGAT eval or sequence training.

Channels saved (all temporal-honest: entity branches trained on the train
period only and scored forward; RGAT is the frozen production checkpoint):
  p        RGAT per-event probability
  agg_<w>  window-aggregate entity score at window w (host+user max)
  seq_<w>  sequence entity score at window w (host+user max)

Saved arrays (per split): the channel columns, the label y, and template id.
Run:  python tools/dump_scores.py runs/<run_dir> [--out runs/<run>/scores.npz]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from eprgat.config import Config
from eprgat.synthetic import generate_dataset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--agg-scales", default="120,480")
    ap.add_argument("--seq-scales", default="240")
    ap.add_argument("--maxlen", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from eprgat.evaluate import _eval_with, _load_model
    from eprgat.graph import prepare_dataset, temporal_split
    from entity_probe import build_entity_windows
    from fusion_probe import entity_temporal_scores
    from sequence_probe import build_entity_sequences, train_sequence_model

    cfg = Config.load(os.path.join(args.run, "config.yaml"))
    device = torch.device(args.device)
    out = args.out or os.path.join(args.run, "scores.npz")

    tab = generate_dataset(cfg)
    bundle = prepare_dataset(cfg, log=lambda m: None)
    n = len(tab); c = tab.cols
    split = bundle.data["event"].split.numpy()
    y_all = bundle.data["event"].y.numpy()
    Xstd = (bundle.data["event"].x.numpy() - bundle.X_mean) / bundle.X_std
    X_t = torch.from_numpy(Xstd.astype(np.float32))

    va_idx = np.where(split == 1)[0]; te_idx = np.where(split == 2)[0]

    # ---------------- RGAT event scores
    model, _ = _load_model(cfg, bundle, os.path.join(args.run, "best.pt"), device)
    _, p_va, _ = _eval_with(model, bundle, cfg, 1, device)
    _, p_te, _ = _eval_with(model, bundle, cfg, 2, device)
    print(f"[dump] RGAT done")

    def node_from_grids(grids, window_min):
        n_win = int(np.ceil((c["ts"].max() + 1.0) / window_min))
        widx = np.clip((c["ts"] // window_min).astype(np.int64), 0, n_win - 1)
        out_arr = np.zeros(n)
        for entity, (sc, ent_id, win_id) in grids.items():
            nent = tab.n_hosts if entity == "host" else tab.n_users
            grid = np.zeros(nent * n_win); grid[ent_id * n_win + win_id] = sc
            ent = c["host"] if entity == "host" else c["user"]
            valid = ent >= 0
            out_arr[valid] = np.maximum(out_arr[valid],
                                        grid[ent[valid] * n_win + widx[valid]])
        return out_arr

    channels = {"p_va": p_va, "p_te": p_te}

    # ---------------- aggregate entity channels (multi-scale)
    for w in [float(x) for x in args.agg_scales.split(",")]:
        grids = {}
        for entity in ("host", "user"):
            Xa, lab, ent_id, win_id, _ = build_entity_windows(tab, entity, w)
            n_win = int(np.ceil((c["ts"].max() + 1.0) / w))
            ws = temporal_split((np.arange(n_win) + 0.5) * w, cfg.graph)
            grids[entity] = (entity_temporal_scores(Xa, lab, win_id, ws, args.seed),
                             ent_id, win_id)
        node = node_from_grids(grids, w)
        channels[f"agg{int(w)}_va"] = node[va_idx]
        channels[f"agg{int(w)}_te"] = node[te_idx]
        print(f"[dump] agg@{int(w)} done")

    # combined multi-scale agg (max) for convenience
    agg_names = [k for k in channels if k.startswith("agg")]
    va_agg = np.maximum.reduce([channels[k] for k in agg_names if k.endswith("_va")])
    te_agg = np.maximum.reduce([channels[k] for k in agg_names if k.endswith("_te")])
    channels["aggmax_va"], channels["aggmax_te"] = va_agg, te_agg

    # ---------------- sequence entity channels
    for w in [float(x) for x in args.seq_scales.split(",")]:
        grids = {}
        for entity in ("host", "user"):
            seqs, lab, ent_id, win_id, n_win = build_entity_sequences(
                tab, entity, w, args.maxlen)
            ws = temporal_split((np.arange(n_win) + 0.5) * w, cfg.graph)
            sc, ap_s = train_sequence_model(seqs, lab, ws[win_id], X_t, cfg,
                                            device, args.seed, epochs=args.epochs)
            grids[entity] = (sc, ent_id, win_id)
            print(f"[dump] seq@{int(w)}[{entity}] AUPRC={ap_s:.4f}")
        node = node_from_grids(grids, w)
        channels[f"seq{int(w)}_va"] = node[va_idx]
        channels[f"seq{int(w)}_te"] = node[te_idx]

    # ---------------- labels + template for the test events
    # template comes from viz_data if present, else -1
    viz = os.path.join(args.run, "viz_data.json")
    if os.path.exists(viz):
        import json
        tpl_te = np.array(json.load(open(viz))["events"]["tpl"])
    else:
        tpl_te = np.full(len(te_idx), -1)

    save = {**channels,
            "y_va": y_all[va_idx], "y_te": y_all[te_idx],
            "tpl_te": tpl_te}
    np.savez_compressed(out, **save)
    print(f"[dump] wrote {out}  channels={[k for k in channels if k.endswith('_te')]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
