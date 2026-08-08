#!/usr/bin/env python
"""Structural diagnosis: how much label signal does the provenance graph expose?

Read-only analysis of the dataset. Nothing computed here is fed to any model —
this script answers "is the topological signal reachable at all, and where":

1. per-relation degree structure of positives vs negatives;
2. label mixing: how positive are the neighbourhoods of positive/negative nodes;
3. k-hop reachability (backward = causal past, forward = consequences):
   which positives have other positives reachable at all, and how many
   negatives are contaminated by nearby positives (the FP risk of topology);
4. naive structural-stat AUCs (degrees etc.) — strength of local topo signal;
5. sampler coverage: how much of a seed's true backward ball does the
   fanout-capped sampler actually retrieve (train and eval fanouts).

Run from the project root:  python tools/diagnose_topology.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score

from eprgat.config import Config
from eprgat.graph import prepare_dataset
from eprgat.sampling import BalancedGraphSamplers
from eprgat.schema import RELATIONS

HOPS = 4


def fmt_row(name, pos_vals, neg_vals):
    def stats(v):
        v = np.asarray(v, dtype=np.float64)
        return (f"mean={v.mean():8.3f}  p50={np.percentile(v, 50):7.2f}  "
                f"p95={np.percentile(v, 95):7.2f}  frac>0={100*(v > 0).mean():5.1f}%")
    print(f"  {name:<22} POS {stats(pos_vals)}")
    print(f"  {'':<22} NEG {stats(neg_vals)}")


def main():
    cfg = Config.load(os.path.join("configs", "full.yaml"))
    print("=" * 78)
    print("TOPOLOGY DIAGNOSTIC (train split; read-only, no model involved)")
    print("=" * 78)
    bundle = prepare_dataset(cfg, log=lambda m: None)
    d = bundle.data["event"]
    y = d.y.numpy().astype(np.int64)
    split = d.split.numpy()
    N = d.num_nodes
    print(f"X feature dim = {d.x.shape[1]}   nodes={N:,}   train pos={int(((split==0)&(y==1)).sum()):,}")

    tr = split == 0
    pos = tr & (y == 1)
    neg = tr & (y == 0)
    pos_idx = np.where(pos)[0]
    neg_idx = np.where(neg)[0]
    rng = np.random.default_rng(0)
    neg_sample = rng.choice(neg_idx, size=min(6000, len(neg_idx)), replace=False)

    # ------------------------------------------------------------------ #
    print("\n[1] per-relation structure (train-split edges only)")
    mats = {}
    for rel in RELATIONS:
        ei = bundle.data["event", rel, "event"].edge_index.numpy()
        m = tr[ei[0]] & tr[ei[1]]
        A = sparse.csr_matrix((np.ones(int(m.sum()), np.float32),
                               (ei[1][m], ei[0][m])), shape=(N, N))
        A.data[:] = 1.0                       # binarise any duplicates
        mats[rel] = A
        indeg = np.asarray(A.sum(axis=1)).ravel()
        outdeg = np.asarray(A.sum(axis=0)).ravel()
        fmt_row(rel + " in-deg", indeg[pos_idx], indeg[neg_sample])
        print(f"  {'':<22} edges={int(m.sum()):,}")

    # ------------------------------------------------------------------ #
    print("\n[2] label mixing per relation (fraction of in-neighbours that are positive)")
    yf = y.astype(np.float32)
    for rel, A in mats.items():
        pos_in = np.asarray(A.dot(yf)).ravel()
        deg = np.asarray(A.sum(axis=1)).ravel()
        has = deg > 0
        pp = pos_in[pos & has] / deg[pos & has]
        pn = pos_in[neg & has] / deg[neg & has]
        print(f"  {rel:<22} pos-dst purity: mean={pp.mean():.3f} frac(any pos nb)={100*(pos_in[pos&has]>0).mean():.1f}%"
              f"   | neg-dst contaminated: mean={pn.mean():.4f} frac(any pos nb)={100*(pos_in[neg&has]>0).mean():.2f}%")

    # ------------------------------------------------------------------ #
    print(f"\n[3] k-hop reachability to OTHER positives, k=1..{HOPS} "
          "(backward = causal past, forward = consequences)")

    def per_seed(seeds, hops, forward):
        """Returns (ball_size_excl_seed, n_pos_in_ball_excl_seed) per seed.
        A rows are dst, cols src: A.T @ b expands to the past (in-edges),
        A @ b expands to the future (out-edges)."""
        sizes = np.zeros(len(seeds), dtype=np.int64)
        npos = np.zeros(len(seeds), dtype=np.int64)
        for i, s in enumerate(seeds):
            b = np.zeros(N, dtype=np.float32); b[s] = 1.0
            visited = b.copy()
            for _ in range(hops):
                nb = np.zeros(N, dtype=np.float32)
                for r in RELATIONS:
                    op = mats[r] if forward else mats[r].T
                    nb += op.dot(b)
                nb = (nb > 0).astype(np.float32) * (1.0 - visited)
                if not nb.any():
                    b = np.zeros(N, dtype=np.float32)
                    break
                visited += nb
                b = nb
            ball = visited
            ball[s] = 0.0
            sizes[i] = int(ball.sum())
            npos[i] = int(ball.dot(yf))
        return sizes, npos

    t0 = time.time()
    b_sz, b_np = per_seed(pos_idx, HOPS, forward=False)
    f_sz, f_np = per_seed(pos_idx, 2, forward=True)
    fb_sz, fb_np = per_seed(neg_sample, HOPS, forward=False)
    dt = time.time() - t0

    print(f"  POS backward k={HOPS}: frac reaching >=1 other pos = {100*(b_np>0).mean():.1f}%   "
          f"mean pos-in-ball={b_np.mean():.1f}  mean ball size={b_sz.mean():.0f}  [{dt:.0f}s]")
    print(f"  POS forward  k=2: frac reaching >=1 other pos = {100*(f_np>0).mean():.1f}%   "
          f"mean pos-in-ball={f_np.mean():.1f}")
    print(f"  NEG backward k={HOPS}: frac with any pos in past  = {100*(fb_np>0).mean():.2f}%   "
          f"mean pos-in-ball={fb_np.mean():.3f}")
    print(f"  POS backward k={HOPS}: pos NEVER reachable from past: {int((b_np==0).sum())}/{len(b_np)}"
          f"  -> these seeds can only be caught via own features or future context")

    # ------------------------------------------------------------------ #
    print("\n[4] naive structural stats — single-stat AUC on train (diagnostic only)")
    stats = {}
    for rel, A in mats.items():
        stats[f"indeg_{rel}"] = np.asarray(A.sum(axis=1)).ravel()[tr]
        stats[f"outdeg_{rel}"] = np.asarray(A.sum(axis=0)).ravel()[tr]
    yt = y[tr]
    rows = []
    for name, v in stats.items():
        if v.min() == v.max():
            continue
        a = roc_auc_score(yt, v)
        rows.append((name, max(a, 1 - a)))
    rows.sort(key=lambda t: -t[1])
    for name, a in rows[:10]:
        print(f"  {name:<32} AUC={a:.3f}")

    # ------------------------------------------------------------------ #
    print("\n[5] sampler coverage of the true backward ball (pos seeds)")
    sampler = BalancedGraphSamplers(
        bundle.data, split, y, cfg.sampling.fanouts, cfg.sampling.batch_seeds,
        cfg.sampling.pos_seed_frac, cfg.sampling.eval_batch_seeds,
        cfg.sampling.eval_fanout_mult, cfg.sampling.num_workers,
        seed=cfg.data.seed, max_frontier=cfg.sampling.max_frontier,
        min_steps_per_epoch=1)
    sample_seeds = rng.choice(pos_idx, size=min(300, len(pos_idx)), replace=False)
    hops = max(len(f) for f in cfg.sampling.fanouts.values())

    def full_backward_ball(s, depth):
        b = np.zeros(N, dtype=np.float32); b[s] = 1.0
        visited = b.copy()
        for _ in range(depth):
            nb = np.zeros(N, dtype=np.float32)
            for r in RELATIONS:
                nb += mats[r].T.dot(b)
            nb = (nb > 0).astype(np.float32) * (1.0 - visited)
            if not nb.any():
                break
            visited += nb
            b = nb
        visited[s] = 0.0
        return np.where(visited > 0)[0]

    cov_train, cov_eval, poscap_train, poscap_eval = [], [], [], []
    ball_sizes = []
    t0 = time.time()
    for s in sample_seeds:
        ball = full_backward_ball(int(s), hops)
        ball_set = set(ball.tolist())
        ball_sizes.append(len(ball_set))
        pos_in_ball = y[ball].sum()

        sampler.sample(np.array([s]), cfg.sampling.fanouts)
        sm = sampler.last_order
        sm_set = set(sm.tolist()) - {int(s)}
        cov_train.append(len(sm_set & ball_set) / max(1, len(ball_set)))
        poscap_train.append(y[list(sm_set)].sum() / max(1, pos_in_ball) if pos_in_ball else 1.0)

        sampler.sample(np.array([s]), sampler.eval_fanouts)
        sm = set(sampler.last_order.tolist()) - {int(s)}
        cov_eval.append(len(sm & ball_set) / max(1, len(ball_set)))
        poscap_eval.append(y[list(sm)].sum() / max(1, pos_in_ball) if pos_in_ball else 1.0)
    print(f"  true {hops}-hop backward ball: mean size={np.mean(ball_sizes):,.0f}  "
          f"p95={np.percentile(ball_sizes, 95):,.0f}  [{time.time()-t0:.0f}s]")
    print(f"  TRAIN fanouts: mean node coverage={100*np.mean(cov_train):.1f}%   "
          f"mean fraction of ball-positives captured={100*np.mean(poscap_train):.1f}%")
    print(f"  EVAL  fanouts: mean node coverage={100*np.mean(cov_eval):.1f}%   "
          f"mean fraction of ball-positives captured={100*np.mean(poscap_eval):.1f}%")
    print(f"  frontier truncations during this section: {sampler.frontier_truncations}")
    print("\nNote: sampled subgraphs are capped by fanouts by design (GraphSAGE-style);")
    print("coverage << 100% with tiny balls is expected; the interesting numbers are")
    print("the positive-capture rate and the reachability rates in section [3].")


if __name__ == "__main__":
    main()
