#!/usr/bin/env python
"""Sequence-level entity probe: attack chains are ORDERED, window aggregates are not.

Valid-account abuse is a short, specific *sequence* by one account
(remote sign-in -> encoded process -> discovery -> file reads -> exfil). The
window-aggregate entity branch (tools/entity_probe.py) scores such a window by
counts/cadence and loses the ordering, which is exactly why valid-account stays
the residual floor. This probe tests whether a small sequence model over each
entity's ordered event stream recovers that signal.

Protocol (matches tools/fusion_probe.py, TEMPORAL-honest):
  * sequence model trains on TRAIN-period entity-windows only, scores val/test;
  * no absolute time, no entity identity — input is the schema feature vector
    of each event in order;
  * scores are fused with the RGAT event score via a combiner fit ONLY on val;
  * reports recall-vs-flagged and the per-template catch, so we can read the
    valid-account line directly.

Analysis-only: nothing retrains or touches the RGAT.

Usage:  python tools/sequence_probe.py runs/<run_dir> [--window 240] [--viz ...]
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


# --------------------------------------------------------------------------- #
# ordered entity-window sequences                                             #
# --------------------------------------------------------------------------- #
def build_entity_sequences(tab, entity: str, window_min: float, maxlen: int):
    """Per (entity, window): ordered node-id list (by ts), capped at maxlen."""
    c = tab.cols
    ts = c["ts"]; n = len(tab)
    ent = c["host"] if entity == "host" else c["user"]
    keep = ent >= 0
    nodes = np.where(keep)[0]
    ts_k = ts[nodes]; ent_k = ent[nodes]
    n_win = int(np.ceil((ts.max() + 1.0) / window_min))
    widx = np.clip((ts_k // window_min).astype(np.int64), 0, n_win - 1)
    key = ent_k * n_win + widx
    order = np.lexsort((ts_k, key))          # stable sort by key, then ts
    nodes = nodes[order]; key = key[order]
    ch = np.flatnonzero(np.diff(key) != 0) + 1
    ukey = key[np.r_[0, ch]]
    ent_id = ukey // n_win; win_id = ukey % n_win
    seqs = []
    for b in np.split(np.arange(len(nodes)), ch):
        ids = nodes[b]
        if len(ids) > maxlen:                # keep the head; attacks are short
            ids = ids[:maxlen]
        seqs.append(ids)
    y = tab.cols["y"]
    label = np.array([int(y[s].max()) for s in seqs], dtype=np.int64)
    return seqs, label, ent_id, win_id, n_win


# --------------------------------------------------------------------------- #
# small sequence model                                                        #
# --------------------------------------------------------------------------- #
def make_model(feat_dim, hidden, device):
    import torch.nn as nn

    class SeqModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(feat_dim, hidden, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(hidden * 2, 32), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(32, 1))

        def forward(self, x, mask):           # x (B,L,F), mask (B,L)
            h, _ = self.gru(x)
            m = mask.unsqueeze(-1).float()
            h = h * m
            mean = h.sum(1) / m.sum(1).clamp(min=1)
            mx = h.masked_fill(~mask.unsqueeze(-1), -1e9).max(1).values
            return self.head(torch.cat([mean, mx], -1)).squeeze(-1)

    return SeqModel().to(device)


def make_model_v2(feat_dim, hidden, device):
    """Sharper variant: bidirectional GRU + learned attention pooling.

    Attention lets the model focus on the suspicious subsequence (e.g. the
    remote-sign-in -> encoded-process -> exfil chain for valid-account abuse)
    instead of averaging it away; the bi-GRU gives each step both its past and
    its consequences."""
    import torch.nn as nn

    class SeqModelV2(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(feat_dim, hidden, batch_first=True, bidirectional=True)
            self.attn = nn.Sequential(
                nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            self.head = nn.Sequential(
                nn.Linear(hidden * 2, 32), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(32, 1))

        def forward(self, x, mask):           # x (B,L,F), mask (B,L)
            h, _ = self.gru(x)                # (B,L,2H)
            a = self.attn(h).squeeze(-1)      # (B,L)
            a = a.masked_fill(~mask, -1e9)
            w = torch.softmax(a, dim=1)       # (B,L)
            ctx = (h * w.unsqueeze(-1)).sum(1)  # (B,2H)
            return self.head(ctx).squeeze(-1)

    return SeqModelV2().to(device)


def collate(seqs, X, device):
    import torch
    L = max(len(s) for s in seqs)
    ids = torch.zeros(len(seqs), L, dtype=torch.long)
    mask = torch.zeros(len(seqs), L, dtype=torch.bool)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.from_numpy(s)
        mask[i, :len(s)] = True
    x = X[ids].to(device)                     # (B,L,F)
    return x, mask.to(device)


def train_sequence_model(seqs, label, wsplit_win, X, cfg, device, seed,
                         epochs=14, hidden=48, batch=64, neg_mult=4,
                         version="v2", focal=True):
    """version: 'v1' mean+max GRU | 'v2' biGRU + attention pooling.
    focal: use focal loss (sharper separation) instead of pos-weighted BCE."""
    import torch
    from sklearn.metrics import average_precision_score
    torch.manual_seed(seed)
    tr = wsplit_win == 0
    pos_idx = np.where(tr & (label == 1))[0]
    neg_idx = np.where(tr & (label == 0))[0]
    if len(pos_idx) == 0:
        return np.full(len(label), label.mean()), float("nan")
    mk = make_model_v2 if version == "v2" else make_model
    model = mk(X.shape[1], hidden, device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    if focal:
        from eprgat.losses import FocalLoss
        lossf = FocalLoss(gamma=2.0, alpha=0.75)
    else:
        pw = torch.tensor([(label[tr] == 0).sum() / max(1, (label[tr] == 1).sum())],
                          device=device)
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    rng = np.random.default_rng(seed)
    model.train()
    for ep in range(epochs):
        neg_samp = rng.choice(neg_idx, size=min(len(neg_idx), neg_mult * len(pos_idx)),
                              replace=False)
        cur = np.concatenate([pos_idx, neg_samp]); rng.shuffle(cur)
        for i in range(0, len(cur), batch):
            b = cur[i:i + batch]
            x, mask = collate([seqs[j] for j in b], X, device)
            yb = torch.from_numpy(label[b].astype(np.float32)).to(device)
            opt.zero_grad()
            loss = lossf(model(x, mask), yb)
            loss.backward(); opt.step()
    model.eval()
    scores = np.zeros(len(label))
    import math
    with torch.no_grad():
        for i in range(0, len(seqs), 256):
            x, mask = collate(seqs[i:i + 256], X, device)
            scores[i:i + 256] = torch.sigmoid(model(x, mask)).cpu().numpy()
    vw = wsplit_win != 0
    ap = average_precision_score(label[vw], scores[vw]) if label[vw].sum() else float("nan")
    return scores, ap


# --------------------------------------------------------------------------- #
def recall_flag_curve(y, score, targets=(0.90, 0.93, 0.95, 0.97)):
    order = np.argsort(-score)
    cum = np.cumsum(y[order]); totpos = int(y.sum()); n = len(y)
    return {t: (int(np.searchsorted(cum, t * totpos) + 1),
                100.0 * (np.searchsorted(cum, t * totpos) + 1) / n) for t in targets}


def auprc(y, s):
    from sklearn.metrics import average_precision_score
    return average_precision_score(y, s)


def main() -> int:
    ap = argparse.ArgumentParser(description="Sequence-level entity probe.")
    ap.add_argument("run")
    ap.add_argument("--window", type=float, default=240.0)
    ap.add_argument("--maxlen", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--version", default="v2", choices=["v1", "v2"],
                    help="seq model: v1 mean+max GRU | v2 biGRU+attention")
    ap.add_argument("--no-focal", action="store_true",
                    help="use pos-weighted BCE instead of focal loss")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--viz", default=None)
    args = ap.parse_args()

    import torch
    from eprgat.evaluate import _eval_with, _load_model
    from eprgat.graph import prepare_dataset, temporal_split

    cfg = Config.load(os.path.join(args.run, "config.yaml"))
    device = torch.device(args.device)
    print(f"[seq] {args.run} window={args.window:.0f}m maxlen={args.maxlen}")

    tab = generate_dataset(cfg)
    bundle = prepare_dataset(cfg, log=lambda m: None)
    n = len(tab); c = tab.cols
    assert bundle.data["event"].x.shape[0] == n
    split = bundle.data["event"].split.numpy()
    y_all = bundle.data["event"].y.numpy()
    # standardised schema features (the model's own input contract)
    X = (bundle.data["event"].x.numpy() - bundle.X_mean) / bundle.X_std

    # ------------------------------------------------ RGAT event scores
    model, _ = _load_model(cfg, bundle, os.path.join(args.run, "best.pt"), device)
    va_idx = np.where(split == 1)[0]; te_idx = np.where(split == 2)[0]
    _, p_va, _ = _eval_with(model, bundle, cfg, 1, device)
    _, p_te, _ = _eval_with(model, bundle, cfg, 2, device)
    y_va = y_all[va_idx]; y_te = y_all[te_idx]

    # ------------------------------------------------ window-aggregate branch
    from entity_probe import build_entity_windows
    from fusion_probe import entity_temporal_scores
    agg_scores = {}
    for entity in ("host", "user"):
        Xa, lab, ent_id, win_id, _ = build_entity_windows(tab, entity, args.window)
        n_win = int(np.ceil((c["ts"].max() + 1.0) / args.window))
        ws = temporal_split((np.arange(n_win) + 0.5) * args.window, cfg.graph)
        agg_scores[entity] = (entity_temporal_scores(Xa, lab, win_id, ws, args.seed),
                              ent_id, win_id, n_win)

    # ------------------------------------------------ sequence branch
    seq_scores = {}
    for entity in ("host", "user"):
        seqs, lab, ent_id, win_id, n_win = build_entity_sequences(
            tab, entity, args.window, args.maxlen)
        ws = temporal_split((np.arange(n_win) + 0.5) * args.window, cfg.graph)
        sc, ap_seq = train_sequence_model(seqs, lab, ws[win_id],
                                          torch.from_numpy(X.astype(np.float32)),
                                          cfg, device, args.seed, epochs=args.epochs,
                                          version=args.version, focal=not args.no_focal)
        print(f"[seq] entity[{entity}] sequence temporal AUPRC={ap_seq:.4f} "
              f"(base {lab.mean():.4f}, {len(seqs):,} windows, {int(lab.sum())} pos)")
        seq_scores[entity] = (sc, ent_id, win_id, n_win)

    # ------------------------------------------------ map to per-node scores
    widx = np.clip((c["ts"] // args.window).astype(np.int64), 0,
                   int(np.ceil((c["ts"].max() + 1.0) / args.window)) - 1)

    def node_score(scores_by_entity):
        out = np.zeros(n)
        for entity in ("host", "user"):
            sc, ent_id, win_id, n_win = scores_by_entity[entity]
            grid = np.zeros((tab.n_hosts if entity == "host" else tab.n_users) * n_win)
            grid[ent_id * n_win + win_id] = sc
            ent = c["host"] if entity == "host" else c["user"]
            valid = ent >= 0
            vals = grid[ent[valid] * n_win + widx[valid]]
            out[valid] = np.maximum(out[valid], vals)
        return out

    seq_node = node_score(seq_scores)
    agg_node = node_score(agg_scores)
    s_va, s_te = seq_node[va_idx], seq_node[te_idx]
    a_va, a_te = agg_node[va_idx], agg_node[te_idx]

    # ------------------------------------------------ fusion + curves
    from sklearn.linear_model import LogisticRegression

    def meta(fv, ft):
        m = LogisticRegression(class_weight="balanced", C=1.0)
        m.fit(fv, y_va); return m.predict_proba(ft)[:, 1]

    methods = {
        "RGAT-only": p_te,
        "fuse(RGAT+agg)": meta(np.stack([p_va, a_va], 1), np.stack([p_te, a_te], 1)),
        "fuse(RGAT+seq)": meta(np.stack([p_va, s_va], 1), np.stack([p_te, s_te], 1)),
        "fuse(RGAT+agg+seq)": meta(np.stack([p_va, a_va, s_va], 1),
                                   np.stack([p_te, a_te, s_te], 1)),
    }
    print(f"\n[seq] TEST: {len(y_te):,} events, {int(y_te.sum())} positives")
    print(f"[seq] {'method':<20} {'AUPRC':>8} | flagged % at recall")
    print(f"[seq] {'':<20} {'':>8} | {'0.90':>7} {'0.93':>7} {'0.95':>7} {'0.97':>7}")
    for name, s in methods.items():
        cu = recall_flag_curve(y_te, s); a = auprc(y_te, s)
        print(f"[seq] {name:<20} {a:>8.4f} | " +
              " ".join(f"{cu[t][1]:>6.2f}%" for t in (0.90, 0.93, 0.95, 0.97)))

    # ------------------------------------------------ per-template
    if args.viz:
        import json
        tpl = np.array(json.load(open(args.viz))["events"]["tpl"])
        TPL = ["T1_phishing", "T2_exploit_web", "T3_valid_account",
               "T4_beacon_persist", "T5_service_lateral"]

        def topk(score, rt):
            order = np.argsort(-score)
            k = int(np.searchsorted(np.cumsum(y_te[order]), rt * y_te.sum()) + 1)
            m = np.zeros(len(score), bool); m[order[:k]] = True; return m

        for rt in (0.93, 0.95):
            print(f"\n[seq] per-template event recall @ overall {rt:.2f}")
            keys = ["RGAT-only", "fuse(RGAT+agg)", "fuse(RGAT+seq)", "fuse(RGAT+agg+seq)"]
            hdr = f"[seq] {'template':<18}" + "".join(f"{k.replace('fuse(RGAT+','').replace(')',''):>12}" for k in keys)
            print(hdr)
            masks = {k: topk(methods[k], rt) for k in keys}
            for t in range(5):
                sel = tpl == t
                if not sel.any():
                    continue
                row = f"[seq] {TPL[t]:<18}"
                for k in keys:
                    r = float(y_te[sel & masks[k]].sum() / max(1, y_te[sel].sum()))
                    row += f"{100*r:>11.1f}%"
                print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
