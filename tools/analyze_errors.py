#!/usr/bin/env python
"""Error analysis: who are the positives the model misses?

Analysis-only: reconstructs incident metadata (template, chain position) for
the exact generated world of a run, joins it with the model's test
probabilities and splits the positives into caught vs missed at the run's
recall-first threshold. Nothing computed here feeds back into training.

Usage:  python tools/analyze_errors.py runs/<run_dir>
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import torch
from scipy import sparse

from eprgat.config import Config
from eprgat.evaluate import _eval_with, _load_model
from eprgat.graph import prepare_dataset
from eprgat.schema import RELATIONS
from eprgat.synthetic import SentinelEventGenerator, generate_dataset

TPL_NAMES = ["T1_phishing", "T2_exploit_web", "T3_valid_account",
             "T4_beacon_persist", "T5_service_lateral"]


class _RngProxy:
    """Delegates to the real Generator but records the template draw — the
    first choice(5, p=[5 probs]) call inside _attack_events."""

    def __init__(self, rng, state):
        self._rng = rng
        self._state = state

    def choice(self, a, *args, **kw):
        res = self._rng.choice(a, *args, **kw)
        if ("tpl" not in self._state and np.ndim(a) == 0 and int(a) == 5
                and len(kw.get("p", [])) == 5):
            self._state["tpl"] = int(res)
        return res

    def __getattr__(self, name):
        return getattr(self._rng, name)


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "runs/20260807_135514_full_rgat"
    cfg = Config.load(os.path.join(run, "config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------ incident metadata
    captured = {}
    orig_ae = SentinelEventGenerator._attack_events

    def patched(self, iid, split_target):
        state = {}
        real = self.rng
        self.rng = _RngProxy(real, state)
        try:
            orig_ae(self, iid, split_target)
        finally:
            self.rng = real
        captured[iid] = state.get("tpl", -1)

    SentinelEventGenerator._attack_events = patched
    tab = generate_dataset(cfg)
    SentinelEventGenerator._attack_events = orig_ae

    cols = tab.cols
    n = len(tab)
    y = cols["y"].astype(np.int64)
    inc = cols["incident"]
    tpl_of_event = np.full(n, -1, dtype=np.int64)
    for iid, tpl in captured.items():
        tpl_of_event[(inc == iid) & (y == 1)] = tpl

    # position within the incident chain (table is sorted by ts)
    pos_frac = np.full(n, -1.0)
    for iid in np.unique(inc[y == 1]):
        m = np.where((inc == iid) & (y == 1))[0]
        pos_frac[m] = np.arange(len(m)) / max(1, len(m) - 1)

    # ------------------------------------------------ model predictions
    bundle = prepare_dataset(cfg, log=lambda m: None)
    assert bundle.data["event"].x.shape[0] == n, "event order mismatch"
    split = bundle.data["event"].split.numpy()

    report = json.load(open(os.path.join(run, "eval_report.json")))
    thr = report["test_alt_recall"]["threshold"]
    print(f"run={run}\nrecall-first threshold (from val): {thr:.3f} "
          f"(reported test recall {report['test_alt_recall']['recall']:.3f})")

    model, _ = _load_model(cfg, bundle, os.path.join(run, "best.pt"), device)
    ty, tp, _ = _eval_with(model, bundle, cfg, 2, device)

    te_mask = split == 2
    te_idx = np.where(te_mask)[0]
    proba = np.full(n, np.nan)
    proba[te_idx] = tp

    pos_te = np.where(te_mask & (y == 1))[0]
    caught = pos_te[proba[pos_te] >= thr]
    missed = pos_te[proba[pos_te] < thr]
    print(f"\ntest positives: {len(pos_te)}   caught {len(caught)}   missed {len(missed)}")

    # ------------------------------------------------ template mix
    print("\ntemplate mix (share of test positives):")
    print(f"{'template':<20}{'all':>8}{'caught':>9}{'missed':>9}{'miss rate':>11}")
    for t in range(5):
        a = int((tpl_of_event[pos_te] == t).sum())
        c = int((tpl_of_event[caught] == t).sum())
        m = int((tpl_of_event[missed] == t).sum())
        if a == 0:
            continue
        print(f"{TPL_NAMES[t]:<20}{100*a/len(pos_te):7.1f}%{100*c/max(1,len(caught)):8.1f}%"
              f"{100*m/max(1,len(missed)):8.1f}%{100*m/max(1,a):10.1f}%")
    other = int((tpl_of_event[pos_te] < 0).sum())
    if other:
        print(f"{'(unknown)':<20}{other:>8}")

    # ------------------------------------------------ chain position
    print("\nchain-relative position of missed vs all positives:")
    for lo, hi in ((0, .25), (.25, .5), (.5, .75), (.75, 1.01)):
        a = int(((pos_frac[pos_te] >= lo) & (pos_frac[pos_te] < hi)).sum())
        m = int(((pos_frac[missed] >= lo) & (pos_frac[missed] < hi)).sum())
        print(f"  [{lo:.2f},{hi:.2f}):  all={100*a/len(pos_te):5.1f}%   "
              f"missed={100*m/max(1,len(missed)):5.1f}%")

    # ------------------------------------------------ reachable context
    print("\ntopological context of test positives (full balls, no sampling):")
    tr = split == 2
    mats = {}
    for rel in RELATIONS:
        ei = bundle.data["event", rel, "event"].edge_index.numpy()
        m2 = tr[ei[0]] & tr[ei[1]]
        mats[rel] = sparse.csr_matrix((np.ones(int(m2.sum()), np.float32),
                                       (ei[1][m2], ei[0][m2])), shape=(n, n))
    yf = y.astype(np.float32)

    def context(seed):
        """(pos in backward-4 ball, pos in forward-2 ball), seed excluded."""
        b = np.zeros(n, dtype=np.float32); b[seed] = 1.0
        seen = b.copy()
        for _ in range(4):
            nb = np.zeros(n, dtype=np.float32)
            for A in mats.values():
                nb += A.T.dot(b)
            nb = (nb > 0).astype(np.float32) * (1.0 - seen)
            if not nb.any():
                break
            seen += nb
            b = nb
        back = float(seen.dot(yf)) - y[seed]
        b = np.zeros(n, dtype=np.float32); b[seed] = 1.0
        seen = b.copy()
        for _ in range(2):
            nb = np.zeros(n, dtype=np.float32)
            for A in mats.values():
                nb += A.dot(b)
            nb = (nb > 0).astype(np.float32) * (1.0 - seen)
            if not nb.any():
                break
            seen += nb
            b = nb
        fwd = float(seen.dot(yf)) - y[seed]
        return back, fwd

    sets = {"caught": caught, "missed": missed}
    stats = {k: ([], []) for k in sets}
    for name, idx in sets.items():
        for s in idx:
            bk, fw = context(int(s))
            stats[name][0].append(bk)
            stats[name][1].append(fw)
    for name, (bk, fw) in stats.items():
        bk, fw = np.array(bk), np.array(fw)
        iso = int(((bk == 0) & (fw == 0)).sum())
        print(f"  {name:<7} n={len(bk):3}  mean pos-in-past={bk.mean():5.2f}  "
              f"mean pos-in-future={fw.mean():5.2f}  "
              f"fully isolated (no pos either way): {iso} ({100*iso/max(1,len(bk)):.0f}%)")

    # ------------------------------------------------ scores of the missed
    pm = proba[missed]
    print("\nmodel probability of the missed positives:")
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        print(f"  p{int(q*100):>2} = {np.quantile(pm, q):.3f}")
    near = int((pm >= thr / 2).sum())
    print(f"  {near}/{len(pm)} missed positives scored >= half the threshold "
          f"(recoverable by better ranking); the rest are far off")

    # worst templates among the isolated misses
    iso_mask = (np.array(stats['missed'][0]) == 0) & (np.array(stats['missed'][1]) == 0)
    if iso_mask.any():
        tpls = tpl_of_event[missed[iso_mask]]
        print("\n  isolated misses by template: " +
              ", ".join(f"{TPL_NAMES[t]}={int((tpls == t).sum())}" for t in range(5)
                        if (tpls == t).any()))


if __name__ == "__main__":
    main()
