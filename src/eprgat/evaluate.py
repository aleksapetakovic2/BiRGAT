"""Final evaluation, explainability outputs and honesty ablations.

Outputs per run directory:
* test metrics (F1, AUROC, AUPRC, MCC, precision@K, confusion matrix);
* PR curve + cumulative precision (triage) curve plots;
* per-relation attention mass — which provenance relations the model leans on;
* **edge-rewire ablation**: the same trained model, but every relation's edge
  destinations are randomly permuted (degrees preserved, topology destroyed).
  If the model truly learned topography, its scores collapse on the rewired
  graph. If they don't, it was leaning on node features alone.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Dict

import numpy as np
import torch

from .config import Config
from .graph import DatasetBundle
from .metrics import Metrics, compute_metrics, format_metrics, search_threshold
from .model import EventProvenanceRGAT
from .sampling import BalancedGraphSamplers
from .schema import RELATIONS
from .trainer import Trainer


def _load_model(cfg: Config, bundle: DatasetBundle, ckpt: str,
                device: torch.device) -> EventProvenanceRGAT:
    from .schema import EDGE_FEATURE_DIM
    edge_dim = EDGE_FEATURE_DIM + (1 if cfg.sampling.reverse_edges else 0)
    model = EventProvenanceRGAT(
        hidden_dim=cfg.model.hidden_dim, num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads, feat_drop=cfg.model.feat_drop,
        attn_drop=cfg.model.attn_drop, use_edge_bias=cfg.model.use_edge_bias,
        residual=cfg.model.residual, edge_dim=edge_dim,
        readout=getattr(cfg.model, "readout", "fusion"),
        deep_input_proj=getattr(cfg.model, "deep_input_proj", False)).to(device)
    model.set_feature_stats(bundle.X_mean, bundle.X_std)
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"])
    model.eval()
    return model, sd


def rewire_graph(bundle: DatasetBundle, seed: int = 0) -> DatasetBundle:
    """Destroy topology while preserving per-relation degree sequences."""
    rng = np.random.default_rng(seed)
    data = copy.deepcopy(bundle.data)
    n = data["event"].num_nodes
    for rel in RELATIONS:
        store = data["event", rel, "event"]
        ei = store.edge_index.numpy()
        if ei.shape[1] == 0:
            continue
        perm = rng.integers(0, n, size=ei.shape[1])
        keep = perm != ei[0]                       # avoid trivial self loops
        ei2 = np.stack([ei[0], np.where(keep, perm, ei[1])])
        store.edge_index = torch.from_numpy(ei2)
    return DatasetBundle(data=data, X_mean=bundle.X_mean, X_std=bundle.X_std,
                         split_counts=bundle.split_counts,
                         leakage_top=bundle.leakage_top,
                         edge_stats=bundle.edge_stats)


def _eval_with(model, bundle, cfg, split, device):
    rng_seed = cfg.train.seed if getattr(cfg.train, "seed", -1) >= 0 \
        else cfg.data.seed
    samplers = BalancedGraphSamplers(
        bundle.data, bundle.data["event"].split.numpy(),
        bundle.data["event"].y.numpy(), cfg.sampling.fanouts,
        cfg.sampling.batch_seeds, cfg.sampling.pos_seed_frac,
        cfg.sampling.eval_batch_seeds, cfg.sampling.eval_fanout_mult,
        cfg.sampling.num_workers, seed=rng_seed,
        max_frontier=cfg.sampling.max_frontier,
        min_steps_per_epoch=cfg.sampling.min_steps_per_epoch,
        reverse_edges=cfg.sampling.reverse_edges)
    trainer = Trainer.__new__(Trainer)              # reuse eval loop only
    trainer.model = model
    trainer.samplers = samplers
    trainer.cfg = cfg
    trainer.device = device
    return trainer.evaluate_split(split)


def evaluate_run(cfg: Config, bundle: DatasetBundle, run_dir: str,
                 device: torch.device, log, do_rewire: bool = True) -> Dict:
    ckpt = os.path.join(run_dir, "best.pt")
    model, sd = _load_model(cfg, bundle, ckpt, device)
    log(f"[eval] loaded checkpoint {ckpt} (epoch {sd.get('epoch')})")

    # threshold re-derived on val (never from test), using the run's
    # operating-point policy
    vy, vp, _ = _eval_with(model, bundle, cfg, 1, device)
    thr = search_threshold(vy, vp, policy=cfg.train.threshold_policy,
                           min_recall=cfg.train.threshold_min_recall)
    val_m = compute_metrics(vy, vp, thr)
    log(format_metrics(val_m, "VALIDATION"))

    ty, tp, rel_mass = _eval_with(model, bundle, cfg, 2, device)
    test_m = compute_metrics(ty, tp, thr)
    log(format_metrics(test_m, "TEST"))

    rel_imp = rel_mass / max(1e-12, rel_mass.sum())
    log("[eval] mean attention per edge, per provenance relation "
        "(what the model leans on, normalised for edge volume):")
    for rel, w in sorted(zip(RELATIONS, rel_imp), key=lambda t: -t[1]):
        bar = "#" * int(40 * w)
        log(f"[eval]   {rel:<16} {100*w:5.1f}%  {bar}")

    report = {"val": val_m.to_dict(), "test": test_m.to_dict(),
              "threshold": thr,
              "threshold_policy": cfg.train.threshold_policy,
              "relation_attention": dict(zip(RELATIONS, rel_imp.tolist()))}

    # report the OTHER operating point as well (threshold also from val,
    # never test): whichever policy is configured, readers see both views
    alt_policy = "f1" if cfg.train.threshold_policy == "recall" else "recall"
    thr_alt = search_threshold(vy, vp, policy=alt_policy,
                               min_recall=cfg.train.threshold_min_recall)
    alt_m = compute_metrics(ty, tp, thr_alt)
    log(format_metrics(alt_m, f"TEST (alt operating point: {alt_policy})"))
    report[f"test_alt_{alt_policy}"] = alt_m.to_dict()

    # ------------------------------------------------------------- ablation
    if do_rewire:
        log("[eval] rewire ablation: permuting edge destinations "
            "(degrees kept, topology destroyed) ...")
        rewired = rewire_graph(bundle, seed=cfg.data.seed + 1)
        ry, rp, _ = _eval_with(model, rewired, cfg, 2, device)
        rew_m = compute_metrics(ry, rp, thr)
        log(format_metrics(rew_m, "TEST (REWIRE ablation)"))
        drop = test_m.auprc - rew_m.auprc
        log(f"[eval] AUPRC drop from rewiring: {test_m.auprc:.4f} -> "
            f"{rew_m.auprc:.4f}  (delta {drop:+.4f}). "
            + ("Model depends on graph structure as intended."
               if drop > 0.02 else
               "WARNING: little structural dependence — model mostly uses "
               "node features."))
        report["rewire_test"] = rew_m.to_dict()

    _plot_pr_curves(run_dir, ty, tp, val_m, test_m)
    with open(os.path.join(run_dir, "eval_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"[eval] report -> {os.path.join(run_dir, 'eval_report.json')}")
    return report


def _plot_pr_curves(run_dir: str, y: np.ndarray, proba: np.ndarray,
                    val_m: Metrics, test_m: Metrics) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve

        prec, rec, _ = precision_recall_curve(y, proba)
        order = np.argsort(-proba)
        k = np.arange(1, len(y) + 1)
        cum_prec = np.cumsum(y[order]) / k

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].plot(rec, prec, lw=1.5)
        axes[0].axhline(test_m.baseline_ap, color="grey", ls="--", lw=1,
                        label=f"baseline AP={test_m.baseline_ap:.3f}")
        axes[0].set_xlabel("recall"); axes[0].set_ylabel("precision")
        axes[0].set_title(f"test PR curve  (AUPRC={test_m.auprc:.3f}, "
                          f"AUROC={test_m.auroc:.3f})")
        axes[0].legend()
        axes[1].plot(100 * k / len(y), cum_prec, lw=1.5)
        axes[1].axhline(test_m.pos_rate, color="grey", ls="--", lw=1)
        axes[1].set_xlabel("% of events investigated (rank order)")
        axes[1].set_ylabel("precision among investigated")
        axes[1].set_xscale("log")
        axes[1].set_title("triage curve (precision vs investigation budget)")
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "test_pr_curves.png"), dpi=130)
        plt.close(fig)
    except Exception as e:
        print(f"[warn] plot failed: {e}")
