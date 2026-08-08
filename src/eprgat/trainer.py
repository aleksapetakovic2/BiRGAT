"""Training loop with very verbose telemetry (watch-it-train friendly).

* per-step lines: focal loss total + pos/neg decomposition, seed F1, LR,
  subgraph size, throughput, GPU memory;
* per-epoch summary tables with val AUROC/AUPRC/F1/P/R/MCC;
* early stopping on val AUPRC, best-model checkpointing, jsonl metric trail,
  loss / AUPRC plots;
* OOM errors are caught and re-raised with the exact knobs to turn.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import Config
from .graph import DatasetBundle
from .losses import FocalLoss
from .metrics import Metrics, compute_metrics, format_metrics, search_threshold
from .model import (
    EventProvenanceRGAT, FlatBatch, MLPBaseline, flatten_hetero_batch,
    merge_flat_batches,
)
from .sampling import BalancedGraphSamplers
from .schema import EDGE_FEATURE_DIM, FEATURE_DIM, RELATIONS


# --------------------------------------------------------------------------- #
class OOMHintError(RuntimeError):
    pass


def _oom_guard(e: RuntimeError) -> None:
    msg = str(e).lower()
    if "out of memory" in msg or "cuda" in msg and "memory" in msg:
        torch.cuda.empty_cache()
        raise OOMHintError(
            "CUDA out of memory. Turn these knobs (configs/*.yaml):\n"
            "  sampling.batch_seeds:          512 -> 256\n"
            "  sampling.fanouts:              e.g. [6,4,2] -> [4,2,2]\n"
            "  sampling.max_frontier:         20000 -> 10000\n"
            "  sampling.eval_batch_seeds:     1024 -> 512\n"
            "  model.hidden_dim:              96 -> 64\n"
            "  train.amp:                     true  (fp16 autocast)\n"
        ) from e
    raise e


def _gpu_mem_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


# --------------------------------------------------------------------------- #
class Trainer:
    def __init__(self, cfg: Config, bundle: DatasetBundle,
                 samplers: BalancedGraphSamplers, run_dir: str,
                 device: torch.device, log: Callable[[str], None]) -> None:
        self.cfg = cfg
        self.bundle = bundle
        self.samplers = samplers
        self.run_dir = run_dir
        self.device = device
        self.log = log

        # bidirectional batches carry one extra edge feature (message
        # direction) — see sampling.BATCH_EDGE_DIM_BIDIR
        edge_dim = EDGE_FEATURE_DIM + (1 if cfg.sampling.reverse_edges else 0)
        self.model = EventProvenanceRGAT(
            hidden_dim=cfg.model.hidden_dim, num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads, feat_drop=cfg.model.feat_drop,
            attn_drop=cfg.model.attn_drop, use_edge_bias=cfg.model.use_edge_bias,
            residual=cfg.model.residual, negative_slope=cfg.model.negative_slope,
            edge_dim=edge_dim, readout=cfg.model.readout,
            deep_input_proj=cfg.model.deep_input_proj,
        ).to(device)
        self.model.set_feature_stats(bundle.X_mean, bundle.X_std)
        self.focal = FocalLoss(cfg.train.focal_gamma, cfg.train.focal_alpha)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=cfg.train.lr,
                                       weight_decay=cfg.train.weight_decay)

        self.steps_per_epoch = samplers.steps_per_epoch()
        self.total_steps = cfg.train.epochs * self.steps_per_epoch
        self.global_step = 0
        self.history: List[dict] = []
        self.train_losses: List[float] = []
        self.val_auprcs: List[float] = []
        self.best_auprc = -1.0
        self.best_epoch = -1
        self.threshold = 0.5

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.log(f"[model] EventProvenanceRGAT  layers={cfg.model.num_layers} "
                 f"hidden={cfg.model.hidden_dim} heads={cfg.model.num_heads} "
                 f"params={n_params:,}")
        self.log(f"[loss ] focal loss  gamma={cfg.train.focal_gamma} "
                 f"alpha={cfg.train.focal_alpha}")
        self.log(f"[thr  ] threshold policy: {cfg.train.threshold_policy}"
                 + (f" (min recall {cfg.train.threshold_min_recall:.0%})"
                    if cfg.train.threshold_policy == "recall" else ""))
        self.log(f"[train] {cfg.train.epochs} epochs x {self.steps_per_epoch} steps "
                 f"(seeds/batch={cfg.sampling.batch_seeds}, "
                 f"pos={samplers.pos_batch}/neg={samplers.neg_batch})")

    # ------------------------------------------------------------- lr schedule
    def _lr(self, step: int) -> float:
        base = self.cfg.train.lr
        w = self.cfg.train.warmup_steps
        if step < w:
            return base * (0.1 + 0.9 * step / max(1, w))
        prog = (step - w) / max(1, self.total_steps - w)
        return base * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))

    # ------------------------------------------------------------- one epoch
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        t0 = time.time()
        sums = {"loss": 0.0, "pos": 0.0, "neg": 0.0, "seeds": 0}
        trunc_before = self.samplers.frontier_truncations
        tp = fp = fn = 0
        scaler = torch.GradScaler("cuda") if (self.cfg.train.amp and
                                              self.device.type == "cuda") else None

        for step, (pos_batch, neg_batch) in enumerate(self.samplers.train_batches()):
            self.global_step += 1
            lr = self._lr(self.global_step)
            for g in self.optim.param_groups:
                g["lr"] = lr

            self.optim.zero_grad(set_to_none=True)
            try:
                fb = merge_flat_batches([
                    flatten_hetero_batch(pos_batch, self.device),
                    flatten_hetero_batch(neg_batch, self.device)])
                with torch.autocast(self.device.type, enabled=bool(scaler)):
                    logits = self.model(fb)
                    # subgraph-wide supervision: edges never cross split
                    # boundaries, so every node sampled around train seeds is
                    # itself a train node. Supervising the whole sampled chain
                    # fragment (not only the seeds) gives far more gradient
                    # signal per batch — especially for the positive context
                    # events that balanced seed sampling rarely picks.
                    if fb.split is not None:
                        m = fb.split == 0
                    else:
                        m = torch.ones(logits.shape[0], dtype=torch.bool,
                                       device=logits.device)
                    y_sup = fb.y[m].float()
                    loss = self.focal(logits[m], y_sup)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optim)
                    nn.utils.clip_grad_norm_(self.model.parameters(),
                                             self.cfg.train.grad_clip)
                    scaler.step(self.optim)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(),
                                             self.cfg.train.grad_clip)
                    self.optim.step()
            except RuntimeError as e:
                _oom_guard(e)

            with torch.no_grad():
                stats = self.focal.decompose(logits.detach()[m], y_sup)
                seed_logits = logits.detach()[:fb.batch_size]
                y = fb.y[:fb.batch_size].float()
                pred = (torch.sigmoid(seed_logits) >= self.threshold)
                tp += int((pred & (y == 1)).sum()); fp += int((pred & (y == 0)).sum())
                fn += int(((~pred.bool()) & (y == 1)).sum())

            sums["loss"] += stats.total; sums["pos"] += stats.pos
            sums["neg"] += stats.neg; sums["seeds"] += stats.n_pos + stats.n_neg
            self.train_losses.append(stats.total)

            if self.global_step % self.cfg.train.log_every == 0 or step == 0:
                seed_f1 = (2 * tp / max(1, 2 * tp + fp + fn))
                line = (
                    f"[train] ep{epoch:>2} step {step+1:>3}/{self.steps_per_epoch} "
                    f"| loss {stats.total:.4f} (pos {stats.pos:.3f}/neg {stats.neg:.3f}) "
                    f"| sup {stats.n_pos}+/{stats.n_neg}- "
                    f"| seedF1 {seed_f1:.3f} "
                    f"| subgraph n={fb.x.size(0):,} e={fb.edge_index.size(1):,} "
                    f"| lr {lr:.2e}")
                if self.device.type == "cuda":
                    line += f" | gpu {_gpu_mem_mb():.0f}MB"
                self.log(line)
                self._jsonl({"kind": "step", "epoch": epoch, "step": step + 1,
                             "loss": stats.total, "loss_pos": stats.pos,
                             "loss_neg": stats.neg, "lr": lr,
                             "subgraph_nodes": fb.x.size(0),
                             "subgraph_edges": fb.edge_index.size(1)})

        dt = time.time() - t0
        trunc = self.samplers.frontier_truncations - trunc_before
        if trunc:
            self.log(f"[warn] frontier cap triggered {trunc}x while sampling "
                     f"this epoch — train subgraphs are randomly truncated; "
                     f"raise sampling.max_frontier or lower fanouts/batch_seeds")
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        return {"loss": sums["loss"] / max(1, step + 1),
                "pos": sums["pos"] / max(1, step + 1),
                "neg": sums["neg"] / max(1, step + 1),
                "seed_f1": 2 * prec * rec / max(1e-12, prec + rec),
                "seconds": dt, "seeds_per_s": sums["seeds"] / max(1e-9, dt)}

    # ------------------------------------------------------------- evaluation
    @torch.no_grad()
    def evaluate_split(self, split: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (y, proba, rel_attention_mass) over all seed nodes of split.

        With ``sampling.eval_passes > 1`` every seed is scored once per
        independently sampled subgraph and the probabilities are averaged —
        this removes neighbourhood-sampling noise from the scores at
        inference time (model weights and data are untouched)."""
        self.model.eval()
        passes = max(1, int(getattr(self.cfg.sampling, "eval_passes", 1)))
        trunc_before = self.samplers.frontier_truncations
        rel_mass = torch.zeros(len(RELATIONS), device=self.device)
        ys = None
        acc = None
        for _ in range(passes):
            ps = []
            ys_pass = []
            for batch in self.samplers.eval_loader(split):
                try:
                    fb = flatten_hetero_batch(batch, self.device)
                    logits, m = self.model(fb, return_attention_mass=True)
                except RuntimeError as e:
                    _oom_guard(e)
                ys_pass.append(fb.y[:fb.batch_size].cpu().numpy())
                ps.append(torch.sigmoid(logits[:fb.batch_size]).cpu().numpy())
                rel_mass += m
            p = np.concatenate(ps)
            acc = p if acc is None else acc + p
            if ys is None:
                ys = np.concatenate(ys_pass)
        trunc = self.samplers.frontier_truncations - trunc_before
        if trunc:
            log = getattr(self, "log", None)
            if log:
                log(f"[warn] frontier cap triggered {trunc}x during this eval "
                    f"— eval subgraphs are randomly truncated; raise "
                    f"sampling.max_frontier or lower eval_batch_seeds")
        return (ys, acc / passes, rel_mass.cpu().numpy())

    # ------------------------------------------------------------- main loop
    def train(self) -> Dict:
        log, cfg = self.log, self.cfg
        for epoch in range(1, cfg.train.epochs + 1):
            ep = self._train_epoch(epoch)

            vy, vp, _ = self.evaluate_split(1)
            self.threshold = search_threshold(
                vy, vp, policy=cfg.train.threshold_policy,
                min_recall=cfg.train.threshold_min_recall)
            vm = compute_metrics(vy, vp, self.threshold)
            self.val_auprcs.append(vm.auprc)

            improved = vm.auprc > self.best_auprc + cfg.train.min_delta
            if improved:
                self.best_auprc, self.best_epoch = vm.auprc, epoch
                self._save_checkpoint("best.pt", epoch, vm)
            patience_left = cfg.train.patience - (epoch - self.best_epoch)

            eta_s = ep["seconds"] * (cfg.train.epochs - epoch)
            log("┌" + "─" * 76)
            log(f"│ EPOCH {epoch:>3}/{cfg.train.epochs}  "
                f"train_loss {ep['loss']:.4f} (pos {ep['pos']:.3f} / neg {ep['neg']:.3f})  "
                f"seedF1 {ep['seed_f1']:.3f}  {ep['seconds']:.1f}s  "
                f"({ep['seeds_per_s']:.0f} seeds/s)  eta ~{eta_s/60:.1f}min")
            log(f"│ VAL  AUROC {vm.auroc:.4f} | AUPRC {vm.auprc:.4f} "
                f"(base {vm.baseline_ap:.4f}) | F1 {vm.f1:.4f} "
                f"(thr {self.threshold:.3f}) | P {vm.precision:.4f} R {vm.recall:.4f} "
                f"| MCC {vm.mcc:.4f}")
            log(f"│ best AUPRC {self.best_auprc:.4f} @ ep{self.best_epoch}   "
                f"{'** improved **' if improved else f'patience {patience_left}/{cfg.train.patience}'}")
            log("└" + "─" * 76)
            self._jsonl({"kind": "epoch", "epoch": epoch, **ep,
                         "val": vm.to_dict(), "threshold": self.threshold,
                         "improved": improved})

            if epoch - self.best_epoch >= cfg.train.patience:
                log(f"[train] early stopping: no AUPRC improvement for "
                    f"{cfg.train.patience} epochs.")
                break

        self._save_checkpoint("last.pt", epoch, vm)
        self._plot_curves()
        return {"best_auprc": self.best_auprc, "best_epoch": self.best_epoch,
                "threshold": self.threshold, "epochs_run": epoch}

    # ------------------------------------------------------------- plumbing
    def _save_checkpoint(self, name: str, epoch: int, vm: Optional[Metrics]) -> None:
        path = os.path.join(self.run_dir, name)
        torch.save({"model": self.model.state_dict(),
                    "epoch": epoch, "threshold": self.threshold,
                    "config": self.cfg.to_dict(),
                    "val_metrics": vm.to_dict() if vm else None}, path)

    def _jsonl(self, obj: dict) -> None:
        with open(os.path.join(self.run_dir, "metrics.jsonl"), "a",
                  encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")

    def _plot_curves(self) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].plot(self.train_losses, lw=0.8)
            axes[0].set_title("train focal loss (per step)")
            axes[0].set_xlabel("step"); axes[0].set_yscale("log")
            axes[1].plot(range(1, len(self.val_auprcs) + 1), self.val_auprcs, "o-")
            axes[1].axhline(self.bundle.split_counts["val"]["pos_rate"],
                            color="grey", ls="--", lw=1, label="baseline AP")
            axes[1].set_title("val AUPRC per epoch"); axes[1].set_xlabel("epoch")
            axes[1].legend()
            fig.tight_layout()
            fig.savefig(os.path.join(self.run_dir, "curves.png"), dpi=130)
            plt.close(fig)
        except Exception as e:                              # plotting is optional
            self.log(f"[warn] could not render curves: {e}")


# --------------------------------------------------------------------------- #
def train_mlp_baseline(cfg: Config, bundle: DatasetBundle, run_dir: str,
                       device: torch.device, log: Callable[[str], None],
                       epochs: int = 25) -> Dict[str, Metrics]:
    """Feature-only reference model (no graph). Quantifies how much signal the
    features alone carry — the RGAT must clearly beat this, otherwise the
    'topology learning' claim is hollow."""
    data = bundle.data
    split = data["event"].split.numpy()
    X = data["event"].x.numpy()
    y = data["event"].y.numpy()
    tr, va, te = split == 0, split == 1, split == 2

    model = MLPBaseline(hidden_dim=max(64, cfg.model.hidden_dim)).to(device)
    model.set_feature_stats(bundle.X_mean, bundle.X_std)
    focal = FocalLoss(cfg.train.focal_gamma, cfg.train.focal_alpha)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                              weight_decay=cfg.train.weight_decay)

    yt = y[tr]
    w = np.where(yt == 1, 0.5 / max(1, yt.sum()), 0.5 / max(1, (yt == 0).sum()))
    sampler = WeightedRandomSampler(w, num_samples=len(yt), replacement=True)
    Xt = torch.from_numpy(X[tr]); Yt = torch.from_numpy(yt)
    loader = DataLoader(torch.utils.data.TensorDataset(Xt, Yt), batch_size=1024,
                        sampler=sampler)

    log("[mlp ] training feature-only baseline (no edges) ...")
    best_auprc, thr = -1.0, 0.5
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device).float()
            optim.zero_grad(set_to_none=True)
            loss = focal(model(xb), yb)
            loss.backward(); optim.step()
        model.eval()
        with torch.no_grad():
            vp = torch.sigmoid(model(torch.from_numpy(X[va]).to(device))).cpu().numpy()
        # same operating-point policy as the RGAT — fair comparison
        thr = search_threshold(y[va], vp, policy=cfg.train.threshold_policy,
                               min_recall=cfg.train.threshold_min_recall)
        va_m = compute_metrics(y[va], vp, thr)
        if va_m.auprc > best_auprc:
            best_auprc = va_m.auprc
            torch.save({"model": model.state_dict()}, os.path.join(run_dir, "mlp_best.pt"))
        if ep % 5 == 0 or ep == 1:
            log(f"[mlp ] ep{ep:>2} val AUPRC {va_m.auprc:.4f}  AUROC {va_m.auroc:.4f}  "
                f"F1 {va_m.f1:.4f}")

    sd = torch.load(os.path.join(run_dir, "mlp_best.pt"), weights_only=False)
    model.load_state_dict(sd["model"]); model.eval()
    with torch.no_grad():
        p_va = torch.sigmoid(model(torch.from_numpy(X[va]).to(device))).cpu().numpy()
        p_te = torch.sigmoid(model(torch.from_numpy(X[te]).to(device))).cpu().numpy()
    out = {"val": compute_metrics(y[va], p_va, thr),
           "test": compute_metrics(y[te], p_te, thr)}
    # the other operating point too, so MLP and RGAT are compared symmetrically
    alt_policy = "f1" if cfg.train.threshold_policy == "recall" else "recall"
    thr_alt = search_threshold(y[va], p_va, policy=alt_policy,
                               min_recall=cfg.train.threshold_min_recall)
    out["test_alt"] = compute_metrics(y[te], p_te, thr_alt)
    out["alt_policy"] = alt_policy
    return out
