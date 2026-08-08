"""Focal loss for extreme class imbalance (~1% incident events).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

Implemented from raw logits (numerically stable). gamma focuses learning on
hard examples; alpha balances the overwhelmingly benign majority class.
Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.8) -> None:
        super().__init__()
        if not (0.0 <= gamma):
            raise ValueError("gamma must be >= 0")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                reduction: str = "mean") -> torch.Tensor:
        """logits: [N] raw scores; targets: [N] in {0,1}."""
        logits = logits.reshape(-1)
        targets = targets.reshape(-1).to(logits.dtype)

        # stable log-loss from logits
        log_p = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        with torch.autocast(logits.device.type, enabled=False):
            p = torch.sigmoid(logits.float())
            p_t = p * targets + (1.0 - p) * (1.0 - targets)
            mod = (1.0 - p_t).pow(self.gamma)
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * mod * log_p.float()
        if reduction == "none":
            return loss
        if reduction == "sum":
            return loss.sum()
        return loss.mean()

    @torch.no_grad()
    def decompose(self, logits: torch.Tensor, targets: torch.Tensor) -> "FocalStats":
        """Per-class contribution breakdown for verbose logging."""
        loss = self.forward(logits, targets, reduction="none")
        pos = targets.bool()
        return FocalStats(
            total=float(loss.mean()),
            pos=float(loss[pos].mean()) if pos.any() else 0.0,
            neg=float(loss[~pos].mean()) if (~pos).any() else 0.0,
            n_pos=int(pos.sum()), n_neg=int((~pos).sum()))


@dataclass
class FocalStats:
    total: float
    pos: float
    neg: float
    n_pos: int
    n_neg: int
