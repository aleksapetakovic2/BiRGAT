import math

import torch
import torch.nn.functional as F

from eprgat.losses import FocalLoss


def test_focal_gamma0_alpha_half_equals_half_bce():
    torch.manual_seed(0)
    logits = torch.randn(256)
    y = (torch.rand(256) < 0.3).float()
    fl = FocalLoss(gamma=0.0, alpha=0.5)
    expected = 0.5 * F.binary_cross_entropy_with_logits(logits, y)
    assert torch.allclose(fl(logits, y), expected, atol=1e-6)


def test_focal_easy_examples_are_downweighted():
    # well-classified positive/negative pair must have ~zero loss at gamma>0
    fl = FocalLoss(gamma=2.0, alpha=0.5)
    easy = torch.tensor([10.0, -10.0])
    hard = torch.tensor([0.1, -0.1])
    y = torch.tensor([1.0, 0.0])
    assert fl(easy, y).item() < 1e-5
    assert fl(hard, y).item() > fl(easy, y).item()


def test_focal_alpha_weights_positive_class():
    # per-class invariant: higher alpha raises the mean positive-sample loss
    # and lowers the mean negative-sample loss (alpha_t definition).
    logits = torch.zeros(1000)
    y = torch.zeros(1000); y[:10] = 1.0
    l_hi = FocalLoss(gamma=2.0, alpha=0.9)(logits, y, reduction="none")
    l_lo = FocalLoss(gamma=2.0, alpha=0.1)(logits, y, reduction="none")
    assert l_hi[:10].mean().item() > l_lo[:10].mean().item()
    assert l_hi[10:].mean().item() < l_lo[10:].mean().item()
    # exact alpha_t scaling at p=0.5: loss(alpha)/loss(1-alpha) per class
    ratio_pos = l_hi[:10].mean().item() / l_lo[:10].mean().item()
    assert abs(ratio_pos - 9.0) < 1e-3          # 0.9 / 0.1


def test_focal_gradient_flows():
    logits = torch.randn(64, requires_grad=True)
    y = (torch.rand(64) < 0.2).float()
    loss = FocalLoss()(logits, y)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_decompose_counts():
    logits = torch.randn(100)
    y = (torch.rand(100) < 0.25).float()
    s = FocalLoss().decompose(logits, y)
    assert s.n_pos + s.n_neg == 100
    assert s.total >= 0
