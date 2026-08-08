import torch

from eprgat.model import (EventProvenanceRGAT, FlatBatch, flatten_hetero_batch,
                          merge_flat_batches)
from eprgat.rgat import RGATConv, RGATLayer
from eprgat.schema import EDGE_FEATURE_DIM, FEATURE_DIM


def _tiny_conv(seed=0):
    torch.manual_seed(seed)
    return RGATConv(in_dim=8, out_dim=8, num_relations=3, num_heads=2,
                    edge_dim=EDGE_FEATURE_DIM, use_edge_bias=True)


def test_shapes_and_attention_mass():
    conv = _tiny_conv()
    x = torch.randn(6, 8)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    edge_type = torch.tensor([0, 1, 2, 0, 1])
    edge_attr = torch.randn(5, EDGE_FEATURE_DIM)
    out, mass, count = conv(x, edge_index, edge_type, edge_attr)
    assert out.shape == (6, 8)
    assert mass.shape == (3,) and count.shape == (3,)
    assert (mass[:2] > 0).all() and mass[2].item() >= 0
    assert count.tolist() == [2.0, 2.0, 1.0]


def test_empty_edges_does_not_crash():
    conv = _tiny_conv()
    x = torch.randn(4, 8)
    out, mass = conv(x, torch.zeros(2, 0, dtype=torch.long),
                     torch.zeros(0, dtype=torch.long), None)
    assert out.shape == (4, 8)
    assert torch.isfinite(out).all()


def test_gradients_flow_through_layer():
    layer = RGATLayer(8, 8, num_relations=3, num_heads=2,
                      edge_dim=EDGE_FEATURE_DIM)
    x = torch.randn(5, 8, requires_grad=True)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    et = torch.tensor([0, 1, 2])
    ea = torch.randn(3, EDGE_FEATURE_DIM)
    out, _, _ = layer(x, ei, et, ea)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_permutation_equivariance():
    """Reordering nodes must reorder outputs identically (no position info)."""
    torch.manual_seed(1)
    layer = RGATLayer(8, 8, num_relations=2, num_heads=2,
                      edge_dim=EDGE_FEATURE_DIM).eval()
    x = torch.randn(6, 8)
    ei = torch.tensor([[0, 1, 2, 4], [1, 2, 3, 5]])
    et = torch.tensor([0, 1, 0, 1])
    ea = torch.randn(4, EDGE_FEATURE_DIM)
    perm = torch.randperm(6)
    inv = torch.empty_like(perm); inv[perm] = torch.arange(6)
    out1, _, _ = layer(x, ei, et, ea)
    ei2 = inv[ei]
    out2, _, _ = layer(x[perm], ei2, et, ea)
    assert torch.allclose(out1[perm], out2, atol=1e-5)


def test_jk_readout_forward_backward():
    torch.manual_seed(3)
    model = EventProvenanceRGAT(hidden_dim=16, num_layers=3, num_heads=2,
                                in_dim=FEATURE_DIM, edge_dim=EDGE_FEATURE_DIM,
                                readout="jk")
    fb = FlatBatch(torch.randn(5, FEATURE_DIM), torch.zeros(5, dtype=torch.long),
                   torch.tensor([[0, 1, 2], [1, 2, 3]]), torch.tensor([0, 1, 2]),
                   torch.randn(3, EDGE_FEATURE_DIM), batch_size=3)
    logits = model(fb)
    assert logits.shape == (5,)
    logits.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters()
               if p.grad is not None)


def test_full_model_forward_and_merge():
    torch.manual_seed(2)
    model = EventProvenanceRGAT(hidden_dim=16, num_layers=2, num_heads=2,
                                in_dim=FEATURE_DIM, edge_dim=EDGE_FEATURE_DIM)
    fb1 = FlatBatch(torch.randn(5, FEATURE_DIM), torch.zeros(5, dtype=torch.long),
                    torch.tensor([[0, 1], [1, 2]]), torch.tensor([0, 1]),
                    torch.randn(2, EDGE_FEATURE_DIM), batch_size=3)
    fb2 = FlatBatch(torch.randn(4, FEATURE_DIM), torch.ones(4, dtype=torch.long),
                    torch.tensor([[0], [3]]), torch.tensor([2]),
                    torch.randn(1, EDGE_FEATURE_DIM), batch_size=2)
    fb = merge_flat_batches([fb1, fb2])
    assert fb.x.shape[0] == 9 and fb.batch_size == 5
    logits, mass = model(fb, return_attention_mass=True)
    assert logits.shape == (9,)
    assert mass.shape[0] == 6
