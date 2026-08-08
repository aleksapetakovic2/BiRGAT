"""Relational Graph Attention (RGAT) layer.

Per relation r the layer keeps its own projection W_r and attention vectors,
following the relational extension of GAT:

    z_i^r       = h_i W_r                                   (per relation)
    e_ij^{r,k}  = LeakyReLU( a_r^k . [z_i^k || z_j^k] ) + edge_bias_ij^k
    alpha_ij^r  = softmax_{j' in N_r(i)}( e_ij' )           (within relation)
    h_i'        = sigma( sum_r sum_{j in N_r(i)} alpha_ij^r z_j^r )

Sentinel-specific adjustment: provenance edges carry temporal features
(time delta between the two events, transferred bytes). A small MLP turns
them into per-head attention biases, so the model can learn e.g. that a
sign-in *immediately* followed by a process start deserves more attention
than one 8 hours earlier.

Memory notes (8 GB VRAM):
* relation projections are applied once per node (N x R x K x d), not once
  per edge — edge-gathered weights would blow up memory;
* only attention coefficients and gathered node embeddings are E-sized.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax


class RGATConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_relations: int,
                 num_heads: int = 4, edge_dim: int = 0, use_edge_bias: bool = True,
                 negative_slope: float = 0.2, attn_drop: float = 0.0) -> None:
        super().__init__()
        if out_dim % num_heads != 0:
            raise ValueError("out_dim must be divisible by num_heads")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.R = num_relations
        self.K = num_heads
        self.d = out_dim // num_heads
        self.negative_slope = negative_slope

        # per-relation projections: [R, K, in_dim, d]
        self.W = nn.Parameter(torch.empty(num_relations, num_heads, in_dim, self.d))
        # self-loop transform (relation-agnostic)
        self.W_self = nn.Linear(in_dim, out_dim, bias=True)
        # per-relation, per-head attention vectors over [z_i || z_j]
        self.a = nn.Parameter(torch.empty(num_relations, num_heads, 2 * self.d))
        self.attn_drop = nn.Dropout(attn_drop)

        self.use_edge_bias = use_edge_bias and edge_dim > 0
        if self.use_edge_bias:
            self.edge_bias = nn.Sequential(
                nn.Linear(edge_dim, num_heads), nn.Tanh())

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.W.view(self.R * self.K, self.in_dim, self.d),
                                gain=gain)
        nn.init.xavier_uniform_(self.a.view(self.R * self.K, 2 * self.d), gain=gain)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor, edge_attr: torch.Tensor | None = None):
        """x [N, in_dim]; edge_index [2, E]; edge_type [E] (ids into 0..R-1).

        Returns (out [N, out_dim], alpha_sum_per_rel [R], edge_count_per_rel [R])
        — the last two are used for explainability logging (mean attention per
        edge and relation).
        """
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        E = src.numel()
        R, K, d = self.R, self.K, self.d
        x_in = x

        # 1) per-relation node projections  ->  Z [N, R, K, d]
        # W[r] is [K, in_dim, d]; bring it to [in_dim, K*d] so column blocks
        # correspond to heads (permute first — a bare reshape would mix heads).
        Z = torch.stack([
            torch.matmul(x_in, self.W[r].transpose(0, 1).reshape(self.in_dim, K * d))
            .view(N, K, d)
            for r in range(R)], dim=1)

        if E == 0:                                           # isolated batch
            return self.W_self(x_in), torch.zeros(R, device=x.device)

        rel = edge_type.long()
        z_src = Z[src, rel]                                  # [E, K, d]
        z_dst = Z[dst, rel]                                  # [E, K, d]

        # 2) attention logits
        logits = torch.cat([z_dst, z_src], dim=-1)           # [E, K, 2d]
        a_r = self.a[rel]                                    # [E, K, 2d]
        e = F.leaky_relu((logits * a_r).sum(dim=-1), self.negative_slope)  # [E, K]
        if self.use_edge_bias and edge_attr is not None:
            e = e + self.edge_bias(edge_attr)                # [E, K]

        # 3) softmax within (dst-node, relation) groups — faithful RGAT
        group = dst * R + rel
        alpha = softmax(e, group, num_nodes=N * R)           # [E, K]
        alpha = self.attn_drop(alpha)

        # 4) weighted message passing (sum aggregation; relations summed)
        msg = alpha.unsqueeze(-1) * z_src                    # [E, K, d]
        out = torch.zeros(N, K, d, device=x.device, dtype=msg.dtype)
        out.index_add_(0, dst, msg)

        # 5) self loop + reshape
        out = out.reshape(N, K * d) + self.W_self(x_in)

        with torch.no_grad(), torch.autocast(x.device.type, enabled=False):
            rel_mass = torch.zeros(R, device=x.device, dtype=torch.float32)
            rel_mass.index_add_(0, rel, alpha.sum(dim=-1).float())
            rel_count = torch.bincount(rel, minlength=R).float()
        return out, rel_mass, rel_count


class RGATLayer(nn.Module):
    """RGATConv + residual + LayerNorm + activation + feature dropout."""

    def __init__(self, in_dim: int, out_dim: int, num_relations: int,
                 num_heads: int = 4, edge_dim: int = 0, use_edge_bias: bool = True,
                 feat_drop: float = 0.0, attn_drop: float = 0.0,
                 negative_slope: float = 0.2, residual: bool = True) -> None:
        super().__init__()
        self.conv = RGATConv(in_dim, out_dim, num_relations, num_heads,
                             edge_dim=edge_dim, use_edge_bias=use_edge_bias,
                             negative_slope=negative_slope, attn_drop=attn_drop)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ELU()
        self.feat_drop = nn.Dropout(feat_drop)
        self.residual = residual and (in_dim == out_dim)
        self.proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, edge_type, edge_attr=None):
        h = self.feat_drop(x)
        out, rel_mass, rel_count = self.conv(h, edge_index, edge_type, edge_attr)
        if self.residual:
            out = out + x
        else:
            out = out + self.proj(x)
        out = self.act(self.norm(out))
        return out, rel_mass, rel_count
