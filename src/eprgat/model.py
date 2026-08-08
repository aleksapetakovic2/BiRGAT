"""Model stack: feature standardisation -> input projection -> N x RGAT -> head.

Also defines `FlatBatch`, the homogeneous view over PyG's sampled HeteroData
batches that the RGAT layers consume, and an MLP baseline used to prove how
much signal lives in features alone (no graph) — the anti-cheating reference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .rgat import RGATLayer
from .schema import EDGE_FEATURE_DIM, FEATURE_DIM, RELATIONS

REL_ORDER: List[str] = RELATIONS
REL_TO_ID: Dict[str, int] = {r: i for i, r in enumerate(REL_ORDER)}


# --------------------------------------------------------------------------- #
@dataclass
class FlatBatch:
    """Homogeneous view of a sampled HeteroData batch."""
    x: torch.Tensor                 # [N, F]
    y: torch.Tensor                 # [N]
    edge_index: torch.Tensor        # [2, E]
    edge_type: torch.Tensor         # [E]
    edge_attr: torch.Tensor         # [E, edge_dim]
    batch_size: int                 # number of seed nodes (first batch_size rows)
    split: Optional[torch.Tensor] = None   # [N] split ids when available

    def to(self, device) -> "FlatBatch":
        return FlatBatch(self.x.to(device), self.y.to(device),
                         self.edge_index.to(device), self.edge_type.to(device),
                         self.edge_attr.to(device), self.batch_size,
                         None if self.split is None else self.split.to(device))


def flatten_hetero_batch(batch, device) -> FlatBatch:
    """Concatenate all relation edge sets of a sampled batch into one
    homogeneous edge set with an edge_type vector."""
    x = batch["event"].x
    y = batch["event"].y
    bs = batch["event"].batch_size
    split = batch["event"].split if "split" in batch["event"] else None
    srcs, dsts, types, attrs = [], [], [], []
    edge_dim = EDGE_FEATURE_DIM
    for rel in REL_ORDER:
        store = batch["event", rel, "event"]
        if store.edge_attr is not None and store.edge_attr.dim() == 2:
            edge_dim = store.edge_attr.size(-1)   # +1 col when bidirectional
        ei = store.edge_index
        if ei is None or ei.numel() == 0:
            continue
        srcs.append(ei[0]); dsts.append(ei[1])
        types.append(torch.full((ei.size(1),), REL_TO_ID[rel], dtype=torch.long))
        attrs.append(store.edge_attr)
    if srcs:
        edge_index = torch.stack([torch.cat(srcs), torch.cat(dsts)])
        edge_type = torch.cat(types)
        edge_attr = torch.cat(attrs)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)
        edge_attr = torch.zeros(0, edge_dim)
    return FlatBatch(x, y, edge_index, edge_type, edge_attr, bs, split).to(device)


def merge_flat_batches(fbs: List[FlatBatch]) -> FlatBatch:
    """Merge several sampled subgraphs (e.g. the balanced pos + neg batches)
    into one disjoint batch for a single fused forward pass."""
    xs, ys, eis, ets, eas, sps = [], [], [], [], [], []
    offset, bs = 0, 0
    have_split = any(fb.split is not None for fb in fbs)
    for fb in fbs:
        xs.append(fb.x); ys.append(fb.y)
        eis.append(fb.edge_index + offset)
        ets.append(fb.edge_type); eas.append(fb.edge_attr)
        if have_split:
            sps.append(fb.split if fb.split is not None
                       else torch.zeros(fb.x.size(0), dtype=torch.long))
        offset += fb.x.size(0)
        bs += fb.batch_size
    return FlatBatch(torch.cat(xs), torch.cat(ys),
                     torch.cat(eis, dim=1) if eis else torch.zeros(2, 0, dtype=torch.long),
                     torch.cat(ets), torch.cat(eas), bs,
                     torch.cat(sps) if have_split else None)


# --------------------------------------------------------------------------- #
class EventProvenanceRGAT(nn.Module):
    def __init__(self, hidden_dim: int = 96, num_layers: int = 3,
                 num_heads: int = 4, feat_drop: float = 0.1,
                 attn_drop: float = 0.1, use_edge_bias: bool = True,
                 residual: bool = True, negative_slope: float = 0.2,
                 in_dim: int = FEATURE_DIM, edge_dim: int = EDGE_FEATURE_DIM,
                 readout: str = "fusion", deep_input_proj: bool = False) -> None:
        super().__init__()
        if readout not in ("fusion", "jk"):
            raise ValueError(f"unknown readout '{readout}'")
        self.readout = readout
        self.register_buffer("feat_mean", torch.zeros(in_dim))
        self.register_buffer("feat_std", torch.ones(in_dim))

        proj = [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                nn.GELU(), nn.Dropout(feat_drop)]
        if deep_input_proj:
            proj += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                     nn.GELU(), nn.Dropout(feat_drop)]
        self.input_proj = nn.Sequential(*proj)

        self.layers = nn.ModuleList([
            RGATLayer(hidden_dim, hidden_dim, num_relations=len(REL_ORDER),
                      num_heads=num_heads, edge_dim=edge_dim,
                      use_edge_bias=use_edge_bias, feat_drop=feat_drop,
                      attn_drop=attn_drop, negative_slope=negative_slope,
                      residual=residual)
            for _ in range(num_layers)])

        # readout keeps the classifier a fair "features AND topology" model:
        # the original feature embedding h0 is always part of the head input,
        # so per-event attribute signal never has to survive every
        # message-passing round — any gain over the features-only baseline is
        # attributable to the provenance graph.
        #   fusion: head over [last layer || h0]
        #   jk:     head over [all layer outputs || h0] (JumpingKnowledge
        #           concat) — lets the head mix shallow and deep views, which
        #           helps when chains are shorter than the network is deep.
        head_in = hidden_dim * (num_layers + 1 if readout == "jk" else 2)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim // 2), nn.GELU(),
            nn.Dropout(feat_drop), nn.Linear(hidden_dim // 2, 1))

    def set_feature_stats(self, mean, std) -> None:
        self.feat_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.feat_std.copy_(torch.as_tensor(std, dtype=torch.float32))

    def forward(self, fb: FlatBatch, return_attention_mass: bool = False):
        x = (fb.x - self.feat_mean) / self.feat_std
        h0 = self.input_proj(x)
        h = h0
        hs = []
        rel_mass = torch.zeros(len(REL_ORDER), device=x.device)
        rel_count = torch.zeros(len(REL_ORDER), device=x.device)
        for layer in self.layers:
            h, m, c = layer(h, fb.edge_index, fb.edge_type, fb.edge_attr)
            rel_mass = rel_mass + m
            rel_count = rel_count + c
            if self.readout == "jk":
                hs.append(h)
        if self.readout == "jk":
            logits = self.head(torch.cat(hs + [h0], dim=-1)).squeeze(-1)
        else:
            logits = self.head(torch.cat([h, h0], dim=-1)).squeeze(-1)
        if return_attention_mass:
            # mean attention weight per edge of each relation (what the model
            # leans on per provenance edge, normalised for edge volume)
            mean_attn = rel_mass / rel_count.clamp(min=1.0)
            return logits, mean_attn
        return logits


class MLPBaseline(nn.Module):
    """Feature-only baseline: same features, no graph.

    If RGAT does not clearly beat this, the features alone carry the signal
    and the generator is leaking (or the graph is useless). Both are valuable
    things to know."""

    def __init__(self, hidden_dim: int = 128, in_dim: int = FEATURE_DIM,
                 feat_drop: float = 0.1) -> None:
        super().__init__()
        self.register_buffer("feat_mean", torch.zeros(in_dim))
        self.register_buffer("feat_std", torch.ones(in_dim))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(feat_drop),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(feat_drop),
            nn.Linear(hidden_dim, 1))

    def set_feature_stats(self, mean, std) -> None:
        self.feat_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.feat_std.copy_(torch.as_tensor(std, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.feat_mean) / self.feat_std
        return self.net(x).squeeze(-1)
