"""Neighbourhood sampling with balanced positive/negative seed nodes.

Self-contained k-hop sampler (NumPy CSR structures + torch indexing) with the
same batch contract as PyG's ``NeighborLoader`` — but without the fragile
``torch-sparse`` / ``pyg-lib`` C++ kernels, which have no wheels for recent
Python/torch builds on Windows.

Semantics (faithful to provenance graphs):

* seeds are the events to classify; the sampler expands their **k-hop causal
  past** — for every frontier node and relation it keeps at most
  ``fanouts[rel][hop]`` incoming edges (uniform without replacement when a
  node has more). Message passing is directed src -> dst, so a seed's
  receptive field is exactly the chain of events that led to it. With
  ``reverse_edges=True`` the expansion also walks outgoing edges and emits
  them reversed, so the receptive field additionally covers the chain the
  seed went on to produce (early-chain events whose evidence only exists
  downstream need this — detection-time provenance is bidirectional);
* each sampled subgraph is returned as a small ``HeteroData`` whose first
  ``batch_size`` nodes are the seeds (the model reads ``logits[:batch_size]``);
* a per-hop frontier cap (``sampling.max_frontier``) bounds subgraph growth
  for 8 GB VRAM; when it triggers, the kept frontier is a uniform random
  subset (standard GraphSAGE-style truncation).

Why balanced seeds: with ~1% incident events, uniform seed sampling would
give batches with a handful of positives — the focal gradient would be
dominated by the benign majority's easy negatives. We therefore draw half the
seeds from incident events and half from benign events every batch, while the
*sampled context* around them stays whatever the graph actually contains.

Why neighbourhood sampling at all: full-batch message passing over millions
of edges does not fit 8 GB VRAM. k-hop sampling with modest fanouts keeps
each subgraph in the tens of thousands of nodes.

Note: sampling runs in-process (``num_workers`` is accepted for API
compatibility but ignored) because the sampler reuses internal buffers.
"""
from __future__ import annotations

from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData

from .schema import EDGE_FEATURE_DIM, RELATIONS

#: with reverse_edges the sampled batches carry one extra edge column
#: (0 = message along causal time, 1 = message reversed against time) so
#: attention can treat past and future evidence differently. This column is
#: a sampling/model-time detail — the stored graph edge features (the KQL
#: contract) are unchanged.
BATCH_EDGE_DIM_FWD = EDGE_FEATURE_DIM
BATCH_EDGE_DIM_BIDIR = EDGE_FEATURE_DIM + 1


def _eval_fanouts(fanouts: Dict[str, List[int]], mult: int) -> Dict[str, List[int]]:
    return {r: [min(25, f * mult) for f in fs] for r, fs in fanouts.items()}


# --------------------------------------------------------------------------- #
# per-endpoint CSR for one relation                                           #
# --------------------------------------------------------------------------- #
class _EndpointCSR:
    """CSR over the edges of one relation keyed by one endpoint:
    key='dst' -> node v maps to all incoming (u -> v) edges;
    key='src' -> node u maps to all outgoing (u -> v) edges.
    Also keeps src/dst arrays for endpoint lookups."""

    __slots__ = ("src", "dst", "indptr", "eids")

    def __init__(self, src: np.ndarray, dst: np.ndarray, n_nodes: int,
                 key: str = "dst") -> None:
        if key not in ("dst", "src"):
            raise ValueError("key must be 'dst' or 'src'")
        self.src = np.ascontiguousarray(src, dtype=np.int64)
        self.dst = np.ascontiguousarray(dst, dtype=np.int64)
        key_arr = self.dst if key == "dst" else self.src
        order = np.argsort(key_arr, kind="stable")
        self.eids = order                                  # pos -> global eid
        k_sorted = key_arr[order]
        indptr = np.zeros(n_nodes + 1, dtype=np.int64)
        if len(k_sorted):
            np.add.at(indptr, k_sorted + 1, 1)
        np.cumsum(indptr, out=indptr)
        self.indptr = indptr

    def sample(self, frontier: np.ndarray, fanout: int,
               rng: np.random.Generator) -> np.ndarray:
        """Sample <= fanout incoming edges per frontier node.
        Returns global edge ids."""
        ip = self.indptr
        starts = ip[frontier]
        ends = ip[frontier + 1]
        deg = ends - starts
        total = int(deg.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64)

        row = np.repeat(np.arange(len(frontier)), deg)
        within = np.arange(total) - np.repeat(np.cumsum(deg) - deg, deg)
        pos = starts[row] + within

        if deg.max() > fanout:
            # uniform without replacement per node: keep the `fanout`
            # smallest random keys within each row (fully vectorised)
            keys = rng.random(total)
            order = np.lexsort((keys, row))
            rs = row[order]
            change = np.flatnonzero(np.diff(rs) != 0) + 1
            base = np.zeros(total, dtype=np.int64)
            base[change] = change
            np.maximum.accumulate(base, out=base)
            krank = np.empty(total, dtype=np.int64)
            krank[order] = np.arange(total) - base
            sel = krank < np.minimum(fanout, deg[row])
        else:
            sel = np.ones(total, dtype=bool)               # keep everything
        return self.eids[pos[sel]]


# --------------------------------------------------------------------------- #
# sampler                                                                     #
# --------------------------------------------------------------------------- #
class BalancedGraphSamplers:
    def __init__(self, data: HeteroData, split_id: np.ndarray, y: np.ndarray,
                 fanouts: Dict[str, List[int]], batch_seeds: int,
                 pos_seed_frac: float, eval_batch_seeds: int,
                 eval_fanout_mult: int = 2, num_workers: int = 0,
                 seed: int = 0, max_frontier: int = 20000,
                 min_steps_per_epoch: int = 120,
                 reverse_edges: bool = False) -> None:
        self.data = data
        self.rng = np.random.default_rng(seed + 777)
        self.fanouts = fanouts
        self.reverse_edges = reverse_edges
        self.eval_fanouts = _eval_fanouts(fanouts, eval_fanout_mult)
        self.batch_seeds = batch_seeds
        self.pos_batch = max(1, int(batch_seeds * pos_seed_frac))
        self.neg_batch = batch_seeds - self.pos_batch
        self.eval_batch_seeds = eval_batch_seeds
        self.max_frontier = max_frontier
        self.min_steps_per_epoch = min_steps_per_epoch
        self.num_hops = max(len(f) for f in fanouts.values())

        ev = data["event"]
        self.n_nodes = ev.num_nodes
        self.x = ev.x
        self.y_t = ev.y
        self.ts = ev.ts if "ts" in ev else None
        self.split_t = ev.split if "split" in ev else None

        self.csr_in: Dict[str, _EndpointCSR] = {}
        self.csr_out: Dict[str, _EndpointCSR] = {}
        self.eattr: Dict[str, torch.Tensor] = {}
        for rel in RELATIONS:
            store = data["event", rel, "event"]
            ei = store.edge_index
            if ei is None or ei.numel() == 0:
                src = np.empty(0, dtype=np.int64)
                dst = np.empty(0, dtype=np.int64)
            else:
                einp = ei.numpy()
                src, dst = einp[0], einp[1]
            self.csr_in[rel] = _EndpointCSR(src, dst, self.n_nodes, key="dst")
            if self.reverse_edges:
                self.csr_out[rel] = _EndpointCSR(src, dst, self.n_nodes, key="src")
            self.eattr[rel] = store.edge_attr

        # reusable local-id buffer (reset after every sampled subgraph)
        self._local = np.full(self.n_nodes, -1, dtype=np.int64)
        # diagnostics: how often the per-hop frontier cap had to truncate
        self.frontier_truncations = 0
        # diagnostics: global node ids of the most recent sample() result
        self.last_order: np.ndarray = np.empty(0, dtype=np.int64)

        self.idx: Dict[int, Dict[str, np.ndarray]] = {}
        for s, name in ((0, "train"), (1, "val"), (2, "test")):
            m = split_id == s
            pos = np.where(m & (y == 1))[0]
            neg = np.where(m & (y == 0))[0]
            self.idx[s] = {"pos": pos, "neg": neg, "all": np.where(m)[0]}
        if len(self.idx[0]["pos"]) == 0:
            raise ValueError("no positive train seeds — add incidents")

    # ------------------------------------------------------------------ #
    def steps_per_epoch(self) -> int:
        # every positive seed is seen at least once per epoch (oversampled
        # with replacement when positives don't fill a batch), and the epoch
        # is stretched to at least `min_steps_per_epoch` balanced batches so
        # benign context is not starved
        pos_steps = max(1, int(np.ceil(len(self.idx[0]["pos"]) / self.pos_batch)))
        return max(pos_steps, self.min_steps_per_epoch)

    # ------------------------------------------------------------------ #
    def sample(self, seeds: np.ndarray, fanouts: Dict[str, List[int]]) -> HeteroData:
        """Sample the k-hop causal past of `seeds` (and, when
        ``reverse_edges`` is set, also their causal future); returns a
        HeteroData whose first rows are the (deduplicated) seeds.

        Future edges are emitted into the subgraph REVERSED — the downstream
        event sends the message upstream — so a seed aggregates both the
        chain that produced it and the chain it went on to produce."""
        seeds = np.unique(np.asarray(seeds, dtype=np.int64))
        n_seeds = len(seeds)
        local = self._local
        parts: List[np.ndarray] = [seeds]
        touched: List[np.ndarray] = [seeds]
        local[seeds] = np.arange(n_seeds)

        selected: Dict[str, List[np.ndarray]] = {r: [] for r in RELATIONS}
        selected_rev: Dict[str, List[np.ndarray]] = {r: [] for r in RELATIONS}
        frontier = seeds
        n_registered = n_seeds
        for h in range(self.num_hops):
            src_parts: List[np.ndarray] = []
            hop_edges: Dict[str, np.ndarray] = {}
            hop_redges: Dict[str, np.ndarray] = {}
            for rel in RELATIONS:
                fs = fanouts[rel]
                if h >= len(fs) or len(frontier) == 0:
                    continue
                eids = self.csr_in[rel].sample(frontier, fs[h], self.rng)
                if len(eids) > 0:
                    hop_edges[rel] = eids
                    src_parts.append(self.csr_in[rel].src[eids])
                if self.reverse_edges:
                    reids = self.csr_out[rel].sample(frontier, fs[h], self.rng)
                    if len(reids) > 0:
                        hop_redges[rel] = reids
                        src_parts.append(self.csr_out[rel].dst[reids])

            if not src_parts:
                break

            # register new (unvisited) neighbour nodes as the next frontier
            cand = np.unique(np.concatenate(src_parts))
            cand = cand[local[cand] < 0]
            if len(cand) > self.max_frontier:
                self.frontier_truncations += 1
                cand = self.rng.choice(cand, size=self.max_frontier, replace=False)
            if len(cand):
                local[cand] = np.arange(n_registered, n_registered + len(cand))
                n_registered += len(cand)
                parts.append(cand)
                touched.append(cand)

            # keep only edges whose endpoints are both inside the subgraph
            # (drops edges into frontier-truncated nodes)
            for rel, eids in hop_edges.items():
                s = self.csr_in[rel].src[eids]
                keep = local[s] >= 0
                if keep.any():
                    selected[rel].append(eids[keep])
            for rel, eids in hop_redges.items():
                d = self.csr_out[rel].dst[eids]
                keep = local[d] >= 0
                if keep.any():
                    selected_rev[rel].append(eids[keep])

            frontier = cand
            if len(frontier) == 0:
                break

        order = np.concatenate(parts)
        self.last_order = order.copy()
        nodes_t = torch.from_numpy(order)
        batch = HeteroData()
        batch["event"].x = self.x[nodes_t]
        batch["event"].y = self.y_t[nodes_t]
        if self.ts is not None:
            batch["event"].ts = self.ts[nodes_t]
        batch["event"].batch_size = int(n_seeds)
        if self.split_t is not None:
            batch["event"].split = self.split_t[nodes_t]
        for rel in RELATIONS:
            store = batch["event", rel, "event"]
            idx_parts: List[np.ndarray] = []
            eid_parts: List[np.ndarray] = []
            dir_parts: List[np.ndarray] = []
            if selected[rel]:
                e = np.unique(np.concatenate(selected[rel]))
                s = self.csr_in[rel].src[e]
                d = self.csr_in[rel].dst[e]
                idx_parts.append(np.stack([local[s], local[d]]))
                eid_parts.append(e)
                if self.reverse_edges:
                    dir_parts.append(np.zeros(len(e), dtype=np.float32))
            if selected_rev[rel]:
                e = np.unique(np.concatenate(selected_rev[rel]))
                s = self.csr_out[rel].src[e]
                d = self.csr_out[rel].dst[e]
                # reversed: future event (d) sends its message to past (s)
                idx_parts.append(np.stack([local[d], local[s]]))
                eid_parts.append(e)
                if self.reverse_edges:
                    dir_parts.append(np.ones(len(e), dtype=np.float32))
            if idx_parts:
                store.edge_index = torch.from_numpy(
                    np.concatenate(idx_parts, axis=1).astype(np.int64))
                attrs = self.eattr[rel][torch.from_numpy(np.concatenate(eid_parts))]
                if self.reverse_edges:
                    # extra direction column (0 = with time, 1 = against time)
                    attrs = torch.cat([attrs, torch.from_numpy(
                        np.concatenate(dir_parts)).unsqueeze(-1)], dim=-1)
                store.edge_attr = attrs
            else:
                store.edge_index = torch.zeros(2, 0, dtype=torch.long)
                store.edge_attr = torch.zeros(
                    0, BATCH_EDGE_DIM_BIDIR if self.reverse_edges
                    else BATCH_EDGE_DIM_FWD)

        for t in touched:                                   # reset the buffer
            local[t] = -1
        return batch

    # ------------------------------------------------------------------ #
    def train_batches(self) -> Iterator[Tuple[HeteroData, HeteroData]]:
        """Yields (pos_batch, neg_batch) HeteroData subgraphs, 1:1 per step."""
        steps = self.steps_per_epoch()
        pos_pool = self.idx[0]["pos"]
        need_pos = steps * self.pos_batch
        replace = len(pos_pool) < need_pos
        pos = self.rng.choice(pos_pool, size=need_pos, replace=replace)
        neg_pool = self.idx[0]["neg"]
        neg = self.rng.choice(neg_pool, size=steps * self.neg_batch,
                              replace=steps * self.neg_batch > len(neg_pool))

        for s in range(steps):
            p = pos[s * self.pos_batch:(s + 1) * self.pos_batch]
            ng = neg[s * self.neg_batch:(s + 1) * self.neg_batch]
            yield self.sample(p, self.fanouts), self.sample(ng, self.fanouts)

    def eval_loader(self, split: int) -> Iterator[HeteroData]:
        idx = self.idx[split]["all"]
        for i in range(0, len(idx), self.eval_batch_seeds):
            yield self.sample(idx[i:i + self.eval_batch_seeds], self.eval_fanouts)
