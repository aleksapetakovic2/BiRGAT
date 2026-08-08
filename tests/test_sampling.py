import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from eprgat.config import Config
from eprgat.graph import prepare_dataset
from eprgat.sampling import BalancedGraphSamplers
from eprgat.schema import RELATIONS


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    cfg = Config()
    cfg.data.days = 4
    cfg.data.n_hosts = 40
    cfg.data.n_users = 30
    cfg.data.n_incidents = 8
    cfg.data.attack_events_range = [15, 30]
    cfg.graph.split_train_frac = 0.55
    cfg.graph.split_val_frac = 0.20
    cfg.graph.split_gap_hours = 2.0
    cfg.data.data_dir = str(tmp_path_factory.mktemp("data"))
    bundle = prepare_dataset(cfg, log=lambda m: None)
    split = bundle.data["event"].split.numpy()
    y = bundle.data["event"].y.numpy()
    samplers = BalancedGraphSamplers(
        bundle.data, split, y, cfg.sampling.fanouts, cfg.sampling.batch_seeds,
        cfg.sampling.pos_seed_frac, cfg.sampling.eval_batch_seeds,
        cfg.sampling.eval_fanout_mult, cfg.sampling.num_workers,
        seed=cfg.data.seed, max_frontier=cfg.sampling.max_frontier)
    return cfg, bundle, samplers, split, y


def test_sample_contract(setup):
    cfg, bundle, samplers, split, y = setup
    seeds = np.where((split == 0))[0][:77]
    batch = samplers.sample(seeds, cfg.sampling.fanouts)
    ev = batch["event"]
    assert ev.batch_size == len(np.unique(seeds))
    assert ev.x.shape[0] == ev.y.shape[0]
    assert ev.x.shape[0] >= ev.batch_size
    # seeds are the first rows, deduplicated and sorted
    assert (ev.y[:ev.batch_size].numpy() == y[np.unique(seeds)]).all()
    # seeds carry their split id (subgraph-wide supervision relies on it)
    assert "split" in batch["event"]
    assert (batch["event"].split[:ev.batch_size].numpy() == 0).all()
    for rel in RELATIONS:
        store = batch["event", rel, "event"]
        ei = store.edge_index
        assert ei.shape[0] == 2
        assert store.edge_attr.shape[0] == ei.shape[1]
        if ei.shape[1]:
            assert (ei >= 0).all() and (ei < ev.x.shape[0]).all()


def test_sampled_edges_point_backwards_in_time(setup):
    """Provenance edges are causal: sampled src must not happen after dst."""
    cfg, bundle, samplers, split, y = setup
    seeds = np.where((split == 0) & (y == 1))[0]
    batch = samplers.sample(seeds, cfg.sampling.fanouts)
    ts = batch["event"].ts.numpy()
    for rel in RELATIONS:
        ei = batch["event", rel, "event"].edge_index.numpy()
        if ei.shape[1]:
            assert (ts[ei[0]] <= ts[ei[1]] + 1e-9).all(), rel


def test_fanout_cap_holds(setup):
    cfg, bundle, samplers, split, y = setup
    seeds = np.random.default_rng(0).choice(np.where(split == 0)[0], 128, replace=False)
    batch = samplers.sample(seeds, cfg.sampling.fanouts)
    n = batch["event"].x.shape[0]
    for rel in RELATIONS:
        cap = max(cfg.sampling.fanouts[rel])
        dst = batch["event", rel, "event"].edge_index[1].numpy()
        indeg = np.bincount(dst, minlength=n)
        assert indeg.max() <= cap, (rel, indeg.max(), cap)


def test_train_batches_balanced(setup):
    cfg, bundle, samplers, split, y = setup
    steps = samplers.steps_per_epoch()
    assert steps >= 1
    seen_steps = 0
    for pos_b, neg_b in samplers.train_batches():
        seen_steps += 1
        assert (pos_b["event"].y[:pos_b["event"].batch_size].numpy() == 1).all()
        assert (neg_b["event"].y[:neg_b["event"].batch_size].numpy() == 0).all()
    assert seen_steps == steps


def test_eval_loader_covers_split(setup):
    cfg, bundle, samplers, split, y = setup
    total = 0
    for s in (1, 2):
        n_expected = (split == s).sum()
        n = sum(b["event"].batch_size for b in samplers.eval_loader(s))
        assert n == n_expected, (s, n, n_expected)
        total += n
    assert total > 0


def test_local_id_buffer_is_reset(setup):
    cfg, bundle, samplers, split, y = setup
    seeds = np.where(split == 0)[0][:50]
    samplers.sample(seeds, cfg.sampling.fanouts)
    assert (samplers._local == -1).all()


def _chain_data():
    """Tiny hand-built chain 0 -> 1 -> 2 (PROCESS_CHILD) for semantic tests."""
    import torch
    from torch_geometric.data import HeteroData
    from eprgat.schema import EDGE_FEATURE_DIM, FEATURE_DIM
    data = HeteroData()
    data["event"].x = torch.randn(3, FEATURE_DIM)
    data["event"].y = torch.tensor([1, 1, 0])
    data["event"].ts = torch.tensor([0.0, 1.0, 2.0])
    data["event"].split = torch.zeros(3, dtype=torch.long)
    for rel in RELATIONS:
        if rel == "PROCESS_CHILD":
            data["event", rel, "event"].edge_index = torch.tensor([[0, 1], [1, 2]])
            data["event", rel, "event"].edge_attr = torch.zeros(2, EDGE_FEATURE_DIM)
        else:
            data["event", rel, "event"].edge_index = torch.zeros(2, 0, dtype=torch.long)
            data["event", rel, "event"].edge_attr = torch.zeros(0, EDGE_FEATURE_DIM)
    return data


def test_reverse_edges_expose_causal_future():
    """With reverse_edges, a chain-start seed must reach its descendants, and
    future edges must be emitted reversed (future event sends the message)."""
    data = _chain_data()
    split = np.zeros(3, dtype=np.int64)
    y = np.array([1, 1, 0])
    fanouts = {r: [4, 4] for r in RELATIONS}

    fwd = BalancedGraphSamplers(data, split, y, fanouts, 4, 0.5, 4,
                                seed=0, reverse_edges=False)
    b = fwd.sample(np.array([0]), fanouts)
    assert b["event"].x.shape[0] == 1              # seed 0 has no causal past

    rev = BalancedGraphSamplers(data, split, y, fanouts, 4, 0.5, 4,
                                seed=0, reverse_edges=True)
    b = rev.sample(np.array([0]), fanouts)
    assert b["event"].x.shape[0] == 3              # seed 0 sees its future chain
    assert b["event"].batch_size == 1
    ei = b["event", "PROCESS_CHILD", "event"].edge_index
    ts = b["event"].ts.numpy()
    assert ei.shape[1] >= 1
    # at least one message flows from a later event to an earlier one
    assert (ts[ei[0].numpy()] > ts[ei[1].numpy()]).any()
    # every edge stays inside the subgraph
    assert (ei >= 0).all() and (ei < 3).all()
    # bidirectional batches carry the extra direction edge column and it
    # flags exactly the against-time messages
    ea = b["event", "PROCESS_CHILD", "event"].edge_attr
    from eprgat.schema import EDGE_FEATURE_DIM
    assert ea.shape[1] == EDGE_FEATURE_DIM + 1
    against_time = ts[ei[0].numpy()] > ts[ei[1].numpy()]
    assert (ea[:, -1].numpy() == against_time.astype(np.float32)).all()


def test_forward_only_batches_keep_schema_edge_width(setup):
    cfg, bundle, samplers, split, y = setup
    from eprgat.schema import EDGE_FEATURE_DIM
    seeds = np.where((split == 0) & (y == 1))[0]
    batch = samplers.sample(seeds, cfg.sampling.fanouts)
    for rel in RELATIONS:
        assert batch["event", rel, "event"].edge_attr.shape[-1] == EDGE_FEATURE_DIM
