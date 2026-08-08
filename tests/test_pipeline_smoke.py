import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from eprgat.config import Config
from eprgat.graph import prepare_dataset
from eprgat.schema import RELATIONS


def _small_cfg(tmp):
    cfg = Config()
    cfg.data.days = 4
    cfg.data.n_hosts = 40
    cfg.data.n_users = 30
    cfg.data.n_incidents = 8
    cfg.data.attack_events_range = [15, 30]
    cfg.data.data_dir = str(tmp)
    # smoke-like split fractions so every split receives incident events
    cfg.graph.split_train_frac = 0.55
    cfg.graph.split_val_frac = 0.20
    cfg.graph.split_gap_hours = 2.0
    return cfg


def test_prepare_dataset(tmp_path):
    cfg = _small_cfg(tmp_path)
    bundle = prepare_dataset(cfg, log=lambda m: None)
    d = bundle.data

    assert d["event"].x.shape[0] == d["event"].y.shape[0]
    assert torch.isfinite(d["event"].x).all()
    for rel in RELATIONS:
        ei = d["event", rel, "event"].edge_index
        ea = d["event", rel, "event"].edge_attr
        assert ei.shape[0] == 2 and ea.shape[0] == ei.shape[1]
        assert (ei >= 0).all() and (ei < d["event"].num_nodes).all()

    split = d["event"].split.numpy()
    y = d["event"].y.numpy()
    for s in (0, 1, 2):
        m = split == s
        assert m.sum() > 0
        assert y[m].sum() > 0, f"split {s} has no positives"

    # no edges across splits
    for rel in RELATIONS:
        ei = d["event", rel, "event"].edge_index.numpy()
        assert (split[ei[0]] == split[ei[1]]).all()

    # standardisation stats are finite
    assert np.isfinite(bundle.X_mean).all() and np.isfinite(bundle.X_std).all()
    assert (bundle.X_std > 0).all()
