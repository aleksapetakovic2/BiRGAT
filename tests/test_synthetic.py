import numpy as np

from eprgat.config import Config
from eprgat.schema import FEATURE_DIM
from eprgat.synthetic import generate_dataset


def _small_cfg():
    cfg = Config()
    cfg.data.days = 4
    cfg.data.n_hosts = 40
    cfg.data.n_users = 30
    cfg.data.n_incidents = 8
    cfg.data.attack_events_range = [15, 30]
    # smoke-like split fractions so every split receives incident events
    # even on a short timeline (chains span up to ~12 h)
    cfg.graph.split_train_frac = 0.55
    cfg.graph.split_val_frac = 0.20
    cfg.graph.split_gap_hours = 2.0
    return cfg


def test_generator_basics():
    tab = generate_dataset(_small_cfg())
    c = tab.cols
    N = len(tab)
    assert N > 1000

    # sorted by time
    assert (np.diff(c["ts"]) >= -1e-9).all()

    # labels and incident ids are consistent
    assert ((c["incident"] >= 0) == (c["y"] == 1)).all()
    pos_rate = c["y"].mean()
    assert 0.002 < pos_rate < 0.05, f"positive rate {pos_rate} out of range"

    # lineage references point backwards in time
    v = c["parent"] >= 0
    assert (c["ts"][c["parent"][v]] <= c["ts"][v] + 1e-6).all()
    v = c["signin"] >= 0
    assert (c["ts"][c["signin"][v]] <= c["ts"][v] + 1e-6).all()


def test_feature_matrix_builds():
    from eprgat.graph import build_node_features
    from eprgat.schema import ETYPE_ID, FEATURE_OFFSETS
    tab = generate_dataset(_small_cfg())
    X = build_node_features(tab)
    assert X.shape == (len(tab), FEATURE_DIM)
    assert np.isfinite(X).all()
    # every event carries exactly one event-type one-hot
    s, e = FEATURE_OFFSETS["etype_onehot"]
    assert (X[:, s:e].sum(axis=1) == 1.0).all()
    assert X[:, s + ETYPE_ID["SignIn"]].sum() > 0
