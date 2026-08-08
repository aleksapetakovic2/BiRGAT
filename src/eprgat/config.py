"""Configuration dataclasses + YAML loading.

Everything that can change lives here; a resolved snapshot of the config is
saved into every run directory for reproducibility.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from .schema import RELATIONS


# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    seed: int = 1337
    preset: str = "full"                 # informational; yaml carries the numbers
    days: int = 28                       # simulated timeline length
    n_hosts: int = 350
    n_users: int = 300
    n_incidents: int = 75                # attack scenarios injected over the timeline
    # target observable events per incident: templates are extended with
    # follow-on behaviour until they reach at least `lo` (hi = soft cap)
    attack_events_range: List[int] = field(default_factory=lambda: [25, 60])
    data_dir: str = "data"               # cache dir for generated .pt artifacts
    regenerate: bool = False             # ignore cache


@dataclass
class GraphConfig:
    host_seq_window_min: float = 10.0    # HOST_SEQUENCE join window
    user_seq_window_min: float = 15.0    # USER_SEQUENCE join window
    net_trigger_window_min: float = 10.0  # NET_TRIGGER correlation window on dst host
    net_trigger_max_links: int = 2
    config_trigger_min_min: float = 5.0  # CONFIG_TRIGGER: min delay to spawned process
    config_trigger_max_h: float = 24.0   # CONFIG_TRIGGER: max delay (persistence firing)
    config_trigger_max_links: int = 2
    session_max_h: float = 12.0          # SignIn session lifetime for SESSION_ACTION edges
    max_out_degree_per_rel: int = 4      # per-(node, relation) outgoing cap (KQL would cap too)
    split_train_frac: float = 0.65
    split_val_frac: float = 0.17         # remainder (~0.18) is test
    split_gap_hours: float = 6.0         # edge-dead zone between splits (leakage firewall)
    leakage_auc_threshold: float = 0.95  # fail loudly if any single feature exceeds this


@dataclass
class ModelConfig:
    model: str = "rgat"                  # rgat | mlp (mlp = feature-only baseline)
    hidden_dim: int = 96
    num_layers: int = 3
    num_heads: int = 4
    feat_drop: float = 0.10
    attn_drop: float = 0.10
    use_edge_bias: bool = True           # temporal edge features modulate attention
    residual: bool = True
    negative_slope: float = 0.2          # LeakyReLU in attention
    # head input: "fusion" = [last layer || raw features];
    #             "jk"     = [all layer outputs || raw features] (JumpingKnowledge)
    readout: str = "fusion"
    # 2-layer feature encoder instead of 1: gives the model a stronger
    # features-only pathway, which is what isolated positives (no incident
    # context within sampling reach) rely on exclusively.
    deep_input_proj: bool = False


@dataclass
class SamplingConfig:
    # per-relation fanouts (len = number of sampled hops). Smaller on poor VRAM.
    fanouts: Dict[str, List[int]] = field(default_factory=lambda: {
        "PROCESS_CHILD": [6, 4, 2],
        "SESSION_ACTION": [6, 4, 2],
        "HOST_SEQUENCE": [4, 2, 2],
        "USER_SEQUENCE": [4, 2, 2],
        "NET_TRIGGER": [6, 4, 3],
        "CONFIG_TRIGGER": [4, 2, 2],
    })
    eval_fanout_mult: int = 2            # eval uses fanouts * mult (more context, stable scores)
    batch_seeds: int = 512               # seeds per training batch (pos+neg combined)
    pos_seed_frac: float = 0.5           # balanced seeds: half malicious, half benign
    eval_batch_seeds: int = 1024
    max_frontier: int = 20000            # per-hop cap on newly sampled nodes (VRAM guard)
    # an epoch covers every positive seed at least once AND at least this many
    # balanced batches, so the model also sees plenty of benign context.
    # Careful: with ~1% positives, large values re-seed the same positives
    # many times per epoch and the model memorises them (val AUPRC collapses).
    # 40 keeps positive re-seeding around ~7x/epoch at pos_seed_frac 0.4.
    min_steps_per_epoch: int = 40
    # also expand the seeds' causal FUTURE (out-edges). Future edges are
    # emitted into the subgraph reversed, so messages flow from consequence
    # back to cause — a seed then aggregates both its past and what it led to.
    # Detection-time provenance is bidirectional; early-chain events whose
    # evidence only exists downstream need this.
    reverse_edges: bool = False
    # eval seeds are scored eval_passes times on independently sampled
    # subgraphs and the probabilities averaged — removes noise from the
    # random neighbourhood sampling at scoring time (training unaffected).
    eval_passes: int = 1
    num_workers: int = 0                 # accepted for API compat; sampling is in-process


@dataclass
class TrainConfig:
    epochs: int = 40
    # training RNG seed (init, dropout, sampling). -1 = use data.seed, which
    # reproduces the historical single-seed behaviour. Setting it lets several
    # models train on the SAME dataset (data.seed controls generation and
    # caching) for honest seed-ensemble averaging.
    seed: int = -1
    lr: float = 2e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 150
    grad_clip: float = 1.0
    focal_gamma: float = 2.0             # focal loss modulation factor
    focal_alpha: float = 0.80            # weight of the positive (incident) class
    patience: int = 8                    # early stopping on val AUPRC
    min_delta: float = 1e-4
    # operating-point policy for the val-tuned threshold (applied identically
    # to the RGAT and the MLP baseline):
    #   f1     — threshold maximising F1 (balanced);
    #   recall — highest threshold still reaching recall >= threshold_min_recall
    #            (use when missing a true positive is costlier than a false alarm)
    threshold_policy: str = "f1"
    threshold_min_recall: float = 0.90
    log_every: int = 5                   # verbose step logging cadence
    amp: bool = False                    # fp16 autocast on CUDA (optional)
    device: str = "auto"                 # auto | cuda | cpu
    run_name: str = ""
    run_mlp_baseline: bool = True        # feature-only reference model


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Config":
        cfg = Config()
        for group_name, group_dc in (
            ("data", cfg.data), ("graph", cfg.graph), ("model", cfg.model),
            ("sampling", cfg.sampling), ("train", cfg.train),
        ):
            section = d.get(group_name, {}) or {}
            for k, v in section.items():
                if hasattr(group_dc, k):
                    setattr(group_dc, k, v)
                else:
                    print(f"[config] WARNING: unknown key {group_name}.{k} ignored")
        return cfg

    @staticmethod
    def load(path: Optional[str]) -> "Config":
        if not path:
            return Config()
        with open(path, "r", encoding="utf-8") as f:
            return Config.from_dict(yaml.safe_load(f) or {})


def apply_cli_overrides(cfg: Config, overrides: Dict[str, Any]) -> Config:
    """Apply flat `group.key=value` overrides coming from the CLI."""
    for dotted, value in (overrides or {}).items():
        if "." not in dotted:
            raise ValueError(f"--set expects group.key=value, got '{dotted}'")
        group, key = dotted.split(".", 1)
        if not hasattr(cfg, group):
            raise ValueError(f"unknown config group '{group}'")
        g = getattr(cfg, group)
        if not hasattr(g, key):
            raise ValueError(f"unknown config key '{dotted}'")
        cur = getattr(g, key)
        # cast to the current type (bool/int/float/str), keep dicts/lists raw
        if isinstance(cur, bool):
            value = str(value).lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int):
            value = int(value)
        elif isinstance(cur, float):
            value = float(value)
        elif isinstance(cur, (dict, list)):
            value = json.loads(value)
        setattr(g, key, value)
    return cfg
