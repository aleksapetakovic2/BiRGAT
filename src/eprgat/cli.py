"""Command-line entry points: train / evaluate / data-only."""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from typing import List, Optional

import numpy as np
import torch

from .config import Config, apply_cli_overrides
from .graph import prepare_dataset
from .model import merge_flat_batches  # noqa: F401  (import-time sanity)
from .sampling import BalancedGraphSamplers
from .schema import describe_schema
from .trainer import Trainer, train_mlp_baseline


# --------------------------------------------------------------------------- #
class RunLogger:
    """Prints to console and appends to <run_dir>/train.log with timestamps."""

    def __init__(self, run_dir: str) -> None:
        os.makedirs(run_dir, exist_ok=True)
        self.path = os.path.join(run_dir, "train.log")
        self._f = open(self.path, "a", encoding="utf-8", buffering=1)

    def __call__(self, msg: str) -> None:
        line = f"{_dt.datetime.now().strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        self._f.write(line + "\n")

    def close(self) -> None:
        self._f.close()


def resolve_device(cfg: Config, log) -> torch.device:
    want = cfg.train.device
    if want == "cuda" or (want == "auto" and torch.cuda.is_available()):
        dev = torch.device("cuda")
        prop = torch.cuda.get_device_properties(0)
        log(f"[env ] CUDA device: {prop.name}  VRAM {prop.total_memory/2**30:.1f} GB  "
            f"torch {torch.__version__}")
        return dev
    log(f"[env ] using CPU (torch {torch.__version__})")
    return torch.device("cpu")


def make_run_dir(cfg: Config) -> str:
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = cfg.train.run_name or f"{cfg.data.preset}_{cfg.model.model}"
    return os.path.join("runs", f"{ts}_{name}")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def training_seed(cfg: Config) -> int:
    """RNG seed for training/eval sampling; data.seed keeps controlling the
    dataset itself (so ensembles with different train seeds share data)."""
    return cfg.train.seed if getattr(cfg.train, "seed", -1) >= 0 else cfg.data.seed


# --------------------------------------------------------------------------- #
def main_train(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Train the event-provenance RGAT incident detector.")
    ap.add_argument("--config", type=str, default=None,
                    help="YAML config (configs/smoke.yaml | medium.yaml | full.yaml)")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="GROUP.KEY=VALUE",
                    help="override any config value, e.g. --set train.epochs=5")
    ap.add_argument("--data-only", action="store_true",
                    help="only generate/build the graph, then exit")
    args = ap.parse_args(argv)

    overrides = {}
    for item in args.overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects group.key=value, got '{item}'")
        k, v = item.split("=", 1)
        overrides[k.strip()] = v.strip()
    cfg = apply_cli_overrides(Config.load(args.config), overrides)

    run_dir = make_run_dir(cfg)
    log = RunLogger(run_dir)
    log("=" * 78)
    log("EVENT-PROVENANCE RGAT  |  focal loss  |  neighbourhood sampling")
    log("=" * 78)
    log(describe_schema())
    cfg.save(os.path.join(run_dir, "config.yaml"))
    log(f"[run ] run dir: {run_dir}")

    seed_everything(training_seed(cfg))
    device = resolve_device(cfg, log)

    bundle = prepare_dataset(cfg, log)
    if args.data_only:
        log("[run ] --data-only: stopping after dataset build.")
        return 0

    split_id = bundle.data["event"].split.numpy()
    y = bundle.data["event"].y.numpy()

    if cfg.model.model == "mlp":
        out = train_mlp_baseline(cfg, bundle, run_dir, device, log)
        from .metrics import format_metrics
        log(format_metrics(out["test"], "TEST (MLP baseline)"))
        return 0

    samplers = BalancedGraphSamplers(
        bundle.data, split_id, y, cfg.sampling.fanouts, cfg.sampling.batch_seeds,
        cfg.sampling.pos_seed_frac, cfg.sampling.eval_batch_seeds,
        cfg.sampling.eval_fanout_mult, cfg.sampling.num_workers,
        seed=training_seed(cfg), max_frontier=cfg.sampling.max_frontier,
        min_steps_per_epoch=cfg.sampling.min_steps_per_epoch,
        reverse_edges=cfg.sampling.reverse_edges)
    trainer = Trainer(cfg, bundle, samplers, run_dir, device, log)
    trainer.train()

    from .evaluate import evaluate_run
    evaluate_run(cfg, bundle, run_dir, device, log, do_rewire=True)

    if cfg.train.run_mlp_baseline:
        from .metrics import format_metrics
        out = train_mlp_baseline(cfg, bundle, run_dir, device, log)
        log(format_metrics(out["test"], "TEST (MLP feature-only baseline)"))
        log(format_metrics(out["test_alt"],
                           f"TEST (MLP baseline, alt operating point: {out['alt_policy']})"))
        log("[run ] Compare: if RGAT test AUPRC >> MLP AUPRC, the model is "
            "reading TOPOLOGY, not just features — exactly the point of "
            "event-centric provenance graphs.")

    log("[run ] done.")
    log.close()
    return 0


def main_evaluate(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Re-evaluate a trained run directory.")
    ap.add_argument("--run", required=True, help="path to a runs/<...> directory")
    ap.add_argument("--no-rewire", action="store_true")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="GROUP.KEY=VALUE",
                    help="override config for this re-evaluation only "
                         "(e.g. --set sampling.eval_passes=4); the run's saved "
                         "training config is not modified")
    args = ap.parse_args(argv)

    overrides = {}
    for item in args.overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects group.key=value, got '{item}'")
        k, v = item.split("=", 1)
        overrides[k.strip()] = v.strip()

    cfg = Config.load(os.path.join(args.run, "config.yaml"))
    cfg = apply_cli_overrides(cfg, overrides)
    log = RunLogger(args.run)
    if overrides:
        log(f"[eval] re-evaluation overrides: {overrides}")
    device = resolve_device(cfg, log)
    bundle = prepare_dataset(cfg, log)
    from .evaluate import evaluate_run
    evaluate_run(cfg, bundle, args.run, device, log,
                 do_rewire=not args.no_rewire)
    log.close()
    return 0
