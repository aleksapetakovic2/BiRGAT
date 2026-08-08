#!/usr/bin/env python
"""Entry point: python train.py --config configs/full.yaml [--set k=v ...]"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from eprgat.cli import main_train  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main_train())
