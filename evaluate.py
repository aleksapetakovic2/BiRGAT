#!/usr/bin/env python
"""Entry point: python evaluate.py --run runs/<dir> [--no-rewire]"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from eprgat.cli import main_evaluate  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main_evaluate())
