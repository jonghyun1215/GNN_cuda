#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
GNN_CUDA_ROOT = Path(__file__).resolve().parents[1]


def bootstrap_pythonpath() -> None:
    for path in (WORKSPACE_ROOT,):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
