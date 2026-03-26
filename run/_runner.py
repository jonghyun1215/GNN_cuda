#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
from pathlib import Path

from _bootstrap import bootstrap_pythonpath


DEFAULT_DATA_ROOT = os.environ.get("GNN_DATASET_ROOT", "/root/workspace/mnt/dataset_npz")
DEFAULT_FRAMEWORK_ENVS = {
    "pyg": os.environ.get("GNN_PYG_ENV", "cu128_pyg"),
    "dgl": os.environ.get("GNN_DGL_ENV", "cu128_dgl"),
}


def _dataset_paths(data_root: str) -> list[Path]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"dataset directory not found: {root}")
    dataset_paths = sorted(root.glob("*.npz"))
    if not dataset_paths:
        raise FileNotFoundError(f"no .npz datasets found in: {root}")
    return dataset_paths


def _current_conda_env() -> str:
    env_name = os.environ.get("CONDA_DEFAULT_ENV", "")
    if env_name:
        return env_name
    prefix = os.environ.get("CONDA_PREFIX", "")
    if prefix:
        return Path(prefix).name
    return ""


def _ensure_framework_env(framework: str) -> None:
    target_env = DEFAULT_FRAMEWORK_ENVS[framework]
    current_env = _current_conda_env()
    if current_env == target_env:
        return
    conda = shutil.which("conda")
    if conda is None:
        raise RuntimeError("conda not found; activate the target env manually or source setup_environment first")
    script_path = str(Path(sys.argv[0]).resolve())
    cmd = [conda, "run", "--no-capture-output", "-n", target_env, "python", script_path, *sys.argv[1:]]
    print(f"Dispatching to conda env '{target_env}' for framework '{framework}'")
    os.execvpe(conda, cmd, os.environ.copy())


def dispatch_model(targets: dict[str, str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--framework", type=str, default="pyg", choices=sorted(targets))
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    args, remaining = parser.parse_known_args()

    _ensure_framework_env(args.framework)
    bootstrap_pythonpath()
    module = importlib.import_module(targets[args.framework])
    model_main = module.main
    data_root = str(args.data_root or DEFAULT_DATA_ROOT)

    if args.dataset == "all":
        dataset_paths = _dataset_paths(data_root)
        for index, dataset_path in enumerate(dataset_paths, start=1):
            print(f"[{index}/{len(dataset_paths)}] dataset={dataset_path.stem} path={dataset_path}")
            sys.argv = [sys.argv[0], "--dataset", dataset_path.stem, "--data_root", data_root, *remaining]
            rc = int(model_main())
            if rc != 0:
                return rc
        return 0

    argv = [sys.argv[0], "--data_root", data_root, *remaining]
    if args.dataset is not None:
        argv[1:1] = ["--dataset", args.dataset]
    sys.argv = argv
    return int(model_main())
