#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
PROFILE_SCRIPT = RUN_DIR / "profile_spmm_migration.py"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run profile_spmm_migration.py over ft_host_alloc values.")
    parser.add_argument("--dataset", required=True, help="dataset name")
    parser.add_argument("--ft_matrix", required=True, choices=("uvm", "hmm"), help="feature memory mode")
    parser.add_argument("--allocs", default="20,40,60,80", help="comma-separated ft_host_alloc sweep values")
    parser.add_argument("--framework", default="pyg", choices=("pyg", "dgl"), help="frontend/backend stack")
    parser.add_argument("--model", default="gcn", choices=("gcn", "gin", "sag", "graphsage"), help="model to profile")
    parser.add_argument("--dim", type=int, default=128, help="base feature / hidden / output dimension")
    parser.add_argument("--num_layers", type=int, default=1, help="number of layers")
    parser.add_argument("--adj_matrix", default="device", choices=("device", "uvm", "hmm"), help="adjacency memory mode")
    parser.add_argument("--weight", default="device", choices=("device", "uvm"), help="weight/output memory mode")
    parser.add_argument("--warmup", type=int, default=1, help="warmup iterations")
    parser.add_argument("--iters", type=int, default=5, help="measured iterations")
    parser.add_argument("--device", default="cuda:0", help="CUDA device string")
    parser.add_argument(
        "--output_dir",
        default=str(RUN_DIR.parent / "report" / "nsys_spmm_sweep"),
        help="directory for nsys artifacts",
    )
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="extra args passed after `--`")
    args = parser.parse_args()

    allocs = [item.strip() for item in args.allocs.split(",") if item.strip()]
    if not allocs:
        raise ValueError("--allocs must contain at least one value")

    extra_args = list(args.extra_args or [])
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    for alloc in allocs:
        cmd = [
            sys.executable,
            str(PROFILE_SCRIPT),
            "--dataset",
            args.dataset,
            "--ft_matrix",
            args.ft_matrix,
            "--ft_host_alloc",
            alloc,
            "--framework",
            args.framework,
            "--model",
            args.model,
            "--dim",
            str(args.dim),
            "--num_layers",
            str(args.num_layers),
            "--adj_matrix",
            args.adj_matrix,
            "--weight",
            args.weight,
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
            "--device",
            args.device,
            "--output_dir",
            args.output_dir,
            *extra_args,
        ]
        print(f"\n=== ft_host_alloc={alloc} ===", flush=True)
        subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
