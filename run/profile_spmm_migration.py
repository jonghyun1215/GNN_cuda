#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = os.environ.get("GNN_DATASET_ROOT", "/root/workspace/mnt/dataset_npz")
DEFAULT_FRAMEWORK_ENVS = {
    "pyg": os.environ.get("GNN_PYG_ENV", "cu128_pyg"),
    "dgl": os.environ.get("GNN_DGL_ENV", "cu128_dgl"),
}
MODEL_DIRS = {
    "gcn": "GCN",
    "gin": "GIN",
    "sag": "GraphSAGE",
    "graphsage": "GraphSAGE",
}


@dataclass(frozen=True)
class TimeRange:
    start_ns: int
    end_ns: int

    @property
    def duration_ns(self) -> int:
        return max(0, self.end_ns - self.start_ns)


def _run(cmd: list[str], *, stdout_path: Path | None = None, stderr_path: Path | None = None) -> None:
    kwargs = {"check": True, "text": True}
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        kwargs["stdout"] = stdout_path.open("w", encoding="utf-8")
    if stderr_path is not None:
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        kwargs["stderr"] = stderr_path.open("w", encoding="utf-8")
    try:
        subprocess.run(cmd, **kwargs)
    finally:
        if "stdout" in kwargs:
            kwargs["stdout"].close()
        if "stderr" in kwargs:
            kwargs["stderr"].close()


def _check_tool(name: str) -> None:
    if subprocess.run(["bash", "-lc", f"command -v {name} >/dev/null 2>&1"]).returncode != 0:
        raise RuntimeError(f"{name} not found in PATH")


def _normalize_model(model: str) -> str:
    key = model.strip().lower()
    if key not in MODEL_DIRS:
        raise ValueError(f"unsupported model: {model}")
    return key


def _inference_script(framework: str, model: str) -> Path:
    model_dir = MODEL_DIRS[_normalize_model(model)]
    prefix = "GNN_PyG_cuda" if framework == "pyg" else "GNN_DGL_cuda"
    script_path = REPO_ROOT / prefix / model_dir / "inference.py"
    if not script_path.exists():
        raise FileNotFoundError(f"inference script not found: {script_path}")
    return script_path


def _artifact_base(output_dir: Path, framework: str, model: str, dataset: str) -> Path:
    dataset_token = dataset.replace("/", "_")
    model_token = _normalize_model(model)
    return output_dir / f"{framework}_{model_token}_{dataset_token}_spmm_um"


def _build_profiled_cmd(args: argparse.Namespace) -> list[str]:
    env_name = args.conda_env or DEFAULT_FRAMEWORK_ENVS[args.framework]
    script_path = _inference_script(args.framework, args.model)
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        env_name,
        "python",
        str(script_path),
        "--dataset",
        args.dataset,
        "--data_root",
        args.data_root,
        "--dim",
        str(args.dim),
        "--num_layers",
        str(args.num_layers),
        "--adj_matrix",
        args.adj_matrix,
        "--ft_matrix",
        args.ft_matrix,
        "--weight",
        args.weight,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--device",
        args.device,
        "--nvtx",
    ]
    extra_args = list(args.extra_args or [])
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    cmd.extend(extra_args)
    return cmd


def _export_sqlite(rep_path: Path, sqlite_base: Path) -> Path:
    subprocess.run(
        [
            "nsys",
            "export",
            "--type",
            "sqlite",
            "--force-overwrite",
            "true",
            "--output",
            str(sqlite_base),
            str(rep_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    sqlite_path = sqlite_base.with_suffix(".sqlite")
    if not sqlite_path.exists():
        raise FileNotFoundError(f"expected sqlite export not found: {sqlite_path}")
    return sqlite_path


def _fetch_nvtx_ranges(sqlite_path: Path) -> tuple[list[TimeRange], list[TimeRange]]:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        table_names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "NVTX_EVENTS" not in table_names:
            raise RuntimeError("NVTX_EVENTS table not found in sqlite export")
        has_string_ids = "StringIds" in table_names
        if has_string_ids:
            query = """
            SELECT
                n.start,
                n.end,
                COALESCE(NULLIF(n.text, ''), s.value, '') AS name
            FROM NVTX_EVENTS AS n
            LEFT JOIN StringIds AS s ON n.textId = s.id
            WHERE n.end IS NOT NULL
            ORDER BY n.start
            """
        else:
            query = """
            SELECT start, end, COALESCE(text, '') AS name
            FROM NVTX_EVENTS
            WHERE end IS NOT NULL
            ORDER BY start
            """
        iteration_ranges: list[TimeRange] = []
        aggregation_ranges: list[TimeRange] = []
        for start_ns, end_ns, name in conn.execute(query):
            range_name = str(name or "")
            time_range = TimeRange(int(start_ns), int(end_ns))
            if range_name == "iteration":
                iteration_ranges.append(time_range)
            elif range_name.endswith("/aggregation"):
                aggregation_ranges.append(time_range)
        return iteration_ranges, aggregation_ranges
    finally:
        conn.close()


def _overlaps(left: TimeRange, right: TimeRange) -> bool:
    return left.start_ns < right.end_ns and left.end_ns > right.start_ns


def _csv_rows_from_stats(text: str, required_cols: tuple[str, ...]) -> list[dict[str, str]]:
    lines = text.splitlines()
    header_idx = None
    for idx, line in enumerate(lines):
        if all(col in line for col in required_cols):
            header_idx = idx
            break
    if header_idx is None:
        return []
    return list(csv.DictReader(lines[header_idx:]))


def _to_float(value: str) -> float:
    cleaned = str(value or "").strip().replace(",", "")
    return float(cleaned) if cleaned else 0.0


def _um_total_bytes(sqlite_path: Path, time_range: TimeRange) -> float:
    cmd = [
        "nsys",
        "stats",
        "--report",
        "um_total_sum",
        "--format",
        "csv",
        "--filter-time",
        f"{time_range.start_ns}/{time_range.end_ns}",
        str(sqlite_path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        return 0.0

    rows = _csv_rows_from_stats(
        result.stdout,
        ("Total HtoD Migration Size", "Total DtoH Migration Size"),
    )
    if not rows:
        return 0.0
    row = rows[0]
    return _to_float(row.get("Total HtoD Migration Size", "0")) + _to_float(row.get("Total DtoH Migration Size", "0"))


def _summarize(sqlite_path: Path) -> tuple[float, float]:
    iteration_ranges, aggregation_ranges = _fetch_nvtx_ranges(sqlite_path)
    if not aggregation_ranges:
        raise RuntimeError("no NVTX aggregation ranges found; make sure the target command ran with --nvtx")

    if not iteration_ranges:
        iteration_ranges = aggregation_ranges

    per_iter_spmm_ns: list[float] = []
    per_iter_migrated_bytes: list[float] = []
    for iteration in iteration_ranges:
        matching_aggs = [agg for agg in aggregation_ranges if _overlaps(agg, iteration)]
        if not matching_aggs:
            continue
        iter_spmm_ns = float(sum(agg.duration_ns for agg in matching_aggs))
        iter_bytes = float(sum(_um_total_bytes(sqlite_path, agg) for agg in matching_aggs))
        per_iter_spmm_ns.append(iter_spmm_ns)
        per_iter_migrated_bytes.append(iter_bytes)

    if not per_iter_spmm_ns:
        raise RuntimeError("no aggregation ranges overlapped measured iteration ranges")

    avg_spmm_ns = sum(per_iter_spmm_ns) / float(len(per_iter_spmm_ns))
    avg_migrated_bytes = sum(per_iter_migrated_bytes) / float(len(per_iter_migrated_bytes))
    return avg_spmm_ns, avg_migrated_bytes


def _write_summary(path: Path, spmm_ns: float, migrated_bytes: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "Summary Report:\n"
        f"spmm_ns, {spmm_ns:.3f}\n"
        f"migrated_bytes, {migrated_bytes:.3f}\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile GNN inference with nsys and print average SpMM time and migrated bytes during SpMM."
    )
    parser.add_argument("--framework", type=str, default="pyg", choices=("pyg", "dgl"), help="frontend/backend stack")
    parser.add_argument("--model", type=str, default="gcn", choices=("gcn", "gin", "sag", "graphsage"), help="model to profile")
    parser.add_argument("--dataset", type=str, required=True, help="dataset name")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT, help="directory containing <dataset>.npz")
    parser.add_argument("--dim", type=int, default=128, help="base feature / hidden / output dimension")
    parser.add_argument("--num_layers", type=int, default=1, help="number of layers")
    parser.add_argument("--adj_matrix", type=str, default="device", choices=("device", "uvm", "hmm"), help="adjacency memory mode")
    parser.add_argument("--ft_matrix", type=str, default="uvm", choices=("device", "uvm", "hmm"), help="feature memory mode")
    parser.add_argument("--weight", type=str, default="device", choices=("device", "uvm"), help="weight/output memory mode")
    parser.add_argument("--warmup", type=int, default=1, help="warmup iterations")
    parser.add_argument("--iters", type=int, default=5, help="measured iterations")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device string")
    parser.add_argument("--conda_env", type=str, default="", help="optional env override")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(REPO_ROOT / "report" / "nsys_spmm"),
        help="directory for nsys artifacts",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="extra inference args passed after `--`",
    )
    args = parser.parse_args()

    _check_tool("nsys")
    _check_tool("conda")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_base = _artifact_base(output_dir, args.framework, args.model, args.dataset)
    rep_path = artifact_base.with_suffix(".nsys-rep")
    sqlite_base = artifact_base.with_name(f"{artifact_base.name}_sqlite")
    sqlite_path = sqlite_base.with_suffix(".sqlite")
    target_stdout = artifact_base.with_name(f"{artifact_base.name}_target.stdout")
    target_stderr = artifact_base.with_name(f"{artifact_base.name}_target.stderr")
    summary_path = artifact_base.with_name(f"{artifact_base.name}_summary.txt")

    profiled_cmd = _build_profiled_cmd(args)
    profile_cmd = [
        "nsys",
        "profile",
        "--trace",
        "cuda,nvtx",
        "--sample",
        "none",
        "--cuda-um-cpu-page-faults=true",
        "--cuda-um-gpu-page-faults=true",
        "--force-overwrite",
        "true",
        "-o",
        str(artifact_base),
        *profiled_cmd,
    ]
    _run(profile_cmd, stdout_path=target_stdout, stderr_path=target_stderr)
    if not rep_path.exists():
        raise FileNotFoundError(f"expected nsys report not found: {rep_path}")

    sqlite_path = _export_sqlite(rep_path, sqlite_base)
    avg_spmm_ns, avg_migrated_bytes = _summarize(sqlite_path)
    _write_summary(summary_path, avg_spmm_ns, avg_migrated_bytes)

    print("Summary Report:")
    print(f"spmm_ns, {avg_spmm_ns:.3f}")
    print(f"migrated_bytes, {avg_migrated_bytes:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
