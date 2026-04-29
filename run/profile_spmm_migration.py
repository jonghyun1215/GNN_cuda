#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sqlite3
import subprocess
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


@dataclass(frozen=True)
class AddressRange:
    start: int
    end: int

    @property
    def is_valid(self) -> bool:
        return self.start >= 0 and self.end > self.start


@dataclass(frozen=True)
class UmTotals:
    htod_bytes: float = 0.0
    dtoh_bytes: float = 0.0
    gpu_faults: float = 0.0

    def __add__(self, other: "UmTotals") -> "UmTotals":
        return UmTotals(
            htod_bytes=self.htod_bytes + other.htod_bytes,
            dtoh_bytes=self.dtoh_bytes + other.dtoh_bytes,
            gpu_faults=self.gpu_faults + other.gpu_faults,
        )


@dataclass(frozen=True)
class NvtxRanges:
    iterations: list[TimeRange]
    aggregations: list[TimeRange]


@dataclass(frozen=True)
class Summary:
    spmm_ns: float
    um_totals: UmTotals


def _run(cmd: list[str], *, stdout_path: Path | None = None, stderr_path: Path | None = None) -> None:
    kwargs = {"check": True, "text": True}
    env = os.environ.copy()
    python_paths = [str(REPO_ROOT.parent), str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    kwargs["env"] = env
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


def _prefetch_to_location(prefetch: int) -> str:
    if int(prefetch) == 0:
        return "none"
    if int(prefetch) == 1:
        return "cuda"
    raise ValueError("--prefetch expects 0 or 1")


def _inference_script(framework: str, model: str) -> Path:
    model_dir = MODEL_DIRS[_normalize_model(model)]
    prefix = "GNN_PyG_cuda" if framework == "pyg" else "GNN_DGL_cuda"
    script_path = REPO_ROOT / prefix / model_dir / "inference.py"
    if not script_path.exists():
        raise FileNotFoundError(f"inference script not found: {script_path}")
    return script_path


def _artifact_base(
    output_dir: Path,
    framework: str,
    model: str,
    dataset: str,
    dim: int,
    ft_matrix: str,
    ft_host_alloc: float,
    prefetch_to: str,
) -> Path:
    dataset_token = dataset.replace("/", "_")
    model_token = _normalize_model(model)
    dim_token = f"dim{int(dim)}"
    host_token = f"host{ft_host_alloc:g}".replace(".", "p")
    prefetch_token = f"prefetch{prefetch_to}"
    return output_dir / (
        f"{framework}_{model_token}_{dataset_token}_{dim_token}_{ft_matrix}_"
        f"{host_token}_{prefetch_token}_spmm_um"
    )


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
        "--preferred_location",
        args.resolved_prefetch_location,
        "--prefetch_to",
        args.resolved_prefetch_location,
        "--ft_host_alloc",
        str(args.ft_host_alloc),
    ]
    extra_args = list(args.extra_args or [])
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    cmd.extend(extra_args)
    return cmd


def _export_sqlite(rep_path: Path, sqlite_base: Path) -> Path:
    result = subprocess.run(
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
    candidates = [sqlite_base]
    sqlite_with_suffix = sqlite_base.with_suffix(".sqlite")
    if sqlite_with_suffix != sqlite_base:
        candidates.append(sqlite_with_suffix)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    candidate_text = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "expected sqlite export not found; looked for "
        f"{candidate_text}\nnsys stdout:\n{result.stdout}\nnsys stderr:\n{result.stderr}"
    )


def _fetch_nvtx_ranges(sqlite_path: Path) -> NvtxRanges:
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
        return NvtxRanges(
            iterations=iteration_ranges,
            aggregations=aggregation_ranges,
        )
    finally:
        conn.close()


def _overlaps(left: TimeRange, right: TimeRange) -> bool:
    return left.start_ns < right.end_ns and left.end_ns > right.start_ns


def _parse_feature_address_range(stdout_path: Path) -> AddressRange | None:
    if not stdout_path.exists():
        return None
    start_addr = None
    end_addr = None
    pattern = re.compile(r"^Feature Address (Start|End):\s*(\d+)\s*$")
    for line in stdout_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line.strip())
        if match is None:
            continue
        if match.group(1) == "Start":
            start_addr = int(match.group(2))
        else:
            end_addr = int(match.group(2))
    if start_addr is None or end_addr is None:
        return None
    feature_range = AddressRange(start=start_addr, end=end_addr)
    return feature_range if feature_range.is_valid else None


def _um_total_stats(sqlite_path: Path, time_range: TimeRange, feature_range: AddressRange) -> UmTotals:
    direct_totals = _um_total_stats_from_sqlite(sqlite_path, time_range, feature_range)
    if direct_totals is not None:
        return direct_totals
    return UmTotals()


def _um_total_stats_all_time(sqlite_path: Path, feature_range: AddressRange) -> UmTotals:
    direct_totals = _um_total_stats_all_time_from_sqlite(sqlite_path, feature_range)
    if direct_totals is not None:
        return direct_totals
    return UmTotals()


def _um_total_stats_from_sqlite(
    sqlite_path: Path,
    time_range: TimeRange,
    feature_range: AddressRange,
) -> UmTotals | None:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        table_names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "CUPTI_ACTIVITY_KIND_MEMCPY" not in table_names:
            return None

        htod_bytes = 0.0
        dtoh_bytes = 0.0
        for copy_kind, total_bytes in conn.execute(
            """
            SELECT copyKind, COALESCE(SUM(bytes), 0)
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE migrationCause IS NOT NULL
              AND start < ?
              AND end > ?
              AND virtualAddress IS NOT NULL
              AND virtualAddress < ?
              AND (virtualAddress + bytes) > ?
              AND copyKind IN (11, 12)
            GROUP BY copyKind
            """,
            (
                int(time_range.end_ns),
                int(time_range.start_ns),
                int(feature_range.end),
                int(feature_range.start),
            ),
        ):
            if int(copy_kind) == 11:
                htod_bytes = float(total_bytes or 0)
            elif int(copy_kind) == 12:
                dtoh_bytes = float(total_bytes or 0)

        gpu_faults = 0.0
        if "CUDA_UM_GPU_PAGE_FAULT_EVENTS" in table_names:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(numberOfPageFaults), 0)
                FROM CUDA_UM_GPU_PAGE_FAULT_EVENTS
                WHERE start < ?
                  AND end > ?
                  AND address >= ?
                  AND address < ?
                """,
                (
                    int(time_range.end_ns),
                    int(time_range.start_ns),
                    int(feature_range.start),
                    int(feature_range.end),
                ),
            ).fetchone()
            gpu_faults = float(row[0] or 0)

        return UmTotals(htod_bytes=htod_bytes, dtoh_bytes=dtoh_bytes, gpu_faults=gpu_faults)
    finally:
        conn.close()


def _um_total_stats_all_time_from_sqlite(
    sqlite_path: Path,
    feature_range: AddressRange,
) -> UmTotals | None:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        table_names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "CUPTI_ACTIVITY_KIND_MEMCPY" not in table_names:
            return None

        htod_bytes = 0.0
        dtoh_bytes = 0.0
        for copy_kind, total_bytes in conn.execute(
            """
            SELECT copyKind, COALESCE(SUM(bytes), 0)
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE migrationCause IS NOT NULL
              AND virtualAddress IS NOT NULL
              AND virtualAddress < ?
              AND (virtualAddress + bytes) > ?
              AND copyKind IN (11, 12)
            GROUP BY copyKind
            """,
            (
                int(feature_range.end),
                int(feature_range.start),
            ),
        ):
            if int(copy_kind) == 11:
                htod_bytes = float(total_bytes or 0)
            elif int(copy_kind) == 12:
                dtoh_bytes = float(total_bytes or 0)

        gpu_faults = 0.0
        if "CUDA_UM_GPU_PAGE_FAULT_EVENTS" in table_names:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(numberOfPageFaults), 0)
                FROM CUDA_UM_GPU_PAGE_FAULT_EVENTS
                WHERE address >= ?
                  AND address < ?
                """,
                (
                    int(feature_range.start),
                    int(feature_range.end),
                ),
            ).fetchone()
            gpu_faults = float(row[0] or 0)

        return UmTotals(htod_bytes=htod_bytes, dtoh_bytes=dtoh_bytes, gpu_faults=gpu_faults)
    finally:
        conn.close()


def _summarize(sqlite_path: Path, feature_range: AddressRange, *, ft_matrix: str) -> Summary:
    nvtx_ranges = _fetch_nvtx_ranges(sqlite_path)
    aggregation_ranges = nvtx_ranges.aggregations
    iteration_ranges = nvtx_ranges.iterations
    if not aggregation_ranges:
        raise RuntimeError("no NVTX aggregation ranges found; make sure the target command ran with --nvtx")

    if not iteration_ranges:
        iteration_ranges = aggregation_ranges

    per_iter_spmm_ns: list[float] = []
    per_iter_um_totals: list[UmTotals] = []
    for iteration in iteration_ranges:
        matching_aggs = [agg for agg in aggregation_ranges if _overlaps(agg, iteration)]
        if not matching_aggs:
            continue
        iter_spmm_ns = float(sum(agg.duration_ns for agg in matching_aggs))
        if ft_matrix == "hmm":
            iter_um_totals = UmTotals()
            for agg in matching_aggs:
                iter_um_totals = iter_um_totals + _um_total_stats(sqlite_path, agg, feature_range)
        elif ft_matrix == "uvm":
            # UVM prefetch and first-touch migrations may occur before the
            # measured iteration NVTX range. Keep SpMM timing tied to
            # aggregation ranges, but count all feature-range UM events in the
            # profile so prefetch-driven migration is not dropped.
            iter_um_totals = _um_total_stats_all_time(sqlite_path, feature_range)
        else:
            iter_um_totals = _um_total_stats(sqlite_path, iteration, feature_range)
        per_iter_spmm_ns.append(iter_spmm_ns)
        per_iter_um_totals.append(iter_um_totals)

    if not per_iter_spmm_ns:
        raise RuntimeError("no aggregation ranges overlapped measured iteration ranges")

    avg_spmm_ns = sum(per_iter_spmm_ns) / float(len(per_iter_spmm_ns))
    avg_um_totals = UmTotals(
        htod_bytes=sum(item.htod_bytes for item in per_iter_um_totals) / float(len(per_iter_um_totals)),
        dtoh_bytes=sum(item.dtoh_bytes for item in per_iter_um_totals) / float(len(per_iter_um_totals)),
        gpu_faults=sum(item.gpu_faults for item in per_iter_um_totals) / float(len(per_iter_um_totals)),
    )
    return Summary(spmm_ns=avg_spmm_ns, um_totals=avg_um_totals)


def _write_summary(path: Path, summary: Summary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "Summary Report:\n"
        f"spmm_ns, {summary.spmm_ns:.3f}\n"
        f"HtoD_bytes, {summary.um_totals.htod_bytes:.3f}\n"
        f"DtoH_bytes, {summary.um_totals.dtoh_bytes:.3f}\n"
        f"GPU_faults, {summary.um_totals.gpu_faults:.3f}\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile GNN inference with nsys and print average SpMM time plus UM migration metrics during SpMM."
    )
    parser.add_argument("--framework", type=str, default="pyg", choices=("pyg", "dgl"), help="frontend/backend stack")
    parser.add_argument("--model", type=str, default="gcn", choices=("gcn", "gin", "sag", "graphsage"), help="model to profile")
    parser.add_argument("--dataset", type=str, required=True, help="dataset name")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT, help="directory containing <dataset>.npz")
    parser.add_argument("--dim", type=int, default=128, help="base feature / hidden / output dimension")
    parser.add_argument("--num_layers", type=int, default=1, help="number of layers")
    parser.add_argument("--adj_matrix", type=str, default="device", choices=("device", "uvm", "hmm"), help="adjacency memory mode")
    parser.add_argument("--ft_matrix", type=str, required=True, choices=("device", "uvm", "hmm"), help="feature memory mode")
    parser.add_argument(
        "--ft_host_alloc",
        type=float,
        default=0.0,
        help="target percent of the feature matrix that should not fit in remaining effective GPU memory",
    )
    parser.add_argument("--weight", type=str, default="device", choices=("device", "uvm"), help="weight/output memory mode")
    parser.add_argument("--prefetch", type=int, default=0, choices=(0, 1), help="0 disables prefetch hints; 1 uses cuda")
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
    if args.ft_host_alloc < 0.0 or args.ft_host_alloc >= 100.0:
        raise ValueError("--ft_host_alloc expects a value in [0, 100)")
    if args.ft_host_alloc > 0.0 and args.model != "gcn":
        raise ValueError("--ft_host_alloc is currently implemented for --model gcn")
    args.resolved_prefetch_location = _prefetch_to_location(args.prefetch)

    _check_tool("nsys")
    _check_tool("conda")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_base = _artifact_base(
        output_dir,
        args.framework,
        args.model,
        args.dataset,
        int(args.dim),
        args.ft_matrix,
        args.ft_host_alloc,
        args.resolved_prefetch_location,
    )
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
        "--cuda-memory-usage=true",
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
    feature_range = _parse_feature_address_range(target_stdout)
    if feature_range is None:
        raise RuntimeError(
            f"feature address range not found in target output: {target_stdout}; "
            "rebuild/run the target after the feature-address logging patch"
        )
    summary = _summarize(sqlite_path, feature_range, ft_matrix=args.ft_matrix)
    _write_summary(summary_path, summary)

    print("Summary Report:")
    print(f"spmm_ns, {summary.spmm_ns:.3f}")
    print(f"HtoD_bytes, {summary.um_totals.htod_bytes:.3f}")
    print(f"DtoH_bytes, {summary.um_totals.dtoh_bytes:.3f}")
    print(f"GPU_faults, {summary.um_totals.gpu_faults:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
