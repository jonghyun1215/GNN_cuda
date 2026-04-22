#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import time
from collections import defaultdict

import torch


class PhaseSummary:
    def __init__(self, device: torch.device):
        self.device = device
        self._totals_ns: dict[str, float] = defaultdict(float)
        self._iteration_totals_ns: list[dict[str, float]] = []
        self._last_iteration_snapshot_ns: dict[str, float] = {}

    @contextlib.contextmanager
    def measure(self, key: str, *, use_cuda_events: bool = True):
        if use_cuda_events and self.device.type == "cuda":
            stream = torch.cuda.current_stream(self.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            try:
                yield
            finally:
                end.record(stream)
                end.synchronize()
                self._totals_ns[key] += float(start.elapsed_time(end)) * 1e6
            return

        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._totals_ns[key] += (time.perf_counter() - t0) * 1e9

    def total_ns(self, key: str) -> float:
        return float(self._totals_ns.get(key, 0.0))

    def avg_ns(self, key: str, *, iters: int) -> float:
        if iters <= 0:
            return 0.0
        return self.total_ns(key) / float(iters)

    def record_iteration(self, keys: tuple[str, ...]) -> None:
        snapshot = {}
        for key in keys:
            total = float(self._totals_ns.get(key, 0.0))
            prev = float(self._last_iteration_snapshot_ns.get(key, 0.0))
            snapshot[key] = total - prev
            self._last_iteration_snapshot_ns[key] = total
        self._iteration_totals_ns.append(snapshot)

    def iteration_values(self, key: str) -> list[float]:
        return [float(entry.get(key, 0.0)) for entry in self._iteration_totals_ns]

    def reset(self, *, preserve: tuple[str, ...] = ("graph_prep",)) -> None:
        preserved = {key: self._totals_ns[key] for key in preserve if key in self._totals_ns}
        self._totals_ns.clear()
        self._totals_ns.update(preserved)
        self._iteration_totals_ns.clear()
        self._last_iteration_snapshot_ns.clear()


def print_summary_report(
    summary: PhaseSummary,
    *,
    iters: int,
    infer_avg_ns: float,
) -> None:
    spmm_avg = summary.avg_ns("spmm", iters=iters)

    print("Summary Report:")
    print(f"spmm, {spmm_avg:.3f}")
    spmm_iters = summary.iteration_values("spmm")
    if spmm_iters:
        print("SpMM Iteration Report:")
        for idx, value in enumerate(spmm_iters, start=1):
            print(f"iter_{idx}, {value:.3f}")
