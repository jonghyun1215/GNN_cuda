#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from .agg_ops import linear_forward_, spmm_sum_forward_, tensor_add_inplace_
from .allocator import (
    COMPUTE_MEMORY_MODES,
    GRAPH_MEMORY_MODES,
    ManagedAllocationConfig,
    allocate_empty,
    allocate_like_mode,
    apply_managed_policy,
    is_uvm_mode,
    normalize_memory_mode,
    pointer_info,
    prefetch_managed,
)
from .graph_utils import build_plain_csr, load_src_dst_features, resolve_dataset_path


def _autocast_context(use_amp: bool, device: torch.device):
    if use_amp and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _cuda_profiler_start(device: torch.device):
    if device.type != "cuda":
        return
    try:
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStart()
    except Exception:
        pass


def _cuda_profiler_stop(device: torch.device):
    if device.type != "cuda":
        return
    try:
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStop()
    except Exception:
        pass


@contextlib.contextmanager
def _nvtx_range(enabled: bool, name: str):
    if enabled and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled and torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()


def _configure_tf32(enabled: bool, device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)
    torch.set_float32_matmul_precision("high" if enabled else "highest")


def _describe_memory(name: str, tensor: torch.Tensor) -> str:
    info = pointer_info(tensor)
    return (
        f"{name}:\tdevice={info.get('device_type')}:{info.get('device_index')} "
        f"type={info.get('pointer_type')} managed={info.get('is_managed')}"
    )


def _apply_policy_if_uvm(
    tensor: torch.Tensor,
    *,
    memory_mode: str,
    managed_cfg: ManagedAllocationConfig,
    device: torch.device,
    read_mostly: bool,
) -> None:
    if not is_uvm_mode(memory_mode):
        return
    apply_managed_policy(
        tensor,
        device=device,
        preferred_location=managed_cfg.preferred_location,
        accessed_by_cpu=managed_cfg.accessed_by_cpu,
        accessed_by_cuda=managed_cfg.accessed_by_cuda,
        read_mostly=read_mostly,
    )
    prefetch_managed(tensor, location=managed_cfg.prefetch_to, device=device)


@dataclass
class GINLayerState:
    weight1: torch.Tensor
    bias1: torch.Tensor
    weight2: torch.Tensor
    bias2: torch.Tensor
    eps: torch.Tensor
    agg_buffer: torch.Tensor
    hidden_buffer: torch.Tensor
    out_buffer: torch.Tensor


def _init_linear_params(in_dim: int, out_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    linear = nn.Linear(in_dim, out_dim, bias=True)
    return linear.weight.detach().t().contiguous().cpu(), linear.bias.detach().contiguous().cpu()


def _build_layer_dims(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> list[tuple[int, int]]:
    n_layers = max(1, int(num_layers))
    if n_layers == 1:
        return [(in_dim, out_dim)]
    dims = [(in_dim, hidden_dim)]
    for _ in range(n_layers - 2):
        dims.append((hidden_dim, hidden_dim))
    dims.append((hidden_dim, out_dim))
    return dims


def _make_layer_state(
    in_dim: int,
    out_dim: int,
    *,
    num_nodes: int,
    device: torch.device,
    memory_mode: str,
    managed_cfg: ManagedAllocationConfig,
    learn_eps: bool,
) -> GINLayerState:
    weight1_cpu, bias1_cpu = _init_linear_params(in_dim, out_dim)
    weight2_cpu, bias2_cpu = _init_linear_params(out_dim, out_dim)
    eps_cpu = torch.zeros((1,), dtype=torch.float32)
    weight1 = allocate_like_mode(weight1_cpu, memory_mode=memory_mode, device=device)
    bias1 = allocate_like_mode(bias1_cpu, memory_mode=memory_mode, device=device)
    weight2 = allocate_like_mode(weight2_cpu, memory_mode=memory_mode, device=device)
    bias2 = allocate_like_mode(bias2_cpu, memory_mode=memory_mode, device=device)
    eps = allocate_like_mode(eps_cpu, memory_mode=memory_mode, device=device)
    if not learn_eps:
        eps.requires_grad_(False)
    agg_buffer = allocate_empty((num_nodes, in_dim), dtype=torch.float32, device=device, memory_mode=memory_mode)
    hidden_buffer = allocate_empty((num_nodes, out_dim), dtype=torch.float32, device=device, memory_mode=memory_mode)
    out_buffer = allocate_empty((num_nodes, out_dim), dtype=torch.float32, device=device, memory_mode=memory_mode)
    for tensor in (weight1, bias1, weight2, bias2, eps, agg_buffer, hidden_buffer, out_buffer):
        _apply_policy_if_uvm(
            tensor,
            memory_mode=memory_mode,
            managed_cfg=managed_cfg,
            device=device,
            read_mostly=False,
        )
    return GINLayerState(
        weight1=weight1,
        bias1=bias1,
        weight2=weight2,
        bias2=bias2,
        eps=eps,
        agg_buffer=agg_buffer,
        hidden_buffer=hidden_buffer,
        out_buffer=out_buffer,
    )


def run_gin_inference(
    *,
    framework_label: str,
    select_device: Callable[[str], torch.device],
    load_graph_features,
    load_kind: str,
    post_layer_relu: bool,
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora", help="dataset name")
    parser.add_argument("--data_root", type=str, default=os.environ.get("GNN_DATASET_ROOT", "/root/workspace/mnt/dataset_npz"), help="directory containing <dataset>.npz")
    parser.add_argument("--dim", type=int, default=128, help="base dim for input/hidden/output")
    parser.add_argument("--feat_dim", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--hidden_dim", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out_dim", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--num_layers", type=int, default=2, help="number of GIN layers")
    parser.add_argument("--learn_eps", action="store_true", help="allocate epsilon as a learnable-style state tensor")
    parser.add_argument("--use_npz_features", action="store_true", help="use sparse features saved in npz if present")
    parser.add_argument("--memory_mode", type=str, default="uvm", help="shared fallback backend for graph/compute allocation: device or uvm; legacy aliases managed/torch_cuda are also accepted")
    parser.add_argument("--graph_memory_mode", type=str, default=None, help="graph backend for CSR/features inputs: device, uvm, hmm; legacy aliases managed/torch_cuda/host_mapped are also accepted")
    parser.add_argument("--compute_memory_mode", type=str, default=None, help="compute backend for weights/outputs/scratch: device or uvm; legacy aliases managed/torch_cuda are also accepted")
    parser.add_argument("--preferred_location", type=str, default="cuda", choices=["none", "cpu", "cuda"], help="UVM preferred location hint")
    parser.add_argument("--accessed_by_cpu", action="store_true", help="mark UVM tensors as CPU-accessible")
    parser.add_argument("--accessed_by_cuda", action="store_true", help="mark UVM tensors as GPU-accessible")
    parser.add_argument("--read_mostly_graph", action="store_true", help="mark graph structures as read-mostly")
    parser.add_argument("--prefetch_to", type=str, default="cuda", choices=["none", "cpu", "cuda"], help="prefetch UVM tensors before compute")
    parser.add_argument("--amp", action="store_true", help="reserved; CUDA prototype currently runs fp32")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True, help="allow TF32 Tensor Core path for float32 GEMM on CUDA")
    parser.add_argument("--nvtx", action="store_true", help="enable NVTX ranges for profiling")
    parser.add_argument("--warmup", type=int, default=1, help="warmup iterations")
    parser.add_argument("--iters", type=int, default=5, help="inference iterations")
    parser.add_argument("--device", type=str, default="cuda:0", help="device string")
    args = parser.parse_args()
    base_dim = int(args.dim)
    if args.feat_dim is None:
        args.feat_dim = base_dim
    if args.hidden_dim is None:
        args.hidden_dim = base_dim
    if args.out_dim is None:
        args.out_dim = base_dim
    device = select_device(args.device)
    _configure_tf32(bool(args.tf32), device)
    if args.amp:
        raise ValueError("The CUDA GIN prototype only supports float32 today.")
    nvtx_enabled = bool(args.nvtx and device.type == "cuda")
    shared_memory_mode = normalize_memory_mode(args.memory_mode, allow_hmm=True)
    graph_memory_mode = normalize_memory_mode(args.graph_memory_mode or shared_memory_mode, allow_hmm=True)
    try:
        compute_memory_mode = normalize_memory_mode(args.compute_memory_mode or shared_memory_mode, allow_hmm=False)
    except ValueError as exc:
        raise ValueError(
            f"compute_memory_mode expects one of {COMPUTE_MEMORY_MODES}; use --graph_memory_mode hmm with --compute_memory_mode uvm/device"
        ) from exc
    args.memory_mode = shared_memory_mode
    args.graph_memory_mode = graph_memory_mode
    args.compute_memory_mode = compute_memory_mode
    print(args)
    if graph_memory_mode == "hmm" and device.type != "cuda":
        raise ValueError(f"graph_memory_mode expects one of {GRAPH_MEMORY_MODES} on CUDA; 'hmm' requires a CUDA device")
    managed_cfg = ManagedAllocationConfig(
        preferred_location=str(args.preferred_location),
        accessed_by_cpu=bool(args.accessed_by_cpu),
        accessed_by_cuda=bool(args.accessed_by_cuda or device.type == "cuda"),
        read_mostly_graph=bool(args.read_mostly_graph),
        prefetch_to=str(args.prefetch_to),
    )

    graph_path = resolve_dataset_path(str(args.dataset), data_root=str(args.data_root))
    loaded = load_graph_features(
        graph_path,
        feat_dim=args.feat_dim,
        use_npz_features=args.use_npz_features,
    )
    src, dst, features_cpu, num_nodes = load_src_dst_features(loaded, load_kind=load_kind)
    num_edges = int(src.numel())

    csr = build_plain_csr(src, dst, num_nodes=num_nodes, add_self_loops=False, transpose_for_incoming=True)
    features_cpu = features_cpu.contiguous().to(dtype=torch.float32)

    row_ptr = allocate_like_mode(csr.row_ptr, memory_mode=graph_memory_mode, device=device)
    col_ind = allocate_like_mode(csr.col_ind, memory_mode=graph_memory_mode, device=device)
    features = allocate_like_mode(features_cpu, memory_mode=graph_memory_mode, device=device)
    for tensor in (row_ptr, col_ind):
        _apply_policy_if_uvm(
            tensor,
            memory_mode=graph_memory_mode,
            managed_cfg=managed_cfg,
            device=device,
            read_mostly=managed_cfg.read_mostly_graph,
        )
    _apply_policy_if_uvm(
        features,
        memory_mode=graph_memory_mode,
        managed_cfg=managed_cfg,
        device=device,
        read_mostly=False,
    )

    layer_states = [
        _make_layer_state(
            in_dim=in_dim,
            out_dim=out_dim,
            num_nodes=int(num_nodes),
            device=device,
            memory_mode=compute_memory_mode,
            managed_cfg=managed_cfg,
            learn_eps=bool(args.learn_eps),
        )
        for in_dim, out_dim in _build_layer_dims(int(features.size(1)), int(args.hidden_dim), int(args.out_dim), int(args.num_layers))
    ]

    def forward_once() -> torch.Tensor:
        x = features
        for idx, layer in enumerate(layer_states):
            tag = f"layer{idx + 1}"
            with _nvtx_range(nvtx_enabled, f"{tag}/aggregation"):
                spmm_sum_forward_(row_ptr, col_ind, x, layer.agg_buffer)
            with _nvtx_range(nvtx_enabled, f"{tag}/epilogue"):
                tensor_add_inplace_(layer.agg_buffer, x, alpha=1.0 + float(layer.eps.item()))
            with _nvtx_range(nvtx_enabled, f"{tag}/dense_update"):
                linear_forward_(
                    layer.agg_buffer,
                    layer.weight1,
                    layer.bias1,
                    layer.hidden_buffer,
                    relu=True,
                )
                linear_forward_(
                    layer.hidden_buffer,
                    layer.weight2,
                    layer.bias2,
                    layer.out_buffer,
                    relu=post_layer_relu and idx != len(layer_states) - 1,
                )
            x = layer.out_buffer
        return x

    with torch.no_grad():
        for _ in range(args.warmup):
            with _autocast_context(False, device):
                forward_once()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        _cuda_profiler_start(device)
        start = time.perf_counter()
        for _ in range(args.iters):
            with _nvtx_range(nvtx_enabled, "iteration"):
                with _autocast_context(False, device):
                    output = forward_once()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        _cuda_profiler_stop(device)

    print(f"Graph:\t{graph_path}")
    print(f"Impl:\t{framework_label}_gin_cuda")
    print(f"Graph Memory Mode:\t{graph_memory_mode}")
    print(f"Compute Memory Mode:\t{compute_memory_mode}")
    print(f"Nodes:\t{num_nodes}")
    print(f"Edges:\t{num_edges}")
    print(_describe_memory("row_ptr", row_ptr))
    print(_describe_memory("col_ind", col_ind))
    print(_describe_memory("features", features))
    print(_describe_memory("weights1_l0", layer_states[0].weight1))
    print(_describe_memory("output", output))
    print(f"Output Sum:\t{float(output.sum().item()):.6f}")
    print(f"Infer (ns):\t{elapsed * 1e9 / args.iters:.3f}")
    return 0
