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
    uses_cuda_memory_hints,
)
from .agg_ops import bias_relu_forward_, gemm_forward_
from .gcn_ops import spmm_gcn_forward_, stage_feature_rows_
from .graph_utils import (
    apply_node_permutation,
    build_gcn_normalized_csr,
    build_pyg_gcn_weighted_csr,
    default_gcn_preprocess_meta_path,
    resolve_dataset_path,
)
from .pyg_gcn_ops import spmm_pyg_gcn_forward_
from .phase_summary import PhaseSummary, print_summary_report


PARTIAL_PREFETCH_FRACTION = 0.80
MIN_PARTIAL_PREFETCH_RUNTIME_SLACK_BYTES = 128 * 1024 * 1024


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


@dataclass
class LayerState:
    weight: torch.Tensor
    bias: torch.Tensor
    agg_buffer: torch.Tensor
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
    weight_memory_mode: str,
    activation_memory_mode: str,
    managed_cfg: ManagedAllocationConfig,
) -> LayerState:
    weight_cpu, bias_cpu = _init_linear_params(in_dim, out_dim)
    weight = allocate_like_mode(weight_cpu, memory_mode=weight_memory_mode, device=device)
    bias = allocate_like_mode(bias_cpu, memory_mode=weight_memory_mode, device=device)
    agg_buffer = allocate_empty((num_nodes, in_dim), dtype=torch.float32, device=device, memory_mode=activation_memory_mode)
    out_buffer = allocate_empty((num_nodes, out_dim), dtype=torch.float32, device=device, memory_mode=activation_memory_mode)
    if is_uvm_mode(weight_memory_mode):
        apply_managed_policy(
            weight,
            device=device,
            preferred_location=managed_cfg.preferred_location,
            accessed_by_cpu=managed_cfg.accessed_by_cpu,
            accessed_by_cuda=managed_cfg.accessed_by_cuda,
            read_mostly=False,
        )
        apply_managed_policy(
            bias,
            device=device,
            preferred_location=managed_cfg.preferred_location,
            accessed_by_cpu=managed_cfg.accessed_by_cpu,
            accessed_by_cuda=managed_cfg.accessed_by_cuda,
            read_mostly=False,
        )
        for tensor in (weight, bias):
            prefetch_managed(tensor, location=managed_cfg.prefetch_to, device=device)
    if is_uvm_mode(activation_memory_mode):
        apply_managed_policy(
            agg_buffer,
            device=device,
            preferred_location=managed_cfg.preferred_location,
            accessed_by_cpu=managed_cfg.accessed_by_cpu,
            accessed_by_cuda=managed_cfg.accessed_by_cuda,
            read_mostly=False,
        )
        apply_managed_policy(
            out_buffer,
            device=device,
            preferred_location=managed_cfg.preferred_location,
            accessed_by_cpu=managed_cfg.accessed_by_cpu,
            accessed_by_cuda=managed_cfg.accessed_by_cuda,
            read_mostly=False,
        )
        for tensor in (agg_buffer, out_buffer):
            prefetch_managed(tensor, location=managed_cfg.prefetch_to, device=device)
    return LayerState(weight=weight, bias=bias, agg_buffer=agg_buffer, out_buffer=out_buffer)


def _describe_memory(name: str, tensor: torch.Tensor) -> str:
    info = pointer_info(tensor)
    return (
        f"{name}:\tdevice={info.get('device_type')}:{info.get('device_index')} "
        f"type={info.get('pointer_type')} managed={info.get('is_managed')}"
    )


def _resolve_preprocess_meta_path(preprocess_meta: str, *, dataset: str, data_root: str) -> str | None:
    mode = str(preprocess_meta)
    if mode == "none":
        return None
    if mode == "auto":
        candidate = default_gcn_preprocess_meta_path(dataset, data_root=data_root)
        return candidate if Path(candidate).exists() else None
    return mode


def run_gcn_inference(
    *,
    framework_label: str,
    select_device: Callable[[str], torch.device],
    load_graph_features,
    load_kind: str,
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora", help="dataset name")
    parser.add_argument("--data_root", type=str, default=os.environ.get("GNN_DATASET_ROOT", "/root/workspace/mnt/dataset_npz"), help="directory containing <dataset>.npz")
    parser.add_argument("--dim", type=int, default=128, help="base dim for input/hidden/output")
    parser.add_argument("--feat_dim", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--hidden_dim", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out_dim", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--num_layers", type=int, default=2, help="number of GCN layers")
    parser.add_argument("--use_npz_features", action="store_true", help="use sparse features saved in npz if present")
    parser.add_argument("--memory_mode", type=str, default="uvm", help="shared fallback backend for adjacency/features/weight allocation: device, uvm, or hmm; legacy aliases managed/torch_cuda are also accepted")
    parser.add_argument("--graph_memory_mode", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--adj_matrix", type=str, default=None, help="adjacency backend for CSR inputs and graph-normalization vectors: device, uvm, hmm")
    parser.add_argument("--ft_matrix", type=str, default=None, help="feature backend for node feature tensors: device, uvm, hmm")
    parser.add_argument("--weight", type=str, default=None, help="compute backend for weights/outputs/scratch: device or uvm")
    parser.add_argument("--activation", type=str, default=None, help="activation backend for aggregation/output buffers: device or uvm")
    parser.add_argument("--compute_memory_mode", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--spmm_mode", type=str, default="plain", choices=["naive", "plain", "optimized"], help="SpMM kernel variant: naive, plain, or optimized")
    parser.add_argument("--gcn_kernel_impl", type=str, default="legacy_fused", choices=["legacy_fused", "pyg_baseline"], help="GCN aggregation implementation: legacy fused normalization kernel or PyG-baseline-style weighted CSR kernel")
    parser.add_argument("--hmm_mode", type=str, default="plain", choices=["plain", "optimized"], help=argparse.SUPPRESS)
    parser.add_argument("--preprocess_meta", type=str, default="none", help=argparse.SUPPRESS)
    parser.add_argument("--pretouch_passes", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--preferred_location", type=str, default="none", choices=["none", "cpu", "cuda"], help="UVM preferred location hint")
    parser.add_argument("--accessed_by_cpu", action="store_true", help="mark UVM tensors as CPU-accessible")
    parser.add_argument("--accessed_by_cuda", action="store_true", help="mark UVM tensors as GPU-accessible")
    parser.add_argument("--read_mostly_graph", action="store_true", help="mark graph structures as read-mostly")
    parser.add_argument("--prefetch_to", type=str, default="none", choices=["none", "cpu", "cuda"], help="prefetch UVM tensors before compute")
    parser.add_argument("--amp", action="store_true", help="reserved; CUDA prototype currently runs fp32")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True, help="allow TF32 Tensor Core path for float32 GEMM on CUDA")
    parser.add_argument("--nvtx", action="store_true", help="enable NVTX ranges for profiling")
    parser.add_argument("--warmup", type=int, default=1, help="warmup iterations")
    parser.add_argument("--iters", type=int, default=5, help="inference iterations")
    parser.add_argument("--device", type=str, default="cuda:0", help="device string")
    parser.add_argument(
        "--ft_host_alloc",
        type=float,
        default=0.0,
        help="target percent of the feature matrix that should not fit in the remaining effective GPU memory",
    )
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
        raise ValueError("The CUDA GCN prototype only supports float32 today.")
    nvtx_enabled = bool(args.nvtx and device.type == "cuda")
    phase_summary = PhaseSummary(device)
    shared_memory_mode = normalize_memory_mode(args.memory_mode, allow_hmm=True)
    graph_fallback_mode = args.graph_memory_mode or shared_memory_mode
    adj_memory_mode = normalize_memory_mode(args.adj_matrix or graph_fallback_mode, allow_hmm=True)
    ft_memory_mode = normalize_memory_mode(args.ft_matrix or graph_fallback_mode, allow_hmm=True)
    try:
        weight_memory_mode = normalize_memory_mode(args.weight or args.compute_memory_mode or shared_memory_mode, allow_hmm=False)
    except ValueError as exc:
        raise ValueError(
            f"weight expects one of {COMPUTE_MEMORY_MODES}; use --adj_matrix/--ft_matrix hmm with --weight uvm/device"
        ) from exc
    try:
        activation_memory_mode = normalize_memory_mode(args.activation or weight_memory_mode, allow_hmm=False)
    except ValueError as exc:
        raise ValueError(
            f"activation expects one of {COMPUTE_MEMORY_MODES}; HMM outputs are not supported by the current dense path"
        ) from exc
    args.memory_mode = shared_memory_mode
    args.adj_matrix = adj_memory_mode
    args.ft_matrix = ft_memory_mode
    args.weight = weight_memory_mode
    args.activation = activation_memory_mode
    print(args)
    if "hmm" in (adj_memory_mode, ft_memory_mode) and device.type != "cuda":
        raise ValueError(
            f"adj_matrix/ft_matrix expect one of {GRAPH_MEMORY_MODES} on CUDA; 'hmm' requires a CUDA device"
        )
    spmm_mode = str(args.spmm_mode)
    gcn_kernel_impl = str(args.gcn_kernel_impl)
    if spmm_mode == "plain" and str(args.hmm_mode) == "optimized":
        spmm_mode = "optimized"
    spmm_kernel = "naive" if spmm_mode == "naive" else "plain_shared"
    managed_cfg = ManagedAllocationConfig(
        preferred_location=str(args.preferred_location),
        accessed_by_cpu=bool(args.accessed_by_cpu),
        accessed_by_cuda=bool(args.accessed_by_cuda or device.type == "cuda"),
        read_mostly_graph=bool(args.read_mostly_graph),
        prefetch_to=str(args.prefetch_to),
    )
    preprocess_meta_path = _resolve_preprocess_meta_path(
        str(args.preprocess_meta),
        dataset=str(args.dataset),
        data_root=str(args.data_root),
    )
    preprocess_meta = None
    row_schedule_cpu = None
    row_schedule_device = None
    edge_weight = None
    hot_pages_cpu = torch.empty((0,), dtype=torch.long)
    hot_cache_coverage = 0.0
    page_reuse_histogram = torch.empty((0,), dtype=torch.long)
    hot_feature_cache = None
    hot_node_cutoff = 0
    hot_node_fraction = 0.0
    hot_node_access_coverage = 0.0
    rows_per_page = 1
    row_block_size = 1
    window_num_blocks = 1
    optimized_hmm_active = False
    reorder_enabled = False
    ft_host_alloc = float(args.ft_host_alloc)
    if ft_host_alloc < 0.0 or ft_host_alloc >= 100.0:
        raise ValueError("--ft_host_alloc expects a value in [0, 100)")
    ft_reserve_enabled = ft_host_alloc > 0.0
    feature_initial_location = "cpu" if is_uvm_mode(ft_memory_mode) else "none"
    if ft_reserve_enabled:
        if device.type != "cuda":
            raise ValueError("--ft_host_alloc requires a CUDA device")
        if ft_memory_mode == "device":
            raise ValueError("--ft_host_alloc is only meaningful with --ft_matrix uvm|hmm")
    reserve_device_bytes = 0
    target_gpu_feature_bytes = 0
    feature_bytes = 0
    reserve_free_bytes_before = 0
    reserve_free_bytes_after = 0
    reserve_tensor = None
    reserve_activated_bytes = 0
    reserve_activations = 0
    feature_cuda_prefetch_bytes = 0

    def activate_feature_reserve() -> None:
        nonlocal reserve_tensor, reserve_activated_bytes, reserve_activations, reserve_free_bytes_after
        if reserve_device_bytes <= 0 or reserve_tensor is not None:
            return
        try:
            reserve_tensor = torch.empty((reserve_device_bytes,), dtype=torch.uint8, device=device)
        except RuntimeError as exc:
            gib = reserve_device_bytes / float(1024 ** 3)
            raise RuntimeError(
                f"Failed to reserve {reserve_device_bytes} bytes ({gib:.3f} GiB) of extra device memory"
            ) from exc
        reserve_activated_bytes = int(reserve_tensor.numel())
        reserve_activations += 1
        torch.cuda.synchronize(device)
        free_after, _ = torch.cuda.mem_get_info(device)
        reserve_free_bytes_after = int(free_after)

    def release_feature_reserve() -> None:
        nonlocal reserve_tensor
        if reserve_tensor is None:
            return
        reserve_tensor = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)

    with phase_summary.measure("graph_prep", use_cuda_events=False):
        graph_path = resolve_dataset_path(str(args.dataset), data_root=str(args.data_root))
        loaded = load_graph_features(
            graph_path,
            feat_dim=args.feat_dim,
            use_npz_features=args.use_npz_features,
        )
        if load_kind == "pyg":
            edge_index, features_cpu, num_nodes = loaded
            src = edge_index[0].contiguous()
            dst = edge_index[1].contiguous()
        elif load_kind == "dgl":
            src, dst, features_cpu, num_nodes = loaded
        else:
            raise ValueError(f"Unsupported load_kind: {load_kind}")
        num_edges = int(src.numel())

        if preprocess_meta_path is not None:
            preprocess_meta = torch.load(preprocess_meta_path, map_location="cpu")
            if "perm" in preprocess_meta and "inv_perm" in preprocess_meta:
                perm = torch.as_tensor(preprocess_meta["perm"], dtype=torch.long).contiguous()
                inv_perm = torch.as_tensor(preprocess_meta["inv_perm"], dtype=torch.long).contiguous()
                if int(perm.numel()) != int(num_nodes) or int(inv_perm.numel()) != int(num_nodes):
                    raise ValueError(
                        f"preprocess metadata {preprocess_meta_path} is incompatible with dataset {args.dataset}"
                    )
                src, dst, features_cpu = apply_node_permutation(
                    src,
                    dst,
                    features_cpu,
                    perm=perm,
                    inv_perm=inv_perm,
                )
                reorder_enabled = True

        optimized_managed_requested = spmm_mode == "optimized"
        if optimized_managed_requested and preprocess_meta_path is not None:
            row_schedule_cpu = torch.as_tensor(preprocess_meta["row_schedule"], dtype=torch.long).contiguous()
            if int(row_schedule_cpu.numel()) != int(num_nodes):
                raise ValueError(
                    f"preprocess metadata {preprocess_meta_path} is incompatible with dataset {args.dataset}"
                )
            hot_pages_cpu = torch.as_tensor(preprocess_meta.get("hot_pages", []), dtype=torch.long).contiguous()
            hot_cache_coverage = float(preprocess_meta.get("hot_coverage", 0.0))
            hot_node_cutoff = max(0, int(preprocess_meta.get("hot_node_cutoff", 0)))
            hot_node_fraction = float(preprocess_meta.get("hot_node_fraction", 0.0))
            hot_node_access_coverage = float(preprocess_meta.get("hot_node_access_coverage", 0.0))
            page_reuse_histogram = torch.as_tensor(
                preprocess_meta.get("page_reuse_histogram", []),
                dtype=torch.long,
            ).contiguous()
            rows_per_page = max(1, int(preprocess_meta.get("rows_per_page", 1)))
            row_block_size = max(1, int(preprocess_meta.get("row_block_size", 1)))
            window_num_blocks = max(1, int(preprocess_meta.get("window_num_blocks", 1)))
        elif optimized_managed_requested:
            raise ValueError("--spmm_mode optimized requires --preprocess_meta auto|<path>")

        if spmm_mode == "optimized" and ft_memory_mode not in {"hmm", "uvm"}:
            raise ValueError("--spmm_mode optimized currently requires --ft_matrix uvm|hmm")
        if spmm_mode == "optimized" and gcn_kernel_impl not in {"legacy_fused", "pyg_baseline"}:
            raise ValueError("--spmm_mode optimized expects legacy_fused or pyg_baseline")
        features_cpu = features_cpu.contiguous().to(dtype=torch.float32)
        feature_bytes = int(features_cpu.numel() * features_cpu.element_size())
        if gcn_kernel_impl == "pyg_baseline":
            weighted_csr = build_pyg_gcn_weighted_csr(src, dst, num_nodes=num_nodes, add_self_loops=True)
            row_ptr = allocate_like_mode(weighted_csr.row_ptr, memory_mode=adj_memory_mode, device=device)
            col_ind = allocate_like_mode(weighted_csr.col_ind, memory_mode=adj_memory_mode, device=device)
            edge_weight = allocate_like_mode(weighted_csr.edge_weight, memory_mode=adj_memory_mode, device=device)
            deg_inv_sqrt = None
            adj_tensors = (row_ptr, col_ind, edge_weight)
            spmm_kernel = f"{spmm_kernel}_pyg"
        else:
            csr = build_gcn_normalized_csr(src, dst, num_nodes=num_nodes, add_self_loops=True)
            row_ptr = allocate_like_mode(csr.row_ptr, memory_mode=adj_memory_mode, device=device)
            col_ind = allocate_like_mode(csr.col_ind, memory_mode=adj_memory_mode, device=device)
            deg_inv_sqrt = allocate_like_mode(csr.deg_inv_sqrt, memory_mode=adj_memory_mode, device=device)
            adj_tensors = (row_ptr, col_ind, deg_inv_sqrt)
        if uses_cuda_memory_hints(adj_memory_mode):
            for tensor in adj_tensors:
                apply_managed_policy(
                    tensor,
                    device=device,
                    preferred_location=managed_cfg.preferred_location,
                    accessed_by_cpu=managed_cfg.accessed_by_cpu,
                    accessed_by_cuda=managed_cfg.accessed_by_cuda,
                    read_mostly=managed_cfg.read_mostly_graph,
                )
                prefetch_managed(tensor, location=managed_cfg.prefetch_to, device=device)

        if row_schedule_cpu is not None:
            row_schedule_device = row_schedule_cpu.to(device=device, dtype=torch.long, non_blocking=False)

        if row_schedule_device is not None and hot_node_cutoff > 0:
            hot_feature_cache = torch.empty(
                (int(hot_node_cutoff), int(features_cpu.size(1))),
                dtype=torch.float32,
                device=device,
            )
            optimized_hmm_active = True

        layer_states = [
            _make_layer_state(
                in_dim=in_dim,
                out_dim=out_dim,
                num_nodes=int(num_nodes),
                device=device,
                weight_memory_mode=weight_memory_mode,
                activation_memory_mode=activation_memory_mode,
                managed_cfg=managed_cfg,
            )
            for in_dim, out_dim in _build_layer_dims(int(features_cpu.size(1)), int(args.hidden_dim), int(args.out_dim), int(args.num_layers))
        ]
        if ft_reserve_enabled:
            torch.cuda.synchronize(device)
            gpu_fit_fraction = max(0.0, (100.0 - ft_host_alloc) / 100.0)
            target_gpu_feature_bytes = int(feature_bytes * gpu_fit_fraction)
            if managed_cfg.prefetch_to == "cuda":
                if target_gpu_feature_bytes >= MIN_PARTIAL_PREFETCH_RUNTIME_SLACK_BYTES:
                    feature_cuda_prefetch_bytes = int(target_gpu_feature_bytes * PARTIAL_PREFETCH_FRACTION)
                    feature_cuda_prefetch_bytes = max(0, min(feature_cuda_prefetch_bytes, feature_bytes))
                else:
                    feature_cuda_prefetch_bytes = 0
            free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
            reserve_free_bytes_before = int(free_bytes)
            reserve_device_bytes = max(
                0,
                int(free_bytes) - target_gpu_feature_bytes - feature_cuda_prefetch_bytes,
            )
            activate_feature_reserve()

        features = allocate_like_mode(
            features_cpu,
            memory_mode=ft_memory_mode,
            device=device,
            uvm_initial_location=feature_initial_location,
        )
        if uses_cuda_memory_hints(ft_memory_mode):
            feature_preferred_location = (
                "cpu"
                if ft_reserve_enabled and managed_cfg.prefetch_to == "cuda"
                else managed_cfg.preferred_location
            )
            apply_managed_policy(
                features,
                device=device,
                preferred_location=feature_preferred_location,
                accessed_by_cpu=managed_cfg.accessed_by_cpu,
                accessed_by_cuda=managed_cfg.accessed_by_cuda,
                read_mostly=False,
            )
            if feature_initial_location != "none":
                prefetch_managed(features, location=feature_initial_location, device=device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            if managed_cfg.prefetch_to == "cuda":
                if ft_reserve_enabled:
                    prefetch_managed(
                        features,
                        location=managed_cfg.prefetch_to,
                        device=device,
                        nbytes=feature_cuda_prefetch_bytes,
                    )
                else:
                    feature_cuda_prefetch_bytes = feature_bytes
                    prefetch_managed(features, location=managed_cfg.prefetch_to, device=device)
            else:
                prefetch_managed(features, location=managed_cfg.prefetch_to, device=device)

        if hot_feature_cache is not None:
            stage_feature_rows_(
                features,
                hot_feature_cache,
            )
        if optimized_hmm_active:
            if gcn_kernel_impl == "pyg_baseline":
                spmm_kernel = f"optimized_pyg_{ft_memory_mode}"
            else:
                spmm_kernel = f"optimized_{ft_memory_mode}"

    feature_addr_start = int(features.data_ptr())
    feature_addr_end = feature_addr_start + feature_bytes

    def forward_once() -> torch.Tensor:
        def run_aggregation(
            layer: LayerState,
            idx: int,
            tag: str,
            layer_spmm_mode: str,
            x: torch.Tensor,
        ) -> None:
            with _nvtx_range(nvtx_enabled, f"{tag}/aggregation"):
                with phase_summary.measure("spmm"):
                    if gcn_kernel_impl == "pyg_baseline":
                        spmm_pyg_gcn_forward_(
                            row_ptr,
                            col_ind,
                            edge_weight,
                            x,
                            layer.agg_buffer,
                            spmm_mode=layer_spmm_mode,
                            optimized_backend=ft_memory_mode if idx == 0 else "hmm",
                            row_schedule=row_schedule_device if idx == 0 else None,
                            hot_feature_cache=hot_feature_cache if idx == 0 else None,
                            hot_node_cutoff=hot_node_cutoff if idx == 0 else 0,
                        )
                    else:
                        spmm_gcn_forward_(
                            row_ptr,
                            col_ind,
                            deg_inv_sqrt,
                            x,
                            layer.agg_buffer,
                            spmm_mode=layer_spmm_mode,
                            row_schedule=row_schedule_device if idx == 0 else None,
                            hot_feature_cache=hot_feature_cache if idx == 0 else None,
                            hot_node_cutoff=hot_node_cutoff if idx == 0 else 0,
                        )

        x = features
        for idx, layer in enumerate(layer_states):
            tag = f"layer{idx + 1}"
            layer_spmm_mode = spmm_mode
            if idx == 0 and optimized_hmm_active:
                layer_spmm_mode = "optimized"
            elif idx != 0 and layer_spmm_mode == "optimized":
                layer_spmm_mode = "plain"
            if idx == 0:
                activate_feature_reserve()
                try:
                    run_aggregation(layer, idx, tag, layer_spmm_mode, x)
                finally:
                    release_feature_reserve()
            else:
                run_aggregation(layer, idx, tag, layer_spmm_mode, x)
            with _nvtx_range(nvtx_enabled, f"{tag}/dense_update"):
                with phase_summary.measure("gemm"):
                    gemm_forward_(
                        layer.agg_buffer,
                        layer.weight,
                        layer.out_buffer,
                    )
                with phase_summary.measure("epilogue"):
                    bias_relu_forward_(
                        layer.out_buffer,
                        layer.bias,
                        relu=idx != len(layer_states) - 1,
                    )
            x = layer.out_buffer
        return x

    with torch.no_grad():
        for _ in range(max(0, int(args.pretouch_passes))):
            with _autocast_context(False, device):
                forward_once()
        for _ in range(args.warmup):
            with _autocast_context(False, device):
                forward_once()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        phase_summary.reset()

        _cuda_profiler_start(device)
        start = time.perf_counter()
        for _ in range(args.iters):
            with _nvtx_range(nvtx_enabled, "iteration"):
                with _autocast_context(False, device):
                    output = forward_once()
            phase_summary.record_iteration(("spmm", "gemm", "epilogue"))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        _cuda_profiler_stop(device)

    print(f"Graph:\t{graph_path}")
    print(f"Impl:\t{framework_label}_gcn_cuda")
    print(f"Adjacency Memory Mode:\t{adj_memory_mode}")
    print(f"Feature Memory Mode:\t{ft_memory_mode}")
    print(f"Feature Initial Location:\t{feature_initial_location}")
    print(f"Feature Prefetch To:\t{managed_cfg.prefetch_to}")
    print(f"Feature Preferred Location:\t{feature_preferred_location if uses_cuda_memory_hints(ft_memory_mode) else 'none'}")
    print(f"Weight Memory Mode:\t{weight_memory_mode}")
    print(f"Activation Memory Mode:\t{activation_memory_mode}")
    print(f"GCN Kernel Impl:\t{gcn_kernel_impl}")
    print(f"SpMM Mode:\t{spmm_mode}")
    print(f"Pre-touch Passes:\t{int(args.pretouch_passes)}")
    print(f"Feature Host Alloc Target (%):\t{ft_host_alloc:.3f}")
    print(f"Feature Reserve Enabled:\t{ft_reserve_enabled}")
    print(f"Feature Bytes:\t{feature_bytes}")
    print(f"Feature CUDA Prefetch Bytes:\t{feature_cuda_prefetch_bytes}")
    print(f"Feature Address Start:\t{feature_addr_start}")
    print(f"Feature Address End:\t{feature_addr_end}")
    print(f"Target GPU Feature Bytes:\t{target_gpu_feature_bytes}")
    print(f"Reserve Free Memory Before (GB):\t{reserve_free_bytes_before / 1_000_000_000.0:.3f}")
    print(f"Reserve Free Memory After (GB):\t{reserve_free_bytes_after / 1_000_000_000.0:.3f}")
    print(f"Reserved Device Memory (GB):\t{reserve_device_bytes / 1_000_000_000.0:.3f}")
    print(f"Activated Reserved Device Memory (GB):\t{reserve_activated_bytes / 1_000_000_000.0:.3f}")
    print(f"Reserve Activations:\t{reserve_activations}")
    print(f"SpMM Kernel:\t{spmm_kernel}")
    print(f"Preprocess Meta:\t{preprocess_meta_path or 'none'}")
    print(f"Node Reorder:\t{reorder_enabled}")
    print(f"Page Rows Per Host Page:\t{rows_per_page}")
    print(f"Row Block Size:\t{row_block_size}")
    print(f"Window Num Blocks:\t{window_num_blocks}")
    print(f"Hot Feature Cache Pages:\t{int(hot_pages_cpu.numel())}")
    print(f"Hot Feature Cache Coverage:\t{hot_cache_coverage:.6f}")
    print(f"Hot Node Cutoff:\t{hot_node_cutoff}")
    print(f"Hot Node Fraction:\t{hot_node_fraction:.6f}")
    print(f"Hot Node Access Coverage:\t{hot_node_access_coverage:.6f}")
    print(f"Optimized Managed Active:\t{optimized_hmm_active}")
    if int(page_reuse_histogram.numel()) > 0:
        nonzero_bins = torch.nonzero(page_reuse_histogram, as_tuple=False).flatten()
        if int(nonzero_bins.numel()) > 0:
            head = nonzero_bins[:8].tolist()
            print(
                "Page Reuse Histogram Head:\t"
                + ", ".join(f"{int(bin_idx)}->{int(page_reuse_histogram[int(bin_idx)].item())}" for bin_idx in head)
            )
    print(f"Nodes:\t{num_nodes}")
    print(f"Edges:\t{num_edges}")
    print(_describe_memory("row_ptr", row_ptr))
    print(_describe_memory("col_ind", col_ind))
    if edge_weight is not None:
        print(_describe_memory("edge_weight", edge_weight))
    else:
        print(_describe_memory("deg_inv_sqrt", deg_inv_sqrt))
    print(_describe_memory("features", features))
    print(_describe_memory("weights_l0", layer_states[0].weight))
    if reserve_tensor is not None:
        print(_describe_memory("reserve_tensor", reserve_tensor) + f" bytes={reserve_tensor.numel()}")
    print(_describe_memory("output", output))
    print(f"Output Sum:\t{float(output.sum().item()):.6f}")
    infer_avg_ns = elapsed * 1e9 / args.iters
    print(f"Infer (ns):\t{infer_avg_ns:.3f}")
    print_summary_report(phase_summary, iters=int(args.iters), infer_avg_ns=infer_avg_ns)
    return 0
