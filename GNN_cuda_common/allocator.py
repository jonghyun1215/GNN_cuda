#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .loader import load_allocator_module


DTYPE_TO_CODE = {
    torch.float32: 0,
    torch.float64: 1,
    torch.float16: 2,
    torch.int32: 3,
    torch.int64: 4,
}

LOCATION_TO_CODE = {
    "cpu": -1,
    "cuda": 0,
}

GRAPH_MEMORY_MODES = ("device", "uvm", "hmm")
COMPUTE_MEMORY_MODES = ("device", "uvm")
MEMORY_MODE_ALIASES = {
    "device": "device",
    "torch_cuda": "device",
    "uvm": "uvm",
    "managed": "uvm",
    "hmm": "hmm",
    "host_mapped": "hmm",
}


@dataclass
class ManagedAllocationConfig:
    preferred_location: str = "cuda"
    accessed_by_cpu: bool = True
    accessed_by_cuda: bool = True
    read_mostly_graph: bool = True
    prefetch_to: str = "cuda"


def _allocator():
    return load_allocator_module()


def normalize_memory_mode(memory_mode: str, *, allow_hmm: bool) -> str:
    try:
        normalized = MEMORY_MODE_ALIASES[str(memory_mode)]
    except KeyError as exc:
        supported = GRAPH_MEMORY_MODES if allow_hmm else COMPUTE_MEMORY_MODES
        raise ValueError(f"Unsupported memory_mode: {memory_mode}. Expected one of {supported}.") from exc
    if normalized == "hmm" and not allow_hmm:
        raise ValueError("compute memory only supports 'device' or 'uvm'; 'hmm' is graph-only")
    return normalized


def is_uvm_mode(memory_mode: str) -> bool:
    return normalize_memory_mode(memory_mode, allow_hmm=True) == "uvm"


def managed_empty(
    shape: Iterable[int],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if device.type != "cuda":
        raise ValueError("managed_empty requires a CUDA device")
    try:
        dtype_code = DTYPE_TO_CODE[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype for managed allocation: {dtype}") from exc
    return _allocator().managed_empty(list(shape), dtype_code, int(device.index or 0))


def host_mapped_empty(
    shape: Iterable[int],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if device.type != "cuda":
        raise ValueError("host_mapped_empty requires a CUDA device for mapping")
    try:
        dtype_code = DTYPE_TO_CODE[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype for host-mapped allocation: {dtype}") from exc
    return _allocator().host_mapped_empty(list(shape), dtype_code, int(device.index or 0))


def allocate_like_mode(
    cpu_tensor: torch.Tensor,
    *,
    memory_mode: str,
    device: torch.device,
) -> torch.Tensor:
    memory_mode = normalize_memory_mode(memory_mode, allow_hmm=True)
    if memory_mode == "uvm":
        out = managed_empty(cpu_tensor.shape, dtype=cpu_tensor.dtype, device=device)
        out.copy_(cpu_tensor, non_blocking=False)
        return out
    if memory_mode == "hmm":
        out = host_mapped_empty(cpu_tensor.shape, dtype=cpu_tensor.dtype, device=device)
        out.copy_(cpu_tensor, non_blocking=False)
        return out
    if memory_mode == "device":
        return cpu_tensor.to(device=device)
    raise ValueError(f"Unsupported memory_mode: {memory_mode}")


def allocate_empty(
    shape: Iterable[int],
    *,
    dtype: torch.dtype,
    device: torch.device,
    memory_mode: str,
) -> torch.Tensor:
    memory_mode = normalize_memory_mode(memory_mode, allow_hmm=True)
    if memory_mode == "uvm":
        return managed_empty(shape, dtype=dtype, device=device)
    if memory_mode == "hmm":
        return host_mapped_empty(shape, dtype=dtype, device=device)
    if memory_mode == "device":
        return torch.empty(tuple(shape), dtype=dtype, device=device)
    raise ValueError(f"Unsupported memory_mode: {memory_mode}")


def apply_managed_policy(
    tensor: torch.Tensor,
    *,
    device: torch.device,
    preferred_location: str,
    accessed_by_cpu: bool,
    accessed_by_cuda: bool,
    read_mostly: bool,
) -> None:
    mod = _allocator()
    if preferred_location != "none":
        mod.advise_preferred_location_(tensor, LOCATION_TO_CODE[preferred_location], int(device.index or 0))
    if accessed_by_cpu:
        mod.advise_accessed_by_(tensor, LOCATION_TO_CODE["cpu"], int(device.index or 0))
    if accessed_by_cuda:
        mod.advise_accessed_by_(tensor, LOCATION_TO_CODE["cuda"], int(device.index or 0))
    if read_mostly:
        mod.advise_read_mostly_(tensor, True)


def prefetch_managed(tensor: torch.Tensor, *, location: str, device: torch.device) -> None:
    mod = _allocator()
    if location == "none":
        return
    target = LOCATION_TO_CODE[location]
    if target == 0:
        target = int(device.index or 0)
    mod.prefetch_(tensor, target)


def pointer_info(tensor: torch.Tensor) -> dict[str, object]:
    return dict(_allocator().pointer_info(tensor))
