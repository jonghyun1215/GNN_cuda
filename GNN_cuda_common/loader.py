#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_REPO_ROOT = Path(__file__).resolve().parents[1]
_NATIVE_DIR = Path(__file__).resolve().parent / "native"
_BUILD_ROOT = _REPO_ROOT / "build"

_ALLOCATOR_MODULE = None
_GCN_MODULE = None
_AGG_MODULE = None
_PYG_GCN_MODULE = None


def _torch_build_tag() -> str:
    version = torch.__version__.replace("+", "_").replace(".", "_")
    cuda_ver = (torch.version.cuda or "cpu").replace(".", "_")
    return f"torch_{version}_cuda_{cuda_ver}"


def _ensure_build_env() -> None:
    for env_key, candidates in {
        "CC": ("x86_64-conda-linux-gnu-gcc", "x86_64-conda_cos6-linux-gnu-gcc", "gcc"),
        "CXX": ("x86_64-conda-linux-gnu-g++", "x86_64-conda_cos6-linux-gnu-g++", "g++"),
    }.items():
        if env_key in os.environ:
            continue
        for candidate in candidates:
            path = shutil.which(candidate)
            if path:
                os.environ[env_key] = path
                break
    if "CUDAHOSTCXX" not in os.environ and "CXX" in os.environ:
        os.environ["CUDAHOSTCXX"] = os.environ["CXX"]
    if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    os.environ.setdefault("MAX_JOBS", "4")


def _load_native(name: str, sources: list[str]):
    _ensure_build_env()
    build_dir = _BUILD_ROOT / _torch_build_tag() / name
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name=f"{name}_{_torch_build_tag()}",
        sources=[str(_NATIVE_DIR / src) for src in sources],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        build_directory=str(build_dir),
        verbose=False,
    )


def load_allocator_module():
    global _ALLOCATOR_MODULE
    if _ALLOCATOR_MODULE is None:
        _ALLOCATOR_MODULE = _load_native(
            "gnn_uvm_allocator",
            ["uvm_allocator.cpp", "uvm_allocator.cu"],
        )
    return _ALLOCATOR_MODULE


def load_gcn_module():
    global _GCN_MODULE
    if _GCN_MODULE is None:
        _GCN_MODULE = _load_native(
            "gnn_gcn_cuda",
            ["gcn_ops.cpp", "gcn_ops.cu"],
        )
    return _GCN_MODULE


def load_pyg_gcn_module():
    global _PYG_GCN_MODULE
    if _PYG_GCN_MODULE is None:
        _PYG_GCN_MODULE = _load_native(
            "gnn_pyg_gcn_cuda",
            ["pyg_gcn_ops.cpp", "pyg_gcn_ops.cu"],
        )
    return _PYG_GCN_MODULE


def load_agg_module():
    global _AGG_MODULE
    if _AGG_MODULE is None:
        _AGG_MODULE = _load_native(
            "gnn_agg_cuda",
            ["agg_ops.cpp", "agg_ops.cu"],
        )
    return _AGG_MODULE
