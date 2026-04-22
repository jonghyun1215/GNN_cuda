#!/usr/bin/env python3
from __future__ import annotations

import torch

from .loader import load_agg_module


def spmm_sum_forward_(
    row_ptr: torch.Tensor,
    col_ind: torch.Tensor,
    x: torch.Tensor,
    out: torch.Tensor,
    *,
    hmm_optimized: bool = False,
) -> None:
    module = load_agg_module()
    if hmm_optimized:
        module.spmm_sum_hmm_forward_(row_ptr, col_ind, x, out)
        return
    module.spmm_sum_forward_(row_ptr, col_ind, x, out)


def spmm_mean_forward_(
    row_ptr: torch.Tensor,
    col_ind: torch.Tensor,
    x: torch.Tensor,
    out: torch.Tensor,
    *,
    hmm_optimized: bool = False,
) -> None:
    module = load_agg_module()
    if hmm_optimized:
        module.spmm_mean_hmm_forward_(row_ptr, col_ind, x, out)
        return
    module.spmm_mean_forward_(row_ptr, col_ind, x, out)


def gemm_forward_(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
    load_agg_module().gemm_forward_(x, weight, out)


def bias_relu_forward_(out: torch.Tensor, bias: torch.Tensor, relu: bool = False) -> None:
    load_agg_module().bias_relu_forward_(out, bias, bool(relu))


def linear_forward_(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    relu: bool = False,
) -> None:
    gemm_forward_(x, weight, out)
    bias_relu_forward_(out, bias, relu=relu)


def tensor_add_inplace_(dst: torch.Tensor, src: torch.Tensor, alpha: float = 1.0) -> None:
    load_agg_module().tensor_add_inplace_(dst, src, float(alpha))


def relu_inplace_(tensor: torch.Tensor) -> None:
    load_agg_module().relu_inplace_(tensor)
