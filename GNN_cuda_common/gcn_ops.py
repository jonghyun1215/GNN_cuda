#!/usr/bin/env python3
from __future__ import annotations

import torch

from .loader import load_gcn_module


def spmm_gcn_forward_(
    row_ptr: torch.Tensor,
    col_ind: torch.Tensor,
    deg_inv_sqrt: torch.Tensor,
    x: torch.Tensor,
    out: torch.Tensor,
    *,
    spmm_mode: str = "plain",
    row_schedule: torch.Tensor | None = None,
    hot_feature_cache: torch.Tensor | None = None,
    hot_node_cutoff: int = 0,
) -> None:
    module = load_gcn_module()
    if spmm_mode == "optimized":
        module.spmm_gcn_hmm_optimized_forward_(
            row_ptr,
            col_ind,
            deg_inv_sqrt,
            x,
            row_schedule,
            hot_feature_cache,
            int(hot_node_cutoff),
            out,
        )
        return
    if spmm_mode == "plain":
        module.spmm_gcn_hmm_forward_(row_ptr, col_ind, deg_inv_sqrt, x, out)
        return
    module.spmm_gcn_forward_(row_ptr, col_ind, deg_inv_sqrt, x, out)


def stage_feature_pages_(
    page_ids: torch.Tensor,
    x: torch.Tensor,
    out: torch.Tensor,
    *,
    rows_per_page: int,
) -> None:
    load_gcn_module().stage_feature_pages_(page_ids, x, out, int(rows_per_page))


def stage_feature_rows_(
    x: torch.Tensor,
    out: torch.Tensor,
) -> None:
    load_gcn_module().stage_feature_rows_(x, out)
