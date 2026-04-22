#!/usr/bin/env python3
from __future__ import annotations

import torch

from .loader import load_pyg_gcn_module


def spmm_pyg_gcn_forward_(
    row_ptr: torch.Tensor,
    col_ind: torch.Tensor,
    edge_weight: torch.Tensor,
    x: torch.Tensor,
    out: torch.Tensor,
    *,
    spmm_mode: str = "plain",
    optimized_backend: str = "hmm",
    row_schedule: torch.Tensor | None = None,
    hot_feature_cache: torch.Tensor | None = None,
    hot_node_cutoff: int = 0,
) -> None:
    module = load_pyg_gcn_module()
    if spmm_mode == "optimized":
        if optimized_backend == "uvm":
            module.spmm_pyg_gcn_uvm_optimized_forward_(
                row_ptr,
                col_ind,
                edge_weight,
                x,
                row_schedule,
                hot_feature_cache,
                int(hot_node_cutoff),
                out,
            )
        else:
            module.spmm_pyg_gcn_hmm_optimized_forward_(
                row_ptr,
                col_ind,
                edge_weight,
                x,
                row_schedule,
                hot_feature_cache,
                int(hot_node_cutoff),
                out,
            )
        return
    if spmm_mode == "plain":
        module.spmm_pyg_gcn_plain_forward_(row_ptr, col_ind, edge_weight, x, out)
        return
    module.spmm_pyg_gcn_forward_(row_ptr, col_ind, edge_weight, x, out)
