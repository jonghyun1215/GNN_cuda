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
) -> None:
    load_gcn_module().spmm_gcn_forward_(row_ptr, col_ind, deg_inv_sqrt, x, out)

