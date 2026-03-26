#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix


@dataclass
class NormalizedCSR:
    row_ptr: torch.Tensor
    col_ind: torch.Tensor
    deg_inv_sqrt: torch.Tensor
    num_nodes: int
    num_edges: int


@dataclass
class PlainCSR:
    row_ptr: torch.Tensor
    col_ind: torch.Tensor
    num_nodes: int
    num_edges: int


def build_gcn_normalized_csr(
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    num_nodes: int,
    add_self_loops: bool = True,
) -> NormalizedCSR:
    src_np = src.cpu().numpy().astype(np.int64, copy=False)
    dst_np = dst.cpu().numpy().astype(np.int64, copy=False)
    if add_self_loops:
        loop = np.arange(int(num_nodes), dtype=np.int64)
        src_np = np.concatenate([src_np, loop], axis=0)
        dst_np = np.concatenate([dst_np, loop], axis=0)
    values = np.ones(src_np.shape[0], dtype=np.float32)
    csr = coo_matrix((values, (dst_np, src_np)), shape=(int(num_nodes), int(num_nodes))).tocsr()
    csr.sort_indices()
    deg = np.diff(csr.indptr).astype(np.float32, copy=False)
    deg_inv_sqrt = np.power(np.clip(deg, 1.0, None), -0.5).astype(np.float32, copy=False)
    return NormalizedCSR(
        row_ptr=torch.from_numpy(csr.indptr.astype(np.int32, copy=False)).contiguous(),
        col_ind=torch.from_numpy(csr.indices.astype(np.int32, copy=False)).contiguous(),
        deg_inv_sqrt=torch.from_numpy(deg_inv_sqrt).contiguous(),
        num_nodes=int(num_nodes),
        num_edges=int(csr.indices.shape[0]),
    )


def build_plain_csr(
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    num_nodes: int,
    add_self_loops: bool = False,
    transpose_for_incoming: bool = True,
) -> PlainCSR:
    src_np = src.cpu().numpy().astype(np.int64, copy=False)
    dst_np = dst.cpu().numpy().astype(np.int64, copy=False)
    if add_self_loops:
        loop = np.arange(int(num_nodes), dtype=np.int64)
        src_np = np.concatenate([src_np, loop], axis=0)
        dst_np = np.concatenate([dst_np, loop], axis=0)
    if transpose_for_incoming:
        row_np = dst_np
        col_np = src_np
    else:
        row_np = src_np
        col_np = dst_np
    values = np.ones(row_np.shape[0], dtype=np.float32)
    csr = coo_matrix((values, (row_np, col_np)), shape=(int(num_nodes), int(num_nodes))).tocsr()
    csr.sort_indices()
    return PlainCSR(
        row_ptr=torch.from_numpy(csr.indptr.astype(np.int32, copy=False)).contiguous(),
        col_ind=torch.from_numpy(csr.indices.astype(np.int32, copy=False)).contiguous(),
        num_nodes=int(num_nodes),
        num_edges=int(csr.indices.shape[0]),
    )


def load_src_dst_features(loaded, *, load_kind: str):
    if load_kind == "pyg":
        edge_index, features, num_nodes = loaded
        return edge_index[0].contiguous(), edge_index[1].contiguous(), features, num_nodes
    if load_kind == "dgl":
        src, dst, features, num_nodes = loaded
        return src.contiguous(), dst.contiguous(), features, num_nodes
    raise ValueError(f"Unsupported load_kind: {load_kind}")


def resolve_dataset_path(dataset: str, *, data_root: str) -> str:
    if dataset == "all":
        raise ValueError("'all' is only supported by the run/ entrypoints")
    if "/" in dataset or "\\" in dataset or dataset.endswith(".npz"):
        raise ValueError("--dataset only accepts a dataset name or 'all'")
    return str(Path(data_root) / f"{dataset}.npz")
