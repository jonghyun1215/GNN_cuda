#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee


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


@dataclass
class WeightedCSR:
    row_ptr: torch.Tensor
    col_ind: torch.Tensor
    edge_weight: torch.Tensor
    num_nodes: int
    num_edges: int


def default_gcn_preprocess_meta_path(dataset: str, *, data_root: str) -> str:
    return str(Path(data_root) / "gnn_cuda_preprocessed" / "gcn" / f"{dataset}.pt")


def feature_rows_per_page(
    *,
    feat_dim: int,
    page_bytes: int = 4096,
    dtype_bytes: int = 4,
) -> int:
    row_bytes = max(1, int(feat_dim) * int(dtype_bytes))
    return max(1, int(page_bytes) // row_bytes)


def apply_node_permutation(
    src: torch.Tensor,
    dst: torch.Tensor,
    features: torch.Tensor,
    *,
    perm: torch.Tensor,
    inv_perm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    perm = perm.to(dtype=torch.long, device="cpu").contiguous()
    inv_perm = inv_perm.to(dtype=torch.long, device="cpu").contiguous()
    src_out = inv_perm[src.long().cpu()].contiguous()
    dst_out = inv_perm[dst.long().cpu()].contiguous()
    features_out = features[perm].contiguous()
    return src_out, dst_out, features_out


def apply_graph_permutation(
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    inv_perm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_perm = inv_perm.to(dtype=torch.long, device="cpu").contiguous()
    src_out = inv_perm[src.long().cpu()].contiguous()
    dst_out = inv_perm[dst.long().cpu()].contiguous()
    return src_out, dst_out


def build_rcm_permutation(
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    num_nodes: int,
    add_self_loops: bool = True,
    symmetrize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    src_np = src.cpu().numpy().astype(np.int64, copy=False)
    dst_np = dst.cpu().numpy().astype(np.int64, copy=False)
    if add_self_loops:
        loop = np.arange(int(num_nodes), dtype=np.int64)
        src_np = np.concatenate([src_np, loop], axis=0)
        dst_np = np.concatenate([dst_np, loop], axis=0)
    values = np.ones(src_np.shape[0], dtype=np.float32)
    adjacency = coo_matrix((values, (src_np, dst_np)), shape=(int(num_nodes), int(num_nodes))).tocsr()
    if symmetrize:
        adjacency = (adjacency + adjacency.transpose()).tocsr()
    adjacency.sort_indices()
    perm_np = reverse_cuthill_mckee(adjacency, symmetric_mode=True).astype(np.int64, copy=False)
    perm = torch.from_numpy(perm_np).long().contiguous()
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(int(num_nodes), dtype=torch.long)
    return perm, inv_perm


def build_hot_reuse_permutation(
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    num_nodes: int,
    add_self_loops: bool = True,
    topk_per_row: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    src_np = src.cpu().numpy().astype(np.int64, copy=False)
    dst_np = dst.cpu().numpy().astype(np.int64, copy=False)
    if add_self_loops:
        loop = np.arange(int(num_nodes), dtype=np.int64)
        src_np = np.concatenate([src_np, loop], axis=0)
        dst_np = np.concatenate([dst_np, loop], axis=0)
    values = np.ones(src_np.shape[0], dtype=np.float32)
    csr = coo_matrix((values, (dst_np, src_np)), shape=(int(num_nodes), int(num_nodes))).tocsr()
    csr.sort_indices()

    access_count = np.bincount(csr.indices, minlength=int(num_nodes)).astype(np.int64, copy=False)
    hot_order = np.argsort(-access_count, kind="stable")
    source_rank = np.empty(int(num_nodes), dtype=np.int64)
    source_rank[hot_order] = np.arange(int(num_nodes), dtype=np.int64)

    indptr = csr.indptr.astype(np.int64, copy=False)
    indices = csr.indices.astype(np.int64, copy=False)
    degree = np.diff(indptr).astype(np.int64, copy=False)
    pad = np.int64(num_nodes)
    top_ranks = np.full((int(num_nodes), max(1, int(topk_per_row))), pad, dtype=np.int64)

    for row in range(int(num_nodes)):
        row_start = indptr[row]
        row_end = indptr[row + 1]
        cols = indices[row_start:row_end]
        if cols.size == 0:
            continue
        ranked = np.sort(source_rank[cols], kind="stable")
        limit = min(ranked.size, top_ranks.shape[1])
        top_ranks[row, :limit] = ranked[:limit]

    sort_keys = [np.arange(int(num_nodes), dtype=np.int64), -degree]
    for key_idx in range(top_ranks.shape[1] - 1, -1, -1):
        sort_keys.append(top_ranks[:, key_idx])
    perm_np = np.lexsort(tuple(sort_keys)).astype(np.int64, copy=False)
    perm = torch.from_numpy(perm_np).long().contiguous()
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(int(num_nodes), dtype=torch.long)
    return perm, inv_perm


def build_hot_source_nodes(
    src: torch.Tensor,
    *,
    num_nodes: int,
    topk: int,
    add_self_loops: bool = True,
) -> torch.Tensor:
    src_cpu = src.long().cpu()
    if add_self_loops:
        src_cpu = torch.cat([src_cpu, torch.arange(int(num_nodes), dtype=torch.long)], dim=0)
    counts = torch.bincount(src_cpu, minlength=int(num_nodes))
    k = max(0, min(int(topk), int(num_nodes)))
    if k == 0:
        return torch.empty((0,), dtype=torch.long)
    hot_nodes = torch.argsort(counts, descending=True, stable=True)[:k]
    return hot_nodes.contiguous()


def build_hot_partition_feature_permutation(
    src: torch.Tensor,
    *,
    num_nodes: int,
    base_order: torch.Tensor | None = None,
    hot_access_coverage: float = 0.80,
    hot_max_ratio: float = 0.25,
    hot_min_nodes: int = 256,
    add_self_loops: bool = True,
) -> dict[str, object]:
    src_cpu = src.long().cpu()
    if add_self_loops:
        src_cpu = torch.cat([src_cpu, torch.arange(int(num_nodes), dtype=torch.long)], dim=0)
    access_count = torch.bincount(src_cpu, minlength=int(num_nodes)).to(dtype=torch.long).contiguous()
    sorted_nodes = torch.argsort(access_count, descending=True, stable=True).contiguous()
    total_access = int(access_count.sum().item())

    coverage = float(max(0.0, min(1.0, float(hot_access_coverage))))
    if float(hot_max_ratio) <= 0.0:
        max_nodes = int(num_nodes)
    else:
        max_nodes = max(1, min(int(num_nodes), int(np.ceil(float(hot_max_ratio) * int(num_nodes)))))
    min_nodes = max(0, min(int(hot_min_nodes), int(num_nodes)))
    min_nodes = min(min_nodes, max_nodes)

    cutoff = min_nodes
    if total_access > 0 and coverage > 0.0:
        prefix = torch.cumsum(access_count[sorted_nodes], dim=0)
        target = int(np.ceil(float(total_access) * coverage))
        cutoff = max(cutoff, int(torch.searchsorted(prefix, target, right=False).item()) + 1)
    elif total_access > 0:
        cutoff = max(cutoff, 1)
    cutoff = max(min_nodes, min(int(num_nodes), int(cutoff), int(max_nodes)))

    hot_nodes = sorted_nodes[:cutoff].contiguous()
    hot_mask = torch.zeros((int(num_nodes),), dtype=torch.bool)
    if int(hot_nodes.numel()) > 0:
        hot_mask[hot_nodes] = True

    if base_order is None:
        perm = sorted_nodes
    else:
        base_order_cpu = base_order.to(dtype=torch.long, device="cpu").contiguous()
        if int(base_order_cpu.numel()) != int(num_nodes):
            raise ValueError("base_order must contain exactly num_nodes entries")
        hot_prefix = base_order_cpu[hot_mask[base_order_cpu]]
        cold_suffix = base_order_cpu[~hot_mask[base_order_cpu]]
        perm = torch.cat([hot_prefix, cold_suffix], dim=0).contiguous()

    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(int(num_nodes), dtype=torch.long)

    hot_access = int(access_count[hot_nodes].sum().item()) if int(hot_nodes.numel()) > 0 else 0
    realized_coverage = float(hot_access / total_access) if total_access > 0 else 0.0
    return {
        "perm": perm,
        "inv_perm": inv_perm,
        "hot_nodes": hot_nodes,
        "hot_node_cutoff": int(cutoff),
        "hot_node_fraction": float(cutoff / max(1, int(num_nodes))),
        "hot_node_access_coverage": float(realized_coverage),
        "hot_node_access_counts": access_count,
        "hot_total_accesses": int(total_access),
        "hot_access_target": float(coverage),
        "hot_max_ratio": float(hot_max_ratio),
        "hot_min_nodes": int(min_nodes),
    }


def build_page_reuse_schedule_metadata(
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    num_nodes: int,
    feat_dim: int,
    page_bytes: int = 4096,
    row_block_size: int = 64,
    hot_cache_pages: int = 1024,
    signature_topk: int = 4,
    window_num_blocks: int = 4,
    window_cache_pages: int = 8,
    add_self_loops: bool = True,
) -> dict[str, object]:
    src_np = src.cpu().numpy().astype(np.int64, copy=False)
    dst_np = dst.cpu().numpy().astype(np.int64, copy=False)
    if add_self_loops:
        loop = np.arange(int(num_nodes), dtype=np.int64)
        src_np = np.concatenate([src_np, loop], axis=0)
        dst_np = np.concatenate([dst_np, loop], axis=0)
    values = np.ones(src_np.shape[0], dtype=np.float32)
    csr = coo_matrix((values, (dst_np, src_np)), shape=(int(num_nodes), int(num_nodes))).tocsr()
    csr.sort_indices()

    rows_per_page = feature_rows_per_page(feat_dim=int(feat_dim), page_bytes=int(page_bytes))
    num_pages = int((int(num_nodes) + rows_per_page - 1) // rows_per_page)
    row_ptr = csr.indptr.astype(np.int64, copy=False)
    col_ind = csr.indices.astype(np.int64, copy=False)
    page_ids = (col_ind // rows_per_page).astype(np.int64, copy=False)
    page_hist = np.bincount(page_ids, minlength=num_pages).astype(np.int64, copy=False)
    nonzero_page_hist = page_hist[page_hist > 0]
    page_reuse_hist = (
        np.bincount(nonzero_page_hist).astype(np.int64, copy=False)
        if nonzero_page_hist.size > 0
        else np.zeros((1,), dtype=np.int64)
    )
    degree = np.diff(row_ptr).astype(np.int64, copy=False)

    hot_order = np.argsort(-page_hist, kind="stable")
    hot_cache_pages = max(0, min(int(hot_cache_pages), int(num_pages)))
    hot_pages = hot_order[:hot_cache_pages].astype(np.int64, copy=False)
    total_access = int(page_hist.sum())
    hot_coverage = float(page_hist[hot_pages].sum() / total_access) if hot_pages.size > 0 and total_access > 0 else 0.0

    page_rank = np.empty(int(num_pages), dtype=np.int64)
    page_rank[hot_order] = np.arange(int(num_pages), dtype=np.int64)
    signature_topk = max(1, int(signature_topk))
    signatures = np.full((int(num_nodes), signature_topk), int(num_pages), dtype=np.int64)

    for row in range(int(num_nodes)):
        start = row_ptr[row]
        end = row_ptr[row + 1]
        row_pages = page_ids[start:end]
        if row_pages.size == 0:
            continue
        uniq_pages, counts = np.unique(row_pages, return_counts=True)
        order = np.lexsort((uniq_pages, page_rank[uniq_pages], -counts))
        picked = uniq_pages[order][:signature_topk]
        signatures[row, : picked.size] = picked

    sort_keys = [np.arange(int(num_nodes), dtype=np.int64), -degree]
    for key_idx in range(signature_topk - 1, -1, -1):
        sort_keys.append(signatures[:, key_idx])
    row_schedule = np.lexsort(tuple(sort_keys)).astype(np.int64, copy=False)

    block_page_hist = np.zeros((int((int(num_nodes) + int(row_block_size) - 1) // int(row_block_size)), num_pages), dtype=np.int64)
    for block_idx in range(block_page_hist.shape[0]):
        row_slot_start = block_idx * int(row_block_size)
        row_slot_end = min(int(num_nodes), row_slot_start + int(row_block_size))
        for row_slot in range(row_slot_start, row_slot_end):
            row = int(row_schedule[row_slot])
            start = row_ptr[row]
            end = row_ptr[row + 1]
            if start == end:
                continue
            np.add.at(block_page_hist[block_idx], page_ids[start:end], 1)
    block_hot_pages = np.full((block_page_hist.shape[0], signature_topk), int(num_pages), dtype=np.int64)
    for block_idx in range(block_page_hist.shape[0]):
        counts = block_page_hist[block_idx]
        nonzero = np.flatnonzero(counts)
        if nonzero.size == 0:
            continue
        order = np.lexsort((nonzero, page_rank[nonzero], -counts[nonzero]))
        picked = nonzero[order][:signature_topk]
        block_hot_pages[block_idx, : picked.size] = picked

    window_num_blocks = max(1, int(window_num_blocks))
    window_cache_pages = max(1, int(window_cache_pages))
    rows_per_window = int(row_block_size) * int(window_num_blocks)
    num_windows = int((int(num_nodes) + rows_per_window - 1) // rows_per_window)
    window_page_hist = np.zeros((num_windows, num_pages), dtype=np.int64)
    for window_idx in range(num_windows):
        row_slot_start = window_idx * rows_per_window
        row_slot_end = min(int(num_nodes), row_slot_start + rows_per_window)
        for row_slot in range(row_slot_start, row_slot_end):
            row = int(row_schedule[row_slot])
            start = row_ptr[row]
            end = row_ptr[row + 1]
            if start == end:
                continue
            np.add.at(window_page_hist[window_idx], page_ids[start:end], 1)
    window_hot_pages = np.full((num_windows, window_cache_pages), int(num_pages), dtype=np.int64)
    window_hot_coverage_sum = 0.0
    for window_idx in range(num_windows):
        counts = window_page_hist[window_idx]
        total = int(counts.sum())
        nonzero = np.flatnonzero(counts)
        if nonzero.size == 0:
            continue
        order = np.lexsort((nonzero, page_rank[nonzero], -counts[nonzero]))
        picked = nonzero[order][:window_cache_pages]
        window_hot_pages[window_idx, : picked.size] = picked
        if total > 0 and picked.size > 0:
            window_hot_coverage_sum += float(counts[picked].sum() / total)
    avg_window_hot_coverage = float(window_hot_coverage_sum / max(1, num_windows))

    return {
        "rows_per_page": int(rows_per_page),
        "num_pages": int(num_pages),
        "page_histogram": torch.from_numpy(page_hist.copy()).long(),
        "page_reuse_histogram": torch.from_numpy(page_reuse_hist.copy()).long(),
        "hot_pages": torch.from_numpy(hot_pages.copy()).long(),
        "hot_coverage": float(hot_coverage),
        "row_schedule": torch.from_numpy(row_schedule.copy()).long(),
        "row_block_size": int(row_block_size),
        "block_hot_pages": torch.from_numpy(block_hot_pages.copy()).long(),
        "window_num_blocks": int(window_num_blocks),
        "window_cache_pages": int(window_cache_pages),
        "rows_per_window": int(rows_per_window),
        "window_hot_pages": torch.from_numpy(window_hot_pages.copy()).long(),
        "avg_window_hot_coverage": float(avg_window_hot_coverage),
        "signature_topk": int(signature_topk),
        "page_bytes": int(page_bytes),
    }


def build_coaccess_feature_permutation(
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    num_nodes: int,
    row_schedule: torch.Tensor,
    row_block_size: int = 64,
    window_num_blocks: int = 4,
    signature_topk: int = 4,
    add_self_loops: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    src_np = src.cpu().numpy().astype(np.int64, copy=False)
    dst_np = dst.cpu().numpy().astype(np.int64, copy=False)
    if add_self_loops:
        loop = np.arange(int(num_nodes), dtype=np.int64)
        src_np = np.concatenate([src_np, loop], axis=0)
        dst_np = np.concatenate([dst_np, loop], axis=0)
    values = np.ones(src_np.shape[0], dtype=np.float32)
    csr = coo_matrix((values, (dst_np, src_np)), shape=(int(num_nodes), int(num_nodes))).tocsr()
    csr.sort_indices()

    row_ptr = csr.indptr.astype(np.int64, copy=False)
    col_ind = csr.indices.astype(np.int64, copy=False)
    row_schedule_np = row_schedule.to(dtype=torch.long, device="cpu").numpy().astype(np.int64, copy=False)
    rows_per_window = max(1, int(row_block_size) * int(window_num_blocks))
    num_windows = int((int(num_nodes) + rows_per_window - 1) // rows_per_window)
    signature_topk = max(1, int(signature_topk))

    access_count = np.zeros(int(num_nodes), dtype=np.int64)
    top_windows = np.full((int(num_nodes), signature_topk), int(num_windows), dtype=np.int64)
    top_counts = np.zeros((int(num_nodes), signature_topk), dtype=np.int64)

    node_window_entries: list[list[tuple[int, int]]] = [[] for _ in range(int(num_nodes))]
    for window_idx in range(num_windows):
        row_slot_start = window_idx * rows_per_window
        row_slot_end = min(int(num_nodes), row_slot_start + rows_per_window)
        counter: dict[int, int] = {}
        for row_slot in range(row_slot_start, row_slot_end):
            row = int(row_schedule_np[row_slot])
            start = int(row_ptr[row])
            end = int(row_ptr[row + 1])
            if start == end:
                continue
            uniq_nodes, counts = np.unique(col_ind[start:end], return_counts=True)
            for node, count in zip(uniq_nodes.tolist(), counts.tolist()):
                counter[int(node)] = counter.get(int(node), 0) + int(count)
        for node, count in counter.items():
            access_count[node] += int(count)
            node_window_entries[node].append((window_idx, int(count)))

    for node in range(int(num_nodes)):
        if not node_window_entries[node]:
            continue
        entries = sorted(node_window_entries[node], key=lambda item: (-item[1], item[0]))
        limit = min(len(entries), signature_topk)
        for idx in range(limit):
            top_windows[node, idx] = int(entries[idx][0])
            top_counts[node, idx] = int(entries[idx][1])

    sort_keys = [np.arange(int(num_nodes), dtype=np.int64), -access_count]
    for key_idx in range(signature_topk - 1, -1, -1):
        sort_keys.append(-top_counts[:, key_idx])
        sort_keys.append(top_windows[:, key_idx])
    perm_np = np.lexsort(tuple(sort_keys)).astype(np.int64, copy=False)
    perm = torch.from_numpy(perm_np.copy()).long().contiguous()
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(int(num_nodes), dtype=torch.long)
    return perm, inv_perm


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


def build_pyg_gcn_weighted_csr(
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    num_nodes: int,
    add_self_loops: bool = True,
) -> WeightedCSR:
    src_np = src.cpu().numpy().astype(np.int64, copy=False)
    dst_np = dst.cpu().numpy().astype(np.int64, copy=False)
    if add_self_loops:
        loop = np.arange(int(num_nodes), dtype=np.int64)
        src_np = np.concatenate([src_np, loop], axis=0)
        dst_np = np.concatenate([dst_np, loop], axis=0)
    values = np.ones(src_np.shape[0], dtype=np.float32)
    csr = coo_matrix((values, (dst_np, src_np)), shape=(int(num_nodes), int(num_nodes))).tocsr()
    csr.sort_indices()
    row_ptr = csr.indptr.astype(np.int32, copy=False)
    col_ind = csr.indices.astype(np.int32, copy=False)
    degree = np.diff(row_ptr).astype(np.float32, copy=False)
    deg_inv_sqrt = np.power(np.clip(degree, 1.0, None), -0.5).astype(np.float32, copy=False)
    row_ids = np.repeat(np.arange(int(num_nodes), dtype=np.int64), np.diff(row_ptr).astype(np.int64, copy=False))
    edge_weight = (deg_inv_sqrt[row_ids] * deg_inv_sqrt[col_ind.astype(np.int64, copy=False)]).astype(np.float32, copy=False)
    return WeightedCSR(
        row_ptr=torch.from_numpy(row_ptr).contiguous(),
        col_ind=torch.from_numpy(col_ind).contiguous(),
        edge_weight=torch.from_numpy(edge_weight).contiguous(),
        num_nodes=int(num_nodes),
        num_edges=int(col_ind.shape[0]),
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
