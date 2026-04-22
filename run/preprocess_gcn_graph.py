#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from _bootstrap import bootstrap_pythonpath


bootstrap_pythonpath()

from GNN_cuda.GNN_cuda_common.graph_utils import (  # noqa: E402
    apply_graph_permutation,
    build_coaccess_feature_permutation,
    build_hot_partition_feature_permutation,
    build_page_reuse_schedule_metadata,
    default_gcn_preprocess_meta_path,
    resolve_dataset_path,
)


DEFAULT_DATA_ROOT = os.environ.get("GNN_DATASET_ROOT", "/root/workspace/mnt/dataset_npz")


def _load_edges(npz_path: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    graph_obj = np.load(npz_path)
    src = torch.from_numpy(np.array(graph_obj["src_li"], dtype=np.int64)).long().contiguous()
    dst = torch.from_numpy(np.array(graph_obj["dst_li"], dtype=np.int64)).long().contiguous()
    if "num_nodes" in graph_obj.files:
        num_nodes = int(np.array(graph_obj["num_nodes"]).reshape(-1)[0])
    else:
        num_nodes = max(int(src.max().item()), int(dst.max().item())) + 1
    return src, dst, num_nodes


def _preprocess_one(
    dataset: str,
    *,
    data_root: str,
    feat_dim: int,
    page_bytes: int,
    row_block_size: int,
    window_num_blocks: int,
    window_cache_pages: int,
    hot_cache_pages: int,
    signature_topk: int,
    hot_access_coverage: float,
    hot_max_ratio: float,
    hot_min_nodes: int,
) -> Path:
    graph_path = resolve_dataset_path(dataset, data_root=data_root)
    src, dst, num_nodes = _load_edges(graph_path)
    base_meta = build_page_reuse_schedule_metadata(
        src,
        dst,
        num_nodes=num_nodes,
        feat_dim=int(feat_dim),
        page_bytes=int(page_bytes),
        row_block_size=int(row_block_size),
        hot_cache_pages=int(hot_cache_pages),
        signature_topk=int(signature_topk),
        window_num_blocks=int(window_num_blocks),
        window_cache_pages=int(window_cache_pages),
        add_self_loops=True,
    )
    perm, inv_perm = build_coaccess_feature_permutation(
        src,
        dst,
        num_nodes=num_nodes,
        row_schedule=torch.as_tensor(base_meta["row_schedule"], dtype=torch.long),
        row_block_size=int(row_block_size),
        window_num_blocks=int(window_num_blocks),
        signature_topk=int(signature_topk),
        add_self_loops=True,
    )
    hot_partition = build_hot_partition_feature_permutation(
        src,
        num_nodes=num_nodes,
        base_order=perm,
        hot_access_coverage=float(hot_access_coverage),
        hot_max_ratio=float(hot_max_ratio),
        hot_min_nodes=int(hot_min_nodes),
        add_self_loops=True,
    )
    src_reordered, dst_reordered = apply_graph_permutation(
        src,
        dst,
        inv_perm=torch.as_tensor(hot_partition["inv_perm"], dtype=torch.long),
    )
    page_meta = build_page_reuse_schedule_metadata(
        src_reordered,
        dst_reordered,
        num_nodes=num_nodes,
        feat_dim=int(feat_dim),
        page_bytes=int(page_bytes),
        row_block_size=int(row_block_size),
        hot_cache_pages=int(hot_cache_pages),
        signature_topk=int(signature_topk),
        window_num_blocks=int(window_num_blocks),
        window_cache_pages=int(window_cache_pages),
        add_self_loops=True,
    )

    out_path = Path(default_gcn_preprocess_meta_path(dataset, data_root=data_root))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "graph_path": str(graph_path),
        "num_nodes": int(num_nodes),
        "feat_dim": int(feat_dim),
        "strategy": "hot_node_partition+coaccess_feature_reorder",
        "add_self_loops": True,
        "perm": torch.as_tensor(hot_partition["perm"], dtype=torch.long),
        "inv_perm": torch.as_tensor(hot_partition["inv_perm"], dtype=torch.long),
        "hot_nodes": torch.as_tensor(hot_partition["hot_nodes"], dtype=torch.long),
        "hot_node_cutoff": int(hot_partition["hot_node_cutoff"]),
        "hot_node_fraction": float(hot_partition["hot_node_fraction"]),
        "hot_node_access_coverage": float(hot_partition["hot_node_access_coverage"]),
        "hot_node_access_counts": torch.as_tensor(hot_partition["hot_node_access_counts"], dtype=torch.long),
        "hot_total_accesses": int(hot_partition["hot_total_accesses"]),
        "hot_access_target": float(hot_partition["hot_access_target"]),
        "hot_max_ratio": float(hot_partition["hot_max_ratio"]),
        "hot_min_nodes": int(hot_partition["hot_min_nodes"]),
        **page_meta,
    }
    torch.save(payload, out_path)
    print(f"Saved GCN preprocess metadata: {out_path}")
    print(
        f"dataset={dataset} nodes={num_nodes} num_pages={payload['num_pages']} "
        f"hot_pages={int(payload['hot_pages'].numel())} hot_coverage={float(payload['hot_coverage']):.6f} "
        f"hot_nodes={int(payload['hot_node_cutoff'])} hot_node_fraction={float(payload['hot_node_fraction']):.6f} "
        f"hot_node_access_coverage={float(payload['hot_node_access_coverage']):.6f} "
        f"rows_per_page={int(payload['rows_per_page'])} row_block_size={int(payload['row_block_size'])} "
        f"window_num_blocks={int(payload['window_num_blocks'])} window_cache_pages={int(payload['window_cache_pages'])} "
        f"avg_window_hot_coverage={float(payload['avg_window_hot_coverage']):.6f}"
    )
    reuse_hist = torch.as_tensor(payload["page_reuse_histogram"], dtype=torch.long)
    nonzero_bins = torch.nonzero(reuse_hist, as_tuple=False).flatten()
    if int(nonzero_bins.numel()) > 0:
        head = nonzero_bins[:8].tolist()
        print("page_reuse_histogram=" + ", ".join(f"{int(bin_idx)}->{int(reuse_hist[int(bin_idx)].item())}" for bin_idx in head))
    return out_path


def _dataset_names(data_root: str) -> list[str]:
    root = Path(data_root)
    return sorted(path.stem for path in root.glob("*.npz"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="dataset name or all")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT, help="directory containing <dataset>.npz")
    parser.add_argument("--feat_dim", type=int, default=128, help="feature dimension used for page packing")
    parser.add_argument("--page_bytes", type=int, default=4096, help="logical host page size for page reuse packing")
    parser.add_argument("--row_block_size", type=int, default=64, help="destination rows per packed block")
    parser.add_argument("--window_num_blocks", type=int, default=4, help="number of row-blocks per staged hot-page window")
    parser.add_argument("--window_cache_pages", type=int, default=8, help="pages staged per row-block window")
    parser.add_argument("--hot_cache_pages", type=int, default=1024, help="number of hot feature pages to stage on the GPU")
    parser.add_argument("--signature_topk", type=int, default=4, help="number of dominant pages per row signature")
    parser.add_argument("--hot_access_coverage", type=float, default=0.80, help="target source-access coverage for the staged hot-node prefix")
    parser.add_argument("--hot_max_ratio", type=float, default=0.25, help="maximum node fraction that can be promoted into the hot prefix")
    parser.add_argument("--hot_min_nodes", type=int, default=256, help="minimum node count in the hot prefix")
    args = parser.parse_args()

    datasets = _dataset_names(args.data_root) if args.dataset == "all" else [str(args.dataset)]
    if not datasets:
        raise FileNotFoundError(f"no .npz datasets found in: {args.data_root}")
    for dataset in datasets:
        _preprocess_one(
            dataset,
            data_root=str(args.data_root),
            feat_dim=int(args.feat_dim),
            page_bytes=int(args.page_bytes),
            row_block_size=int(args.row_block_size),
            window_num_blocks=int(args.window_num_blocks),
            window_cache_pages=int(args.window_cache_pages),
            hot_cache_pages=int(args.hot_cache_pages),
            signature_topk=int(args.signature_topk),
            hot_access_coverage=float(args.hot_access_coverage),
            hot_max_ratio=float(args.hot_max_ratio),
            hot_min_nodes=int(args.hot_min_nodes),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
