#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def select_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def _load_num_nodes(graph_obj) -> int:
    if "num_nodes" in graph_obj.files:
        return int(np.array(graph_obj["num_nodes"]).reshape(-1)[0])
    src_li = np.array(graph_obj["src_li"], dtype=np.int64)
    dst_li = np.array(graph_obj["dst_li"], dtype=np.int64)
    max_idx = max(int(src_li.max(initial=-1)), int(dst_li.max(initial=-1)))
    return max_idx + 1


def _random_features(num_nodes: int, feat_dim: int) -> torch.Tensor:
    target_dim = feat_dim if feat_dim > 0 else 128
    return torch.randn(num_nodes, target_dim, dtype=torch.float32)


def _dense_features_from_sparse_npz(graph_obj, num_nodes: int, feat_dim: int) -> torch.Tensor:
    required = {"feat_indices", "feat_values", "feat_shape"}
    if not required.issubset(set(graph_obj.files)):
        return _random_features(num_nodes, feat_dim)

    feat_shape = np.array(graph_obj["feat_shape"]).astype(np.int64).reshape(-1)
    if feat_shape.size != 2:
        return _random_features(num_nodes, feat_dim)

    src_rows = int(feat_shape[0])
    src_cols = int(feat_shape[1])
    if src_rows <= 0 or src_cols < 0:
        return _random_features(num_nodes, feat_dim)

    out_dim = feat_dim if feat_dim > 0 else src_cols
    if out_dim <= 0:
        out_dim = 128

    features = torch.zeros((num_nodes, out_dim), dtype=torch.float32)
    feat_indices = np.array(graph_obj["feat_indices"], dtype=np.int64)
    feat_values = np.array(graph_obj["feat_values"], dtype=np.float32)
    if feat_indices.ndim != 2 or feat_indices.shape[1] != 2 or feat_values.ndim != 1:
        return features
    if feat_indices.shape[0] != feat_values.shape[0]:
        return features

    valid = (
        (feat_indices[:, 0] >= 0)
        & (feat_indices[:, 0] < num_nodes)
        & (feat_indices[:, 1] >= 0)
        & (feat_indices[:, 1] < out_dim)
    )
    if not np.any(valid):
        return features

    rows = torch.from_numpy(feat_indices[valid, 0].astype(np.int64, copy=False))
    cols = torch.from_numpy(feat_indices[valid, 1].astype(np.int64, copy=False))
    vals = torch.from_numpy(feat_values[valid])
    features[rows, cols] = vals
    return features


def _load_common(npz_path: str, feat_dim: int, use_npz_features: bool):
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"graph file not found: {path}")

    graph_obj = np.load(str(path))
    src_li = np.array(graph_obj["src_li"], dtype=np.int64)
    dst_li = np.array(graph_obj["dst_li"], dtype=np.int64)
    num_nodes = _load_num_nodes(graph_obj)

    if use_npz_features:
        features = _dense_features_from_sparse_npz(graph_obj, num_nodes, feat_dim)
    else:
        features = _random_features(num_nodes, feat_dim)

    return src_li, dst_li, features, num_nodes


def load_pyg_graph_features_from_npz(npz_path: str, feat_dim: int, use_npz_features: bool):
    src_li, dst_li, features, num_nodes = _load_common(npz_path, feat_dim, use_npz_features)
    edge_index = torch.from_numpy(np.stack([src_li, dst_li], axis=0)).long()
    return edge_index, features, num_nodes


def load_dgl_graph_features_from_npz(npz_path: str, feat_dim: int, use_npz_features: bool):
    src_li, dst_li, features, num_nodes = _load_common(npz_path, feat_dim, use_npz_features)
    src = torch.from_numpy(src_li).long()
    dst = torch.from_numpy(dst_li).long()
    return src, dst, features, num_nodes
