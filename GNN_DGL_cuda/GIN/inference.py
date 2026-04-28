#!/usr/bin/env python3

from GNN_cuda.GNN_cuda_common.gin_inference import run_gin_inference
from GNN_cuda.GNN_cuda_common.npz_utils import load_dgl_graph_features_from_npz, select_device


def main() -> int:
    return run_gin_inference(
        framework_label="dgl",
        select_device=select_device,
        load_graph_features=load_dgl_graph_features_from_npz,
        load_kind="dgl",
        post_layer_relu=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
