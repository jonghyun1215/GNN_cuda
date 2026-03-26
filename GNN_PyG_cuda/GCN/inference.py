#!/usr/bin/env python3

from GNN_cuda.GNN_cuda_common.gcn_inference import run_gcn_inference
from GNN_PyG.common.npz_utils import load_graph_features_from_npz, select_device


def main() -> int:
    return run_gcn_inference(
        framework_label="pyg",
        select_device=select_device,
        load_graph_features=load_graph_features_from_npz,
        load_kind="pyg",
    )


if __name__ == "__main__":
    raise SystemExit(main())
