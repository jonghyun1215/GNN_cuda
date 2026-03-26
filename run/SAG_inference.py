#!/usr/bin/env python3
from __future__ import annotations

from _runner import dispatch_model


def main() -> int:
    return dispatch_model(
        {
            "pyg": "GNN_cuda.GNN_PyG_cuda.GraphSAGE.inference",
            "dgl": "GNN_cuda.GNN_DGL_cuda.GraphSAGE.inference",
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())
