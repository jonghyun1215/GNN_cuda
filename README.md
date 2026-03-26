# GNN CUDA

## Setup

```bash
cd GNN_cuda
source ./setup_environment dataset_path
gnn_cuda_create_envs --build
```

- `source ./setup_environment <dataset_path>`
  - sets `GNN_DATASET_ROOT`
  - exports the workspace and Python paths
  - prints the configured dataset path and available helper commands
- `gnn_cuda_create_envs --build`
  - creates missing conda environments from `env/cu128_pyg.yml` and `env/cu128_dgl.yml`
  - builds the native CUDA extensions for both environments
  - if the envs already exist, it only performs the build step

## Run

Single dataset:

```bash
python run/GCN_inference.py --framework pyg --dataset cora
python run/GIN_inference.py --framework dgl --dataset cora
python run/SAG_inference.py --framework pyg --dataset cora
```

All datasets under the default dataset directory:

```bash
python run/GCN_inference.py --framework pyg --dataset all
```

## Arguments

- `--framework`: backend frontend, `pyg` or `dgl`
- `--dataset`: dataset name or `all`
- `--dim`: base feature / hidden / output dimension, default `128`
- `--num_layers`: number of layers
- `--graph_memory_mode`: graph-input placement, `device`, `uvm`, or `host_mapped`
- `--compute_memory_mode`: weights / outputs / scratch placement, `device` or `uvm`
- `--device`: execution device, for example `cuda:0`
- `--warmup`: number of warmup iterations, default `1`
- `--iters`: number of measured iterations, default `5`

Memory modes:

- `device`: regular CUDA device memory
- `uvm`: `cudaMallocManaged`-based UVM
- `host_mapped`: current host-mapped graph-input path
