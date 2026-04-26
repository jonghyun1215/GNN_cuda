# GNN CUDA

## Setup

```bash
cd GNN_cuda
source ./setup_environment <dataset_path>
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

<!-- Wrapper entrypoints:

```bash
python run/GCN_inference.py --framework pyg --dataset Pubmed
python run/GIN_inference.py --framework dgl --dataset Pubmed
python run/SAG_inference.py --framework pyg --dataset Pubmed
python run/GCN_inference.py --framework pyg --dataset all
```

Direct inference entry used for previous manual experiments:

```bash
conda run -n cu128_pyg python GNN_PyG_cuda/GCN/inference.py \
  --dataset Pubmed \
  --num_layers 1 \
  --dim 128 \
  --adj_matrix device \
  --ft_matrix uvm \
  --weight device \
  --warmup 1 \
  --iters 5 \
  --device cuda:0
```
-->

Use the nsys wrapper below to print average SpMM time and UM migration metrics during SpMM:

```bash
python run/profile_spmm_migration.py \
  --dataset Pubmed \
  --ft_matrix uvm
```

Output:

```text
Summary Report:
spmm_ns, ...
HtoD_bytes, ...
DtoH_bytes, ...
GPU_faults, ...
```

All reported UM metrics are averaged per measured iteration over NVTX `aggregation` ranges.

## Arguments

- `--dataset`: dataset name, required
- `--ft_matrix`: feature placement, `device`, `uvm`, or `hmm`, required
- `--framework`: `pyg` or `dgl`, default `pyg`
- `--model`: `gcn`, `gin`, or `sag`, default `gcn`
- `--dim`: base feature / hidden / output dimension, default `128`
- `--num_layers`: number of layers, default `1`
- `--adj_matrix`: adjacency / CSR placement, `device`, `uvm`, or `hmm`, default `device`
- `--weight`: weights / outputs / scratch placement, `device` or `uvm`, default `device`
- `--device`: execution device, default `cuda:0`
- `--warmup`: number of warmup iterations, default `1`
- `--iters`: number of measured iterations, default `5`

Memory modes:

- `device`: regular CUDA device memory
- `uvm`: `cudaMallocManaged`-based UVM
- `hmm`: ordinary system-memory graph-input path for Linux HMM
