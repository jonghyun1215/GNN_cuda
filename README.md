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

Use the nsys wrapper below to print average SpMM time and feature-range UM migration metrics:

```bash
python run/profile_spmm_migration.py \
  --dataset Pubmed \
  --ft_matrix uvm \
  --ft_host_alloc 20
```

Output:

```text
Summary Report:
spmm_ns, ...
HtoD_bytes, ...
DtoH_bytes, ...
GPU_faults, ...
```

For `--ft_matrix uvm|hmm`, feature data is initialized from CPU memory by default. Use `--prefetch` to choose whether it remains host-resident before SpMM or is explicitly prefetched to CUDA.

To compare UVM without and with CUDA prefetch:

```bash
# CPU-resident UVM feature, no CUDA prefetch
python run/profile_spmm_migration.py \
  --dataset Pubmed \
  --ft_matrix uvm \
  --prefetch 0 \
  --warmup 0 \
  --iters 1

# CPU-resident UVM feature, CUDA prefetch before compute
python run/profile_spmm_migration.py \
  --dataset Pubmed \
  --ft_matrix uvm \
  --prefetch 1 \
  --warmup 0 \
  --iters 1
```

<!-- All reported UM metrics are filtered to the feature matrix virtual-address range. For `--ft_matrix hmm`, migration metrics use the NVTX `aggregation` range. For `--ft_matrix uvm`, migration metrics use the full profile timeline so prefetch-driven migration is included.
This wrapper passes prefetch/preferred-location hints as none when `--prefetch 0`, and cuda when `--prefetch 1`. -->

## Arguments

- `--dataset`: dataset name, required
- `--ft_matrix`: feature placement, `device`, `uvm`, or `hmm`, required
- `--ft_host_alloc`: target percent of the feature matrix that should not fit in remaining effective GPU memory; `0` disables reserve, values above `0` reserve before feature allocation and during SpMM, then release before dense GEMM
- `--prefetch`: `0` disables prefetch/preferred-location hints, `1` uses `cuda`, default `0`; with `--ft_host_alloc > 0`, GCN prefetches up to 80% of the remaining feature GPU budget and skips partial prefetch when the budget is below the 128 MiB runtime slack guard
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
