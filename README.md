# MNIST Vision Transformer in CUDA

A small encoder-only transformer that classifies MNIST. No patching — each of
the `28*28 = 784` pixels is one token, and the pixel intensity (0..255) is
its token id. Every operation is a hand-written CUDA kernel; `cuBLAS` is used
only for matmul. Multi-GPU training is data parallel via MPI + NCCL.

The whole project is one source file: [`src/train_vit.cu`](src/train_vit.cu).

## Architecture

```
pixels [B, 784]
   |
   |  wte[V=256, D]  (intensity embedding)
   |  wpe[T=784, D]  (positional embedding)
   v
encoded [B, T, D]
   |
   |  for l in [0, L):
   |     LN  ->  QKV (D -> 3D) -> attention -> proj (D -> D) -> + residual
   |     LN  ->  MLP (D -> 4D, GELU, 4D -> D)             -> + residual
   v
LN_final  ->  mean over T  [B, D]  ->  head (D -> 10)  ->  softmax CE
```

Default config: `V=256, T=784, L=2, D=64, H=4, head_dim=16, classes=10`.
About 120 000 trainable parameters.

## File layout

```
src/train_vit.cu     -- everything (kernels, forward/backward, training loop)
data/                -- put Kaggle's train.csv here
```

## Build

Needs `nvcc`, CUDA, cuBLAS, NCCL, an MPI install providing `mpicxx`.

```bash
mkdir -p bin
nvcc -O2 -std=c++17 -ccbin mpicxx src/train_vit.cu -o bin/train_vit -lcublas -lnccl
```

## Run

Download the Kaggle digit-recognizer `train.csv` and drop it into `data/`.

```bash
# single GPU
./bin/train_vit data/train.csv 200 8 0.05

# multi-GPU
mpirun -np 2 ./bin/train_vit data/train.csv 200 8 0.05
#                            ^csv          ^^^ ^ ^^^^
#                                       steps  B  lr
```

Output:

```
ranks=2 N=42000 B=8 T=784 L=2 D=64 H=4 C=10 params=124416 steps=200 lr=0.05
step   10 | loss 2.2978 | acc 0.188
step   20 | loss 2.1304 | acc 0.250
...
step  200 | loss 0.7... | acc 0.8...

finished: 200 steps in 4.32s
throughput: 740 img/s global | 370 img/s/GPU
```

## Code map

The source file is six labelled sections.

1. **Helpers** — `CHECK` macro, warp reductions, `Cfg` struct.
2. **Tensor catalog** — `P_*` / `A_*` enums (one slot per tensor), `fill_*_sizes`
   functions, `assign_pointers`.
3. **Kernels** — every elementary operation:
   - `encoder_forward / encoder_backward`
   - `layernorm_forward / layernorm_backward`
   - `bias_add / bias_grad`
   - `attention_qk / attention_softmax / attention_av`
   - `attention_dv / attention_d_attn / attention_d_softmax / attention_dq / attention_dk`
   - `gelu_forward / gelu_backward`
   - `residual_add`
   - `mean_pool_forward / mean_pool_backward`
   - `softmax_ce_forward / softmax_ce_backward`
   - `sgd_step`
   - `atomic_sum`, `count_correct` (reporting only)
4. **cuBLAS wrappers** — `matmul_forward` and `matmul_backward`.
5. **Model** — `model_forward` and `model_backward` chain the kernels.
6. **Main** — MPI/NCCL bootstrap, data loading, training loop, throughput.

## Memory layout

Parameters, gradients, momentum, activations and activation-gradients each
live in a single flat `cudaMalloc` of size `sum(sizes)`. `params[i]`,
`grads[i]`, `acts[i]`, `dacts[i]` are pointers into those buffers, set up
once by `assign_pointers`. Concretely:

- `ncclAllReduce(d_grads, d_grads, total_params, ...)` reduces every gradient
  in one call.
- `sgd_step<<<GRID(total_params), ...>>>` runs the optimiser over every
  parameter in one kernel launch.

## Data parallel

All ranks initialise weights from the same RNG seed → identical parameters,
no `MPI_Bcast` needed. Each rank samples its own random batches. After
backward, `ncclAllReduce` sums gradients across ranks; the SGD step divides
by `B * world` so the effective update is the mean gradient over the global
batch.

To measure scaling, run with `-np 1`, `-np 2`, `-np 4` and compare the
final throughput line.
