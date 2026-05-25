# Почему CUDA-версия медленнее PyTorch — анализ и план

## 1. `cudaMemsetAsync(d_dacts, ...)` каждый шаг — сотни мегабайт

[train_vit.cu:1019](src/train_vit.cu#L1019) зануляет весь буфер активационных
градиентов каждый шаг. При `B=8, T=784, L=2`:

- `A_ATTN_PRE = L*B*H*T*T = 2*8*4*784*784 ≈ 39.3M floats = 157 MB`
- `A_ATTN` = столько же = 157 MB
- плюс всё остальное

Итого **~330–350 MB зануляется на каждом шаге**. На T4 HBM ~250 GB/s это
**~1.4 ms чистого простоя** + L2 кэш постоянно вымывается, что бьёт по всем
последующим kernel'ам. PyTorch ничего такого не делает — он переиспользует
тензоры или зануляет их через `.grad = None`.

Реально нужно занулить **только** то, во что идёт `atomicAdd` / `+=`:

- `d_grads` целиком (~167k float = **670 KB**) — туда атомарно пишут
  `encoder_backward` (dwte, dwpe) и `layernorm_backward` (dgamma, dbeta).
- `dacts[A_RESID2] + (L-1)*BTD` (вход LNF, ~400 KB) — потому что
  `layernorm_backward` делает `dx += dxd`, а этот буфер не префиллится
  memcpy'ем.

Все остальные `dacts` либо пишутся с `=` через `matmul_backward`
(`beta=0` в `cublasSgemm`), либо префиллятся `cudaMemcpyAsync`
([train_vit.cu:680, 711](src/train_vit.cu#L680)) до того, как LayerNorm
накапливает.

**Замена:**

```c
CHECK(cudaMemsetAsync(d_grads, 0, total_params * sizeof(float), stream));
CHECK(cudaMemsetAsync(dacts[A_RESID2] + (size_t)(L-1)*B*T*D, 0,
                      (size_t)B*T*D*sizeof(float), stream));
```

Сэкономит **1–2 ms на шаге** (на T4) плюс перестанет уничтожать L2.

## 2. Attention делает `B` отдельных cuBLAS-вызовов вместо одного

[train_vit.cu:562-582](src/train_vit.cu#L562-L582) и
[train_vit.cu:734-773](src/train_vit.cu#L734-L773) — везде
`for (int b = 0; b < B; b++) cublasSgemmStridedBatched(...)`. Каждый
`cublasSgemm*` стоит ~5–15 µs оверхеда на launch.

Launch'ей на шаг для attention:

- fwd: 2 батч-цикла (QK^T, AV) × B × L = 2·8·2 = **32**
- bwd: 4 батч-цикла (dV, dAttn, dQ, dK) × B × L = 4·8·2 = **64**
- → **96 cuBLAS launch'ей** только для attention при B=8

При 10 µs/launch это **~1 ms на шаге** чистого CPU-overhead'а и пузырей
в стриме. PyTorch'овский `MultiheadAttention` делает по сути
**2 батч-матмуля** на attention.

**Почему стрид сейчас нельзя сложить в один `StridedBatched`?** Q/K/V
лежат в одной [3D] оси interleaved: `qkv[b, t, qkv_idx*D + h*HD + d]`.
Стрид между головами = `HD`, между батч-элементами = `T*3*D`, и они не
кратны — `cublasSgemmStridedBatched` требует *одинакового* шага между
всеми элементами батча.

Два варианта:

- **(a)** Использовать `cublasSgemmBatched` (массив указателей, не stride).
  Поддерживает произвольный layout. Заводим один раз
  `float* d_qkv_ptrs[B*H]` через `cudaMalloc` + заполняем хост-массивом,
  копируем `H2D` единожды при инициализации (указатели не зависят от
  данных, только от base-pointer'а). batchCount = B·H. **6 cuBLAS
  launch'ей** вместо 96.

- **(b)** Переложить QKV из `[B,T,3,H,HD]` в `[3,B,H,T,HD]` (отдельные
  плотные блоки для Q, K, V). Тогда стрид одинаков и хватит обычного
  `cublasSgemmStridedBatched` с batchCount = B·H. Чище для cuBLAS, но
  требует отдельного permute-kernel'а после QKV-проекции (и обратного
  в backward'е). Скорее всего, permute дешевле, чем 90 лишних launch'ей.

Я бы шёл по **(а)** — меньше переписывания, тот же memory layout.

## 3. Softmax/LayerNorm — один warp на строку

[train_vit.cu:323-342](src/train_vit.cu#L323-L342):
`attention_softmax<<<B*H*T, 32, ...>>>` — **32 потока на строку из T=784
элементов**. Каждый поток делает 25 итераций цикла. PyTorch'овский
softmax обычно использует 128–256 потоков на строку с block-level
reduction через shared memory — для T=784 это в 4× меньше итераций на
поток.

При B=8, H=4, L=2: 8·4·784·2 = ~50k строк softmax'а fwd + столько же bwd.
Каждая ~32 итерации load/exp/store. Это **существенный кусок** при T=784
(хоть и не доминирующий).

Простой апгрейд:

```c
__global__ void attention_softmax(float *attn, const float *scores, int B, int H, int T) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int nthr = blockDim.x;  // например 128
    extern __shared__ float smem[];
    // load → reduce max в smem → reduce sum в smem → store
    ...
}
// launch: attention_softmax<<<B*H*T, 128, 2*sizeof(float), stream>>>(...)
```

То же для `attention_d_softmax` и `softmax_ce_forward`. Для
`layernorm_forward` (D=64) — там 32 потока это уже норм, ничего не меняй.

## Прочее, помельче

- **`bias_grad` ([train_vit.cu:314-321](src/train_vit.cu#L314-L321)) —
  последовательная редукция:** один поток суммирует `N=B*T=6272`
  элементов. Для `K=D=64` это всего 64 потока, GPU простаивает. Замени
  на «один блок на k, parallel reduction внутри блока» (или
  `cub::BlockReduce`). Выигрыш ~0.1–0.3 ms.

- **Per-kernel `cudaEventRecord` (41 события/шаг)** — оверхед мал
  (<50 µs/шаг), но если хочешь чистый бенч против PyTorch, добавь
  флаг `--no-fine-events` и не вызывай их в основном замере.

- **cuBLAS на T4** — `CUBLAS_TENSOR_OP_MATH` ничего не даст (нет TF32),
  но `cublasSetWorkspace` с правильным размером (или
  `CUBLAS_PEDANTIC_MATH`-отключение) иногда выбирает чуть лучший
  алгоритм. Маргинально.

## Приоритеты

| Фикс                                          | Ожидаемый выигрыш           | Сложность |
|-----------------------------------------------|-----------------------------|-----------|
| Перестать memset'ить весь d_dacts             | **1–2 ms/step**, меньше L2  | 10 мин    |
| Батчить attention через `cublasSgemmBatched`  | **~1 ms/step**              | 1–2 часа  |
| Softmax с 128–256 thr/block                   | **~0.3–0.6 ms/step**        | 30 мин    |
| Параллельный `bias_grad`                      | ~0.1–0.3 ms/step            | 20 мин    |

Начни с **#1** — почти наверняка он один закроет основную часть разрыва.
И запусти после этого `kaggle_run.ipynb` с тем же конфигом, что и
торч-ноутбук, и сравни `training_log.csv`-колонки по фазам — увидишь,
куда ещё уходит время.
