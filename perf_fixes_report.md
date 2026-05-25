# Отчёт по оптимизациям CUDA-версии train_vit

Дата: 2026-05-25
Файл: [src/train_vit.cu](src/train_vit.cu)
Сборка локально не проверена (нет nvcc на macOS). **Собирать и тестировать на
Kaggle через [kaggle_run.ipynb](kaggle_run.ipynb).**

## Runtime feature flags (новое)

Каждое из 4 нововведений теперь **отключаемо отдельной env-переменной** без
пересборки. По умолчанию все = 1 (новый/быстрый путь). Старый код сохранён
рядом и активируется флагом = 0.

| Env var | Что переключает | Default |
|---|---|---|
| `PERF_LN_BWD_FAST=0/1` | layernorm_backward: register-accumulators (1) vs warp-per-row + atomicAdd-per-row (0) | 1 |
| `PERF_SGEMV_BIAS=0/1` | bias_grad: cublasSgemv (1) vs serial 1-thread-per-column kernel (0) | 1 |
| `PERF_ATTN_BATCHED=0/1` | attention matmuls: cublasSgemmBatched (1) vs per-b cublasSgemmStridedBatched loop (0) | 1 |
| `PERF_NARROW_MEMSET=0/1` | per-step memset: ~1 MB targeted (1) vs ~350 MB full d_dacts (0) | 1 |

Запуск с конкретным фиксом отключённым:

```bash
# отключить только LN bwd fast, остальные включены
PERF_LN_BWD_FAST=0 ./bin/train_vit data/train.csv 200 8 0.001

# отключить всё (полный legacy)
PERF_LN_BWD_FAST=0 PERF_SGEMV_BIAS=0 PERF_ATTN_BATCHED=0 PERF_NARROW_MEMSET=0 \
  ./bin/train_vit data/train.csv 200 8 0.001
```

В stdout первой строкой печатается текущая конфигурация флагов, например:

```
perf flags: LN_BWD_FAST=1 SGEMV_BIAS=1 ATTN_BATCHED=1 NARROW_MEMSET=1
```

## Тайминги для измерения каждого фикса

В `training_log.csv` (и в stdout) есть колонки для оценки эффекта каждого фикса
по отдельности:

| Фикс | Колонки CSV для оценки | Прямота |
|---|---|---|
| **#1 LN bwd** | `tb_l0_ln1, tb_l0_ln2, tb_l1_ln1, tb_l1_ln2, tb_lnf` | прямо |
| **#2 bias_grad** | `tb_l0_qkv, tb_l0_aproj, tb_l0_fc1, tb_l0_fc2` (и l1) | косвенно — bias_grad внутри matmul_backward, его время растворено в общем тайминге matmul-bwd |
| **#3 attn batched** | `tf_l0_attn, tf_l1_attn, tb_l0_attn, tb_l1_attn` | прямо |
| **#4 narrow memset** | `t_zero_ms` (новая колонка, отдельный coarse-таймер) | прямо |

**Новое:** добавлен event `ev_zero` между memset и forward; теперь:

- `t_h2d_ms` — копирование пикселей host→device
- `t_zero_ms` — **только memset** (изолирован от forward)
- `t_fwd_ms` — forward kernels (без memset)
- `t_bwd_ms`, `t_nccl_ms`, `t_adam_ms` — без изменений

**Внимание при сравнении со старыми логами:** в старых логах `tf_enc` (первая
колонка тонкого fwd-тайминга) включал memset; теперь он чистый encoder fwd.
Аналогично `t_fwd_ms` стал меньше за счёт выноса memset.

### Что НЕ покрыто инструментально

`bias_grad` нельзя замерить **изолированно** — он сидит внутри `matmul_backward`
между двумя GEMM-вызовами. Чтобы получить чистый таймер, нужно было бы
добавить event внутри `matmul_backward` (3 события на вызов вместо 1). Сейчас
оценка идёт через **A/B сравнение `tb_l*_qkv/aproj/fc1/fc2`** при
`PERF_SGEMV_BIAS=1` vs `=0`. Самый показательный — `tb_l*_fc1` (там K=4D=256,
самый «толстый» bias).

## Методика бенча: A/B по флагам

Чтобы понять вклад каждого фикса:

```bash
# baseline (всё старое)
PERF_LN_BWD_FAST=0 PERF_SGEMV_BIAS=0 PERF_ATTN_BATCHED=0 PERF_NARROW_MEMSET=0 \
  ./bin/train_vit data/train.csv 50 8 0.001
mv training_log.csv training_log_baseline.csv

# +#1 LN bwd
PERF_LN_BWD_FAST=1 PERF_SGEMV_BIAS=0 PERF_ATTN_BATCHED=0 PERF_NARROW_MEMSET=0 \
  ./bin/train_vit data/train.csv 50 8 0.001
mv training_log.csv training_log_ln.csv

# +#1 +#2
PERF_LN_BWD_FAST=1 PERF_SGEMV_BIAS=1 PERF_ATTN_BATCHED=0 PERF_NARROW_MEMSET=0 \
  ./bin/train_vit data/train.csv 50 8 0.001
mv training_log.csv training_log_ln_bias.csv

# и т.д.
```

Сравнивать колонки попарно — увидишь дельту по каждому фиксу отдельно.

## Что было сделано

Четыре связанных фикса. Каждый адресует одну из проблем, описанных в
[perf_notes.md](perf_notes.md). Фокус — backward, потому что по старым
запускам именно там разрыв с PyTorch.

### 1. `layernorm_backward` — register-accumulators, 32 строки на блок

**Файл:** [train_vit.cu:287-350](src/train_vit.cu#L287-L350)

**Что было.** Один блок (1 warp, 32 потока) обрабатывал одну строку токена.
В конце блок делал `atomicAdd` в `dgamma[d]` и `dbeta[d]` (D=64 ячеек).
При `BT=B*T=6272` строках на каждую ячейку приходило 6272 атомиков.
Атомики на одну ячейку **сериализуются** на уровне L2 → длинная очередь.

**Что стало.** Каждый блок (по-прежнему 32 потока) теперь обрабатывает
`LN_ROWS_PER_BLOCK = 32` строк подряд. Внутри блока вклад в `dgamma/dbeta`
копится в **регистровых аккумуляторах** (`dg_acc[8]`, `db_acc[8]` на поток —
D до 256). В конце блок делает **ровно одну** пачку `atomicAdd` на (block, d).

**Эффект.** Число атомиков на ячейку: `6272 → 196` (×32 меньше). Параллелизма
по блокам всё ещё много (`196 ≥ 40 SM × несколько warp'ов`). Семантически
кернел идентичен старому, регистровая ёмкость 8 слотов покрывает D≤256.

**Изменены 3 launch-сайта:**
- LNF backward — [train_vit.cu:709](src/train_vit.cu#L709)
- LN2 backward — [train_vit.cu:747](src/train_vit.cu#L747)
- LN1 backward — [train_vit.cu:821](src/train_vit.cu#L821)

Везде используется макрос `LN_BWD_GRID(N)`.

### 2. `bias_grad` → `cublasSgemv`

**Файл:** [train_vit.cu:545-549](src/train_vit.cu#L545-L549)

**Что было.** Кастомный кернел: K потоков всего, каждый поток последовательно
суммирует N=BT=6272 элементов с шагом K (плохой strided pattern). На K=64
это **2 warp'а** — занят <1% GPU.

**Что стало.** Один вызов `cublasSgemv`:

```c
cublasSgemv(cublas, CUBLAS_OP_N, OC, N,
            &alpha, dy, OC, g_ones, 1,
            &beta,  db, 1);
```

`g_ones` — заранее заполненный вектор единиц длины BT, лежит в `static
float *g_ones` (выделен в main, см. [train_vit.cu:964](src/train_vit.cu#L964)).
`dy` row-major `[N, OC]` cuBLAS видит как column-major `[OC, N]` с lda=OC,
поэтому `db = dy_cublas @ ones = Σ_n dy[n, :]`.

**Эффект.** За 8 вызовов на шаг (qkv, attproj, fc1, fc2 × L=2) экономится
~0.4-1.2 мс по сравнению со старым кернелом. cuBLAS GEMV — давно оптимизирован
под coalesced reads и нормальную occupancy.

**Старый кернел** `bias_grad` удалён ([train_vit.cu:357-358](src/train_vit.cu#L357)).

### 3. Attention через `cublasSgemmBatched` (forward и backward)

**Файлы:**
- forward: [train_vit.cu:604-627](src/train_vit.cu#L604-L627)
- backward: [train_vit.cu:768-810](src/train_vit.cu#L768-L810)

**Что было.** В forward были 2 `for (b = 0; b < B; b++) cublasSgemmStridedBatched(batchCount=H)`
(QK^T и Attn@V). В backward — 4 таких цикла (dV, dAttn, dQ, dK). При B=8, L=2:
- forward: 2·B·L = **32** cuBLAS launch'ей
- backward: 4·B·L = **64** launch'ей

Каждый — ~5-15 µs CPU overhead'а.

**Почему было нужно так:** Q/K/V interleaved в одной 3D-оси, поэтому страйды
между b-элементами `(T*3*D)` и между головами `(HD)` не сводятся в один,
`cublasSgemmStridedBatched` требует **одинакового** stride между всеми
батч-элементами.

**Что стало.** Использован `cublasSgemmBatched` с **массивами указателей**
(а не stride'ами). Каждый указатель может вести в произвольное место. Указатели
живут в device-памяти `g_attn_ptrs` и считаются **один раз** в main()
после `assign_pointers` ([train_vit.cu:971-1018](src/train_vit.cu#L971-L1018)).

Layout: 12 секций по `L*B*H` указателей, по одной на роль:

```c
enum { PT_Q, PT_K, PT_V, PT_S, PT_A, PT_O,
       PT_dQ, PT_dK, PT_dV, PT_dS, PT_dA, PT_dO, PT_COUNT };
```

Indexing: `g_attn_ptrs[PT_X * LBH + (l*B + b)*H + h]`.

**Эффект.** Forward attention: **32 → 2** launch'а. Backward attention: **64 → 4**.
Экономия CPU-overhead'а на attention: `(32-2)*10µs + (64-4)*10µs ≈ 900 µs/шаг`.
Плюс лучше fill стрима (меньше "пузырей" между мелкими kernels).

### 4. Memset только нужного — d_grads + один RESID2-срез

**Файл:** [train_vit.cu:1104-1124](src/train_vit.cu#L1104-L1124)

**Что было.** На каждом шаге `cudaMemsetAsync(d_dacts, 0, ~350 MB)` — зануление
**всего** буфера активационных градиентов. На T4 (~250 GB/s HBM) — это
~1.4 мс чистого простоя + полное вымывание L2 cache (а L2 на T4 всего 4 MB).

**Что стало.** Зануляем только то, во что идёт `atomicAdd` / `+=`:
- `d_grads` целиком (~670 KB) — туда атомарно пишут `encoder_backward`
  (dwte, dwpe) и `layernorm_backward` (dgamma, dbeta).
- `dacts[A_RESID2] + (L-1)*BTD` (~400 KB) — единственный dacts-срез, который
  читается до того, как `layernorm_backward` накопит в него `+= dxd`. Все
  остальные dacts либо пишутся с `=` (cublas beta=0), либо префиллятся
  D2D memcpy'ями ([train_vit.cu:715-716, 746-748](src/train_vit.cu#L715-L716)).

**Эффект.** Зануление ~350 MB → ~1 MB на шаге. Экономия ~1-2 мс и, что
важнее, **L2 кэш больше не вымывается** между шагами — последующие
ядра (особенно attention с её 314 МБ score/attn-матриц) выигрывают.

## Связанные изменения

- Глобальные file-scope указатели `g_ones`, `g_attn_ptrs` и enum `PT_*`
  объявлены в [train_vit.cu:67-82](src/train_vit.cu#L67-L82).
- `cudaFree` в эпилоге main для обоих буферов — [train_vit.cu:1276](src/train_vit.cu#L1276).
- Старый кернел `bias_grad` физически удалён (заменён комментарием).

## Что НЕ делал (и почему)

1. **`attention_softmax` / `attention_d_softmax` с 128+ потоками/блок.** При
   T=784 один warp/блок даёт ~25 итераций — не идеально, но при 50k блоков
   GPU и так насыщена параллелизмом, прирост ожидаемый ~0.3-0.6 мс. После
   #1-#4 это будет в пропорции маленьким куском backward'а. Оставил на потом.

2. **`encoder_backward` атомики в `dwte[0]`.** На MNIST ~80% пикселей — 0,
   поэтому `dwte[0, :]` получает массу атомиков. Но D=64 даёт параллелизм
   по последней оси, и реальная стоимость оценивается в ~100-300 µs/шаг —
   не доминирующий вклад. Фикс потребовал бы либо shared-memory pre-aggregation,
   либо sort-and-reduce. Дорого относительно эффекта.

3. **TF32 / `CUBLAS_TENSOR_OP_MATH`.** На T4 (Turing) нет TF32-юнитов —
   игнорируется.

## Ожидаемый суммарный эффект

| Фикс | Экономия (оценка) |
|---|---|
| #1 layernorm_backward atomics | 0.8–2 мс/шаг |
| #2 bias_grad → cublasSgemv | 0.4–1.2 мс/шаг |
| #3 attention batched (fwd+bwd) | ~0.9 мс/шаг |
| #4 memset d_dacts | 1–2 мс/шаг + меньше L2-trash |
| **Итого** | **~3-6 мс/шаг** |

Это должно **закрыть разрыв с PyTorch и, скорее всего, обогнать его** на
T4 при B=8-32 (PyTorch с `SDPBackend.MATH` без flash-attention).

## Что проверить после запуска

1. **Сборка через nvcc** — на Kaggle через [kaggle_run.ipynb](kaggle_run.ipynb)
   (cell 2-5: установка OpenMPI, поиск NCCL, компиляция).
2. **Численная эквивалентность** — loss-кривая должна быть идентичной старой
   (с точностью до floating-point noise атомиков). Прогнать 200 шагов и
   сравнить с предыдущим `training_log.csv` (loss column).
3. **Профилирование по фазам.** В `training_log.csv` смотреть:
   - `tb_l*_ln1`, `tb_l*_ln2`, `tb_lnf` — должны просесть в ~3-5×.
   - `tb_l*_qkv`, `tb_l*_aproj`, `tb_l*_fc1`, `tb_l*_fc2` — должны просесть
     за счёт bias_grad замены.
   - `tb_l*_attn` и `tf_l*_attn` — просесть за счёт меньшего числа launch'ей.
   - `t_fwd_ms` — заметно просесть за счёт memset-фикса.
4. **Бенч против PyTorch** — запустить [kaggle_torch_run.ipynb](kaggle_torch_run.ipynb)
   с теми же B/steps, сравнить throughput.

## Если что-то сломалось

Самые рисковые места:

- **Pointer-array layout** в attention. Если cublas ругается на размерности —
  проверить `lda/ldb/ldc` и тип op (CUBLAS_OP_T/N). Backward особенно
  чувствителен — там 4 разных вызова. Перекрёстная проверка с оригиналом
  возможна по конкретным аргументам в [train_vit.cu:778-809](src/train_vit.cu#L778-L809).
- **`cublasSgemv` для bias_grad.** Если db получается нулевым или с NaN —
  проверить, что `g_ones` действительно заполнен единицами (проверка:
  `cudaMemcpy` обратно на хост одной float'ы из g_ones).
- **`layernorm_backward` register accumulator overflow.** Если D > 256 —
  не сработает, нужно увеличить `LN_ACC_MAX`. Для текущего проекта D=64,
  слотов нужно 2 из 8.
- **Memset-fix.** Если loss сразу взрывается в NaN — проверить, что нет
  забытого dacts-среза, который требовал зануления. Стартовый сейв-шаг:
  закомментировать новый узкий memset, добавить обратно полный
  `cudaMemsetAsync(d_dacts, 0, total_acts*sizeof(float), stream)`, и
  если NaN уходит — значит, не угадан срез.

## Что осталось на следующий раунд (см. perf_notes.md)

- softmax kernels с увеличенным block size
- encoder_backward atomics для MNIST sparsity
- возможный переход на flash-attention для T=784 (большой рефактор)
