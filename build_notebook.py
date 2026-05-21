#!/usr/bin/env python3
"""Builds kaggle_run.ipynb from train_vit.cu — embeds the full source."""
import json, pathlib

ROOT = pathlib.Path(__file__).parent
SRC  = (ROOT / "src" / "train_vit.cu").read_text()

# ── profiling cell sources ─────────────────────────────────────────────────────
PROF_SWEEP = """\
import subprocess, threading, time, os
import numpy as np

try:
    import psutil
except ImportError:
    subprocess.run(['pip', 'install', '-q', 'psutil'], check=True)
    import psutil

def _gpu_monitor(stop_evt, records, interval=0.25):
    while not stop_evt.is_set():
        r = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True)
        ts = time.time()
        for line in r.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(',')]
            if len(parts) < 5:
                continue
            try:
                records.append(dict(
                    ts=ts, gpu=int(parts[0]),
                    gpu_util=float(parts[1]),
                    mem_mb=float(parts[2]),
                    mem_total=float(parts[3]),
                    power=float(parts[4]) if 'N/A' not in parts[4] else 0.0,
                ))
            except ValueError:
                pass
        time.sleep(interval)

def run_prof(csv_path, B, steps=50, lr=0.001, np_=1):
    prefix = f'mpirun --allow-run-as-root -np {np_} ' if np_ > 1 else ''
    cmd = f'{prefix}./bin/train_vit {csv_path} {steps} {B} {lr}'
    stop = threading.Event()
    recs, cpu_s = [], []

    def _cpu():
        while not stop.is_set():
            cpu_s.append((time.time(), psutil.cpu_percent()))
            time.sleep(0.25)

    gmon = threading.Thread(target=_gpu_monitor, args=(stop, recs), daemon=True)
    cmon = threading.Thread(target=_cpu, daemon=True)
    gmon.start()
    cmon.start()

    t0 = time.time()
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - t0

    stop.set()
    time.sleep(0.4)

    tput = None
    for line in proc.stdout.splitlines():
        if 'img/s' in line and 'throughput' in line.lower():
            for tok in line.replace('|', ' ').split():
                try:
                    tput = float(tok)
                    break
                except ValueError:
                    pass
            break

    return dict(recs=recs, cpu=cpu_s, elapsed=elapsed,
                tput=tput, stdout=proc.stdout)

CSV = os.environ.get('CSV', '')
assert CSV, 'Run the data cell first (cell 6)'

n_gpus = int(subprocess.check_output(
    'nvidia-smi --query-gpu=name --format=csv,noheader | wc -l',
    shell=True, text=True).strip())
print(f'GPUs detected: {n_gpus}')

prof = {}
BATCH_SIZES = [8, 16, 32, 64]

for B in BATCH_SIZES:
    print(f'  1-GPU  B={B:3d} / 50 steps ...', end=' ', flush=True)
    prof[('1gpu', B)] = run_prof(CSV, B, steps=50, np_=1)
    t = prof[('1gpu', B)]['tput']
    print(f"done  {t} img/s  elapsed={prof[('1gpu', B)]['elapsed']:.1f}s")

if n_gpus >= 2:
    print('  2-GPU  B= 32 / 50 steps (NCCL) ...', end=' ', flush=True)
    prof[('2gpu', 32)] = run_prof(CSV, 32, steps=50, np_=2)
    t = prof[('2gpu', 32)]['tput']
    print(f"done  {t} img/s  elapsed={prof[('2gpu', 32)]['elapsed']:.1f}s")
else:
    print('Single-GPU session - 2-GPU profiling skipped.')

print('Profiling sweep complete.')
"""

PROF_PLOT = """\
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZES = [8, 16, 32, 64]
cmap = plt.cm.tab10(np.linspace(0, 0.8, len(BATCH_SIZES)))

def gpu_ts(recs, gpu_idx=0):
    g = [r for r in recs if r['gpu'] == gpu_idx]
    if not g:
        return [], [], [], []
    t0 = g[0]['ts']
    return (
        [r['ts'] - t0 for r in g],
        [r['gpu_util'] for r in g],
        [r['mem_mb'] / 1024 for r in g],
        [r['power'] for r in g],
    )

# Figure 1: time-series per batch size
fig1, axes = plt.subplots(4, len(BATCH_SIZES), figsize=(5 * len(BATCH_SIZES), 12))
fig1.suptitle('GPU & CPU time-series — 50 steps, 1 GPU', fontsize=13, fontweight='bold')
row_labels = ['GPU util (%)', 'GPU mem (GiB)', 'GPU power (W)', 'CPU util (%)']

for idx, (B, c) in enumerate(zip(BATCH_SIZES, cmap)):
    key = ('1gpu', B)
    if key not in prof:
        for row in range(4):
            axes[row][idx].set_visible(False)
        continue
    exp = prof[key]
    ts, util, mem, pwr = gpu_ts(exp['recs'], 0)

    for row, (data, ylabel) in enumerate([
        (util, 'GPU util %'),
        (mem,  'GPU mem GiB'),
        (pwr,  'GPU power W'),
    ]):
        ax = axes[row][idx]
        ax.plot(ts, data, color=c, linewidth=1.3)
        ax.fill_between(ts, data, alpha=0.12, color=c)
        ax.set_title(f'B={B}', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

    ax_cpu = axes[3][idx]
    if exp['cpu']:
        ct0 = exp['cpu'][0][0]
        ax_cpu.plot([x[0] - ct0 for x in exp['cpu']], [x[1] for x in exp['cpu']],
                    color=c, linewidth=1.3)
    ax_cpu.set_title(f'B={B}', fontsize=10)
    ax_cpu.set_xlabel('Time (s)', fontsize=8)
    ax_cpu.set_ylabel('CPU util %', fontsize=8)
    ax_cpu.grid(alpha=0.3)
    ax_cpu.set_ylim(0, 105)

for row, label in enumerate(row_labels):
    axes[row][0].set_ylabel(label, fontsize=9)

plt.tight_layout()
plt.savefig('profiling_timeseries.png', dpi=130, bbox_inches='tight')
plt.show()
print('Saved: profiling_timeseries.png')

# Figure 2: summary bars vs batch size
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
fig2.suptitle('Summary metrics vs batch size — 50 steps, 1 GPU', fontsize=13, fontweight='bold')

tputs    = [prof.get(('1gpu', B), {}).get('tput') or 0 for B in BATCH_SIZES]
peak_mem = [max((r['mem_mb'] for r in prof.get(('1gpu', B), {}).get('recs', [])
                 if r['gpu'] == 0), default=0) / 1024 for B in BATCH_SIZES]
avg_util, avg_cpu = [], []
for B in BATCH_SIZES:
    rs = [r['gpu_util'] for r in prof.get(('1gpu', B), {}).get('recs', []) if r['gpu'] == 0]
    avg_util.append(np.mean(rs) if rs else 0)
    cv = [v for _, v in prof.get(('1gpu', B), {}).get('cpu', [])]
    avg_cpu.append(np.mean(cv) if cv else 0)

labels = [f'B={B}' for B in BATCH_SIZES]
for ax, vals, ylabel, title in [
    (axes2[0, 0], tputs,    'img/s', 'Throughput (img/s)'),
    (axes2[0, 1], peak_mem, 'GiB',   'Peak GPU Memory (GiB)'),
    (axes2[1, 0], avg_util, '%',     'Avg GPU Utilization (%)'),
    (axes2[1, 1], avg_cpu,  '%',     'Avg CPU Utilization (%)'),
]:
    bars = ax.bar(labels, vals, color=cmap, edgecolor='white', linewidth=0.5)
    ax.bar_label(bars, fmt='%.1f', fontsize=9, padding=2)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig('profiling_summary.png', dpi=130, bbox_inches='tight')
plt.show()
print('Saved: profiling_summary.png')

# Figure 3: NCCL overhead — 1-GPU vs 2-GPU at B=32
if ('2gpu', 32) in prof:
    fig3, (ax_u, ax_m) = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle('NCCL overhead: 1-GPU vs 2-GPU at B=32 (50 steps)',
                  fontsize=13, fontweight='bold')

    ts1,  u1,  m1,  _ = gpu_ts(prof[('1gpu', 32)]['recs'], 0)
    ts2a, u2a, m2a, _ = gpu_ts(prof[('2gpu', 32)]['recs'], 0)
    ts2b, u2b, m2b, _ = gpu_ts(prof[('2gpu', 32)]['recs'], 1)

    for ax, y1, y2a, y2b, ylabel in [
        (ax_u, u1, u2a, u2b, 'GPU util (%)'),
        (ax_m, m1, m2a, m2b, 'GPU mem (GiB)'),
    ]:
        ax.plot(ts1, y1, 'b-', linewidth=1.8, label='1-GPU  GPU0')
        ax.plot(ts2a, y2a, 'r-', linewidth=1.8, label='2-GPU  GPU0')
        ax.plot(ts2b, y2b, 'r--', linewidth=1.3, label='2-GPU  GPU1', alpha=0.8)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

    tput1 = prof[('1gpu', 32)]['tput'] or 0
    tput2 = prof[('2gpu', 32)]['tput'] or 0
    ratio = tput2 / tput1 if tput1 else 0
    ax_u.set_title(
        f'1GPU: {tput1:.0f} img/s  2GPU: {tput2:.0f} img/s  speedup: {ratio:.2f}x'
        ' (dips = NCCL allreduce idle)', fontsize=10)
    ax_m.set_title('GPU memory usage', fontsize=10)

    plt.tight_layout()
    plt.savefig('profiling_nccl.png', dpi=130, bbox_inches='tight')
    plt.show()
    print(f'NCCL: 1GPU={tput1:.0f} img/s  2GPU={tput2:.0f} img/s  speedup={ratio:.2f}x')
else:
    print('No 2-GPU profiling data. Run in a T4x2 session to see NCCL overhead.')

print('All profiling plots saved.')
"""


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}

cells = [
    md("""# MNIST Vision Transformer on 2× T4 (CUDA + MPI + NCCL + Adam)

This notebook compiles and runs the hand-written CUDA ViT from `train_vit.cu`
on a Kaggle session with **two Tesla T4 GPUs**.

Optimizer: **Adam** (β₁=0.9, β₂=0.999, ε=1e-8, lr=1e-3).
Each run saves a `training_log.csv` with columns `step, elapsed_s, loss, accuracy`.

**Setup checklist (do this in the Kaggle UI before running):**

1. **Accelerator → GPU T4 x2** (Settings panel on the right).
2. **Internet → On** (needed once, to `apt-get install` OpenMPI).
3. Optionally add the "Digit Recognizer" dataset; otherwise MNIST is downloaded via torchvision.

Run cells top-to-bottom.
For the **1-GPU session** stop after cell 9 and rename the log (cell 9).
For the **2-GPU session** run through cell 11, then plot with cell 12.
"""),

    md("## 1. Verify the environment\n\nExpect two T4 GPUs."),
    code("!nvidia-smi --query-gpu=index,name,memory.total --format=csv"),

    md("## 2. Install OpenMPI\n\n"
       "Kaggle ships `libnccl2`, `nvcc`, and `cuBLAS` already. "
       "Only `openmpi-bin`/`libopenmpi-dev` are missing.\n\n"
       "⚠️ **Do NOT install `libnccl-dev`** — it conflicts with the version "
       "bundled in Kaggle's PyTorch image."),
    code("import subprocess\n"
         "subprocess.run(['apt-get', '-qq', 'update'], check=True)\n"
         "subprocess.run(['apt-get', 'install', '-y', '-qq',\n"
         "                'openmpi-bin', 'libopenmpi-dev'], check=True)\n"
         "print('OK')\n"),

    md("## 3. Locate the existing NCCL header and library\n\n"
       "NCCL lives inside the PyTorch conda env on Kaggle. "
       "We find `nccl.h` and `libnccl.so` and store their dirs in env vars."),
    code("import os, subprocess\n"
         "\n"
         "hdr = subprocess.check_output(\n"
         "    'find /opt/conda /usr/include /usr/local -name nccl.h 2>/dev/null | head -n1',\n"
         "    shell=True, text=True).strip()\n"
         "lib = subprocess.check_output(\n"
         "    'find /opt/conda /usr/lib /usr/local -name \"libnccl.so*\" 2>/dev/null | head -n1',\n"
         "    shell=True, text=True).strip()\n"
         "\n"
         "assert hdr, 'nccl.h not found'\n"
         "assert lib, 'libnccl.so not found'\n"
         "\n"
         "os.environ['NCCL_INCLUDE'] = os.path.dirname(hdr)\n"
         "os.environ['NCCL_LIB']     = os.path.dirname(lib)\n"
         "print('nccl.h     :', hdr)\n"
         "print('libnccl.so :', lib)\n"),

    md("### Toolchain sanity check"),
    code("!which nvcc mpicxx mpirun\n"
         "!nvcc --version | tail -n 2\n"
         "!mpirun --version | head -n 1\n"
         "!echo \"NCCL include: $NCCL_INCLUDE\"\n"
         "!echo \"NCCL lib    : $NCCL_LIB\"\n"),

    md("## 4. Write the source file\n\n"
       "`%%writefile` makes the notebook self-contained — "
       "no need to upload `train_vit.cu` separately."),
    code("%%writefile train_vit.cu\n" + SRC),

    md("## 5. Compile\n\n"
       "- `-ccbin mpicxx` — MPI C++ wrapper as host compiler.\n"
       "- `-arch=sm_75` — Turing (T4).\n"
       "- `-I$NCCL_INCLUDE` / `-L$NCCL_LIB` / `-rpath` — link against the "
       "NCCL we located above."),
    code("!mkdir -p bin\n"
         "!nvcc -O2 -std=c++17 -ccbin mpicxx -arch=sm_75 \\\n"
         "      -I$NCCL_INCLUDE -L$NCCL_LIB \\\n"
         "      -Xlinker -rpath=$NCCL_LIB \\\n"
         "      train_vit.cu -o bin/train_vit \\\n"
         "      -lcublas -lnccl\n"
         "!ls -lh bin/train_vit\n"
         "!ldd bin/train_vit | grep -E 'nccl|cublas|mpi'\n"),

    md("## 6. Locate (or materialise) the data\n\n"
       "Search order: Kaggle digit-recognizer dataset → any `*train*.csv` "
       "with 785 columns → torchvision MNIST fallback."),
    code("import os, glob, numpy as np\n"
         "\n"
         "known = [\n"
         "    '/kaggle/input/digit-recognizer/train.csv',\n"
         "    '/kaggle/input/fashionmnist/fashion-mnist_train.csv',\n"
         "    '/kaggle/input/fashion-mnist/fashion-mnist_train.csv',\n"
         "]\n"
         "globbed = sorted(set(\n"
         "    glob.glob('/kaggle/input/**/*train*.csv', recursive=True) +\n"
         "    glob.glob('/kaggle/input/**/*Train*.csv', recursive=True)))\n"
         "\n"
         "def valid_csv(path):\n"
         "    try:\n"
         "        with open(path) as f:\n"
         "            return len(f.readline().split(',')) == 785\n"
         "    except Exception:\n"
         "        return False\n"
         "\n"
         "CSV = next((c for c in known + globbed if os.path.exists(c) and valid_csv(c)), None)\n"
         "\n"
         "if CSV is None:\n"
         "    print('Falling back to torchvision MNIST...')\n"
         "    from torchvision.datasets import MNIST\n"
         "    ds = MNIST(root='/kaggle/working/mnist_raw', train=True, download=True)\n"
         "    labels = ds.targets.numpy().astype(np.int32)\n"
         "    pixels = ds.data.numpy().reshape(-1, 784).astype(np.int32)\n"
         "    arr    = np.concatenate([labels[:, None], pixels], axis=1)\n"
         "    header = 'label,' + ','.join(f'pixel{i}' for i in range(784))\n"
         "    CSV    = '/kaggle/working/train.csv'\n"
         "    np.savetxt(CSV, arr, fmt='%d', delimiter=',', header=header, comments='')\n"
         "\n"
         "assert os.path.exists(CSV), CSV\n"
         "print('Using CSV:', CSV)\n"
         "print('Size     :', os.path.getsize(CSV) // (1024*1024), 'MiB')\n"
         "os.environ['CSV'] = CSV\n"
         "!head -c 120 \"$CSV\" ; echo\n"
         "!wc -l \"$CSV\"\n"),

    md("## 7. GPU / memory / NCCL profiling — batch-size sweep\n\n"
       "Runs **50 training steps** at batch sizes B ∈ {8, 16, 32, 64} on 1 GPU, "
       "plus one run with 2 GPUs at B=32 (if a second GPU is present).\n\n"
       "A background thread samples `nvidia-smi` and `psutil` every 250 ms, recording:\n\n"
       "- **GPU compute utilization** (%)\n"
       "- **GPU memory used** (MiB → GiB in plots)\n"
       "- **GPU power draw** (W)\n"
       "- **Host CPU utilization** (%)\n\n"
       "Training stdout is captured so the cell stays clean. "
       "Results are stored in the `prof` dict for the plotting cell below.\n\n"
       "⏱ Expected runtime: ~2–4 min total (50 steps × 4 batch sizes × ~10–30 s each)."),
    code(PROF_SWEEP),

    md("## 7b. Plot profiling results\n\n"
       "Three figures:\n\n"
       "1. **Time-series** — GPU util / mem / power / CPU vs wall-clock time for each B.\n"
       "2. **Summary bars** — throughput, peak memory, avg GPU util, avg CPU util vs B.\n"
       "3. **NCCL overhead** — 1-GPU vs 2-GPU GPU utilization at B=32. "
       "Visible dips in the 2-GPU curve are the allreduce idle windows "
       "(T4s are PCIe-connected, no NVLink)."),
    code(PROF_PLOT),

    md("## 8. Single-GPU run (Adam, 3000 steps)\n\n"
       "Optimizer: Adam lr=1e-3, B=32.  \n"
       "Saves log to `training_log.csv`.  \n"
       "After this cell finishes — run **cell 9** to rename the log, "
       "then start a **new session** for the 2-GPU run."),
    code("!./bin/train_vit $CSV 3000 32 0.001\n"),

    md("## 9. Save the 1-GPU log\n\n"
       "Rename so the 2-GPU run can write a fresh `training_log.csv`."),
    code("import shutil, os\n"
         "shutil.copy('training_log.csv', '1gpu_log.csv')\n"
         "print('Saved as 1gpu_log.csv  —  size:', os.path.getsize('1gpu_log.csv'), 'bytes')\n"),

    md("## 10. Two-GPU run (Adam, 3000 steps)\n\n"
       "`--allow-run-as-root` is required on Kaggle (kernel runs as root).  \n"
       "`-np 2` spawns one process per GPU via `cudaSetDevice(rank % devices)`.  \n"
       "Per-rank batch = 32 → global batch = 64. `adam_step` divides by `B * world` "
       "so the update is the mean gradient over the global batch."),
    code("!mpirun --allow-run-as-root -np 2 ./bin/train_vit $CSV 3000 32 0.001\n"),

    md("## 11. Save the 2-GPU log"),
    code("import shutil, os\n"
         "shutil.copy('training_log.csv', '2gpu_log.csv')\n"
         "print('Saved as 2gpu_log.csv  —  size:', os.path.getsize('2gpu_log.csv'), 'bytes')\n"),

    md("## 12. Loss vs time: 1 GPU vs 2 GPU\n\n"
       "X-axis is **wall-clock time** (seconds), not steps — "
       "this shows the actual speedup from the second GPU."),
    code("import os, pandas as pd, matplotlib.pyplot as plt, matplotlib.ticker as ticker\n"
         "\n"
         "logs = {'1 GPU': '1gpu_log.csv', '2 GPU': '2gpu_log.csv'}\n"
         "\n"
         "fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))\n"
         "\n"
         "for label, path in logs.items():\n"
         "    if not os.path.exists(path):\n"
         "        print(f'Файл не найден: {path}')\n"
         "        continue\n"
         "    df = pd.read_csv(path)\n"
         "    smooth = lambda s: s.rolling(window=10, min_periods=1).mean()\n"
         "    line, = ax_loss.plot(df['elapsed_s'], smooth(df['loss']),\n"
         "                        label=label, linewidth=2)\n"
         "    ax_loss.plot(df['elapsed_s'], df['loss'],\n"
         "                alpha=0.15, color=line.get_color())\n"
         "    line2, = ax_acc.plot(df['elapsed_s'], smooth(df['accuracy']),\n"
         "                        label=label, linewidth=2)\n"
         "    ax_acc.plot(df['elapsed_s'], df['accuracy'],\n"
         "               alpha=0.15, color=line2.get_color())\n"
         "\n"
         "ax_loss.set_xlabel('Время, секунды')\n"
         "ax_loss.set_ylabel('Loss (cross-entropy)')\n"
         "ax_loss.set_title('Loss vs время')\n"
         "ax_loss.legend(); ax_loss.grid(alpha=0.3)\n"
         "\n"
         "ax_acc.set_xlabel('Время, секунды')\n"
         "ax_acc.set_ylabel('Accuracy')\n"
         "ax_acc.set_title('Accuracy vs время')\n"
         "ax_acc.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))\n"
         "ax_acc.legend(); ax_acc.grid(alpha=0.3)\n"
         "\n"
         "plt.suptitle('1 GPU vs 2 GPU — Adam lr=1e-3, B=32, 3000 steps', fontsize=13)\n"
         "plt.tight_layout()\n"
         "plt.savefig('training_curves.png', dpi=150)\n"
         "plt.show()\n"
         "print('График сохранён: training_curves.png')\n"),

    md("## Notes\n\n"
       "- T4s on Kaggle are PCIe-connected (no NVLink) → NCCL uses PCIe/shared-mem. "
       "Speedup on this small model (~167k params) is sub-linear due to allreduce overhead.\n"
       "- Attention scores scale as `L · B · H · T²`: with `T=784, H=4, L=2, B=32` "
       "each of `A_ATTN_PRE` / `A_ATTN` ≈ 0.6 GB — stay well within 15 GB.\n"
       "- Adam adds one extra float buffer (`d_velocity`, same size as params) vs SGD.\n"
       "- The log file is flushed after every reporting step (every 10 steps) "
       "so you can inspect it while training is running."),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = ROOT / "kaggle_run.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({out.stat().st_size} bytes, {len(cells)} cells)")
