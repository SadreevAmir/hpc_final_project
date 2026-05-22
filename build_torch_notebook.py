#!/usr/bin/env python3
"""Build kaggle_torch_run.ipynb with the embedded PyTorch ViT source."""

import json
import pathlib

ROOT = pathlib.Path(__file__).parent
SRC = (ROOT / "src" / "train_vit_torch.py").read_text()


DATA_CELL = """\
import glob
import os
import numpy as np

known = [
    '/kaggle/input/digit-recognizer/train.csv',
    '/kaggle/input/fashionmnist/fashion-mnist_train.csv',
    '/kaggle/input/fashion-mnist/fashion-mnist_train.csv',
]
globbed = sorted(set(
    glob.glob('/kaggle/input/**/*train*.csv', recursive=True) +
    glob.glob('/kaggle/input/**/*Train*.csv', recursive=True)))

def valid_csv(path):
    try:
        with open(path) as f:
            return len(f.readline().split(',')) == 785
    except Exception:
        return False

CSV = next((c for c in known + globbed if os.path.exists(c) and valid_csv(c)), None)

if CSV is None:
    print('Falling back to torchvision MNIST...')
    from torchvision.datasets import MNIST
    ds = MNIST(root='/kaggle/working/mnist_raw', train=True, download=True)
    labels = ds.targets.numpy().astype(np.int32)
    pixels = ds.data.numpy().reshape(-1, 784).astype(np.int32)
    arr = np.concatenate([labels[:, None], pixels], axis=1)
    header = 'label,' + ','.join(f'pixel{i}' for i in range(784))
    CSV = '/kaggle/working/train.csv'
    np.savetxt(CSV, arr, fmt='%d', delimiter=',', header=header, comments='')

assert os.path.exists(CSV), CSV
print('Using CSV:', CSV)
print('Size     :', os.path.getsize(CSV) // (1024 * 1024), 'MiB')
os.environ['CSV'] = CSV
!head -c 120 "$CSV" ; echo
!wc -l "$CSV"
"""


PROF_SWEEP = """\
import os
import subprocess
import threading
import time
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
    if np_ == 1:
        cmd = ['python', 'train_vit_torch.py', csv_path, str(steps), str(B), str(lr),
               '--device', 'cuda', '--log-path', 'torch_profile_log.csv']
    else:
        cmd = ['torchrun', '--standalone', f'--nproc_per_node={np_}',
               '--master_port=29531', 'train_vit_torch.py',
               csv_path, str(steps), str(B), str(lr),
               '--device', 'cuda', '--log-path', 'torch_profile_log.csv']
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
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    stop.set()
    time.sleep(0.4)
    if proc.returncode:
        raise RuntimeError(
            f"profile run failed: {cmd}\\n"
            f"--- stdout ---\\n{proc.stdout}\\n"
            f"--- stderr ---\\n{proc.stderr}"
        )

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
assert CSV, 'Run the data cell first'
n_gpus = int(subprocess.check_output(
    'nvidia-smi --query-gpu=name --format=csv,noheader | wc -l',
    shell=True, text=True).strip())
print(f'GPUs detected: {n_gpus}')

prof = {}
BATCH_SIZES = [8, 16, 32, 64]
for B in BATCH_SIZES:
    print(f'  1-GPU  B={B:3d} / 50 steps ...', end=' ', flush=True)
    prof[('1gpu', B)] = run_prof(CSV, B, steps=50, np_=1)
    print(f"done  {prof[('1gpu', B)]['tput']} img/s  "
          f"elapsed={prof[('1gpu', B)]['elapsed']:.1f}s")

if n_gpus >= 2:
    print('  2-GPU  B= 32 / 50 steps (PyTorch DDP) ...',
          end=' ', flush=True)
    prof[('2gpu', 32)] = run_prof(CSV, 32, steps=50, np_=2)
    print(f"done  {prof[('2gpu', 32)]['tput']} img/s  "
          f"elapsed={prof[('2gpu', 32)]['elapsed']:.1f}s")
else:
    print('Single-GPU session - 2-GPU profiling skipped.')
"""


PROF_PLOT = """\
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZES = [8, 16, 32, 64]
cmap = plt.cm.tab10(np.linspace(0, 0.8, len(BATCH_SIZES)))

def gpu_ts(recs, gpu_idx=0):
    rows = [r for r in recs if r['gpu'] == gpu_idx]
    if not rows:
        return [], [], [], []
    t0 = rows[0]['ts']
    return (
        [r['ts'] - t0 for r in rows],
        [r['gpu_util'] for r in rows],
        [r['mem_mb'] / 1024 for r in rows],
        [r['power'] for r in rows],
    )

fig1, axes = plt.subplots(4, len(BATCH_SIZES), figsize=(5 * len(BATCH_SIZES), 12))
fig1.suptitle('PyTorch GPU and CPU time-series - 50 steps, 1 GPU',
              fontsize=13, fontweight='bold')

for idx, (B, color) in enumerate(zip(BATCH_SIZES, cmap)):
    exp = prof.get(('1gpu', B))
    if not exp:
        for row in range(4):
            axes[row][idx].set_visible(False)
        continue
    ts, util, mem, pwr = gpu_ts(exp['recs'], 0)
    for row, (values, label) in enumerate([
        (util, 'GPU util %'),
        (mem, 'GPU mem GiB'),
        (pwr, 'GPU power W'),
    ]):
        ax = axes[row][idx]
        ax.plot(ts, values, color=color, linewidth=1.3)
        ax.fill_between(ts, values, alpha=0.12, color=color)
        ax.set_title(f'B={B}', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

    ax_cpu = axes[3][idx]
    if exp['cpu']:
        ct0 = exp['cpu'][0][0]
        ax_cpu.plot([x[0] - ct0 for x in exp['cpu']],
                    [x[1] for x in exp['cpu']], color=color, linewidth=1.3)
    ax_cpu.set_title(f'B={B}', fontsize=10)
    ax_cpu.set_xlabel('Time (s)', fontsize=8)
    ax_cpu.set_ylabel('CPU util %', fontsize=8)
    ax_cpu.grid(alpha=0.3)
    ax_cpu.set_ylim(0, 105)

plt.tight_layout()
plt.savefig('torch_profiling_timeseries.png', dpi=130, bbox_inches='tight')
plt.show()

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
fig2.suptitle('PyTorch summary metrics vs batch size - 50 steps, 1 GPU',
              fontsize=13, fontweight='bold')

tputs = [prof.get(('1gpu', B), {}).get('tput') or 0 for B in BATCH_SIZES]
peak_mem = [max((r['mem_mb'] for r in prof.get(('1gpu', B), {}).get('recs', [])
                 if r['gpu'] == 0), default=0) / 1024 for B in BATCH_SIZES]
avg_util, avg_cpu = [], []
for B in BATCH_SIZES:
    recs = prof.get(('1gpu', B), {}).get('recs', [])
    cpu = prof.get(('1gpu', B), {}).get('cpu', [])
    gpu_values = [r['gpu_util'] for r in recs if r['gpu'] == 0]
    avg_util.append(np.mean(gpu_values) if gpu_values else 0)
    avg_cpu.append(np.mean([v for _, v in cpu]) if cpu else 0)

labels = [f'B={B}' for B in BATCH_SIZES]
for ax, values, ylabel, title in [
    (axes2[0, 0], tputs, 'img/s', 'Throughput'),
    (axes2[0, 1], peak_mem, 'GiB', 'Peak GPU memory'),
    (axes2[1, 0], avg_util, '%', 'Average GPU utilization'),
    (axes2[1, 1], avg_cpu, '%', 'Average CPU utilization'),
]:
    bars = ax.bar(labels, values, color=cmap, edgecolor='white', linewidth=0.5)
    ax.bar_label(bars, fmt='%.1f', fontsize=9, padding=2)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('torch_profiling_summary.png', dpi=130, bbox_inches='tight')
plt.show()

if ('2gpu', 32) in prof:
    fig3, (ax_u, ax_m) = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle('PyTorch 1 GPU vs 2 GPU at B=32', fontsize=13, fontweight='bold')
    ts1, u1, m1, _ = gpu_ts(prof[('1gpu', 32)]['recs'], 0)
    ts2a, u2a, m2a, _ = gpu_ts(prof[('2gpu', 32)]['recs'], 0)
    ts2b, u2b, m2b, _ = gpu_ts(prof[('2gpu', 32)]['recs'], 1)
    for ax, one, two_a, two_b, ylabel in [
        (ax_u, u1, u2a, u2b, 'GPU util (%)'),
        (ax_m, m1, m2a, m2b, 'GPU mem (GiB)'),
    ]:
        ax.plot(ts1, one, 'b-', linewidth=1.8, label='1 GPU, GPU0')
        ax.plot(ts2a, two_a, 'r-', linewidth=1.8, label='2 GPU, GPU0')
        ax.plot(ts2b, two_b, 'r--', linewidth=1.3, label='2 GPU, GPU1')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig('torch_profiling_allreduce.png', dpi=130, bbox_inches='tight')
    plt.show()
else:
    print('No 2-GPU profiling data to plot.')
"""


TRAINING_CURVES = """\
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

logs = {'PyTorch 1 GPU': 'torch_1gpu_log.csv',
        'PyTorch 2 GPU': 'torch_2gpu_log.csv'}
fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))

for label, path in logs.items():
    if not os.path.exists(path):
        print('Missing:', path)
        continue
    df = pd.read_csv(path)
    smooth = lambda s: s.rolling(window=10, min_periods=1).mean()
    line, = ax_loss.plot(df['elapsed_s'], smooth(df['loss']),
                         label=label, linewidth=2)
    ax_loss.plot(df['elapsed_s'], df['loss'], alpha=0.15, color=line.get_color())
    line, = ax_acc.plot(df['elapsed_s'], smooth(df['accuracy']),
                        label=label, linewidth=2)
    ax_acc.plot(df['elapsed_s'], df['accuracy'], alpha=0.15, color=line.get_color())

ax_loss.set_xlabel('Wall-clock time, seconds')
ax_loss.set_ylabel('Cross-entropy loss')
ax_loss.set_title('Loss vs time')
ax_loss.legend()
ax_loss.grid(alpha=0.3)
ax_acc.set_xlabel('Wall-clock time, seconds')
ax_acc.set_ylabel('Accuracy')
ax_acc.set_title('Accuracy vs time')
ax_acc.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax_acc.legend()
ax_acc.grid(alpha=0.3)
plt.suptitle('PyTorch 1 GPU vs 2 GPU - Adam lr=1e-3, per-rank B=32, 3000 steps',
             fontsize=13)
plt.tight_layout()
plt.savefig('torch_training_curves.png', dpi=150)
plt.show()
"""


CUDA_TORCH_COMPARE = """\
import os
import pandas as pd
import matplotlib.pyplot as plt

pairs = [
    ('1 GPU', 'cuda_1gpu_log.csv', 'torch_1gpu_log.csv'),
    ('2 GPU', 'cuda_2gpu_log.csv', 'torch_2gpu_log.csv'),
]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for gpu_label, cuda_path, torch_path in pairs:
    for impl, path, style in [('CUDA', cuda_path, '-'), ('PyTorch', torch_path, '--')]:
        if not os.path.exists(path):
            print('Missing optional comparison log:', path)
            continue
        df = pd.read_csv(path)
        axes[0].plot(df['elapsed_s'], df['loss'].rolling(10, min_periods=1).mean(),
                     linestyle=style, linewidth=2, label=f'{impl} {gpu_label}')
        axes[1].plot(df['elapsed_s'],
                     df['accuracy'].rolling(10, min_periods=1).mean(),
                     linestyle=style, linewidth=2, label=f'{impl} {gpu_label}')

axes[0].set_xlabel('Wall-clock time, seconds')
axes[0].set_ylabel('Cross-entropy loss')
axes[0].set_title('Training loss')
axes[1].set_xlabel('Wall-clock time, seconds')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training accuracy')
for ax in axes:
    ax.grid(alpha=0.3)
    ax.legend()
plt.suptitle('CUDA vs PyTorch logs with matching run settings', fontsize=13)
plt.tight_layout()
plt.show()
"""


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = [
    md("""# MNIST Vision Transformer on 2x T4 (PyTorch counterpart)

This notebook runs the PyTorch counterpart of the hand-written CUDA ViT. It is
set up for comparison against `kaggle_run.ipynb`:

- same Kaggle CSV search and torchvision MNIST fallback;
- same model config: V=256, T=784, L=2, D=64, H=4, C=10;
- same train runs: Adam lr=1e-3, 3000 steps, per-rank B=32;
- same log columns and throughput timer boundaries.

The PyTorch script is written in the normal ML style: `nn.Module` layers,
`DataLoader`, `torch.optim.Adam`, and `DistributedDataParallel` for two GPUs.
The architecture and experiment settings stay aligned with the CUDA notebook.
Attention is pinned to PyTorch's SDPA math backend in the source so the run
does not depend on automatic Flash or memory-efficient backend selection.

For throughput, use the same Kaggle accelerator and the same CSV in both
notebooks. Loss/accuracy curves are a training-behavior comparison, not an
exact numeric parity test: C++ and PyTorch RNG streams do not give identical
initial weights or sampled batches from seed 42.

Kaggle setup:

1. Select `GPU T4 x2` in Settings.
2. Enable Internet only if the MNIST fallback must download data.
3. Add the Digit Recognizer dataset when comparing with the CUDA notebook.
"""),
    md("## 1. Verify the environment\n\nExpect two T4 GPUs and a CUDA PyTorch build."),
    code("!nvidia-smi --query-gpu=index,name,memory.total --format=csv\n"
         "import torch\n"
         "print('torch:', torch.__version__)\n"
         "print('cuda available:', torch.cuda.is_available())\n"
         "print('cuda devices:', torch.cuda.device_count())\n"
         "assert torch.cuda.is_available(), 'Enable a Kaggle GPU accelerator'\n"),
    md("## 2. Write the PyTorch source\n\n"
       "The source is embedded so this notebook can run without uploading the repo."),
    code("%%writefile train_vit_torch.py\n" + SRC),
    md("## 3. Check the model contract\n\n"
       "The default PyTorch model should keep the CUDA-comparable parameter count."),
    code("from train_vit_torch import Cfg, DEFAULT_PARAM_COUNT, PixelViT, count_parameters\n"
         "cfg = Cfg()\n"
         "model = PixelViT(cfg)\n"
         "print(cfg)\n"
         "print('parameter count:', count_parameters(model))\n"
         "assert count_parameters(model) == DEFAULT_PARAM_COUNT\n"),
    md("## 4. Locate or materialize the data\n\n"
       "This is the same CSV search order and MNIST fallback as the CUDA notebook."),
    code(DATA_CELL),
    md("## 5. Training smoke test\n\n"
       "Run one normal PyTorch step before profiling. This leaves the real "
       "training traceback visible if the embedded source or CUDA environment "
       "is stale."),
    code("!python train_vit_torch.py \"$CSV\" 1 8 0.001 \\\n"
         "  --device cuda --log-path torch_smoke_log.csv\n"),
    md("## 6. GPU, memory, and DDP profiling\n\n"
       "Runs 50 PyTorch steps for B in {8, 16, 32, 64} on one GPU, plus one "
       "two-GPU run at B=32. The internal throughput timer matches the training "
       "script timer and excludes model/data setup."),
    code(PROF_SWEEP),
    md("## 6b. Plot profiling results"),
    code(PROF_PLOT),
    md("## 7. Single-GPU training run\n\n"
       "Matches the CUDA notebook run: Adam lr=1e-3, 3000 steps, per-rank B=32."),
    code("!python train_vit_torch.py \"$CSV\" 3000 32 0.001 \\\n"
         "  --device cuda --log-path training_log_torch.csv\n"),
    md("## 8. Save the single-GPU PyTorch log"),
    code("import os, shutil\n"
         "shutil.copy('training_log_torch.csv', 'torch_1gpu_log.csv')\n"
         "print('Saved torch_1gpu_log.csv:', os.path.getsize('torch_1gpu_log.csv'), 'bytes')\n"),
    md("## 9. Two-GPU training run\n\n"
       "`torchrun` launches one rank per GPU. Per-rank B=32 means global B=64, "
       "matching the CUDA notebook multi-GPU run."),
    code("!torchrun --standalone --nproc_per_node=2 --master_port=29541 \\\n"
         "  train_vit_torch.py \"$CSV\" 3000 32 0.001 \\\n"
         "  --device cuda --log-path training_log_torch.csv\n"),
    md("## 10. Save the two-GPU PyTorch log"),
    code("import os, shutil\n"
         "shutil.copy('training_log_torch.csv', 'torch_2gpu_log.csv')\n"
         "print('Saved torch_2gpu_log.csv:', os.path.getsize('torch_2gpu_log.csv'), 'bytes')\n"),
    md("## 11. PyTorch loss and accuracy vs time\n\n"
       "The x-axis is wall-clock time, so this is the scaling view."),
    code(TRAINING_CURVES),
    md("""## 12. Optional CUDA vs PyTorch plot

Copy CUDA notebook logs into this session as:

- `cuda_1gpu_log.csv`
- `cuda_2gpu_log.csv`

The plot expects the same CSV dataset and the same run arguments. Compare
throughput and time-to-loss trends. Do not interpret point-by-point loss
differences as a kernel parity failure unless both implementations are fed the
same parameters and sampled batches.
"""),
    code(CUDA_TORCH_COMPARE),
    md("""## Comparison notes

- The CUDA notebook and this notebook use the same data locator and fallback
  materialization to the Kaggle CSV format.
- The reported global batch is not the same between 1 GPU and 2 GPU runs:
  both notebooks use per-rank B=32, so two GPUs use global B=64.
- The PyTorch code uses standard `nn.Module`, `DataLoader`, Adam, and DDP
  rather than copying CUDA memory layout and kernel boundaries.
- PyTorch attention is pinned with `sdpa_kernel(SDPBackend.MATH)` for these
  comparisons.
- For a strict numerical forward/backward parity test, add a shared parameter
  fixture and a shared batch-index fixture to both implementations first.
"""),
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

out = ROOT / "kaggle_torch_run.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({out.stat().st_size} bytes, {len(cells)} cells)")
