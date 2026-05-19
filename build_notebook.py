#!/usr/bin/env python3
"""Builds kaggle_run.ipynb from train_vit.cu — embeds the full source."""
import json, pathlib

ROOT = pathlib.Path(__file__).parent
SRC  = (ROOT / "src" / "train_vit.cu").read_text()

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}

cells = [
    md("""# MNIST Vision Transformer on 2× T4 (CUDA + MPI + NCCL)

This notebook compiles and runs the hand-written CUDA ViT from `src/train_vit.cu`
on a Kaggle session with **two Tesla T4 GPUs**.

**Setup checklist (do this in the Kaggle UI before running):**

1. **Accelerator → GPU T4 x2** (Settings panel on the right).
2. **Add Data → search "Digit Recognizer" → add the competition dataset.** The
   `train.csv` will appear at `/kaggle/input/digit-recognizer/train.csv`.
3. **Internet → On** (needed once, to `apt-get install` OpenMPI).

Then run the cells top-to-bottom.
"""),

    md("## 1. Verify the environment\n\nWe expect two T4 GPUs and a working CUDA toolchain."),
    code("!nvidia-smi --query-gpu=index,name,memory.total --format=csv"),

    md("## 2. Install OpenMPI\n\n"
       "Kaggle's image already ships `libnccl2` (for PyTorch), `nvcc`, and "
       "`cuBLAS`. We only need to add `openmpi-bin`/`libopenmpi-dev` for "
       "`mpirun`/`mpicxx`.\n\n"
       "⚠️ **Do NOT install `libnccl-dev`** — it tries to pull a `libnccl2` "
       "version that conflicts with the one shipped with Kaggle's CUDA/PyTorch "
       "image (`apt` errors out with *held broken packages*). The NCCL headers "
       "we need are already on disk in the PyTorch conda env — we'll locate "
       "them in the next cell."),
    code("import subprocess\n"
         "subprocess.run(['apt-get', '-qq', 'update'], check=True)\n"
         "subprocess.run(['apt-get', 'install', '-y', '-qq',\n"
         "                'openmpi-bin', 'libopenmpi-dev'], check=True)\n"
         "print('OK')\n"),

    md("## 3. Locate the existing NCCL header and library\n\n"
       "On Kaggle, NCCL is usually under `/opt/conda` (the PyTorch env). We "
       "find `nccl.h` and `libnccl.so*` and stash their directories in env "
       "vars so the compile cell can pass them via `-I` and `-L`."),
    code("import os, glob, subprocess\n"
         "\n"
         "hdr = subprocess.check_output(\n"
         "    'find /opt/conda /usr/include /usr/local -name nccl.h 2>/dev/null | head -n1',\n"
         "    shell=True, text=True).strip()\n"
         "lib = subprocess.check_output(\n"
         "    'find /opt/conda /usr/lib /usr/local -name \"libnccl.so*\" 2>/dev/null | head -n1',\n"
         "    shell=True, text=True).strip()\n"
         "\n"
         "assert hdr, 'nccl.h not found — open an issue or `pip install nvidia-nccl-cu12`'\n"
         "assert lib, 'libnccl.so not found'\n"
         "\n"
         "NCCL_INCLUDE = os.path.dirname(hdr)\n"
         "NCCL_LIB     = os.path.dirname(lib)\n"
         "print('nccl.h     :', hdr)\n"
         "print('libnccl.so :', lib)\n"
         "os.environ['NCCL_INCLUDE'] = NCCL_INCLUDE\n"
         "os.environ['NCCL_LIB']     = NCCL_LIB\n"),

    md("### Toolchain sanity check"),
    code("!which nvcc mpicxx mpirun\n"
         "!nvcc --version | tail -n 2\n"
         "!mpirun --version | head -n 1\n"
         "!echo \"NCCL include: $NCCL_INCLUDE\"\n"
         "!echo \"NCCL lib    : $NCCL_LIB\"\n"),

    md("## 4. Drop the source file into the working dir\n\n"
       "We use `%%writefile` so the notebook is fully self-contained — no need "
       "to upload `train_vit.cu` as a dataset."),
    code("%%writefile train_vit.cu\n" + SRC),

    md("## 5. Compile\n\n"
       "- `-ccbin mpicxx` — `nvcc` uses the MPI C++ wrapper as host compiler "
       "(so `<mpi.h>` and the MPI libs are picked up automatically).\n"
       "- `-arch=sm_75` — targets the T4 (Turing).\n"
       "- `-I$NCCL_INCLUDE` / `-L$NCCL_LIB` — point at the NCCL we located above.\n"
       "- `-Xlinker -rpath=$NCCL_LIB` — bake the NCCL path into the binary so "
       "`./bin/train_vit` finds `libnccl.so` at runtime without us having to "
       "export `LD_LIBRARY_PATH`."),
    code("!mkdir -p bin\n"
         "!nvcc -O2 -std=c++17 -ccbin mpicxx -arch=sm_75 \\\n"
         "      -I$NCCL_INCLUDE -L$NCCL_LIB \\\n"
         "      -Xlinker -rpath=$NCCL_LIB \\\n"
         "      train_vit.cu -o bin/train_vit \\\n"
         "      -lcublas -lnccl\n"
         "!ls -lh bin/train_vit\n"
         "!ldd bin/train_vit | grep -E 'nccl|cublas|mpi'\n"),

    md("## 6. Locate (or materialise) the data\n\n"
       "The trainer reads the Kaggle CSV layout: a header row, then "
       "`label, pixel0, pixel1, ..., pixel783` per sample (pixels 0..255).\n\n"
       "**Any 28×28 / 10-class dataset works** — MNIST digits, Fashion-MNIST, "
       "Kuzushiji-MNIST. The trainer doesn't care what the labels mean.\n\n"
       "**Search order:**\n"
       "1. Common known paths: `digit-recognizer/train.csv`, "
       "`fashionmnist/fashion-mnist_train.csv`.\n"
       "2. Any `*train*.csv` under `/kaggle/input` (validated to have 785 "
       "columns).\n"
       "3. **Fallback:** materialise from `torchvision.datasets.MNIST` "
       "(preinstalled, ~10 MB with Internet ON) → `/kaggle/working/train.csv`."),
    code("import os, glob, numpy as np\n"
         "\n"
         "# 1. Known paths first.\n"
         "known = [\n"
         "    '/kaggle/input/digit-recognizer/train.csv',\n"
         "    '/kaggle/input/fashionmnist/fashion-mnist_train.csv',\n"
         "    '/kaggle/input/fashion-mnist/fashion-mnist_train.csv',\n"
         "]\n"
         "# 2. Anything that looks like a train CSV.\n"
         "globbed = sorted(set(\n"
         "    glob.glob('/kaggle/input/**/*train*.csv', recursive=True) +\n"
         "    glob.glob('/kaggle/input/**/*Train*.csv', recursive=True)))\n"
         "\n"
         "def valid_csv(path):\n"
         "    \"\"\"785 cols (label + 784 pixels)?\"\"\"\n"
         "    try:\n"
         "        with open(path) as f:\n"
         "            return len(f.readline().split(',')) == 785\n"
         "    except Exception:\n"
         "        return False\n"
         "\n"
         "CSV = next((c for c in known + globbed if os.path.exists(c) and valid_csv(c)), None)\n"
         "\n"
         "if CSV is None:\n"
         "    if globbed:\n"
         "        print('Found these CSVs but none matched the 785-column layout:')\n"
         "        for p in globbed: print(' ', p)\n"
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

    md("## 7. Single-GPU baseline\n\n"
       "Useful as a reference for the multi-GPU throughput number.\n\n"
       "Arguments: `csv steps batch_size learning_rate`."),
    code("!./bin/train_vit $CSV 200 32 0.05\n"),

    md("## 8. Two-GPU run (MPI + NCCL)\n\n"
       "`--allow-run-as-root` is required because the Kaggle kernel runs as root. "
       "`-np 2` spawns one process per GPU; each process picks its device via "
       "`cudaSetDevice(rank % devices)` (see [src/train_vit.cu:986](train_vit.cu#L986)).\n\n"
       "Per-rank batch stays at 32, so the **global** batch is 64. The `sgd_step` "
       "kernel divides by `B * world` so the effective update is the mean over "
       "the global batch."),
    code("!mpirun --allow-run-as-root -np 2 ./bin/train_vit $CSV 200 32 0.05\n"),

    md("## 9. Quick scaling test (optional)\n\n"
       "Compare `throughput` lines between `-np 1` and `-np 2` to see the speedup. "
       "For this tiny model on PCIe-connected T4s the all-reduce overhead is "
       "non-trivial, so expect sub-linear scaling — but it should still be "
       "noticeably faster per global step."),
    code("import subprocess, re, os\n"
         "\n"
         "def throughput(np_):\n"
         "    cmd  = ['mpirun', '--allow-run-as-root', '-np', str(np_),\n"
         "            './bin/train_vit', os.environ['CSV'], '300', '32', '0.05']\n"
         "    res  = subprocess.run(cmd, capture_output=True, text=True)\n"
         "    print(res.stdout)\n"
         "    if res.returncode != 0:\n"
         "        print('--- STDERR ---'); print(res.stderr)\n"
         "        raise SystemExit(f'exit {res.returncode}')\n"
         "    m = re.search(r'throughput:\\s*([\\d.]+)\\s*img/s\\s*global', res.stdout)\n"
         "    return float(m.group(1)) if m else float('nan')\n"
         "\n"
         "t1 = throughput(1)\n"
         "t2 = throughput(2)\n"
         "print(f'\\n1 GPU : {t1:8.0f} img/s')\n"
         "print(f'2 GPUs: {t2:8.0f} img/s   ({t2/t1:.2f}x)')\n"),

    md("## Notes\n\n"
       "- T4s on Kaggle are connected over PCIe (no NVLink), so NCCL falls back to "
       "PCIe + (sometimes) shared memory. That caps the speedup on a tiny model "
       "like this one — the all-reduce is on ~124 k parameters per step.\n"
       "- If you bump `B` too high you'll OOM the activation buffer: it scales as "
       "`L · B · H · T²` for the attention scores (`A_ATTN_PRE`, `A_ATTN`). With "
       "`T=784, H=4, L=2`, each is `B · 4.9M` floats ≈ `B · 19.6 MiB`.\n"
       "- The CUDA build is committed to one source file on purpose — see "
       "[`train_vit.cu`](train_vit.cu) and the README for the section map."),
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
