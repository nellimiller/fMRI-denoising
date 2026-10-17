# fMRI Denoising

An experimental 4D fMRI masked autoencoder for self-supervised reconstruction
and exploratory signal decomposition. The current data workflow targets
Natural Scenes Dataset (NSD) functional runs and uses anatomical grey matter
(GM), white matter (WM), and cerebrospinal fluid (CSF) masks.

## Status and interpretation

The model reconstructs the observed BOLD signal as the sum of two learned
outputs:

```text
reconstruction = clean_head + noise_head
```

Training uses masked reconstruction together with full reconstruction,
tissue-based regularizers, and a temporal FFT term. There is no clean fMRI
target in this workflow. Consequently, reconstruction and anatomical priors
alone do not uniquely identify the clean and noise components. Treat the
separation as an experimental hypothesis, not validated denoising; inspect
the component metrics and visualizations before drawing conclusions.

## Environment

The project requires Python 3.12 or newer and uses `uv`:

```powershell
uv sync
```

The training environment pins PyTorch 2.6.0, torchvision 0.21.0, and
torchaudio 2.6.0 to their CUDA 12.4 builds. The CUDA runtime is bundled with
these PyTorch wheels; the cluster still needs an NVIDIA driver compatible
with CUDA 12.4. See the Slurm setup below for environment and GPU checks.

## NSD workflow

### 1. Prepare subject manifests

The preprocessing command creates GM/WM/CSF masks and a manifest of temporal
windows. It references BOLD NIfTI files instead of copying large runs.

```powershell
uv run python -m src.data_preprocessing.nsd.preprocess `
  --raw-root "C:\path\to\nsd_data" `
  --output-root "C:\path\to\nsd_processed" `
  --subject subj01 `
  --window-length 40 `
  --stride 20
```

Expected processed-data layout:

```text
nsd_processed/
└── subj01/
    ├── anat/
    │   ├── gm.nii.gz
    │   ├── wm.nii.gz
    │   └── csf.nii.gz
    └── manifest.json
```

Manifest paths are relative to the processed-data root, so the dataset can be
moved between machines as long as the referenced raw data remains at the
corresponding relative location. The loader also accepts existing absolute
paths when they still resolve.

The preprocessing expects the NSD layout described in
[the NSD preprocessing guide](src/data_preprocessing/nsd/README.md). Review
the label sets in `src/data_preprocessing/nsd/preprocess.py` if the
segmentation convention differs from the one used for the data.

### 2. Run a short training trial

From the repository root:

```powershell
uv run python -m src.train_nsd `
  --data_dir "C:\path\to\nsd_processed" `
  --crop_size 64 64 64 `
  --epochs 1 `
  --num_workers 4 `
  --precision 32-true
```

The data module keeps complete subject/run groups on one side of the
train/validation split to avoid leakage between overlapping windows. At least
two distinct subject/run groups are needed. Note that the default patch size
and three encoder merges impose a minimum padded grid of 112 voxels per spatial
axis, so crops smaller than that do not necessarily reduce model memory.

Useful options include:

| Option | Purpose |
| --- | --- |
| `--seed` | Seed model, split, and worker initialization (default `1234`). |
| `--crop_size X Y Z` | Crop the volume around the tissue-mask center. |
| `--precision` | Lightning precision mode; use `32-true` for a conservative smoke run. |
| `--gradient_clip_val` | Clip gradient norm (default `1.0`). |
| `--checkpoint_dir` | Save best validation and latest checkpoints. |
| `--visualization_dir` | Save validation slice panels. |
| `--use_wandb` | Enable Weights & Biases metric logging. |

Run `uv run python -m src.train_nsd --help` for the complete option list.

### Local GPU smoke training with one real NSD window

First verify that PyTorch can see your local NVIDIA GPU:

```powershell
nvidia-smi
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU detected')"
```

If `torch.cuda.is_available()` is false, install a CUDA-enabled PyTorch build
compatible with your GPU and driver, following the official PyTorch
installation instructions for your system. The following diagnostic requests
the GPU explicitly and should fail rather than silently run on CPU.

Generate a small manifest with short windows and at most one window per run:

```powershell
uv run python -m src.data_preprocessing.nsd.preprocess `
  --raw-root "C:\path\to\nsd_data" `
  --output-root "C:\path\to\nsd_processed_smoke" `
  --subject subj01 `
  --window-length 8 `
  --stride 10000
```

Then train repeatedly on one real window:

```powershell
uv run python -m src.utils.train_nsd_overfit `
  --data_dir "C:\path\to\nsd_processed_smoke" `
  --output_dir "C:\path\to\fmri_runs\local_smoke" `
  --subject subj01 `
  --entry 0 `
  --time_frames 8 `
  --crop_size 32 32 32 `
  --patch_size 4 `
  --epochs 2 `
  --repeats 4
```

This checks real-data loading, GPU forward/backward execution, and produces
initial qualitative outputs. It intentionally reuses one sample, so it is not
a validation or denoising-quality experiment. A latest checkpoint is saved
under the output directory, and validation slice panels are saved in its
`visualizations/` subdirectory. Increase the spatial crop, temporal length, and
number of training samples only after checking runtime and GPU memory. The
regular training entry point uses patch size 14 and can require substantially
more memory than this diagnostic.

## Monitoring outputs

The trainer logs total loss and its masked reconstruction, full reconstruction,
flatness, orthogonality, and FFT components. It also logs clean/noise output
energy and the masked-token fraction. Checkpoints are written to
`checkpoints_nsd/` by default. A validation slice panel for a fixed sample is
saved to `validation_visualizations/` after each validation epoch; the
randomized mask means the reconstruction may vary between epochs.

## Adding evaluation metrics

Metrics computable from model predictions and the original inputs can usually
be calculated after training: load a saved checkpoint, run inference on
held-out data, and compute them in an evaluation script. Add a metric before
training if it should affect optimization, be logged during training, control
early stopping/checkpoint selection, or guide decisions during the run.

In either case, retain the checkpoint and evaluation data, plus needed
metadata such as TR. Without clean ground-truth fMRI, metrics against observed
BOLD quantify reconstruction or consistency, not denoising accuracy; they do
not prove that the clean output is noise-free.

Before a longer run, check that:

- losses and gradients remain finite;
- the clean and noise heads both have nontrivial energy;
- the total reconstruction and residuals look plausible in the saved panels;
- validation performance is interpreted at the subject/run split level.

## Slurm cluster pilot

The cluster helpers are in `scripts/cluster/`. They assume the repository is
checked out on shared storage and that `uv` is available. Ask the cluster
administrator which Python/CUDA modules, GPU request syntax, partition,
account, memory limit, and persistent storage should be used; the Slurm
directives below are a starting point, not universal cluster settings.

### 1. Transfer the code

Prefer transferring code with Git rather than copying a local virtual
environment:

```bash
git clone <your-repository-url> fMRI-denoising
cd fMRI-denoising
```

If you have uncommitted changes, commit/push them first or transfer those
specific source changes separately. Do not transfer `.venv/`, caches, model
checkpoints, or local datasets as part of the code checkout.

### 2. Make the data available to compute nodes

Follow the NSD data agreement and your institution's storage/transfer rules.
Prefer an approved transfer service or cluster data-transfer node for large
datasets; avoid routing the full dataset through a personal computer unless
that is the approved and practical route. Keep raw data read-only on storage
that compute nodes can access.

On the cluster, define the raw-data, processed-data, and persistent-run paths:

```bash
export NSD_RAW=/path/to/nsd_data
export NSD_PROCESSED=/path/to/nsd_processed_pilot
export RUN_DIR=/path/to/persistent/fmri_runs/nsd_pilot
mkdir -p "$RUN_DIR"
```

Generate a deliberately small pilot manifest. A long stride selects at most
one short window per run; at least two distinct subject/run groups are required
for the training/validation split:

```bash
uv run python -m src.data_preprocessing.nsd.preprocess \
  --raw-root "$NSD_RAW" \
  --output-root "$NSD_PROCESSED" \
  --subject subj01 \
  --window-length 8 \
  --stride 10000
```

Create manifests where the cluster data will live and keep that directory
layout stable: the manifest records relative paths from the processed-data
root to the raw BOLD files. Verify the groups, sample shape, and affine checks
before submitting a GPU job.

### 3. Create the training environment

Run this in a shell with any required cluster Python/CUDA modules loaded:

```bash
bash scripts/cluster/setup_env.sh
```

This runs `uv sync --locked --python 3.12`, creating or updating the ignored
`.venv/` from `uv.lock`. The project pins PyTorch, torchvision, and torchaudio
to CUDA 12.4 wheels from the official PyTorch CUDA 12.4 package index. The
setup script checks the PyTorch CUDA runtime version. A login node may not
expose GPUs; the Slurm job checks the runtime version and GPU visibility again
on the allocated GPU node. The cluster's NVIDIA driver must support CUDA 12.4.

### 4. Adjust and submit the Slurm pilot

Review `scripts/cluster/nsd_pilot.slurm` before submitting. Its example request
uses one GPU, four CPUs, 32 GB of memory, and two hours. Change the GPU
directive (`--gres=gpu:1`), partition/account, time, CPUs, and memory as
required by your cluster.

Submit from the repository root with the data and output paths exported:

```bash
sbatch \
  --export=ALL,NSD_PROCESSED="$NSD_PROCESSED",RUN_DIR="$RUN_DIR",PILOT_TIME_FRAMES=8,PILOT_EPOCHS=1 \
  scripts/cluster/nsd_pilot.slurm
```

The job verifies CUDA, loads one real window, checks that there are at least
two subject/run groups, and runs the normal `src.train_nsd` model for one
epoch. It writes Slurm output files in the submission directory and saves
checkpoints and visualizations under `$RUN_DIR`. This is a small-data
integration/resource pilot, not evidence of scientific denoising performance.

Check job status and outputs using the cluster's Slurm tools (typically
`squeue`, `sacct`, and the `slurm-*.out` log). Keep the output directory
persistent. W&B is not enabled by this script unless configured separately;
the current trainer's default logging behavior should be checked to ensure the
metrics you need are persisted.

## Tests and diagnostics

Run the automated tests with:

```powershell
uv run pytest
```

They include a small synthetic CPU training smoke test. This verifies basic
execution and gradient flow, not scientific denoising quality.

For a data-backed one-window overfit diagnostic, see
[the NSD preprocessing guide](src/data_preprocessing/nsd/README.md). It
deliberately repeats one window and must not be treated as a generalization
experiment.

## Code map

```text
src/
├── data/
│   ├── nsd_dataset.py              # Manifest-backed NSD loader
│   └── synthetic.py                # Synthetic data for smoke tests
├── data_preprocessing/nsd/
│   └── preprocess.py               # NSD masks and temporal manifests
├── models/
│   ├── mae_core.py                 # Masked hierarchical Swin encoder
│   ├── decoder.py                  # Additive clean/noise output heads
│   └── lightning_model.py          # Training steps, metrics, optimizer
├── utils/
│   ├── losses.py                   # Reconstruction and structural losses
│   ├── masking.py                  # Token masking
│   ├── visualization.py            # Validation slice panels
│   └── train_nsd_overfit.py        # Single-window diagnostic
├── train_nsd.py                    # Canonical NSD training entry point
└── test_pipeline.py                # Manual synthetic forward/backward check
```
