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
