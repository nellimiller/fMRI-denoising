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

The project requires Python 3.13 or newer and uses `uv`:

```powershell
uv sync
```

The project installs PyTorch from the default Python package index, without
forcing a particular CUDA build. `uv.lock` records the resolved packages;
the NVIDIA driver on the cluster must support the CUDA runtime bundled with
the selected PyTorch wheel. The Slurm job checks that CUDA is available on its
allocated GPU.

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

The data module holds out complete subjects for validation, so no windows from
a validation subject are used for gradient updates. By default, a seeded
20%-of-subjects split is used; set `--validation_subjects subj08` to choose the
held-out subject explicitly (the IDs must also be included in `--subjects`, if
that option is set). At least two subjects are required. Note that the default
patch size and three encoder merges impose a minimum padded grid of 112 voxels
per spatial axis, so crops smaller than that do not necessarily reduce model
memory.

Useful options include:

| Option | Purpose |
| --- | --- |
| `--seed` | Seed model, split, and worker initialization (default `1234`). |
| `--subjects` | Subject IDs to include; by default all manifests are loaded. |
| `--validation_subjects` | Explicit subject IDs to reserve for validation. |
| `--crop_size X Y Z` | Crop the volume around the tissue-mask center. |
| `--patch_size` / `--embed_dim` | Model capacity; defaults are `14` / `96`. |
| `--precision` | Lightning precision mode; use `32-true` for a conservative smoke run. |
| `--gradient_clip_val` | Clip gradient norm (default `1.0`). |
| `--log_dir` | Persist CSV metrics (default `logs_nsd/`). |
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
The FFT component is calculated in float32 so mixed-precision training supports
the default 40-frame sequence length on CUDA.

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

### Post-training evaluation with GLMdenoise beta maps

Evaluation can be implemented and run after training; it does not have to be
part of the optimization loop, and it need not change the unsupervised MAE
objective. Decide the evaluation split and save the checkpoint, subject/run
IDs, preprocessing details, and TR before a long run. The validation subject
is used for checkpoint selection, so reserve a different, untouched test
subject for final evaluation if reporting a held-out result. With eight
subjects, a pilot could use six for training, one for validation, and one for
testing; subject-level cross-validation can use the data more fully later.

Single-trial GLMdenoise beta maps are not directly interchangeable with the
model's input or output: the model takes 4D BOLD windows plus tissue masks and
predicts time-series clean/noise components, whereas each beta map is a 3D
trial coefficient. A reasonable external analysis is to apply a frozen model
to held-out BOLD, form its clean time series, run the same trial GLM on those
time series, and compare the resulting trial betas to the GLMdenoise beta
maps. Keep trial design, scaling, registration, and voxel masks aligned. This
measures agreement with a GLMdenoise-derived reference, not ground-truth
denoising: the reference is itself estimated from fMRI and may share data and
noise with the model input. Report map correlation and reliability alongside
scale-sensitive errors, and compare against betas from the original BOLD and
appropriate GLM baselines. Do not feed isolated beta volumes into this
4D-temporal model as though they were BOLD frames.

For unsupervised evaluation without beta targets, report held-out-subject
masked reconstruction error and residuals, temporal/spectral summaries,
clean/noise energy and stability across seeds, plus visual checks in GM, WM,
and CSF. These are reconstruction/consistency diagnostics, not evidence that
the clean head is biologically clean: only the sum of the two heads is directly
constrained by reconstruction, so their decomposition is not identifiable
from the current objective alone.

### Run held-out evaluation on Slurm

After training, choose a best-validation checkpoint and a test subject that
was not used for training or validation. From the repository root, set the
processed-data path, checkpoint, and persistent output directory, then submit
the evaluation job. The defaults evaluate `subj08` with 40-frame windows:

```bash
export NSD_PROCESSED=/path/to/nsd_processed_40f_s20
export CHECKPOINT=/path/to/run/checkpoints/best-validation-checkpoint.ckpt
export EVAL_DIR=/path/to/persistent/evaluations/nsd_test
export EVAL_SUBJECTS=subj08
export EVAL_TRAIN_VISUAL_SUBJECTS=subj01
export EVAL_VALIDATION_VISUAL_SUBJECTS=subj07

sbatch scripts/cluster/nsd_evaluate.slurm
```

The evaluator loads model hyperparameters from the Lightning checkpoint and
writes `metrics_by_window.csv`, `metrics_by_run.csv`,
`metrics_by_subject.csv`, and `summary.json` to `EVAL_DIR`. It also saves up
to three example slice panels per subject under
`EVAL_DIR/visualizations/{train,validation,test}/`. `EVAL_SUBJECTS` are the
test subjects included in the quantitative metrics. Optional
`EVAL_TRAIN_VISUAL_SUBJECTS` and `EVAL_VALIDATION_VISUAL_SUBJECTS` add
training/validation subjects for visual comparison only; they are excluded
from the metric CSVs. Each panel shows observed BOLD, the predicted clean
head, reconstruction, noise head, reconstruction residual, and GM/WM/CSF
masks. Input, clean, and reconstruction panels use the same intensity scale.
The predicted clean head is an experimental output, not a validated
clean-signal estimate. Examples are selected across each subject's windows
and show the middle time point on the axial slice with the most tissue-mask
voxels.

Metrics include full and masked MSE/MAE/RMSE and correlation, tissue-specific
MSE, and clean and noise output energy. The run and subject summaries average
windows and runs respectively; the overall summary averages subjects equally.
Use `EVAL_TIME_FRAMES` and optional `EVAL_CROP_SIZE="X Y Z"` to match the
preprocessed windows and training crop, `EVAL_MASK_REPEATS` to average
different deterministic tube-mask draws, `EVAL_SEED` to choose their
reproducible seed, and `EVAL_VISUAL_EXAMPLES_PER_SUBJECT` to change the number
of panels (set it to `0` to disable). These remain
reconstruction/consistency metrics against observed BOLD, not measures of
clean-signal recovery.

Before a longer run, check that:

- losses and gradients remain finite;
- the clean and noise heads both have nontrivial energy;
- the total reconstruction and residuals look plausible in the saved panels;
- validation performance is interpreted at the subject split level.

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
export NSD_PROCESSED=/path/to/nsd_processed_pilot_8subjects
export RUN_DIR=/path/to/persistent/fmri_runs/nsd_pilot
mkdir -p "$RUN_DIR"
```

Generate a deliberately small pilot manifest for **each** subject. A fresh
processed directory avoids mixing manifests with different window lengths.
The long stride selects at most one short window per run; the full eight-subject
set must still have at least two distinct subject/run groups for the
training/validation split. House rules require computations to run as Slurm
jobs, so submit preprocessing as a CPU job (do not run this loop on the login
node):

```bash
PREP_JOB_ID=$(sbatch --parsable \
  --export=ALL,NSD_RAW="$NSD_RAW",NSD_PROCESSED="$NSD_PROCESSED",PILOT_TIME_FRAMES=8,PILOT_STRIDE=10000 \
  scripts/cluster/nsd_preprocess_pilot.slurm)
PREP_JOB_ID="${PREP_JOB_ID%%;*}"
echo "Preprocessing job: $PREP_JOB_ID"
```

Create manifests where the cluster data will live and keep that directory
layout stable: the manifest records relative paths from the processed-data
root to the raw BOLD files. The preprocessing job writes one manifest per
subject and validates image geometry. The training job refuses to start if a
requested subject manifest is missing or has no windows.

### 3. Install uv and create the training environment

Load the cluster's Python 3.13 module and, if available, its `uv` module.
Confirm the Python version:

```bash
python3 --version
```

If `uv` is not provided as a module, install it in your user account using the
official installer, then add its default install location to `PATH`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

If the cluster blocks downloads, ask the administrator for its approved `uv`
installation method. Once Python 3.13 and `uv` are available, create the
environment from the repository root:

```bash
bash scripts/cluster/setup_env.sh
```

This runs `uv sync --locked --python 3.13`, creating or updating the ignored
`.venv/` from `uv.lock`. PyTorch is resolved from the default package index;
there is no CUDA 12.4-only index or version pin. The installed wheel includes
its CUDA runtime dependencies, and the cluster's NVIDIA driver must support
that runtime. The login node need not expose GPUs; the pilot verifies GPU
visibility on the allocated compute node.

### 4. Adjust and submit the Slurm pilot

Review `scripts/cluster/nsd_pilot.slurm` before submitting. Its example request
uses one GPU, four CPUs, 32 GB of memory, and two hours. Change the GPU
directive (`--gres=gpu:1`), partition/account, time, CPUs, and memory as
required by your cluster. The preprocessing job in
`scripts/cluster/nsd_preprocess_pilot.slurm` requests four CPUs, 32 GB of
memory, and four hours; adjust those directives to your cluster as well.

Submit the GPU pilot as a dependent job, so it only starts if preprocessing
completes successfully. These commands submit jobs; the data preprocessing and
training themselves run on Slurm compute nodes:

```bash
PILOT_JOB_ID=$(sbatch --parsable \
  --dependency="afterok:$PREP_JOB_ID" \
  --export=ALL,NSD_PROCESSED="$NSD_PROCESSED",RUN_DIR="$RUN_DIR",PILOT_TIME_FRAMES=8,PILOT_EPOCHS=1,PILOT_VALIDATION_SUBJECTS=subj08 \
  scripts/cluster/nsd_pilot.slurm)
PILOT_JOB_ID="${PILOT_JOB_ID%%;*}"
echo "Training job: $PILOT_JOB_ID (after preprocessing job $PREP_JOB_ID)"
```

By default, the job verifies all eight manifests and CUDA visibility, loads a
real sample, and trains one epoch over the requested subjects. For this
resource/integration pilot it uses an 8-frame window, a 32-voxel cube,
`patch_size=4`, and `embed_dim=24` (override with `PILOT_*` variables as
needed). The default seeded split reserves two subjects for validation; set
`PILOT_VALIDATION_SUBJECTS=subj08` to hold out one specific subject. This is
intentionally a smaller model than the normal training
defaults (`patch_size=14`, `embed_dim=96`); do not use its checkpoint as a
production checkpoint. It checks end-to-end data loading, optimization,
validation, checkpointing, visualization, and metric writing, but does not
establish denoising performance.

Expected outputs include:

- `$RUN_DIR/checkpoints/last.ckpt` and a best-validation checkpoint selected
  by `val_total`.
- `$RUN_DIR/visualizations/validation_epoch_0000.png`.
- CSV metrics under
  `$RUN_DIR/logs/nsd-8subject-one-epoch/version_N/metrics.csv` (`version_0`
  for a fresh run directory).
- `slurm-fmri-nsd-pilot-<job-id>.out` and `.err` in the submission directory.

The pilot enables W&B as well as local CSV metrics. Ensure the cluster job can
reach W&B and is authenticated before submission (for example, configure
`WANDB_API_KEY` through the cluster's approved secret mechanism, or use an
existing W&B login). Set `PILOT_WANDB_PROJECT` to override the project name.
The regular training entry point can likewise opt into W&B with `--use_wandb`.

Check both job statuses and outputs using the cluster's Slurm tools (typically
`squeue`, `sacct`, and the `slurm-*.out` logs). Keep the output directory
persistent. If preprocessing fails, Slurm will leave the dependent training
job unstarted. The training split uses unseen subjects for validation.
Without clean ground-truth fMRI, validation loss measures reconstruction and
consistency, not denoising accuracy; see the evaluation notes above before
interpreting it scientifically.

### Full training after the pilot

The one-epoch pilot checked execution, but its short windows and small model
are not a full training run. The separate `scripts/cluster/nsd_train.slurm`
uses the regular model defaults (`patch_size=14`, `embed_dim=96`), 40-frame
windows, batch size 1, and 100 epochs. It defaults to subjects 1-6 for
training and subject 7 for validation; subject 8 is not loaded, leaving it
available for a later final test. Change the split only before starting the
run, and choose a held-out test subject distinct from the validation subjects.

Preprocess a fresh full-window directory using a Slurm CPU job. The stride of
20 creates overlapping windows; this can produce many more samples than the
pilot, so check the preprocessing duration and storage:

```bash
export FULL_PROCESSED=/hpc/faststorage/miller/fMRI-denoising/nsd_processed_40f_s20
export FULL_RUN_DIR=/hpc/faststorage/miller/fMRI-denoising/runs/nsd_full_40f
mkdir -p "$FULL_RUN_DIR/logs/slurm"

FULL_PREP_JOB_ID=$(sbatch --parsable \
  --output="$FULL_RUN_DIR/logs/slurm/preprocess-%j.out" \
  --error="$FULL_RUN_DIR/logs/slurm/preprocess-%j.err" \
  --export=ALL,NSD_RAW="$NSD_RAW",NSD_PROCESSED="$FULL_PROCESSED",PILOT_TIME_FRAMES=40,PILOT_STRIDE=20 \
  scripts/cluster/nsd_preprocess_pilot.slurm)
FULL_PREP_JOB_ID="${FULL_PREP_JOB_ID%%;*}"
echo "Full-window preprocessing job: $FULL_PREP_JOB_ID"
```

Once submitted, preprocessing runs on the compute node. The following
dependent training job will not start unless preprocessing succeeds. Ensure
W&B credentials are available to the job through the cluster-approved method:

```bash
FULL_TRAIN_JOB_ID=$(sbatch --parsable \
  --dependency="afterok:$FULL_PREP_JOB_ID" \
  --output="$FULL_RUN_DIR/logs/slurm/train-%j.out" \
  --error="$FULL_RUN_DIR/logs/slurm/train-%j.err" \
  --export=ALL,NSD_PROCESSED="$FULL_PROCESSED",RUN_DIR="$FULL_RUN_DIR",TRAIN_SUBJECTS="subj01 subj02 subj03 subj04 subj05 subj06",VALIDATION_SUBJECTS=subj07,TRAIN_TIME_FRAMES=40,TRAIN_EPOCHS=100 \
  scripts/cluster/nsd_train.slurm)
FULL_TRAIN_JOB_ID="${FULL_TRAIN_JOB_ID%%;*}"
echo "Full training job: $FULL_TRAIN_JOB_ID"
```

The 100-epoch default is a starting configuration, not a runtime estimate.
Ask for a wall-time limit supported by the cluster and watch epoch duration.
The learning-rate cosine schedule uses the configured epoch count, so set
`TRAIN_EPOCHS` to the intended total before starting. If the job reaches its
time limit, it currently does not automatically resume from a checkpoint;
do not assume that a checkpoint is a resumable run without explicitly
continuing with the intended checkpoint path.

Monitor both jobs with:

```bash
squeue -j "$FULL_PREP_JOB_ID,$FULL_TRAIN_JOB_ID" -o "%.18i %.24j %.8T %.10M %R"
sacct -j "$FULL_PREP_JOB_ID,$FULL_TRAIN_JOB_ID" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
tail -f "$FULL_RUN_DIR/logs/slurm/train-${FULL_TRAIN_JOB_ID}.out"
```

Training checkpoints are under `$FULL_RUN_DIR/checkpoints`, local CSV metrics
under `$FULL_RUN_DIR/logs`, validation panels under
`$FULL_RUN_DIR/visualizations`, and W&B metrics in the configured project.
Inspect validation curves and outputs before treating this as a useful model;
successful completion alone is not evidence of denoising quality.

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
