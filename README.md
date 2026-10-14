fMRI-denoising
Deep Learning-Based Denoising of Ultra-High-Field 7 Tesla Functional Magnetic Resonance Imaging Data

## Proposed Architecture: Structural Masked Autoencoder for 7T fMRI

This project implements a 4D fMRI masked autoencoder baseline based on the SWIFT/Swin Transformer family and extends it with an additive decomposition objective for denoising.

The encoder is trained as an unsupervised masked autoencoder, while the decoder splits its output into two heads:

- `Y_clean`: the denoised neural/BOLD component in grey matter
- `Y_noise`: physiological and scanner-related nuisance signal routed to CSF/WM

The total reconstruction satisfies:

Y_total = Y_clean + Y_noise

The training objective follows the structural decomposition described in the project notes:

L_total = L_recon + lambda_1 L_flat + lambda_2 L_ortho

`L_flat` is computed as temporal variance over time for voxels in WM/CSF for the clean branch, and `L_ortho` is implemented as a deep aCompCor-style summary penalty: a single temporal noise vector averaged over WM/CSF is projected against each GM voxel time series, avoiding the memory-heavy token-to-token covariance matrix. This keeps the model fully self-supervised while remaining computationally stable for high-resolution 7T data. The first implementation targets the Natural Scenes Dataset and uses anatomical masks derived from the structural scan (GM, WM, CSF) to build spatiotemporal masking and penalty terms.

## Directory Hierarchy

fMRI-denoising/
├── .gitignore
├── notes.txt
├── pyproject.toml
├── uv.lock
├── README.md
└── src/
    ├── main.py
    ├── test_pipeline.py
    ├── data/
    │   ├── __init__.py
    │   ├── dataset.py
    │   └── preprocess.py
    ├── models/
    │   ├── __init__.py
    │   ├── decoder.py
    │   ├── lightning_model.py
    │   ├── mae_core.py
    │   ├── patch_embed.py
    │   └── swift_official.py
    └── utils/
        ├── __init__.py
        ├── losses.py
        └── masking.py


## Root Directory
Files managing the environment, dependencies, project metadata
### pyproject.toml & uv.lock :
define project dependencies ensuring reproducible environment using uv package manager
### .gitignore :
specifies intentionally untracked files to ignore for clean version control
### notes.txt :
local scratchpad for tracking next steps and project to-dos


## Core Execution (src/)
### main.py :
Primary entry point for training. Configures PyTorch Lightning Trainer, initializes the fMRIDataModule to handle batching and splitting, sets up WandbLogger for experiment tracking, establishes ModelCheckpoint callbacks to save the best-performing models.
### test_pipeline.py :
A diagnostic script that verifies the functionality of the model and loss engines by running a simulated forward and backward pass using small, randomly generated dummy fMRI tensors.


## Data Handling (src/data/)
### dataset.py :
Contains the custom fMRIDataset class. It uses nibabel to load 4D BOLD signals and 3D anatomical masks (Grey Matter, White Matter, CSF). The script performs critical pre-processing, including global Z-score normalization, sequence length padding/truncation (enforcing 40 timeframes), and tensor merging using einops to match BOLD and mask dimensions.
### preprocess.py :
Contains a preprocessing script acting as one-time data preparation step. Loops through all patient folders, converts the structual DICOMs to a T1 NIfTI, run 


## Model Architecture (src/models/)
### mae_core.py :
defines the central HierarchicalSwinMAE Autoencoder. It ties together the patch embedding, the 4-stage Swin Transformer encoder pyramid, and the cascading decoder into a single, cohesive architecture.
### lightning_model.py :
The PyTorch Lightning wrapper (fMRIDenoisingLightning). It defines the specific training and validation step logic, dynamic cropping to match prediction and target dimensions, and configures the optimizer (AdamW) with a Cosine Annealing learning rate scheduler.
### decoder.py :
Contains the CascadingDecoder class. It walks the 4-stage feature representations back up the pyramid using 3D transposed convolutions, progressively inflating the tokens back into original voxel space.
### patch_embed.py :
Contains the logic responsible for tokenizing the complex 4D fMRI data so it can be processed by the Transformer blocks.
### swift_official.py :
Houses the official Swin Transformer blocks (BasicLayer) and merging configurations (MERGING_MODE) utilized by the core MAE encoder stages.

## Utilities & Operations (src/utils/)
### masking.py :
Contains the core logic for the masking strategy (e.g. apply_swin_mask), ensuring the specific 75% tube masking ratio is properly applied to tokens before they hit the encoder.
### losses.py :
Defines the custom dual-engine loss function (fMRIMaskedLoss) which computes and combines the Masked Mean Squared Error (MSE) and Temporal Fast Fourier Transform (FFT) loss to guide reconstruction.

## First NSD training run

Preprocess each subject into a portable manifest, then launch the canonical
training entry point from the repository root:

```powershell
uv run python -m src.data_preprocessing.nsd.preprocess `
  --raw-root "C:\path\to\nsd_data" `
  --output-root "C:\path\to\nsd_processed" `
  --subject subj01 `
  --window-length 40 `
  --stride 20

uv run python -m src.train_nsd `
  --data_dir "C:\path\to\nsd_processed" `
  --crop_size 64 64 64 `
  --epochs 1 `
  --num_workers 4 `
  --precision 32-true
```

The training script splits complete subject/run groups rather than overlapping
temporal windows, logs every reconstruction and structural loss component,
saves the last and best checkpoints, and writes fixed validation slice panels
to `validation_visualizations/`. Use `--use_wandb` to additionally log metrics
to Weights & Biases.

The decomposition remains an unsupervised research hypothesis: reconstruction
alone cannot uniquely identify clean and noise components. Inspect the
clean/noise energy metrics and validation panels before interpreting a run as
successful denoising.