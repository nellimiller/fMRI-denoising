# fMRI-denoising
Deep Learning-Based Denoising of Ultra-High-Field 7 Tesla Functional Magnetic Resonance Imaging Data

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
    │   └── dataset.py
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
- pyproject.toml & uv.lock : define project dependencies ensuring reproducible environment using uv package manager
- .gitignore : specifies intentionally untracked files to ignore for clean version control
- notes.txt : local scratchpad for tracking next steps and project to-dos

## Core Execution (src/)
- main.py : Primary entry point for training. Configures PyTorch Lightning Trainer, initializes the fMRIDataModule to handle batching and splitting, sets up WandbLogger for experiment tracking, establishes ModelCheckpoint callbacks to save the best-performing models.
-test_pipeline.py : A diagnostic script that verifies the functionality of the model and loss engines by running a simulated forward and backward pass using small, randomly generated dummy fMRI tensors.

## Data Handling (src/data/)
- dataset.py : Contains the custom fMRIDataset class. It uses nibabel to load 4D BOLD signals and 3D anatomical masks (Grey Matter, White Matter, CSF). The script performs critical pre-processing, including global Z-score normalization, sequence length padding/truncation (enforcing 40 timeframes), and tensor merging using einops to match BOLD and mask dimensions.

## Model Architecture (src/models/)
- mae_core.py : defines the central HierarchicalSwinMAE Autoencoder. It ties together the patch embedding, the 4-stage Swin Transformer encoder pyramid, and the cascading decoder into a single, cohesive architecture.
- lightning_model.py : The PyTorch Lightning wrapper (fMRIDenoisingLightning). It defines the specific training and validation step logic, dynamic cropping to match prediction and target dimensions, and configures the optimizer (AdamW) with a Cosine Annealing learning rate scheduler.
- decoder.py : Contains the CascadingDecoder class. It walks the 4-stage feature representations back up the pyramid using 3D transposed convolutions, progressively inflating the tokens back into original voxel space.
- patch_embed.py : Contains the logic responsible for tokenizing the complex 4D fMRI data so it can be processed by the Transformer blocks.
- swift_official.py : Houses the official Swin Transformer blocks (BasicLayer) and merging configurations (MERGING_MODE) utilized by the core MAE encoder stages.

## Utilities & Operations (src/utils/)
- maksing.py : Contains the core logic for the masking strategy (e.g. apply_swin_mask), ensuring the specific 75% tube masking ratio is properly applied to tokens before they hit the encoder.
- losses.py : Defines the custom dual-engine loss function (fMRIMaskedLoss) which computes and combines the Masked Mean Squared Error (MSE) and Temporal Fast Fourier Transform (FFT) loss to guide reconstruction.