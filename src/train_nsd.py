"""Training script for the Natural Scenes Dataset (NSD) Masked Autoencoder.

Run from the project root (recommended):
    uv run python -m src.train_nsd --data_dir "C:/path/to/nsd_root" --batch_size 1 --epochs 100

If running directly from the src/ folder:
    uv run python train_nsd.py --data_dir "C:/path/to/nsd_root"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, Subset

try:
    from .data.nsd_dataset import NSDDataset
    from .models.lightning_model import fMRIDenoisingLightning
    from .utils.visualization import ValidationVisualizationCallback
except ImportError:
    from data.nsd_dataset import NSDDataset
    from models.lightning_model import fMRIDenoisingLightning
    from utils.visualization import ValidationVisualizationCallback


class NSDDataModule(L.LightningDataModule):
    """LightningDataModule coordinating the loading and splitting of 4D fMRI tensors.

    Args:
        data_dir: Path to the preprocessed NSD root directory containing subject manifests.
        batch_size: Number of temporal windows per batch. Defaults to 1 due to 4D tensor size.
        time_frames: Sequence length (T) for the sliding window. Defaults to 40.
        num_workers: Subprocesses for data loading.
        val_split: Fraction of the dataset reserved for validation.
        subjects: Optional list of specific subject IDs to load (e.g., ["subj01"]).
    """
    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 1,
        time_frames: int = 40,
        num_workers: int = 4,
        val_split: float = 0.2,
        subjects: list[str] | None = None,
        crop_size: tuple[int, int, int] | None = None,
        seed: int = 1234,
    ) -> None:
        """Initialize dataloader hyperparameters."""
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.time_frames = time_frames
        self.num_workers = num_workers
        self.val_split = val_split
        self.subjects = subjects
        self.crop_size = crop_size
        self.seed = seed

    def setup(self, stage: str | None = None) -> None:
        """Parse JSON manifests and split the dataset into training and validation sets."""
        self.full_dataset = NSDDataset(
            self.data_dir,
            subjects=self.subjects,
            time_frames=self.time_frames,
            crop_size=self.crop_size,
        )

        total_len = len(self.full_dataset)
        if total_len == 0:
            raise RuntimeError(f"No subjects found in {self.data_dir}")

        groups: dict[tuple[str, str], list[int]] = {}
        for index, entry in enumerate(self.full_dataset.entries):
            key = (str(entry["subject"]), str(entry["run"]))
            groups.setdefault(key, []).append(index)

        generator = torch.Generator().manual_seed(self.seed)
        group_keys = list(groups)
        if len(group_keys) < 2:
            raise ValueError("At least two subject/run groups are required for validation.")
        permutation = torch.randperm(len(group_keys), generator=generator).tolist()
        val_groups = min(len(group_keys) - 1, max(1, round(len(group_keys) * self.val_split)))
        val_keys = {group_keys[i] for i in permutation[:val_groups]}
        val_indices = [i for key, indices in groups.items() if key in val_keys for i in indices]
        train_indices = [i for key, indices in groups.items() if key not in val_keys for i in indices]
        if not train_indices:
            raise ValueError("The subject/run split left no training examples.")
        self.train_set = Subset(self.full_dataset, train_indices)
        self.val_set = Subset(self.full_dataset, val_indices)

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader with pinned memory for fast GPU transfers."""
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="NSD root folder containing subject subfolders")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--time_frames", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--crop_size", nargs=3, type=int, default=None)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--precision", default="16-mixed")

    # Logging & Checkpointing params
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--project", default="7T-fMRI-Denoising")
    parser.add_argument("--name", default="SwinMAE-NSD")
    parser.add_argument("--checkpoint_dir", default="checkpoints_nsd")
    parser.add_argument("--visualization_dir", default="validation_visualizations")
    parser.add_argument("--visualize_every_n_epochs", type=int, default=1)

    return parser.parse_args()


def main() -> None:
    """Execute the PyTorch Lightning training pipeline."""
    args = parse_args()
    L.seed_everything(args.seed, workers=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("Preparing data module...")
    datamodule = NSDDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        time_frames=args.time_frames,
        num_workers=args.num_workers,
        crop_size=tuple(args.crop_size) if args.crop_size else None,
        seed=args.seed,
    )
    # Explicit setup call ensures dataset metrics are available before model init if needed
    datamodule.setup()

    print("Initializing model...")
    model = fMRIDenoisingLightning(
        in_chans=4,
        patch_size=14,
        embed_dim=96,
        mask_ratio=0.75,
        lr=args.lr,
    )

    logger = None
    if args.use_wandb:
        logger = WandbLogger(project=args.project, name=args.name, log_model="all")

    # Saves the top 3 models based on lowest validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        dirpath=checkpoint_dir,
        filename="swinmae-nsd-{epoch:02d}-{val_loss:.4f}",
        save_last=True,
    )
    visualization_callback = ValidationVisualizationCallback(
        output_dir=args.visualization_dir,
        every_n_epochs=args.visualize_every_n_epochs,
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[checkpoint_callback, visualization_callback],
        accelerator="auto",
        devices="auto",
        # 16-mixed precision halves VRAM usage while maintaining gradient stability
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=1,
    )

    print("Starting training...")
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
