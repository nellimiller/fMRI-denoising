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
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

from data.nsd_dataset import NSDDataset
from models.lightning_model import fMRIDenoisingLightning


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
    ) -> None:
        """Initialize dataloader hyperparameters."""
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.time_frames = time_frames
        self.num_workers = num_workers
        self.val_split = val_split
        self.subjects = subjects

    def setup(self, stage: str | None = None) -> None:
        """Parse JSON manifests and split the dataset into training and validation sets."""
        self.full_dataset = NSDDataset(
            self.data_dir,
            subjects=self.subjects,
            time_frames=self.time_frames
        )

        total_len = len(self.full_dataset)
        if total_len == 0:
            raise RuntimeError(f"No subjects found in {self.data_dir}")

        val_len = int(total_len * self.val_split)
        train_len = total_len - val_len

        # Guardrail for extremely small dataset testing
        if val_len == 0:
            val_len = 1
            train_len = total_len - 1

        self.train_set, self.val_set = random_split(self.full_dataset, [train_len, val_len])

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

    # Logging & Checkpointing params
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--project", default="7T-fMRI-Denoising")
    parser.add_argument("--name", default="SwinMAE-NSD")
    parser.add_argument("--checkpoint_dir", default="checkpoints_nsd")

    return parser.parse_args()


def main() -> None:
    """Execute the PyTorch Lightning training pipeline."""
    args = parse_args()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("Preparing data module...")
    datamodule = NSDDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        time_frames=args.time_frames,
        num_workers=args.num_workers,
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
        dirpath=args.checkpoint_dir,
        filename="swinmae-nsd-{epoch:02d}-{val_loss:.4f}",
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        devices="auto",
        # 16-mixed precision halves VRAM usage while maintaining gradient stability
        precision="16-mixed",
        log_every_n_steps=1,
    )

    print("Starting training...")
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
