"""
Training script for Natural Scenes Dataset (NSD)

Run from the project root (recommended):
    uv run python -m src.train_nsd --data_dir C:/path/to/nsd_root --batch_size 1 --epochs 100

If you prefer running from the src/ folder use:
    uv run python train_nsd.py --data_dir C:/path/to/nsd_root

The script constructs a Lightning DataModule that uses data.nsd_dataset.NSDDataset and trains the fMRIDenoisingLightning model.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split

from data.nsd_dataset import NSDDataset
from models.lightning_model import fMRIDenoisingLightning


class NSDDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 1,
        time_frames: int = 40,
        num_workers: int = 4,
        val_split: float = 0.2,
        subjects: Optional[list] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.time_frames = time_frames
        self.num_workers = num_workers
        self.val_split = val_split
        self.subjects = subjects

    def setup(self, stage=None):
        # Build the dataset
        self.full_dataset = NSDDataset(self.data_dir, subjects=self.subjects, time_frames=self.time_frames)

        total_len = len(self.full_dataset)
        if total_len == 0:
            raise RuntimeError(f"No subjects found in {self.data_dir}")

        val_len = int(total_len * self.val_split)
        train_len = total_len - val_len
        if val_len == 0:
            # ensure at least one validation example
            val_len = 1
            train_len = total_len - 1

        self.train_set, self.val_set = random_split(self.full_dataset, [train_len, val_len])

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="NSD root folder containing subject subfolders")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--time_frames", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--project", default="7T-fMRI-Denoising")
    parser.add_argument("--name", default="SwinMAE-NSD")
    parser.add_argument("--checkpoint_dir", default="checkpoints_nsd")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("Preparing data module...")
    datamodule = NSDDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        time_frames=args.time_frames,
        num_workers=args.num_workers,
    )
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
        precision="16-mixed",
        log_every_n_steps=1,
    )

    print("Starting training...")
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
