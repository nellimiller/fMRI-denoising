"""Small isolated NSD overfit test.

Run from the repository root:

    uv run python -m src.utils.train_nsd_overfit \
      --data_dir "C:\\fMRI Denoising\\NSD\\nsd_processed" \
      --output_dir "C:\\fMRI Denoising\\NSD\\overfit_test" \
      --time_frames 8 \
      --crop_size 64 64 32 --epochs 10

This intentionally repeats one temporal window. It is a diagnostic for data
loading, model shapes, gradient flow, and whether the loss can decrease. It
is not a meaningful denoising experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

try:
    from ..data.nsd_dataset import NSDDataset
    from ..models.lightning_model import fMRIDenoisingLightning
    from .visualization import ValidationVisualizationCallback
except ImportError:
    from data.nsd_dataset import NSDDataset
    from models.lightning_model import fMRIDenoisingLightning
    from utils.visualization import ValidationVisualizationCallback


class RepeatedDataset(Dataset):
    """A diagnostic dataset that endlessly repeats a single sample.
    
    Used to force the network to memorize a single target, verifying that 
    gradients flow end-to-end and loss can mathematically reach zero.
    
    Args:
        sample: The single tensor sample to repeat.
        repeats: How many times to yield the sample per epoch.
    """

    def __init__(self, sample: torch.Tensor, repeats: int) -> None:
        """Initialize the diagnostic dataset."""
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        self.sample = sample
        self.repeats = repeats

    def __len__(self) -> int:
        """Return the fixed number of repeats."""
        return self.repeats

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return the identical sample regardless of the index requested."""
        return self.sample


class OverfitDataModule(L.LightningDataModule):
    """LightningDataModule wrapper for the overfit testing pipeline.

    Args:
        dataset: A base dataset from which to extract the first sample.
        repeats: Number of times to repeat the extracted sample in training.
        batch_size: DataLoader batch size.
    """
    
    def __init__(self, dataset: Dataset, repeats: int, batch_size: int) -> None:
        """Initialize the data module."""
        super().__init__()
        self.dataset = dataset
        self.repeats = repeats
        self.batch_size = batch_size

    def setup(self, stage: str | None = None) -> None:
        """Extract a single sample and set up identical train/val sets."""
        sample = self.dataset[0]
        self.train_set = RepeatedDataset(sample, self.repeats)
        self.val_set = RepeatedDataset(sample, 1)

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        return DataLoader(self.val_set, batch_size=1, shuffle=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--subject", default="subj01")
    parser.add_argument("--entry", type=int, default=0)
    parser.add_argument("--time_frames", type=int, default=40)
    parser.add_argument("--crop_size", nargs=3, type=int, default=(64, 64, 32))
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=96)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    return parser.parse_args()


def main() -> None:
    """Execute the single-batch overfitting sanity check."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the real dataset just to extract a single valid crop
    dataset = NSDDataset(
        args.data_dir,
        subjects=[args.subject],
        time_frames=args.time_frames,
        crop_size=tuple(args.crop_size),
    )

    if not 0 <= args.entry < len(dataset):
        raise IndexError(f"entry must be between 0 and {len(dataset) - 1}")

    # Select one real temporal window, then repeat it deliberately for overfitting.
    one_sample = dataset[args.entry]
    print(f"Test sample shape: {tuple(one_sample.shape)}")

    data = OverfitDataModule(
        dataset=RepeatedDataset(one_sample, 1),
        repeats=args.repeats,
        batch_size=1,
    )
    data.setup()

    model = fMRIDenoisingLightning(
        in_chans=4,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        mask_ratio=args.mask_ratio,
        lr=args.lr,
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=True,
        default_root_dir=str(output_dir),
        callbacks=[
            ModelCheckpoint(save_last=True, save_top_k=0),
            ValidationVisualizationCallback(output_dir / "visualizations"),
        ],
        limit_val_batches=1,
        log_every_n_steps=1,
        precision="32-true",
    )

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()