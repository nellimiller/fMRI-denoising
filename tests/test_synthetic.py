"""Smoke tests and CI verificatons for the fMRI denoising pipeline."""

from __future__ import annotations

import lightning as L
import torch
from torch.utils.data import DataLoader, random_split

from data.synthetic import SyntheticNSDDataset
from models.lightning_model import fMRIDenoisingLightning


def test_synthetic_dataset_shape() -> None:
    """Verify the synthetic dataset generates correct 5D tensor shapes."""
    ds = SyntheticNSDDataset(n_subjects=2, time_frames=8, shape=(16, 16, 8), seed=1)

    assert len(ds) == 2

    item = ds[0]
    assert isinstance(item, torch.Tensor)

    # Expected shape: [Channels (BOLD+GM+WM+CSF), Time, X, Y, Z]
    assert item.shape == (4, 8, 16, 16, 8)


class SyntheticModule(L.LightningDataModule):
    """LightningDataModule wrapper for the SyntheticNSDDataset.
    
    Handles training/validation spilts and DataLoader instantiation for testing.
    
    Args:
        dataset: The initialized synthetic dataset.
        batch_size: Number of subjects per batch. Defaults to 1.
    """

    def __init__(self, dataset, batch_size: int = 1) -> None:
        """Initialize the data module."""
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def setup(self, stage: str | None = None) -> None:
        """Split the dataset into an 80/20 train/validation split."""
        total = len(self.dataset)
        val = max(1, int(total * 0.2))
        train = total - val

        # random_split ensures the network sees different synthetic subjects during validation
        self.train_set, self.val_set = random_split(self.dataset, [train, val])

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        return DataLoader(self.val_set, batch_size=self.batch_size)


def test_training_smoke_cpu() -> None:
    """Run a full forward, backward, and loss computation pass on CPU.
    
    This 'smoke test' ensures that the entire Lighning module, Swin Transformer,
    and dual-engine loss functions compile and execute successfully without
    requiring a GPU or massive NIfTI files."""
    ds = SyntheticNSDDataset(n_subjects=4, time_frames=8, shape=(16, 16, 8), seed=2)
    dm = SyntheticModule(ds, batch_size=1)
    dm.setup()

    # embed_dim=96 aligns cleanly with SwiFT num_heads (must be divisible by 3, 6, 12, 24)
    model = fMRIDenoisingLightning(
        in_chans=4,
        patch_size=4,
        embed_dim=96,
        mask_ratio=0.5,
        lr=1e-3
    )

    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        limit_train_batches=1,
        limit_val_batches=1,
        logger=False,
        enable_checkpointing=False,
        precision=32,
    )

    # If trainer.fit completes without raising an exception (e.g., shape mismatches
    # or OOM errors), the core pipeline geometry is functionally sound.
    # Should run without raising
    trainer.fit(model, datamodule=dm)