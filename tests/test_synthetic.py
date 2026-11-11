"""Smoke tests and CI verificatons for the fMRI denoising pipeline."""

from __future__ import annotations

import lightning as L
import torch
from torch.utils.data import DataLoader, random_split

from data.synthetic import SyntheticNSDDataset
from models.lightning_model import fMRIDenoisingLightning
from train_nsd import NSDDataModule
from utils.losses import StructuralDenoisingLoss
from utils.visualization import crop_to_input


def test_visualization_crops_padded_predictions() -> None:
    sample = torch.zeros(4, 8, 30, 27, 19)
    padded_prediction = torch.ones(1, 1, 8, 32, 32, 24)

    cropped = crop_to_input(padded_prediction, sample)

    assert cropped.shape == (1, 1, 8, 30, 27, 19)
    assert torch.equal(cropped, torch.ones_like(cropped))


def test_synthetic_dataset_shape() -> None:
    """Verify the synthetic dataset generates correct 5D tensor shapes."""
    ds = SyntheticNSDDataset(n_subjects=2, time_frames=8, shape=(16, 16, 8), seed=1)

    assert len(ds) == 2

    item = ds[0]
    assert isinstance(item, torch.Tensor)

    # Expected shape: [Channels (BOLD+GM+WM+CSF), Time, X, Y, Z]
    assert item.shape == (4, 8, 16, 16, 8)


def test_structural_loss_fft_supports_half_precision_40_frames() -> None:
    clean = torch.randn(1, 1, 40, 2, 2, 2, dtype=torch.float16, requires_grad=True)
    noise = torch.randn_like(clean, requires_grad=True)
    target_bold = torch.randn_like(clean)
    gm = torch.ones_like(clean)
    wm = torch.zeros_like(clean)
    csf = torch.zeros_like(clean)
    target = torch.cat((target_bold, gm, wm, csf), dim=1)

    total, _, _, _, fft = StructuralDenoisingLoss()(clean, noise, target)

    assert total.dtype == torch.float32
    assert torch.isfinite(total)
    assert torch.isfinite(fft)
    total.backward()
    assert clean.grad is not None and torch.isfinite(clean.grad).all()
    assert noise.grad is not None and torch.isfinite(noise.grad).all()


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


def test_nsd_split_holds_out_complete_subjects(monkeypatch) -> None:
    class FakeNSDDataset:
        def __init__(self, *args, **kwargs) -> None:
            self.entries = [
                {"subject": subject, "run": f"run{run}"}
                for subject in ("subj01", "subj02", "subj03")
                for run in (1, 2)
            ]

        def __len__(self) -> int:
            return len(self.entries)

    monkeypatch.setattr("train_nsd.NSDDataset", FakeNSDDataset)
    datamodule = NSDDataModule("unused", validation_subjects=["subj03"])
    datamodule.setup()

    train_subjects = {
        datamodule.full_dataset.entries[index]["subject"]
        for index in datamodule.train_set.indices
    }
    val_subjects = {
        datamodule.full_dataset.entries[index]["subject"]
        for index in datamodule.val_set.indices
    }
    assert train_subjects == {"subj01", "subj02"}
    assert val_subjects == {"subj03"}