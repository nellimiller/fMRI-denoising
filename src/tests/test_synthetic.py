import tempfile
import torch
import lightning as L
from torch.utils.data import DataLoader, random_split

from data.synthetic import SyntheticNSDDataset
from models.lightning_model import fMRIDenoisingLightning


def test_synthetic_dataset_shape():
    ds = SyntheticNSDDataset(n_subjects=2, time_frames=8, shape=(16, 16, 8), seed=1)
    assert len(ds) == 2
    item = ds[0]
    assert isinstance(item, torch.Tensor)
    assert item.shape == (4, 8, 16, 16, 8)


class SyntheticModule(L.LightningDataModule):
    def __init__(self, dataset, batch_size=1):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def setup(self, stage=None):
        total = len(self.dataset)
        val = max(1, int(total * 0.2))
        train = total - val
        self.train_set, self.val_set = random_split(self.dataset, [train, val])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)


def test_training_smoke_cpu():
    # Small synthetic dataset
    ds = SyntheticNSDDataset(n_subjects=4, time_frames=8, shape=(16, 16, 8), seed=2)
    dm = SyntheticModule(ds, batch_size=1)
    dm.setup()

    # Use embed_dim aligned with SwiFT num_heads (divisible by 3,6,12,24) -> 96 is a good small choice
    model = fMRIDenoisingLightning(in_chans=4, patch_size=4, embed_dim=96, mask_ratio=0.5, lr=1e-3)

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

    # Should run without raising
    trainer.fit(model, datamodule=dm)
    # If it completes, we consider the smoke test passed
