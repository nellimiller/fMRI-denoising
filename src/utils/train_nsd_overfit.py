"""Small isolated NSD overfit test.

Run from ``src``:

    uv run python train_nsd_overfit.py \
      --data_dir "C:\\fMRI Denoising\\NSD\\nsd_processed" \
      --output_dir "C:\\fMRI Denoising\\NSD\\overfit_test" \
      --crop_size 64 64 32 --epochs 10

This intentionally repeats one temporal window. It is a diagnostic for data
loading, model shapes, gradient flow, and whether the loss can decrease. It
is not a meaningful denoising experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader, Dataset

from data.nsd_dataset import NSDDataset
from models.lightning_model import fMRIDenoisingLightning


class RepeatedDataset(Dataset):
    def __init__(self, sample: object, repeats: int) -> None:
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        self.sample = sample
        self.repeats = repeats

    def __len__(self) -> int:
        return self.repeats

    def __getitem__(self, index: int):
        return self.sample


class OverfitDataModule(L.LightningDataModule):
    def __init__(self, dataset: Dataset, repeats: int, batch_size: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.repeats = repeats
        self.batch_size = batch_size

    def setup(self, stage=None) -> None:
        sample = self.dataset[0]
        repeated = RepeatedDataset(sample, self.repeats)
        self.train_set = repeated
        self.val_set = RepeatedDataset(sample, 1)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
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
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = NSDDataset(
        args.data_dir,
        subjects=[args.subject],
        time_frames=args.time_frames,
        crop_size=tuple(args.crop_size),
    )
    if not 0 <= args.entry < len(dataset):
        raise IndexError(f"entry must be between 0 and {len(dataset) - 1}")

    # Select one real window, then repeat it deliberately for overfitting.
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
        limit_val_batches=1,
        log_every_n_steps=1,
        precision="32-true",
    )
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
