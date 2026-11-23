"""Smoke tests and CI verifications for the fMRI denoising pipeline."""

from __future__ import annotations

import json

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader, random_split

from data.channels import BRAIN, DYN, NUM_CHANNELS
from data.synthetic import SyntheticNSDDataset
from models.lightning_model import fMRIDenoisingLightning, split_weight_decay_groups
from train_nsd import NSDDataModule, parse_args, validate_args, write_hparams
from utils.visualization import crop_to_input


def test_visualization_crops_padded_predictions() -> None:
    sample = torch.zeros(6, 8, 30, 27, 19)
    padded_prediction = torch.ones(1, 1, 8, 32, 32, 24)

    cropped = crop_to_input(padded_prediction, sample)

    assert cropped.shape == (1, 1, 8, 30, 27, 19)
    assert torch.equal(cropped, torch.ones_like(cropped))


def test_synthetic_dataset_matches_real_sample_contract() -> None:
    ds = SyntheticNSDDataset(n_subjects=2, time_frames=8, shape=(16, 16, 8), seed=1)

    assert len(ds) == 2
    item = ds[0]
    assert item["x"].shape == (NUM_CHANNELS, 8, 16, 16, 8)
    assert float(item["tr"]) == 1.0
    brain = item["x"][BRAIN, 0] > 0
    assert (item["x"][DYN][:, ~brain] == 0).all()


class SyntheticModule(L.LightningDataModule):
    """LightningDataModule wrapper for the SyntheticNSDDataset."""

    def __init__(self, dataset, batch_size: int = 1) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def setup(self, stage: str | None = None) -> None:
        total = len(self.dataset)
        val = max(1, int(total * 0.2))
        self.train_set, self.val_set = random_split(self.dataset, [total - val, val])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=self.batch_size)


def test_training_smoke_cpu() -> None:
    """One Lightning fit step (train + validation) of the full model on CPU."""
    ds = SyntheticNSDDataset(n_subjects=4, time_frames=8, shape=(16, 16, 8), seed=2)
    dm = SyntheticModule(ds, batch_size=1)
    dm.setup()

    model = fMRIDenoisingLightning(patch_size=4, embed_dim=24, mask_ratio=0.5, lr=1e-3, n_comp=2)
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        limit_train_batches=1,
        limit_val_batches=1,
        logger=False,
        enable_checkpointing=False,
        precision=32,
        enable_progress_bar=False,
    )
    trainer.fit(model, datamodule=dm)
    logged = set(trainer.callback_metrics)
    for name in ("train_total", "train_rec", "train_tissue", "train_band", "train_ortho", "val_total", "val_rec"):
        assert name in logged, name
    assert not any(k.endswith(("_fft", "_flat", "_structural_total")) for k in logged)


def test_weight_decay_excludes_bias_norm_mask_token_and_relative_bias() -> None:
    model = fMRIDenoisingLightning(patch_size=4, embed_dim=24)
    groups = split_weight_decay_groups(model, 0.05)
    decay_ids = {id(p) for p in groups[0]["params"]}
    no_decay_ids = {id(p) for p in groups[1]["params"]}
    assert groups[0]["weight_decay"] == 0.05 and groups[1]["weight_decay"] == 0.0
    assert decay_ids.isdisjoint(no_decay_ids)
    assert decay_ids | no_decay_ids == {id(p) for p in model.parameters()}
    special = ("mask_token", "relative_position_bias_table", "norm")
    for name, parameter in model.named_parameters():
        in_no_decay = id(parameter) in no_decay_ids
        if name.endswith("bias") or any(s in name for s in special):
            assert in_no_decay, name
        elif parameter.ndim > 1:
            assert not in_no_decay, name


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

    train_subjects = {datamodule.full_dataset.entries[i]["subject"] for i in datamodule.train_set.indices}
    val_subjects = {datamodule.full_dataset.entries[i]["subject"] for i in datamodule.val_set.indices}
    assert train_subjects == {"subj01", "subj02"}
    assert val_subjects == {"subj03"}
    assert datamodule.train_subjects == ["subj01", "subj02"] and datamodule.val_subjects == ["subj03"]


def test_cli_rejects_invalid_configuration_and_overlapping_test_subjects() -> None:
    base = ["--data_dir", "x"]
    validate_args(parse_args(base))
    with pytest.raises(ValueError):
        validate_args(parse_args(base + ["--band_low", "0.2", "--band_high", "0.1"]))
    with pytest.raises(ValueError):
        validate_args(parse_args(base + ["--subjects", "subj01", "subj08", "--test_subjects", "subj08"]))
    with pytest.raises(ValueError):
        validate_args(parse_args(base + ["--tissue_loss_weight", "-1"]))


def test_hparams_json_records_resolved_configuration(tmp_path, monkeypatch) -> None:
    class FakeNSDDataset:
        def __init__(self, *args, **kwargs) -> None:
            self.entries = [{"subject": s, "run": "run1", "tr": 1.0} for s in ("subj01", "subj02", "subj07")]

        def __len__(self) -> int:
            return len(self.entries)

    monkeypatch.setattr("train_nsd.NSDDataset", FakeNSDDataset)
    args = parse_args(["--data_dir", "x", "--validation_subjects", "subj07", "--test_subjects", "subj08"])
    datamodule = NSDDataModule("x", validation_subjects=["subj07"])
    datamodule.setup()
    model = fMRIDenoisingLightning(patch_size=4, embed_dim=24)

    record = write_hparams(tmp_path / "hparams.json", args, datamodule, model, {"best_checkpoint": "a.ckpt"})
    loaded = json.loads((tmp_path / "hparams.json").read_text())
    assert loaded["split"] == {
        "train_subjects": ["subj01", "subj02"],
        "validation_subjects": ["subj07"],
        "test_subjects": ["subj08"],
    }
    assert loaded["args"]["ortho_loss_weight"] == 0.1 and loaded["model_hparams"]["n_comp"] == 3
    assert loaded["dataset"]["tr_seconds"] == [1.0] and loaded["best_checkpoint"] == "a.ckpt"
    assert record["software"]["torch"]
