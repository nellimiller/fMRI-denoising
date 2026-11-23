"""Training script for the Natural Scenes Dataset (NSD) Masked Autoencoder.

Run from the project root (recommended)::

    uv run python -m src.train_nsd --data_dir "C:/path/to/nsd_root" --batch_size 1 --epochs 100 \
        --subjects subj01 subj02 subj07 --validation_subjects subj07 --test_subjects subj08

Every run writes ``hparams.json`` (all resolved arguments, the subject split, dataset
sizes, TR values, model hyperparameters and software versions) next to the checkpoints.
After training, the best checkpoint is evaluated with J-invariant K-pass inference on the
validation and test subjects (unless ``--skip_final_eval``).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
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
        val_split: Fraction of subjects reserved for validation.
        subjects: Optional list of specific subject IDs to load (e.g., ["subj01"]).
        validation_subjects: Optional subject IDs to hold out from training.
        crop_size: Optional spatial crop ``(X, Y, Z)``.
        seed: Seed for the automatic subject-level split.
        detrend: Linear detrending mode of the PSC target (see :class:`NSDDataset`).
        tr_override: Repetition time (s) replacing the NIfTI header value.
    """

    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 1,
        time_frames: int = 40,
        num_workers: int = 4,
        val_split: float = 0.2,
        subjects: list[str] | None = None,
        validation_subjects: list[str] | None = None,
        crop_size: tuple[int, int, int] | None = None,
        seed: int = 1234,
        detrend: str = "run",
        tr_override: float | None = None,
    ) -> None:
        """Initialize dataloader hyperparameters."""
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.time_frames = time_frames
        self.num_workers = num_workers
        self.val_split = val_split
        self.subjects = subjects
        self.validation_subjects = validation_subjects
        self.crop_size = crop_size
        self.seed = seed
        self.detrend = detrend
        self.tr_override = tr_override
        self.train_subjects: list[str] = []
        self.val_subjects: list[str] = []

    def setup(self, stage: str | None = None) -> None:
        """Parse JSON manifests and split the dataset into training and validation sets."""
        self.full_dataset = NSDDataset(
            self.data_dir,
            subjects=self.subjects,
            time_frames=self.time_frames,
            crop_size=self.crop_size,
            detrend=self.detrend,
            tr_override=self.tr_override,
        )

        total_len = len(self.full_dataset)
        if total_len == 0:
            raise RuntimeError(f"No subjects found in {self.data_dir}")

        subject_indices: dict[str, list[int]] = {}
        for index, entry in enumerate(self.full_dataset.entries):
            subject_indices.setdefault(str(entry["subject"]), []).append(index)

        available_subjects = sorted(subject_indices)
        if len(available_subjects) < 2:
            raise ValueError("At least two subjects are required for unseen-subject validation.")

        if self.validation_subjects is not None:
            val_subjects = set(self.validation_subjects)
            missing_subjects = val_subjects - set(available_subjects)
            if missing_subjects:
                raise ValueError(
                    f"Validation subjects have no dataset windows: {sorted(missing_subjects)}"
                )
            if not val_subjects or val_subjects == set(available_subjects):
                raise ValueError("Validation must include at least one, but not all, subjects.")
        else:
            generator = torch.Generator().manual_seed(self.seed)
            permutation = torch.randperm(len(available_subjects), generator=generator).tolist()
            val_count = min(
                len(available_subjects) - 1,
                max(1, round(len(available_subjects) * self.val_split)),
            )
            val_subjects = {available_subjects[i] for i in permutation[:val_count]}

        train_subjects = set(available_subjects) - val_subjects
        val_indices = [
            index for subject in sorted(val_subjects) for index in subject_indices[subject]
        ]
        train_indices = [
            index for subject in sorted(train_subjects) for index in subject_indices[subject]
        ]
        if not train_indices:
            raise ValueError("The subject split left no training examples.")
        self.train_subjects = sorted(train_subjects)
        self.val_subjects = sorted(val_subjects)
        print(f"Subject split: train={self.train_subjects}; validation={self.val_subjects}")
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True, help="NSD root folder containing subject subfolders")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--time_frames", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--subjects", nargs="+", default=None, help="Training + validation subject IDs (default: all manifests)")
    parser.add_argument(
        "--validation_subjects",
        nargs="+",
        default=None,
        help="Subjects held out entirely for validation (default: seeded subject-level split)",
    )
    parser.add_argument(
        "--test_subjects",
        nargs="*",
        default=[],
        help="Subjects never used for training or checkpoint selection; evaluated after training",
    )
    parser.add_argument("--crop_size", nargs=3, type=int, default=None)
    parser.add_argument("--detrend", default="run", choices=["run", "window", "run+window", "none"])
    parser.add_argument("--tr", type=float, default=None, help="Repetition time override in seconds (default: NIfTI header)")
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--embed_dim", type=int, default=96)
    parser.add_argument("--full_res_channels", type=int, default=24)
    parser.add_argument(
        "--decoder_skips",
        default="full",
        choices=["full", "none"],
        help="Ablation: 'none' removes the stage skips and voxel stem (bottleneck-only decoder)",
    )
    parser.add_argument(
        "--free_noise_head",
        action="store_true",
        help="Ablation: free per-frame noise head instead of the regressor-subspace head",
    )
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    # Identifiable objective
    parser.add_argument("--rec_loss_weight", type=float, default=1.0)
    parser.add_argument("--tissue_loss_weight", type=float, default=1.0)
    parser.add_argument("--band_loss_weight", type=float, default=1.0)
    parser.add_argument("--ortho_loss_weight", type=float, default=0.1)
    parser.add_argument("--n_comp", type=int, default=3, help="aCompCor components in the nuisance basis")
    parser.add_argument("--band_low", type=float, default=0.01)
    parser.add_argument("--band_high", type=float, default=0.15)
    parser.add_argument("--erosion_iterations", type=int, default=1)
    parser.add_argument(
        "--soft_bandpass",
        action="store_true",
        help="Ablation: no architectural GM gate / bandpass; the loss alone enforces the priors",
    )
    parser.add_argument(
        "--prior_warmup_steps",
        type=int,
        default=0,
        help="Ramp tissue/band/ortho loss terms from 0 to full weight over this many steps (0 = off)",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Stop after this many validation epochs without improvement; 0 disables it.",
    )

    # Final evaluation
    parser.add_argument("--skip_final_eval", action="store_true")
    parser.add_argument("--eval_K", type=int, default=4, help="Complementary masks for inference")
    parser.add_argument("--final_eval_windows_per_subject", type=int, default=24, help="0 = all windows")
    parser.add_argument("--eval_dir", default=None, help="Default: <checkpoint_dir>/final_evaluation")

    # Logging & Checkpointing params
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--project", default="7T-fMRI-Denoising")
    parser.add_argument("--name", default="SwinMAE-NSD")
    parser.add_argument("--log_dir", default="logs_nsd")
    parser.add_argument("--checkpoint_dir", default="checkpoints_nsd")
    parser.add_argument("--visualization_dir", default="validation_visualizations")
    parser.add_argument("--visualize_every_n_epochs", type=int, default=1)

    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid hyperparameters early."""
    if not 0.0 <= args.mask_ratio < 1.0:
        raise ValueError("--mask_ratio must be in [0, 1).")
    if args.early_stopping_patience < 0:
        raise ValueError("--early_stopping_patience cannot be negative.")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("--lr must be finite and positive.")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("--weight_decay must be finite and non-negative.")
    weights = (
        args.rec_loss_weight,
        args.tissue_loss_weight,
        args.band_loss_weight,
        args.ortho_loss_weight,
    )
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("Loss weights must be finite and non-negative.")
    if not 0 < args.band_low < args.band_high:
        raise ValueError("Require 0 < --band_low < --band_high.")
    if args.prior_warmup_steps < 0:
        raise ValueError("--prior_warmup_steps cannot be negative.")
    if args.n_comp < 0 or args.erosion_iterations < 0:
        raise ValueError("--n_comp and --erosion_iterations cannot be negative.")
    overlap = set(args.test_subjects) & set(args.subjects or [])
    if overlap:
        raise ValueError(f"Test subjects must not be in --subjects: {sorted(overlap)}")


def _git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[1]
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best effort
        return None


def write_hparams(
    path: Path,
    args: argparse.Namespace,
    datamodule: NSDDataModule,
    model: fMRIDenoisingLightning,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the fully resolved run configuration as JSON and return it."""
    entries = datamodule.full_dataset.entries
    record: dict[str, Any] = {
        "args": vars(args),
        "split": {
            "train_subjects": datamodule.train_subjects,
            "validation_subjects": datamodule.val_subjects,
            "test_subjects": list(args.test_subjects),
        },
        "dataset": {
            "num_train_windows": len(datamodule.train_set),
            "num_validation_windows": len(datamodule.val_set),
            "tr_seconds": sorted({float(e.get("tr", args.tr or 0.0)) for e in entries}),
            "tr_override": args.tr,
            "detrend": args.detrend,
        },
        "model_hparams": dict(model.hparams),
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "lightning": L.__version__,
            "git_commit": _git_commit(),
        },
        "command": sys.argv,
    }
    if extra:
        record.update(extra)
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return record


def run_final_evaluation(args: argparse.Namespace, checkpoint: Path, datamodule: NSDDataModule) -> dict[str, Any]:
    """Evaluate ``checkpoint`` on the validation and test subjects with K-pass inference."""
    try:
        from . import evaluate_nsd
    except ImportError:
        import evaluate_nsd

    eval_dir = Path(args.eval_dir) if args.eval_dir else Path(args.checkpoint_dir) / "final_evaluation"
    namespace = argparse.Namespace(
        data_dir=args.data_dir,
        checkpoint=str(checkpoint),
        subjects=list(args.test_subjects),
        validation_subjects=list(datamodule.val_subjects),
        train_visual_subjects=datamodule.train_subjects[:1],
        output_dir=str(eval_dir),
        time_frames=args.time_frames,
        crop_size=args.crop_size,
        num_workers=args.num_workers,
        K=args.eval_K,
        mask_repeats=1,
        seed=args.seed,
        max_windows_per_subject=args.final_eval_windows_per_subject,
        visual_examples_per_subject=3,
        detrend=args.detrend,
        tr=args.tr,
        amp=False,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return evaluate_nsd.run_evaluation(namespace)


def main(argv: list[str] | None = None) -> None:
    """Execute the PyTorch Lightning training pipeline."""
    args = parse_args(argv)
    validate_args(args)

    L.seed_everything(args.seed, workers=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing data module...")
    datamodule = NSDDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        time_frames=args.time_frames,
        num_workers=args.num_workers,
        subjects=args.subjects,
        validation_subjects=args.validation_subjects,
        crop_size=tuple(args.crop_size) if args.crop_size else None,
        seed=args.seed,
        detrend=args.detrend,
        tr_override=args.tr,
    )
    # Explicit setup call ensures dataset metrics are available before model init if needed
    datamodule.setup()

    print("Initializing model...")
    model = fMRIDenoisingLightning(
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        mask_ratio=args.mask_ratio,
        lr=args.lr,
        weight_decay=args.weight_decay,
        rec_loss_weight=args.rec_loss_weight,
        tissue_loss_weight=args.tissue_loss_weight,
        band_loss_weight=args.band_loss_weight,
        ortho_loss_weight=args.ortho_loss_weight,
        n_comp=args.n_comp,
        band_low=args.band_low,
        band_high=args.band_high,
        erosion_iterations=args.erosion_iterations,
        hard_bandpass=not args.soft_bandpass,
        full_res_channels=args.full_res_channels,
        decoder_skips=args.decoder_skips,
        free_noise_head=args.free_noise_head,
        prior_warmup_steps=args.prior_warmup_steps,
        validation_seed=args.seed,
    )

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    hparams_path = checkpoint_dir / "hparams.json"
    write_hparams(hparams_path, args, datamodule, model)
    print(f"Wrote resolved hyperparameters to {hparams_path}")

    loggers = [CSVLogger(save_dir=str(log_dir), name=args.name)]
    if args.use_wandb:
        loggers.append(WandbLogger(project=args.project, name=args.name))

    # The Lightning module logs the combined objective as val_total.
    checkpoint_callback = ModelCheckpoint(
        monitor="val_total",
        mode="min",
        save_top_k=3,
        dirpath=checkpoint_dir,
        filename="swinmae-nsd-{epoch:02d}-{val_total:.4f}",
        save_last=True,
    )
    visualization_callback = ValidationVisualizationCallback(
        output_dir=args.visualization_dir,
        every_n_epochs=args.visualize_every_n_epochs,
        K=args.eval_K,
        seed=args.seed,
    )
    callbacks = [checkpoint_callback, visualization_callback]
    if args.early_stopping_patience:
        callbacks.append(
            EarlyStopping(
                monitor="val_total",
                mode="min",
                patience=args.early_stopping_patience,
                check_finite=True,
                verbose=True,
            )
        )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        logger=loggers,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        # 16-mixed precision halves VRAM usage while maintaining gradient stability
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=1,
    )

    print("Starting training...")
    trainer.fit(model, datamodule=datamodule)

    if not trainer.is_global_zero:
        return
    best = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    extra: dict[str, Any] = {
        "best_checkpoint": best,
        "best_val_total": (
            float(checkpoint_callback.best_model_score)
            if checkpoint_callback.best_model_score is not None
            else None
        ),
    }
    if best and not args.skip_final_eval:
        print(f"Running final K={args.eval_K} evaluation on validation/test subjects with {best}")
        summary = run_final_evaluation(args, Path(best), datamodule)
        extra["final_evaluation"] = {
            "dir": str(Path(args.eval_dir) if args.eval_dir else checkpoint_dir / "final_evaluation"),
            "splits": summary.get("splits"),
        }
    write_hparams(hparams_path, args, datamodule, model, extra)


if __name__ == "__main__":
    main()
