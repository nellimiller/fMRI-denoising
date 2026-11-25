"""PyTorch Lightning module for training the 4D fMRI Masked Autoencoder."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.optim as optim

try:
    from ..data.channels import NUM_CHANNELS
    from ..utils.losses import IdentifiableDenoisingLoss
    from .mae_core import HierarchicalSwinMAE
except ImportError:
    from data.channels import NUM_CHANNELS
    from models.mae_core import HierarchicalSwinMAE
    from utils.losses import IdentifiableDenoisingLoss


def split_weight_decay_groups(
    model: torch.nn.Module, weight_decay: float
) -> list[dict[str, Any]]:
    """AdamW parameter groups: no decay for biases, norm layers, the mask token and
    relative position bias tables (all 1-D parameters or explicitly named)."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        excluded = (
            parameter.ndim <= 1
            or name.endswith(".bias")
            or "mask_token" in name
            or "relative_position_bias_table" in name
            or "norm" in name.lower()
        )
        (no_decay if excluded else decay).append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class fMRIDenoisingLightning(L.LightningModule):
    """Lightning module coordinating the SwinMAE architecture and identifiable loss.

    Batches are dicts ``{"x": [B, 6, T, X, Y, Z], "tr": [B], ...}`` as produced by
    :class:`data.nsd_dataset.NSDDataset`.

    Args:
        in_chans: Input channels (6: ``dyn, static_anat, gm, wm, csf, brain``).
        patch_size: Base spatial patch size for the encoder.
        embed_dim: Initial embedding dimension for the Swin blocks.
        mask_ratio: Fraction of valid tubes hidden during training (``K = 1 / (1 - ratio)``
            complementary passes reproduce it at inference; 0.75 <-> K = 4).
        lr: Maximum learning rate for the Cosine Annealing schedule.
        weight_decay: L2 penalty for AdamW (not applied to biases / norms / mask token /
            relative position bias tables).
        rec_loss_weight, tissue_loss_weight, band_loss_weight, ortho_loss_weight: Loss weights.
        n_comp: aCompCor components in the nuisance basis.
        band_low, band_high: Hemodynamic pass band in Hz.
        erosion_iterations: Erosion of WM/CSF for regressors and the tissue penalty.
        hard_bandpass: Gate and band-limit the clean head architecturally.
        decoder_skips: ``"full"`` or ``"none"`` (ablation: bottleneck-only decoder).
        free_noise_head: Ablation: free per-frame noise head instead of ``sum_j beta_j r_j``.
        prior_warmup_steps: Linearly ramp the tissue / band / ortho terms from 0 to full
            weight over this many optimiser steps (0 = off). All three priors are minimised by
            an all-zero clean output, which can keep the clean head collapsed early in training.
        validation_seed: Base seed for reproducible validation masks.
    """

    def __init__(
        self,
        in_chans: int = NUM_CHANNELS,
        patch_size: int = 14,
        embed_dim: int = 96,
        mask_ratio: float = 0.75,
        lr: float = 1.5e-4,
        weight_decay: float = 0.05,
        rec_loss_weight: float = 1.0,
        tissue_loss_weight: float = 0.1,
        band_loss_weight: float = 0.1,
        ortho_loss_weight: float = 0.05,
        n_comp: int = 3,
        band_low: float = 0.01,
        band_high: float = 0.15,
        erosion_iterations: int = 1,
        hard_bandpass: bool = True,
        full_res_channels: int = 24,
        decoder_skips: str = "full",
        free_noise_head: bool = False,
        prior_warmup_steps: int = 0,
        prior_warmup_epochs: int = 2,
        min_clean_var_frac: float = 0.05,
        w_min_var: float = 1.0,
        validation_seed: int = 1234,
    ) -> None:
        """Initialize hyperparameters, core model, and loss function."""
        super().__init__()
        self.save_hyperparameters()

        self.model = HierarchicalSwinMAE(
            patch_size=self.hparams.patch_size,
            in_chans=self.hparams.in_chans,
            embed_dim=self.hparams.embed_dim,
            mask_ratio=self.hparams.mask_ratio,
            n_comp=self.hparams.n_comp,
            band_low=self.hparams.band_low,
            band_high=self.hparams.band_high,
            erosion_iterations=self.hparams.erosion_iterations,
            hard_bandpass=self.hparams.hard_bandpass,
            full_res_channels=self.hparams.full_res_channels,
            decoder_skips=self.hparams.decoder_skips,
            free_noise_head=self.hparams.free_noise_head,
        )

        self.loss_fn = IdentifiableDenoisingLoss(
            rec_weight=self.hparams.rec_loss_weight,
            tissue_weight=self.hparams.tissue_loss_weight,
            band_weight=self.hparams.band_loss_weight,
            ortho_weight=self.hparams.ortho_loss_weight,
            erosion_iterations=self.hparams.erosion_iterations,
            band_low=self.hparams.band_low,
            band_high=self.hparams.band_high,
            min_clean_var_frac=self.hparams.min_clean_var_frac,
            w_min_var=self.hparams.w_min_var,
        )

    def forward(
        self,
        x: torch.Tensor,
        tr: torch.Tensor | float = 1.0,
        token_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Single masked forward pass (see :meth:`HierarchicalSwinMAE.forward`)."""
        return self.model(x, tr, token_mask=token_mask)

    @torch.no_grad()
    def denoise_complementary(
        self,
        x: torch.Tensor,
        tr: torch.Tensor | float = 1.0,
        K: int = 4,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """J-invariant full-volume inference with K complementary tube masks."""
        return self.model.denoise_complementary(x, tr, K=K, generator=generator)

    @torch.no_grad()
    def denoise_single(
        self,
        x: torch.Tensor,
        tr: torch.Tensor | float = 1.0,
        mask_ratio: float | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Single-pass inference (ablation; not J-invariant). ``mask_ratio=0`` = unmasked."""
        return self.model.denoise_single(x, tr, mask_ratio=mask_ratio, generator=generator)

    def _shared_step(self, batch: dict[str, Any], phase: str) -> torch.Tensor:
        """Forward pass and loss computation shared by training and validation."""
        x, tr = batch["x"], batch["tr"]
        out = self(x, tr)
        scale = 1.0
        min_var_scale = 0.0
        if phase == "train":
            step_warmup = int(self.hparams.prior_warmup_steps)
            epoch_warmup = int(self.hparams.prior_warmup_epochs)
            if step_warmup > 0:
                scale = min(scale, self.global_step / step_warmup)
            if epoch_warmup > 0:
                train_batches = max(1, int(self.trainer.num_training_batches))
                scale = min(scale, self.global_step / (train_batches * epoch_warmup))
            self.log("prior_scale", scale, batch_size=x.shape[0])
            min_var_scale = float(self.global_step >= 1)
        losses = self.loss_fn(out, x, tr, prior_scale=scale, min_var_scale=min_var_scale)

        batch_size = x.shape[0]
        for name, value in losses.items():
            self.log(
                f"{phase}_{name}",
                value,
                prog_bar=name == "total",
                sync_dist=True,
                batch_size=batch_size,
            )
            self.log(
                f"{phase}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        valid = out["valid"].float().sum().clamp_min(1.0)
        self.log(
            f"{phase}_mask_fraction",
            out["token_mask"].float().sum() / valid,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            f"{phase}/mask_fraction",
            out["token_mask"].float().sum() / valid,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return losses["total"]

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Process a single training batch."""
        return self._shared_step(batch, phase="train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Process a single validation batch with a reproducible mask."""
        x = batch["x"]
        seed = (
            int(self.hparams.validation_seed)
            + int(self.global_rank) * 1_000_003
            + batch_idx
        )
        devices = (
            [x.device.index if x.device.index is not None else torch.cuda.current_device()]
            if x.device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed(seed)
            return self._shared_step(batch, phase="val")

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure AdamW (selective weight decay) and a cosine annealing schedule."""
        optimizer = optim.AdamW(
            split_weight_decay_groups(self, self.hparams.weight_decay),
            lr=self.hparams.lr,
            betas=(0.9, 0.95),
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer.max_epochs else 100,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
