"""PyTorch Lightning module for training the 4D fMRI Masked Autoencoder."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.optim as optim

from models.mae_core import HierarchicalSwinMAE
from utils.losses import fMRIMaskedLoss

class fMRIDenoisingLightning(L.LightningModule):
    """Lightning module coordinating the SwinMAE architecture and dual-engine loss.

    Args:
        in_chans: Number of input channels (1 BOLD + 3 anatomical masks).
        patch_size: Base spatial patch size for the encoder.
        embed_dim: Initial embedding dimension for the Swin blocks.
        mask_ratio: Fraction of tokens to drop during training.
        lr: Maximum learning rate for the Cosine Annealing schedule.
        weight_decay: L2 penalty factor for AdamW.
    """
    def __init__(
        self,
        in_chans: int = 4,
        patch_size: int = 14,
        embed_dim: int = 96,
        mask_ratio: float = 0.75,
        lr: float = 1.5e-4,
        weight_decay: float = 0.05
    ) -> None:
        """Initialize hyperparameters, core model, and loss function."""
        super().__init__()
        self.save_hyperparameters()

        self.model = HierarchicalSwinMAE(
            patch_size=self.hparams.patch_size,
            in_chans=self.hparams.in_chans,
            embed_dim=self.hparams.embed_dim,
            mask_ratio=self.hparams.mask_ratio
        )

        self.loss_fn = fMRIMaskedLoss(
            patch_size=self.hparams.patch_size, 
            fft_weight=0.1
        )

    def forward(
        self, x: torch.Tensor, tube_masking: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute the forward pass through the Masked Autoencoder.

        Args:
            x: Input 4D fMRI tensor '[B, C, T, X, Y, Z]'.
            tube_masking: If True, drops identical spatial patches across all timeframes.

        Returns:
            Tuple of '(reconstructed_total, clean_pred, noise_pred, binary_mask)'.
        """
        return self.model(x, tube_masking=tube_masking, return_heads=True)

    def _shared_step(self, batch: torch.Tensor, phase: str) -> torch.Tensor:
        """Handle forward pass, padding removal, and loss computation.
        
        Extracts redundant logic from both training and validation steps.
        """
        pred, clean_pred, noise_pred, mask = self(batch, tube_masking=True)

        # Dynamic cropping: Swin blocks often require spatial padding to divide evenly.
        # Crops the predictions back to the original raw target shape before loss computation.
        X, Y, Z = batch.shape[3], batch.shape[4], batch.shape[5]
        
        pred_cropped = pred[:, :, :, :X, :Y, :Z]
        clean_cropped = clean_pred[:, :, :, :X, :Y, :Z]
        noise_cropped = noise_pred[:, :, :, :X, :Y, :Z]

        # Compute dual-engine loss using anatomical tissue priors (channels 1, 2, 3)
        total_loss, mse, fft = self.loss_fn(
            pred_cropped,
            target=batch,
            mask=mask,
            clean_pred=clean_cropped,
            noise_pred=noise_cropped,
            gm_mask=batch[:, 1:2],
            wm_mask=batch[:, 2:3],
            csf_mask=batch[:, 3:4],
        )

        self.log(f"{phase}_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log(f"{phase}_mse", mse, sync_dist=True)
        self.log(f"{phase}_fft", fft, sync_dist=True)

        return total_loss

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Process a single training batch."""
        return self._shared_step(batch, phase="train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Process a single validation batch."""
        return self._shared_step(batch, phase="val")
    
    def configure_optimizers(self) -> dict[str, Any]:
        """Configure the AdamW optimizer and Cosine Annealing schedule."""
        # AdamW with high weight decay is the industry standard for Vision Transformers
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95),
        )
        
        # Transformers are notoriously unstable during early training.
        # Cosine annealing lowers the LR to settle into deep minimums.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs if self.trainer.max_epochs else 100, 
            eta_min=1e-6,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }