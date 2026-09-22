"""Masked MSE and structural tissue-based decomposition loss functions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class StructuralDenoisingLoss(nn.Module):
    """Calculates decomposition penalties based on anatomical tissue masks.

    Enforces that predicted clean BOLD signals have low temporal variance in
    White Matter (WM) and CSF (which contain physiological noise), and that 
    Grey Matter (GM) signals are orthogonal to the global noise summary.

    Args:
        patch_size: Spatial patch dimension.
        fft_weight: Penalty weight for frequency domain mismatch.
        flat_weight: Penalty weight for temporal variance in WM/CSF.
        ortho_weight: Penalty weight for GM/Noise orthogonality (Deep aCompCor).
        merge_stages: Number of hierarchical downsampling stages.
    """

    def __init__(
        self,
        patch_size: int = 14,
        fft_weight: float = 0.1,
        flat_weight: float = 1.0,
        ortho_weight: float = 1e-3,
        merge_stages: int = 3,
    ) -> None:
        """Initialize loss weights and structural hyper-parameters."""
        super().__init__()
        self.patch_size = patch_size
        self.fft_weight = fft_weight
        self.flat_weight = flat_weight
        self.ortho_weight = ortho_weight
        self.grid_multiple = 2 ** merge_stages

    def _extract_target_and_masks(
        self, 
        target: torch.Tensor, 
        gm_mask: torch.Tensor | None = None, 
        wm_mask: torch.Tensor | None = None, 
        csf_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unpack concatenated targets or fall back to empty masks."""
        # If dataset batches masks together with BOLD (Channels = 4)
        if target.shape[1] >= 4:
            target_bold = target[:, :1]
            gm_mask = target[:, 1:2] if gm_mask is None else gm_mask
            wm_mask = target[:, 2:3] if wm_mask is None else wm_mask
            csf_mask = target[:, 3:4] if csf_mask is None else csf_mask
            return target_bold, gm_mask, wm_mask, csf_mask

        target_bold = target
        gm_mask = torch.zeros_like(target_bold) if gm_mask is None else gm_mask
        wm_mask = torch.zeros_like(target_bold) if wm_mask is None else wm_mask
        csf_mask = torch.zeros_like(target_bold) if csf_mask is None else csf_mask

        return target_bold, gm_mask, wm_mask, csf_mask

    #def _masked_variance_over_time(self, x, mask):
    #    # x: [B, C, T, X, Y, Z], mask: same shape; average temporal variance over voxels in mask
    #    x = x * mask
    #    var = x.var(dim=2, unbiased=False)
    #    denom = mask.sum(dim=(2, 3, 4, 5), keepdim=True).clamp_min(1.0)
    #    # var is [B, C, X, Y, Z]; we only care about the central channel and its spatial mask
    #    return (var * mask[:, :, 0:1]).sum() / (mask[:, :, 0:1].sum().clamp_min(1.0))

    def forward(
        self, 
        clean_pred: torch.Tensor, 
        noise_pred: torch.Tensor, 
        target: torch.Tensor, 
        gm_mask: torch.Tensor | None = None, 
        wm_mask: torch.Tensor | None = None, 
        csf_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the weighted structural loss components."""
        target_bold, gm_mask, wm_mask, csf_mask = self._extract_target_and_masks(
            target, gm_mask, wm_mask, csf_mask
        )

        gm_mask = gm_mask.float()
        wm_mask = wm_mask.float()
        csf_mask = csf_mask.float()

        # Combine WM and CSF into a single physiological noise mask
        tissue_mask = torch.maximum(wm_mask, csf_mask)

        # 1. Standard reconstruction loss
        total_pred = clean_pred + noise_pred
        recon_loss = F.mse_loss(total_pred, target_bold)

        # 2. L_flat: Penalize temporal variance of the *clean* prediction inside WM/CSF
        clean_tissue = clean_pred * tissue_mask
        clean_var = clean_tissue.var(dim=2, unbiased=False)
        tissue_weight = tissue_mask[:, :, 0:1] if tissue_mask.dim() == 6 else tissue_mask
        flat_loss = (clean_var * tissue_weight).sum() / (tissue_weight.sum().clamp_min(1.0))

        # 3. L_ortho (Deep aCompCor-style loss)
        # Extract the average temporal noise summary from the WM+CSF volume
        noise_summary = (noise_pred * tissue_mask).sum(dim=(3, 4, 5), keepdim=True) / (
            tissue_mask.sum(dim=(3, 4, 5), keepdim=True).clamp_min(1.0)
        )

        # Squeeze spatial dimensions to get [B, C, T], then isolate the sequence [B, T]
        noise_vector = noise_summary.squeeze(-1).squeeze(-1).squeeze(-1)[:, 0, :]

        # Penalize correlation between GM signal and the global noise summary
        gm_voxels = (clean_pred * gm_mask).reshape(clean_pred.shape[0], -1, clean_pred.shape[2])
        gm_scores = torch.matmul(gm_voxels, noise_vector.unsqueeze(-1))
        ortho_loss = gm_scores.pow(2).mean()

        # 4. Fast Fourier Transform Loss (Spectrum Matching)
        pred_fft = torch.fft.rfft(total_pred, dim=2)
        target_fft = torch.fft.rfft(target_bold, dim=2)
        fft_loss = F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))

        total_loss = (
            recon_loss 
            + self.flat_weight * flat_loss 
            + self.ortho_weight * ortho_loss 
            + self.fft_weight * fft_loss
        )

        return total_loss, recon_loss, flat_loss, ortho_loss, fft_loss


class fMRIMaskedLoss(nn.Module):
    """Combines MAE token masking logic with the Structural Denoising Loss."""

    def __init__(
        self, 
        patch_size: int = 14, 
        fft_weight: float = 0.1, 
        flat_weight: float = 1.0, 
        ortho_weight: float = 1e-3, 
        merge_stages: int = 3
    ) -> None:
        """Initialize the loss wrapper."""
        super().__init__()
        self.patch_size = patch_size
        self.grid_multiple = 2 ** merge_stages
        self.structural_loss = StructuralDenoisingLoss(
            patch_size=patch_size,
            fft_weight=fft_weight,
            flat_weight=flat_weight,
            ortho_weight=ortho_weight,
            merge_stages=merge_stages,
        )

        #self.fft_weight = fft_weight
        #self.flat_weight = flat_weight
        #self.ortho_weight = ortho_weight
        
        
    def forward(
        self, 
        pred: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
        target: torch.Tensor, 
        mask: torch.Tensor | None = None, 
        clean_pred: torch.Tensor | None = None, 
        noise_pred: torch.Tensor | None = None, 
        gm_mask: torch.Tensor | None = None, 
        wm_mask: torch.Tensor | None = None, 
        csf_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute structural loss alongside Masked-MSE (calculating MSE only on dropped patches)."""
        if isinstance(pred, (tuple, list)):
            total_pred, clean_pred, noise_pred = pred
        else:
            total_pred = pred
            clean_pred = pred if clean_pred is None else clean_pred
            noise_pred = torch.zeros_like(pred) if noise_pred is None else noise_pred

        # MAE Masked MSE: calculate reconstruction loss ONLY on the dropped tokens
        if mask is not None and total_pred.shape[1] == 1:
            B, C, T, X, Y, Z = total_pred.shape
            P = self.patch_size
            M = self.grid_multiple

            # Recalculate grid dimensions based on patch size and pooling stages
            min_h = (X + P - 1) // P
            min_w = (Y + P - 1) // P
            min_d = (Z + P - 1) // P

            Hp = ((min_h + M - 1) // M) * M
            Wp = ((min_w + M - 1) // M) * M
            Dp = ((min_d + M - 1) // M) * M

            # Reshape the 1D token mask back into the 4D voxel space
            mask_grid = rearrange(mask, 'b (t hp wp dp) -> b 1 t hp wp dp', t=T, hp=Hp, wp=Wp, dp=Dp)
            voxel_mask = repeat(mask_grid, 'b 1 t hp wp dp -> b 1 t (hp p1) (wp p2) (dp p3)', p1=P, p2=P, p3=P)
            voxel_mask = voxel_mask[:, :, :, :X, :Y, :Z]

            mse_loss = ((total_pred - target[:, :1]) ** 2) * voxel_mask
            masked_mse = mse_loss.sum() / (voxel_mask.sum().clamp_min(1.0))
        else:
            masked_mse = F.mse_loss(total_pred, target[:, :1])

        total_loss, recon_loss, flat_loss, ortho_loss, fft_loss = self.structural_loss(
            clean_pred=clean_pred,
            noise_pred=noise_pred,
            target=target,
            gm_mask=gm_mask,
            wm_mask=wm_mask,
            csf_mask=csf_mask,
        )
        
        return total_loss, masked_mse, fft_loss