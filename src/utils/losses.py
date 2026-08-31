### Masked MSE and anatomical decomposition loss.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class StructuralDenoisingLoss(nn.Module):
    def __init__(
        self,
        patch_size=14,
        fft_weight=0.1,
        flat_weight=1.0,
        ortho_weight=1e-3,
        merge_stages=3,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.fft_weight = fft_weight
        self.flat_weight = flat_weight
        self.ortho_weight = ortho_weight
        self.grid_multiple = 2 ** merge_stages

    def _extract_target_and_masks(self, target, gm_mask=None, wm_mask=None, csf_mask=None):
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

    def _masked_variance_over_time(self, x, mask):
        # x: [B, C, T, X, Y, Z], mask: same shape; average temporal variance over voxels in mask
        x = x * mask
        var = x.var(dim=2, unbiased=False)
        denom = mask.sum(dim=(2, 3, 4, 5), keepdim=True).clamp_min(1.0)
        # var is [B, C, X, Y, Z]; we only care about the central channel and its spatial mask
        return (var * mask[:, :, 0:1]).sum() / (mask[:, :, 0:1].sum().clamp_min(1.0))

    def forward(self, clean_pred, noise_pred, target, gm_mask=None, wm_mask=None, csf_mask=None):
        target_bold, gm_mask, wm_mask, csf_mask = self._extract_target_and_masks(target, gm_mask, wm_mask, csf_mask)

        gm_mask = gm_mask.float()
        wm_mask = wm_mask.float()
        csf_mask = csf_mask.float()
        tissue_mask = torch.maximum(wm_mask, csf_mask)

        total_pred = clean_pred + noise_pred
        recon_loss = F.mse_loss(total_pred, target_bold)

        # L_flat: variance over time in WM + CSF for the clean output
        clean_tissue = clean_pred * tissue_mask
        clean_var = clean_tissue.var(dim=2, unbiased=False)
        tissue_weight = tissue_mask[:, :, 0:1] if tissue_mask.dim() == 6 else tissue_mask
        flat_loss = (clean_var * tissue_weight).sum() / (tissue_weight.sum().clamp_min(1.0))

        # L_ortho (Deep aCompCor-style): match each GM voxel's time series with a single noise summary vector
        # summary vector = average temporal signal in WM+CSF noise volume
        noise_summary = (noise_pred * tissue_mask).sum(dim=(3, 4, 5), keepdim=True) / (
            tissue_mask.sum(dim=(3, 4, 5), keepdim=True).clamp_min(1.0)
        )
        noise_summary = noise_summary.squeeze(-1).squeeze(-1).squeeze(-1)  # [B, C, T]
        noise_vector = noise_summary[:, 0, :]  # [B, T]

        gm_voxels = (clean_pred * gm_mask).reshape(clean_pred.shape[0], -1, clean_pred.shape[2])
        gm_scores = torch.matmul(gm_voxels, noise_vector.unsqueeze(-1))
        ortho_loss = (gm_scores.pow(2).mean(dim=1)).mean()

        pred_fft = torch.fft.rfft(total_pred, dim=2)
        target_fft = torch.fft.rfft(target_bold, dim=2)
        fft_loss = F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))

        total_loss = recon_loss + self.flat_weight * flat_loss + self.ortho_weight * ortho_loss + self.fft_weight * fft_loss
        return total_loss, recon_loss, flat_loss, ortho_loss, fft_loss


class fMRIMaskedLoss(nn.Module):
    def __init__(self, patch_size=14, fft_weight=0.1, flat_weight=1.0, ortho_weight=1e-3, merge_stages=3):
        super().__init__()
        self.patch_size = patch_size
        self.fft_weight = fft_weight
        self.flat_weight = flat_weight
        self.ortho_weight = ortho_weight
        self.grid_multiple = 2 ** merge_stages
        self.structural_loss = StructuralDenoisingLoss(
            patch_size=patch_size,
            fft_weight=fft_weight,
            flat_weight=flat_weight,
            ortho_weight=ortho_weight,
            merge_stages=merge_stages,
        )

    def forward(self, pred, target, mask=None, clean_pred=None, noise_pred=None, gm_mask=None, wm_mask=None, csf_mask=None):
        if isinstance(pred, (tuple, list)):
            total_pred, clean_pred, noise_pred = pred
        else:
            total_pred = pred
            if clean_pred is None:
                clean_pred = pred
            if noise_pred is None:
                noise_pred = torch.zeros_like(pred)

        if mask is not None and total_pred.shape[1] == 1:
            B, C, T, X, Y, Z = total_pred.shape
            P = self.patch_size
            M = self.grid_multiple

            min_h = (X + P - 1) // P
            min_w = (Y + P - 1) // P
            min_d = (Z + P - 1) // P

            Hp = ((min_h + M - 1) // M) * M
            Wp = ((min_w + M - 1) // M) * M
            Dp = ((min_d + M - 1) // M) * M

            mask_grid = rearrange(mask, 'b (t hp wp dp) -> b 1 t hp wp dp', t=T, hp=Hp, wp=Wp, dp=Dp)
            voxel_mask = repeat(mask_grid, 'b 1 t hp wp dp -> b 1 t (hp p1) (wp p2) (dp p3)', p1=P, p2=P, p3=P)
            voxel_mask = voxel_mask[:, :, :, :X, :Y, :Z]
            mse_loss = ((total_pred - target[:, :1]) ** 2) * voxel_mask
            masked_mse = mse_loss.sum() / (voxel_mask.sum().clamp_min(1.0))
        else:
            masked_mse = F.mse_loss(total_pred, target[:, :1])

        if clean_pred is None or noise_pred is None:
            clean_pred = total_pred
            noise_pred = torch.zeros_like(total_pred)

        total_loss, recon_loss, flat_loss, ortho_loss, fft_loss = self.structural_loss(
            clean_pred=clean_pred,
            noise_pred=noise_pred,
            target=target,
            gm_mask=gm_mask,
            wm_mask=wm_mask,
            csf_mask=csf_mask,
        )
        return total_loss, masked_mse, fft_loss