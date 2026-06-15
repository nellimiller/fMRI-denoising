### Masked MSE and FFT frequency penalty for scanner drift.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class fMRIMaskedLoss(nn.Module):
    def __init__(self, patch_size=14, fft_weight=0.1, merge_stages=3):
        super().__init__()
        self.patch_size = patch_size
        self.fft_weight = fft_weight
        self.grid_multiple = 2 ** merge_stages

    def forward(self, pred, target, mask):
        B, C, T, X, Y, Z = pred.shape
        P = self.patch_size
        M = self.grid_multiple
        
        # 1. Replicate the padding math to find the Padded Grid Dimensions
        min_h = (X + P - 1) // P
        min_w = (Y + P - 1) // P
        min_d = (Z + P - 1) // P
        
        Hp = ((min_h + M - 1) // M) * M
        Wp = ((min_w + M - 1) // M) * M
        Dp = ((min_d + M - 1) // M) * M
        
        # 2. Inflate the 1D mask back to the Padded Spatial Grid
        # Transforms 8192 tokens -> [Batch, 1, 2, 16, 16, 16]
        mask_grid = rearrange(mask, 'b (t hp wp dp) -> b 1 t hp wp dp', 
                              t=T, hp=Hp, wp=Wp, dp=Dp)
        
        # 3. Blow up the grid blocks into full 14x14x14 voxel patches 
        # Resulting shape: [Batch, 1, 2, 224, 224, 224]
        voxel_mask = repeat(mask_grid, 'b 1 t hp wp dp -> b 1 t (hp p1) (wp p2) (dp p3)', 
                            p1=P, p2=P, p3=P)
        
        # 4. Crop the mask down to the raw brain size
        # Resulting shape: [Batch, 1, 2, 210, 210, 114]
        voxel_mask = voxel_mask[:, :, :, :X, :Y, :Z]
        
        # 5. Masked Spatial Loss (Rebuilding Anatomy)
        mse_loss = (pred - target) ** 2
        masked_mse = (mse_loss * voxel_mask).sum() / (voxel_mask.sum() + 1e-8)
        
        # 6. Temporal FFT Loss (Rebuilding BOLD Dynamics)
        pred_fft = torch.fft.rfft(pred, dim=2)
        target_fft = torch.fft.rfft(target, dim=2)
        fft_loss = F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))
        
        # 7. Combine for Total Loss
        total_loss = masked_mse + (self.fft_weight * fft_loss)
        
        return total_loss, masked_mse, fft_loss