### Decoder blocks with lightweight CNN upsamlpler for the 4D Swin Transformer.

import torch
import torch.nn as nn
from einops import rearrange

class LightCNNDecoder(nn.Module):
    def __init__(self, embed_dim=96, out_chans=4, patch_size=14):
        super().__init__()
        self.patch_size = patch_size

        # Upsampler: upsampling of 15x15x9 grid back to 201x201x126 in one step
        # for arranging in 3D grid: X/W = 210/14 = 15, Y/H = 210/14 = 15, Z/D = 114/14 = 9
        self.upsample = nn.ConvTranspose3d(
            in_channels=embed_dim,
            out_channels=embed_dim // 2,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Smoothing convolution to refine the upsampled output and remove "blocky" artifacts
        # predicting the final 4 BOLD / Anatomy channels
        self.smooth = nn.Sequential(
            nn.Conv3d(embed_dim // 2, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(32, out_chans, kernel_size=3, padding=1)
        )

    def forward(self, x, spatial_grid, T):
        # x shape entering decoder: [Batch, Sequence_Length, Embed_Dim]
        B, L, E = x.shape
        H_p, W_p, D_p = spatial_grid

        # Unflatten sequence back into 3D spatial grid
        # Collapse Batch and Time for ConvTranspose3d
        x = rearrange(x, 'b (t hp wp dp) e -> (b t) e hp wp dp', 
                      t=T, hp=H_p, wp=W_p, dp=D_p)
        
        # Upsample to voxel space
        x = self.upsample(x)  # [B*T, embed_dim // 2, H, W, D]

        # Smooth the upsampled output and project to 4 channels
        x = self.smooth(x)

        # Restore time dimension to output [Batch, Channels, Time, Height, Width, Depth]
        x = rearrange(x, '(b t) c h w d -> b c t h w d', b=B, t=T)

        return x