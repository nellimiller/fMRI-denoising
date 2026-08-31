### Decoder blocks with Cascading CNN upsampler for the 4D Swin Transformer.

import torch
import torch.nn as nn
from einops import rearrange


class CascadingDecoder(nn.Module):
    def __init__(self, embed_dim=96, out_chans=4, patch_size=14):
        super().__init__()
        self.patch_size = patch_size

        # 4 stages means deep tokens arrive with 8x the original dimension (768)
        deep_dim = embed_dim * 8

        # Stage 1: 768 -> 384
        self.up1 = nn.ConvTranspose3d(deep_dim, deep_dim // 2, kernel_size=2, stride=2)
        # Stage 2: 384 -> 192
        self.up2 = nn.ConvTranspose3d(deep_dim // 2, deep_dim // 4, kernel_size=2, stride=2)
        # Stage 3: 192 -> 96
        self.up3 = nn.ConvTranspose3d(deep_dim // 4, embed_dim, kernel_size=2, stride=2)

        # Final Inflator: 96 -> Voxels
        self.final_inflate = nn.ConvTranspose3d(
            in_channels=embed_dim,
            out_channels=embed_dim // 2,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.shared = nn.Sequential(
            nn.Conv3d(embed_dim // 2, 32, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.clean_head = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(16, 1, kernel_size=1),
        )

        self.noise_head = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(16, 1, kernel_size=1),
        )

    def forward(self, x, spatial_grid, T):
        B, C, D, H, W, Time = x.shape
        x = rearrange(x, 'b c d h w t -> (b t) c h w d')

        # Walk back up the 4-stage pyramid
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.final_inflate(x)
        x = self.shared(x)

        clean = self.clean_head(x)
        noise = self.noise_head(x)

        clean = rearrange(clean, '(b t) c h w d -> b c t h w d', b=B, t=T)
        noise = rearrange(noise, '(b t) c h w d -> b c t h w d', b=B, t=T)
        return clean, noise