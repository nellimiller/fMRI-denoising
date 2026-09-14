"""Decoder blocks with Cascading CNN upsampler for the 4D Swin Transformer."""

import torch
import torch.nn as nn
from einops import rearrange


class CascadingDecoder(nn.Module):
    """Cascading 3D CNN decoder for separating clean BOLD signals from noise.
    
    Upsamples deep Swin Transformer tokens back to original voxel space
    and splits the representation into dual additive components (clean and noise).
    
    Args:
        embed_dim: Base embedding dimension of the Swin Transformer. Defaults to 96.
        out_chans: Number of output channels (unused legacy argument). Defaults to 4.
        patch_size: Spatial patch size used in the encoder. Defaults to 14.
    """

    def __init__(
        self,
        embed_dim: int=96,
        out_chans: int = 4,
        patch_size: int = 14
    ) -> None:
        """Intitialize the upsampling pyramid and dual projection heads."""
        super().__init__()
        self.patch_size = patch_size

        # Deepest encoder stage outputs 8x the base embedding dimension
        deep_dim = embed_dim * 8

        self.up1 = nn.ConvTranspose3d(deep_dim, deep_dim // 2, kernel_size=2, stride=2) # Stage 1: 768 -> 384
        self.up2 = nn.ConvTranspose3d(deep_dim // 2, deep_dim // 4, kernel_size=2, stride=2) # Stage 2: 384 -> 192
        self.up3 = nn.ConvTranspose3d(deep_dim // 4, embed_dim, kernel_size=2, stride=2) # Stage 3: 192 -> 96

        # Inflate tokens back to raw voxel resolution
        self.final_inflate = nn.ConvTranspose3d(
            in_channels=embed_dim,
            out_channels=embed_dim // 2,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Shared feature extraction before splitting into components
        self.shared = nn.Sequential(
            nn.Conv3d(embed_dim // 2, 32, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Dual heads for additive signal decomposition
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

    def forward(
        self, x: torch.Tensor, spatial_grid: torch.Tensor, T: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Upsample tokens and predict clean/noise signal components.
        
        Args:
            x: Deep latent tensor of shape '(B, C, D, H, W, Time)'.
            spatial_grid: Positional grid tensor (unused in this block).
            T: Total number of timeframes.
            
        Returns:
            A tuple containing:
            - clean: Predicted neural BOLD signal '(B, 1, T, H, W, D)'.
            - noise: Predicted physiological/termal noise '(B, 1, T, H, W, D)'.
        """
        B, C, D, H, W, Time = x.shape

        # Merge batch and time dimensions for standard 3D spatial convolutions
        x = rearrange(x, 'b c d h w t -> (b t) c h w d')

        # Cascade through the upsampling pyramid
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.final_inflate(x)

        x = self.shared(x)

        # Project into seperate additive components
        clean = self.clean_head(x)
        noise = self.noise_head(x)

        # Seperate batch and time dimensions, restoring the 4D temporal structure
        clean = rearrange(clean, '(b t) c h w d -> b c t h w d', b=B, t=T)
        noise = rearrange(noise, '(b t) c h w d -> b c t h w d', b=B, t=T)
        
        return clean, noise