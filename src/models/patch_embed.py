"""Dynamic padding and tokenization of 4D fMRI volumes for Swin Transformers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class fMRIPatchEmbed4D(nn.Module):
    """Slices 4D fMRI volumes into token embeddings with dynamic spatial padding.
    
    Ensures input dimensions are divisible by the patch size adn the downsampling
    requirements of the hierarchical Swin Transformer stages.
    
    Args:
        patch_size: 3D cube voxel dimension grouped into a single token.
            Lower values increase spatial resolution but consume more GPU memory.
        in_chans: Number of input channels. Defaults to 4 (1 BOLD sequence + 
            3 spatial tissue masks: GM, WM, CSF) to mitigate physiological aliasing.
        embed_dim: Size of the latent token vector.
        merge_stages: Number of hierarchical downsampling stages in the network.
    """

    def __init__(
        self,
        patch_size: int = 14,
        in_chans: int = 4,
        embed_dim: int = 96,
        merge_stages: int = 3
    ) -> None:
        """Initialize the spatial patch embedding projection."""
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Calculate the required grid multiple to ensure dimensions survive pooling
        self.grid_multiple = 2 ** merge_stages

        # 3D convolution converts spatial patches directly into latent tokens
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Pad the 4D volume and project it into a flattened token sequence.

        Args:
            x: Input tensor of shape '(B, C, T, H, W, D)'.

        Returns:
            A tuple containing:
            - The flattened token sequence `(B, L, E)`.
            - The spatial grid dimensions `(H_p, W_p, D_p)` for later reshaping.
        """
        B, C, T, H, W, D = x.shape
        P = self.patch_size
        M = self.grid_multiple

        # Calculate the absolute minimum patches needed without cropping
        min_h = (H + P - 1) // P
        min_w = (W + P - 1) // P
        min_d = (D + P - 1) // P

        # Round up to the nearest multiple of the hierarchical pooling stages
        target_h_patches = ((min_h + M - 1) // M) * M
        target_w_patches = ((min_w + M - 1) // M) * M
        target_d_patches = ((min_d + M - 1) // M) * M

        # Calculate target voxel dimensions
        target_h = target_h_patches * P
        target_w = target_w_patches * P
        target_d = target_d_patches * P

        # Calculate padding required to hit those targets
        pad_h = target_h - H
        pad_w = target_w - W
        pad_d = target_d - D

        # Apply spatial padding (starts from the last dimension: D, then W, then H)
        x = F.pad(x, (0, pad_d, 0, pad_w, 0, pad_h))
        H_pad, W_pad, D_pad = x.shape[3:]

        # Merge Batch and Time for 3D spatial convolution
        x = rearrange(x, "b c t h w d -> (b t) c h w d")
        x = self.proj(x) 
        
        # Flatten spatial dimensions into a 1D token sequence and restore Batch
        x = rearrange(x, "(b t) e hp wp dp -> b (t hp wp dp) e", b=B, t=T)
        
        spatial_grid = (H_pad // P, W_pad // P, D_pad // P)
        
        return x, spatial_grid