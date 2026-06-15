### This file dynamically pads 4D volumes and slices them into token embeddings.
### Ensures that the input dimensions are divisible by the patch size, allowing for seamless tokenization.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Channel size of 4 to solve physiological aliasing problem
# Channel 1 = BOLD fMRI sequence, Channel 2 = Grey Matter spatial mask, Channel 3 = White Matter spatial mask, Channel 4 = CSF spatial mask

# Patch size: 3D cube voxels grouped into single token. Reducing it -> saves GPU memory but lowers spatial resolution.
# Embedding dimension: Size of the latent token vector. Higher values capture more complex features but increase computational load.

class fMRIPatchEmbed4D(nn.Module):
    def __init__(self, patch_size=14, in_chans=4, embed_dim=96, merge_stages=3):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Calculate required grid multiple
        self.grid_multiple = 2 ** merge_stages 

        # 3D convolution to convert spatial patches into latent tokens
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x):
        # Input: [Batch, Channels, Time, Height, Width, Depth]
        B, C, T, H, W, D = x.shape
        P = self.patch_size
        M = self.grid_multiple

        # Calculate the absolute minimum patches needed without cropping
        min_h = (H + P - 1) // P
        min_w = (W + P - 1) // P
        min_d = (D + P - 1) // P

        # Round up to the nearest multiple of M
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

        # Apply padding
        x = F.pad(x, (0, pad_d, 0, pad_w, 0, pad_h))
        _, _, _, H_pad, W_pad, D_pad = x.shape

        x = rearrange(x, 'b c t h w d -> (b t) c h w d')
        x = self.proj(x) 
        
        x = rearrange(x, '(b t) e hp wp dp -> b (t hp wp dp) e', b=B, t=T)
        
        spatial_grid = (H_pad // P, W_pad // P, D_pad // P)
        return x, spatial_grid