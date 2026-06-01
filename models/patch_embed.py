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
    def __init__(self, patch_size=14, in_chans=4, embed_dim=96):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

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

        # Calculate padding to ensure divisibility by patch size
        pad_h = (P - H % P) % P
        pad_w = (P - W % P) % P
        pad_d = (P - D % P) % P

        # Apply padding (right-to-left: back, front, bottom, top, left, right)
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            x = F.pad(x, (0, pad_d, 0, pad_w, 0, pad_h))

        _, _, _, H_pad, W_pad, D_pad = x.shape

        # Temporarily collapse Batch and Time to process through standard Conv3d
        x = rearrange(x, 'b c t h w d -> (b t) c h w d')
        x = self.proj(x)  # [B*T, embed_dim, H_pad/P, W_pad/P, D_pad/P]

        # Flatten spatial patches into 1D sequence line
        # Output: [Batch, (Time * H_patches * W_patches * D_patches), embed_dim]
        x = rearrange(x, '(b t) e hp wp dp -> b (t hp wp dp) e', b=B, t=T)
        
        # Return tokens and padded grid dimensions for positional embeddings later
        spatial_grid = (H_pad // P, W_pad // P, D_pad // P)
        return x, spatial_grid