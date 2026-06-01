### Central MAE model that integrates the patch embedding, Swin blocks, and decoder into one cohesive Autoencoder.

import torch
import torch.nn as nn
from models.patch_embed import fMRIPatchEmbed4D
from utils.masking import apply_swin_mask
from models.decoder import LightCNNDecoder

class SwinMAE(nn.Module):
    def __init__(self, patch_size=14, in_chans=4, embed_dim=96, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        
        # Front-End: Ingestion and Masking
        self.patch_embed = fMRIPatchEmbed4D(patch_size, in_chans, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        

        torch.nn.init.normal_(self.mask_token, std=0.02)  # Initialize mask token with std=0.02

        # The Encoder: Placeholder
        self.encoder = nn.Identity()

        # The Back-End Decoder: Upsampling and Reconstruction
        self.decoder = LightCNNDecoder(embed_dim, in_chans, patch_size)

    
    def forward(self, x, tube_masking=False):
        # Extract number of timepoints
        T = x.shape[2]

        # Embed raw fMRI volume into tokens
        tokens, spatial_grid = self.patch_embed(x)

        # Apply Swin Masking
        x_masked, mask = apply_swin_mask(
            x=tokens,
            mask_token=self.mask_token,
            mask_ratio=self.mask_ratio,
            tube_masking=tube_masking,
            T=T
        )

        # Apply Transformer Encoder
        latent = self.encoder(x_masked)

        # Rebuild brain volume
        reconstruction = self.decoder(latent, spatial_grid, T)

        return reconstruction, mask