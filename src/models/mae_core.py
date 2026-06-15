### Central MAE model that integrates the patch embedding, official and adapted Swin blocks, and decoder into one cohesive Autoencoder.

import torch
import torch.nn as nn
from einops import rearrange
from models.patch_embed import fMRIPatchEmbed4D
from utils.masking import apply_swin_mask
from models.decoder import CascadingDecoder

# Import the official blocks
from models.swift_official import BasicLayer, MERGING_MODE

class HierarchicalSwinMAE(nn.Module):
    def __init__(self, patch_size=14, in_chans=4, embed_dim=96, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        
        # 3 merge stages configured here
        self.patch_embed = fMRIPatchEmbed4D(patch_size, in_chans, embed_dim, merge_stages=3)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, 150000, embed_dim))
        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)

        # The 4-Stage Official SwiFT Pyramid
        self.stage1 = BasicLayer(
            dim=embed_dim, depth=2, num_heads=3, window_size=(3, 3, 3, 3), 
            drop_path=0.0, downsample=MERGING_MODE["mergingv2"]
        )
        self.stage2 = BasicLayer(
            dim=embed_dim * 2, depth=2, num_heads=6, window_size=(3, 3, 3, 3), 
            drop_path=0.0, downsample=MERGING_MODE["mergingv2"]
        )
        self.stage3 = BasicLayer(
            dim=embed_dim * 4, depth=2, num_heads=12, window_size=(3, 3, 3, 3), 
            drop_path=0.0, downsample=MERGING_MODE["mergingv2"]
        )
        self.stage4 = BasicLayer(
            dim=embed_dim * 8, depth=2, num_heads=24, window_size=(3, 3, 3, 3), 
            drop_path=0.0, downsample=None # Final stage
        )
        
        self.decoder = CascadingDecoder(embed_dim, in_chans, patch_size)

    def forward(self, x, tube_masking=False):
        T = x.shape[2] 
        
        tokens, spatial_grid = self.patch_embed(x)
        H_p, W_p, D_p = spatial_grid
        
        x_masked, mask = apply_swin_mask(tokens, self.mask_token, self.mask_ratio, tube_masking, T)
        
        L = x_masked.shape[1]
        x_masked = x_masked + self.pos_embed[:, :L, :]
        
        x_encoded = rearrange(x_masked, 'b (t hp wp dp) c -> b c dp hp wp t', 
                              t=T, hp=H_p, wp=W_p, dp=D_p)
        
        # Plunge into the 4 stages
        x_encoded = self.stage1(x_encoded)
        x_encoded = self.stage2(x_encoded)
        x_encoded = self.stage3(x_encoded)
        latent = self.stage4(x_encoded)
        
        reconstruction = self.decoder(latent, spatial_grid, T)
        return reconstruction, mask