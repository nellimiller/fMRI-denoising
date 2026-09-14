"""Central Masked Autoencoder integrating patch embedding, Swin blocks, and decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from models.decoder import CascadingDecoder
from models.patch_embed import fMRIPatchEmbed4D
from models.swift_official import MERGING_MODE, BasicLayer
from utils.masking import apply_swin_mask


class HierarchicalSwinMAE(nn.Module):
    """4D Swin Transformer Masked Autoencoder for fMRI denoising.

    Args:
        patch_size: Base spatial patch size. Defaults to 14.
        in_chans: Number of input channels (e.g., 1 BOLD + 3 masks). Defaults to 4.
        embed_dim: Initial embedding dimension for the Swin blocks. Defaults to 96.
        mask_ratio: Fraction of tokens to mask during training. Defaults to 0.75.
    """

    def __init__(
        self,
        patch_size: int = 14,
        in_chans: int = 4,
        embed_dim: int = 96,
        mask_ratio: float = 0.75,
    ) -> None:
        """Initialize the embedding, masking, Swin pyramid, and decoder."""
        super().__init__()
        self.mask_ratio = mask_ratio

        self.patch_embed = fMRIPatchEmbed4D(patch_size, in_chans, embed_dim, merge_stages=3)
        
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # 1D Positional embedding (150000 capacity covers max sequence length)
        self.pos_embed = nn.Parameter(torch.zeros(1, 150000, embed_dim))
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 4-stage Swin Transformer encoder pyramid
        self.stage1 = BasicLayer(
            dim=embed_dim,
            depth=2,
            num_heads=3,
            window_size=(3, 3, 3, 3),
            drop_path=0.0,
            downsample=MERGING_MODE["mergingv2"],
        )
        self.stage2 = BasicLayer(
            dim=embed_dim * 2,
            depth=2,
            num_heads=6,
            window_size=(3, 3, 3, 3),
            drop_path=0.0,
            downsample=MERGING_MODE["mergingv2"],
        )
        self.stage3 = BasicLayer(
            dim=embed_dim * 4,
            depth=2,
            num_heads=12,
            window_size=(3, 3, 3, 3),
            drop_path=0.0,
            downsample=MERGING_MODE["mergingv2"],
        )
        self.stage4 = BasicLayer(
            dim=embed_dim * 8,
            depth=2,
            num_heads=24,
            window_size=(3, 3, 3, 3),
            drop_path=0.0,
            downsample=None,  # Final stage
        )

        self.decoder = CascadingDecoder(embed_dim, in_chans, patch_size)

    def forward(
        self, x: torch.Tensor, tube_masking: bool = False, return_heads: bool = False
    ) -> tuple[torch.Tensor, ...]:
        """Execute the autoencoder forward pass.

        Args:
            x: Input 4D fMRI tensor '[B, C, T, X, Y, Z]'.
            tube_masking: If True, applies identical spatial masks across all timeframes.
            return_heads: If True, returns separated clean and noise predictions.

        Returns:
            Tuple containing the reconstructed tensor and the binary mask.
            If 'return_heads' is True, also returns the isolated clean and noise tensors.
        """
        T = x.shape[2]

        tokens, spatial_grid = self.patch_embed(x)
        H_p, W_p, D_p = spatial_grid

        # Apply MAE token dropping strategy
        x_masked, mask = apply_swin_mask(tokens, self.mask_token, self.mask_ratio, tube_masking, T)

        # Inject positional embeddings up to the current sequence length
        L = x_masked.shape[1]
        x_masked = x_masked + self.pos_embed[:, :L, :]

        # Reshape 1D token sequence back into 4D spatial-temporal grid for Swin stages
        x_encoded = rearrange(
            x_masked,
            "b (t hp wp dp) c -> b c dp hp wp t",
            t=T,
            hp=H_p,
            wp=W_p,
            dp=D_p,
        )

        # Extract hierarchical features through the encoder pyramid
        x_encoded = self.stage1(x_encoded)
        x_encoded = self.stage2(x_encoded)
        x_encoded = self.stage3(x_encoded)
        latent = self.stage4(x_encoded)

        # Decode latent representation into clean and noise components
        clean_pred, noise_pred = self.decoder(latent, spatial_grid, T)
        reconstruction = clean_pred + noise_pred

        if return_heads:
            return reconstruction, clean_pred, noise_pred, mask
            
        return reconstruction, mask