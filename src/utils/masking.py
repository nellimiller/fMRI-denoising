"""Token masking logic for the Masked Autoencoder.

Generates random probabilistic masks and overwrites dropped tokens 
with a learnable placeholder while maintaining sequence length.
"""

from __future__ import annotations

import torch


def apply_swin_mask(
    x: torch.Tensor,
    mask_token: torch.Tensor,
    mask_ratio: float = 0.75,
    tube_masking: bool = False,
    T: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace a percentage of tokens with a learnable MASK token.

    Maintains exact sequence length to allow reshaping back into the 4D spatial-temporal grid.

    Args:
        x: Tensor of embedded tokens `[Batch, Sequence_Length, Embed_Dim]`.
        mask_token: Learnable parameter `[1, 1, Embed_Dim]`.
        mask_ratio: Fraction of tokens to mask out.
        tube_masking: If True, drops the exact same spatial patches across all timeframes.
        T: Number of timeframes. Required if `tube_masking` is True.

    Returns:
        A tuple containing:
        - x_masked: Full-length sequence with masked tokens replaced.
        - mask: Binary mask indicating which tokens were dropped (1=masked, 0=unmasked).
    """
    B, L, D = x.shape

    if tube_masking:
        if T is None:
            raise ValueError("T (number of timepoints) must be provided for tube masking.")
        
        # Calculate how many spatial tokens per timepoint
        S = L // T

        # Generate random noise ONLY for spatial grid [Batch, Spatial_Patches]
        noise_spatial = torch.rand(B, S, device=x.device)

        # Repreat noise across temporal dimension to create a continuoustube mask
        # Shape: [Batch, Time, Spatial_Patches]
        noise = noise_spatial.unsqueeze(1).repeat(1, T, 1)

        # Flatten it back into the 1D token sequence [Batch, Sequence_Length]
        noise = noise.view(B, L)

    else:
        # Unstructured random masking: Generate noise for each token independently
        noise = torch.rand(B, L, device=x.device)

   
    # Create binary mask (1= masked, 0=unmasked) based on noise and mask ratio
    mask = (noise < mask_ratio).float() 

    # Expand mask to match token embedding dimension
    mask_expanded = mask.unsqueeze(-1).expand(-1, -1, D)

    # Apply mask: keep original tokens where mask is 0, insert mask_token where mask is 1
    x_masked = x * (1 - mask_expanded) + mask_expanded * mask_token

    return x_masked, mask