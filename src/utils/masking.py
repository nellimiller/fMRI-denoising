### Logic for injecting learnable masks into the model.
### This files generates random probabilistic masks masks and overwrite data with a placeholder.

import torch

def apply_swin_mask(x, mask_token, mask_ratio=0.75, tube_masking=False, T=None):
    '''
    Replaces a percentage of tokes with a learnable MASK token.
    Maintains exact sequence length for reshaping the 4D grid.

    Args:
        x: Tensor of embedded tokens [Batch, Sequence_Length, Embed_Dim]
        mask_token: Learnable parameter [1, 1 Embed_Dim]
        mask_ratio: Percentage of tokens to mask out
        tube_masking: If True, applies same spatial mask across the temporal dimension
        T: Number of timepoints (required if tube_masking=True)

    Returns:
        x_masked: Full-length sequencs with masked tokesn replaced
        mask: binary mask indicating which tokens were masked (1=masked, 0=unmasked)
    '''

    B, L, D = x.shape

    if tube_masking:
        if T is None:
            raise ValueError("T (number of timepoints) must be provided for tube masking.")
        
        # Calculate how many spatial tokens per timepoint
        S = L // T

        # Generate random noise ONLY for spatial grid []Batch, Spatial_Patches]
        noise_spatial = torch.rand(B, S, device=x.device)

        # Repreat noise across temporal dimension to create tube mask [Batch, Time, Spatial_Patches]
        noise = noise_spatial.unsqueeze(1).repeat(1, T, 1)

        # Flatten it back into 1D sequence [Batch, Sequence_Length]
        noise = noise.view(B, L)


    else:
        # Unstructured random masking: Generate noise for each token independently
        noise = torch.rand(B, L, device=x.device)

   
    # Create binary mask (1= masked, 0=unmasked) based on noise and mask ratio
    mask = (noise < mask_ratio).float() 

    # Expand mask to match token embedding dimension
    mask_expanded = mask.unsqueeze(-1).expand(-1, -1, D)

    # Apply mask
    x_masked = x * (1 - mask_expanded) + mask_expanded * mask_token

    return x_masked, mask