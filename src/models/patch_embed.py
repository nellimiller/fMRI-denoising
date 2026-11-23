"""Dynamic padding and tokenization of 4D fMRI volumes for Swin Transformers.

The embedding has two pathways so that masking can be restricted to the dynamic
(BOLD) channel:

* ``proj_dyn`` embeds the per-frame dynamic PSC patch; masked tubes replace this
  embedding with the learnable mask token.
* ``proj_anat`` embeds the time-invariant anatomy (``static_anat``, GM, WM, CSF,
  brain) once per volume. It is added to every token, masked or not, so the
  encoder always sees the subject's true geometry.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def padded_grid(size: tuple[int, int, int], patch_size: int, grid_multiple: int) -> tuple[int, int, int]:
    """Patch-grid shape after rounding up to ``grid_multiple`` (hierarchical pooling)."""
    return tuple(
        (((s + patch_size - 1) // patch_size + grid_multiple - 1) // grid_multiple) * grid_multiple
        for s in size
    )


class fMRIPatchEmbed4D(nn.Module):
    """Slices 4D fMRI volumes into token embeddings with dynamic spatial padding.

    Args:
        patch_size: 3D cube voxel dimension grouped into a single token.
        embed_dim: Size of the latent token vector.
        merge_stages: Number of hierarchical downsampling stages in the network.
        dyn_chans: Channels of the masked dynamic input (1: PSC).
        anat_chans: Channels of the always-visible anatomy input
            (``static_anat, gm, wm, csf, brain``).
    """

    def __init__(
        self,
        patch_size: int = 14,
        embed_dim: int = 96,
        merge_stages: int = 3,
        dyn_chans: int = 1,
        anat_chans: int = 5,
    ) -> None:
        """Initialize the spatial patch embedding projections."""
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Calculate the required grid multiple to ensure dimensions survive pooling
        self.grid_multiple = 2**merge_stages

        self.proj_dyn = nn.Conv3d(dyn_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.proj_anat = nn.Conv3d(anat_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def padded_shape(self, spatial: tuple[int, int, int]) -> tuple[int, int, int]:
        """Voxel shape after padding to a whole, pooling-compatible patch grid."""
        grid = padded_grid(spatial, self.patch_size, self.grid_multiple)
        return tuple(g * self.patch_size for g in grid)

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        """Zero-pad the trailing three (spatial) dims to the padded shape."""
        spatial = tuple(x.shape[-3:])
        target = self.padded_shape(spatial)
        pad_h, pad_w, pad_d = (t - s for t, s in zip(target, spatial))
        return F.pad(x, (0, pad_d, 0, pad_w, 0, pad_h))

    def forward(
        self,
        dyn: torch.Tensor,
        anat: torch.Tensor,
        brain: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int], torch.Tensor]:
        """Tokenize padded volumes.

        Args:
            dyn: Dynamic channel ``(B, 1, T, H, W, D)``, already padded.
            anat: Anatomy channels ``(B, C_a, H, W, D)``, already padded.
            brain: Boolean/0-1 brain mask ``(B, 1, H, W, D)``, already padded.

        Returns:
            * dynamic tokens ``(B, T * S, E)`` ordered ``(t, hp, wp, dp)``,
            * anatomy tokens ``(B, S, E)`` ordered ``(hp, wp, dp)``,
            * the patch grid ``(H_p, W_p, D_p)``,
            * ``valid`` ``(B, S)`` bool: grid cells containing at least one brain voxel.
        """
        B, _, T, H, W, D = dyn.shape
        P = self.patch_size
        if H % P or W % P or D % P:
            raise ValueError(f"Spatial shape {(H, W, D)} is not divisible by patch size {P}; pad first.")

        dyn_tokens = self.proj_dyn(rearrange(dyn, "b c t h w d -> (b t) c h w d"))
        dyn_tokens = rearrange(dyn_tokens, "(b t) e hp wp dp -> b (t hp wp dp) e", b=B, t=T)

        anat_tokens = self.proj_anat(anat)
        anat_tokens = rearrange(anat_tokens, "b e hp wp dp -> b (hp wp dp) e")

        valid = F.max_pool3d((brain > 0).float(), kernel_size=P, stride=P) > 0
        valid = rearrange(valid, "b 1 hp wp dp -> b (hp wp dp)")

        grid = (H // P, W // P, D // P)
        return dyn_tokens, anat_tokens, grid, valid
