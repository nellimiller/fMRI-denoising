"""Central Masked Autoencoder: patch embedding, Swin pyramid, U-Net decoder, dual heads.

Signal model (percent-signal-change space, inside the parenchymal brain mask)::

    X~(v, t) = S_clean(v, t) + N_phys(v, t) + e(v, t)

* ``S_clean = g(v) * B[ clean_head ]`` : soft-GM gated, band-limited (0.01-0.15 Hz).
* ``N_phys = sum_j beta_j(v) r_j(t)`` : static coupling maps times fixed nuisance
  regressors from *visible* eroded WM/CSF voxels (eroded WM mean, eroded CSF mean,
  top-K aCompCor components).
* ``e`` : unmodeled, spatially incoherent / high-frequency (thermal) residual.

Masking hides tubes of the *dynamic* channel only; the anatomy channels are always
visible through the second embedding pathway and the voxel-resolution stem.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

try:
    from ..data.channels import (
        ANAT_BRAIN,
        ANAT_CSF,
        ANAT_GM,
        ANAT_WM,
        ANATOMY_CHANNELS,
        DYN,
        NUM_ANATOMY_CHANNELS,
        NUM_CHANNELS,
    )
    from ..utils.masking import apply_swin_mask, denoise_complementary, token_mask_to_voxel_mask
    from ..utils.spectral import bandpass
    from ..utils.tissue import nuisance_regressors, soft_support
    from .decoder import CascadingDecoder
    from .patch_embed import fMRIPatchEmbed4D
    from .swift_official import MERGING_MODE, BasicLayer
except ImportError:
    from data.channels import (
        ANAT_BRAIN,
        ANAT_CSF,
        ANAT_GM,
        ANAT_WM,
        ANATOMY_CHANNELS,
        DYN,
        NUM_ANATOMY_CHANNELS,
        NUM_CHANNELS,
    )
    from models.decoder import CascadingDecoder
    from models.patch_embed import fMRIPatchEmbed4D
    from models.swift_official import MERGING_MODE, BasicLayer
    from utils.masking import apply_swin_mask, denoise_complementary, token_mask_to_voxel_mask
    from utils.spectral import bandpass
    from utils.tissue import nuisance_regressors, soft_support


class HierarchicalSwinMAE(nn.Module):
    """4D Swin Transformer Masked Autoencoder with U-Net decoder and asymmetric heads.

    Args:
        patch_size: Base spatial patch size. Defaults to 14.
        in_chans: Number of input channels; must be 6 (``dyn, static_anat, gm, wm, csf, brain``).
        embed_dim: Initial embedding dimension for the Swin blocks. Defaults to 96.
        mask_ratio: Fraction of valid tubes to hide during training. Defaults to 0.75.
        n_comp: aCompCor components in the nuisance basis (``J = 2 + n_comp``).
        band_low, band_high: Hemodynamic pass band (Hz) of the clean head.
        erosion_iterations: Erosion of WM/CSF before regressor extraction.
        decoder_skips: ``"full"`` (default) or ``"none"`` (ablation: bottleneck-only decoder).
        free_noise_head: Ablation: free per-frame noise head instead of ``sum_j beta_j r_j``.
        hard_bandpass: Band-limit/gate the clean head architecturally (default). If
            ``False`` the clean head is ungated and unfiltered and the loss alone
            must enforce the priors (ablation).
        window_size: 4D Swin window ``(D, H, W, T)``.
        num_heads: Heads per stage; stage widths are ``E, 2E, 4E, 8E``.
        full_res_channels: Channels of the voxel-resolution decoder layers.
    """

    def __init__(
        self,
        patch_size: int = 14,
        in_chans: int = NUM_CHANNELS,
        embed_dim: int = 96,
        mask_ratio: float = 0.75,
        n_comp: int = 3,
        band_low: float = 0.01,
        band_high: float = 0.15,
        erosion_iterations: int = 1,
        hard_bandpass: bool = True,
        window_size: tuple[int, int, int, int] = (3, 3, 3, 3),
        num_heads: tuple[int, int, int, int] = (3, 6, 12, 24),
        full_res_channels: int = 24,
        decoder_skips: str = "full",
        free_noise_head: bool = False,
    ) -> None:
        """Initialize the embedding, masking, Swin pyramid, and decoder."""
        super().__init__()
        if in_chans != NUM_CHANNELS:
            raise ValueError(f"in_chans must be {NUM_CHANNELS} (dyn, static_anat, gm, wm, csf, brain).")
        self.mask_ratio = mask_ratio
        self.n_comp = n_comp
        self.n_regressors = 2 + n_comp
        self.band_low = band_low
        self.band_high = band_high
        self.erosion_iterations = erosion_iterations
        self.hard_bandpass = hard_bandpass
        self.free_noise_head = free_noise_head
        self.patch_size = patch_size

        self.patch_embed = fMRIPatchEmbed4D(
            patch_size, embed_dim, merge_stages=3, dyn_chans=1, anat_chans=NUM_ANATOMY_CHANNELS
        )

        # Learnable mask token replaces the *dynamic* embedding of hidden tubes.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Position enters only through translation-equivariant relative position bias in the
        # window attention and through the anatomy embedding; there is no absolute pos_embed.
        dims = [embed_dim * 2**i for i in range(4)]
        self.stage1 = BasicLayer(dims[0], 2, num_heads[0], window_size, drop_path=0.0, downsample=None)
        self.stage2 = BasicLayer(dims[1], 2, num_heads[1], window_size, drop_path=0.0, downsample=None)
        self.stage3 = BasicLayer(dims[2], 2, num_heads[2], window_size, drop_path=0.0, downsample=None)
        self.stage4 = BasicLayer(dims[3], 2, num_heads[3], window_size, drop_path=0.0, downsample=None)
        merge = MERGING_MODE["mergingv2"]
        self.merge1 = merge(dim=dims[0], spatial_dims=4, c_multiplier=2)
        self.merge2 = merge(dim=dims[1], spatial_dims=4, c_multiplier=2)
        self.merge3 = merge(dim=dims[2], spatial_dims=4, c_multiplier=2)

        self.decoder = CascadingDecoder(
            embed_dim,
            patch_size,
            n_regressors=self.n_regressors,
            anat_chans=NUM_ANATOMY_CHANNELS,
            full_res_channels=full_res_channels,
            skips=decoder_skips,
            free_noise_head=free_noise_head,
        )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _merge(module: nn.Module, feature: torch.Tensor) -> torch.Tensor:
        """Apply a PatchMergingV2 to a ``(b, c, d, h, w, t)`` feature map."""
        feature = rearrange(feature, "b c d h w t -> b d h w t c")
        feature = module(feature)
        return rearrange(feature, "b d h w t c -> b c d h w t")

    def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad the sample and split it into dynamic, anatomy and brain tensors."""
        if x.ndim != 6 or x.shape[1] != NUM_CHANNELS:
            raise ValueError(f"Expected input [B, {NUM_CHANNELS}, T, X, Y, Z], got {tuple(x.shape)}.")
        dyn = self.patch_embed.pad(x[:, DYN : DYN + 1])
        anat = self.patch_embed.pad(x[:, list(ANATOMY_CHANNELS), 0])  # static: frame 0 suffices
        brain = anat[:, ANAT_BRAIN : ANAT_BRAIN + 1]
        return dyn, anat, brain

    @torch.no_grad()
    def valid_cells(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Maskable spatial cells ``(B, S)`` (cells holding >= 1 brain voxel) and the patch grid."""
        _, anat, brain = self._split(x)
        P = self.patch_size
        valid = torch.nn.functional.max_pool3d((brain > 0).float(), P, P) > 0
        grid = tuple(valid.shape[-3:])
        return rearrange(valid, "b 1 h w d -> b (h w d)"), grid

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        x: torch.Tensor,
        tr: torch.Tensor | float = 1.0,
        token_mask: torch.Tensor | None = None,
        mask_ratio: float | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Execute the autoencoder forward pass.

        Args:
            x: Sample ``[B, 6, T, X, Y, Z]``: ``[dyn, static_anat, gm, wm, csf, brain]``.
            tr: Repetition time in seconds, scalar or ``[B]`` (sets the pass-band bins).
            token_mask: Optional explicit spatial tube mask ``[B, S]`` (``True`` = hidden),
                e.g. one of ``K`` complementary masks. Sampled randomly when ``None``.
            mask_ratio: Overrides ``self.mask_ratio`` for the sampled mask.
            generator: RNG for the sampled mask.

        Returns:
            Dict (all volumes cropped to the input ``X, Y, Z``):

            * ``clean`` ``[B,1,T,X,Y,Z]``: ``S_clean`` (gated, band-limited);
            * ``noise`` ``[B,1,T,X,Y,Z]``: ``N_phys``;
            * ``recon`` ``clean + noise``;
            * ``clean_pre`` ``[B,1,T,X,Y,Z]``: head output before gating / filtering
              (what the tissue and band penalties act on);
            * ``beta`` ``[B,J,X,Y,Z]`` and ``regressors`` ``[B,J,T]``;
            * ``token_mask`` / ``valid`` ``[B,S]`` and ``voxel_mask`` ``[B,1,X,Y,Z]`` (hidden);
            * ``grid`` the patch grid.
        """
        B, _, T, X, Y, Z = x.shape
        dyn, anat, brain = self._split(x)
        padded = tuple(dyn.shape[-3:])

        dyn_tokens, anat_tokens, grid, valid = self.patch_embed(dyn, anat, brain)
        S = anat_tokens.shape[1]

        ratio = self.mask_ratio if mask_ratio is None else mask_ratio
        dyn_tokens, mask = apply_swin_mask(
            dyn_tokens, self.mask_token, ratio, True, T, valid=valid, token_mask=token_mask, generator=generator
        )
        spatial_mask = mask.view(B, T, S)[:, 0] > 0

        # Anatomy is added to every token, masked or not.
        tokens = dyn_tokens + anat_tokens.unsqueeze(1).expand(B, T, S, -1).reshape(B, T * S, -1)

        h = rearrange(tokens, "b (t hp wp dp) c -> b c dp hp wp t", t=T, hp=grid[0], wp=grid[1], dp=grid[2])
        s1 = self.stage1(h)
        s2 = self.stage2(self._merge(self.merge1, s1))
        s3 = self.stage3(self._merge(self.merge2, s2))
        s4 = self.stage4(self._merge(self.merge3, s3))

        raw, second = self.decoder({"s1": s1, "s2": s2, "s3": s3, "s4": s4}, anat, T)

        # --- regressors from visible eroded WM/CSF voxels only (no gradient) ---
        hidden = token_mask_to_voxel_mask(spatial_mask, grid, self.patch_size, padded)
        regressors = nuisance_regressors(
            dyn,
            anat[:, ANAT_WM : ANAT_WM + 1],
            anat[:, ANAT_CSF : ANAT_CSF + 1],
            ~hidden,
            n_comp=self.n_comp,
            erosion_iterations=self.erosion_iterations,
        )

        brain_f = (brain > 0).float()
        gate = soft_support(anat[:, ANAT_GM : ANAT_GM + 1]) * brain_f  # (B,1,X,Y,Z)
        raw = raw.float() * brain_f.unsqueeze(2)
        if self.hard_bandpass:
            clean = gate.unsqueeze(2) * bandpass(raw, tr, self.band_low, self.band_high, time_dim=2)
            # Gate and filter make penalties on `clean` vanish by construction, so the tissue /
            # band losses act on the un-gated, un-filtered head output instead.
            clean_pre = raw
        else:
            clean = raw
            clean_pre = raw

        if self.free_noise_head:
            beta = torch.zeros(B, self.n_regressors, *padded, device=x.device)  # unused in this ablation
            noise = second.float()
        else:
            beta = second
            noise = torch.einsum("bjxyz,bjt->btxyz", beta.float(), regressors.float()).unsqueeze(1)
        noise = noise * brain_f.unsqueeze(2)

        def crop(volume: torch.Tensor) -> torch.Tensor:
            return volume[..., :X, :Y, :Z]

        return {
            "clean": crop(clean),
            "noise": crop(noise),
            "recon": crop(clean + noise),
            "clean_pre": crop(clean_pre),
            "beta": crop(beta.float()),
            "regressors": regressors,
            "token_mask": spatial_mask,
            "valid": valid,
            "voxel_mask": crop(hidden),
            "grid": grid,
        }

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def denoise_single(
        self,
        x: torch.Tensor,
        tr: torch.Tensor | float = 1.0,
        mask_ratio: float | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Single-pass inference (ablation): one random tube mask, all voxels kept.

        With ``mask_ratio=0`` nothing is hidden (full visibility). Visible voxels see their own
        measurements, so the outputs are *not* J-invariant: fits are optimistic and identity
        shortcuts are possible. Provided for the inference ablation only.
        """
        out = self(x, tr, mask_ratio=mask_ratio, generator=generator)
        out["count"] = torch.ones_like(out["voxel_mask"], dtype=torch.float32).unsqueeze(2)
        return out

    @torch.no_grad()
    def denoise_complementary(
        self,
        x: torch.Tensor,
        tr: torch.Tensor | float = 1.0,
        K: int = 4,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """J-invariant full-volume inference with ``K`` complementary tube masks.

        Each voxel is averaged only over the ``K - 1`` passes in which it was hidden
        (hidden ratio per pass ``(K - 1) / K``; ``0.75`` for ``K = 4``).

        Returns:
            ``clean``, ``noise``, ``recon`` ``[B,1,T,X,Y,Z]``, ``count`` ``[B,1,1,X,Y,Z]``,
            the valid cells and the grid.
        """
        valid, grid = self.valid_cells(x)

        def forward_fn(token_mask: torch.Tensor) -> dict[str, torch.Tensor]:
            return self(x, tr, token_mask=token_mask)

        result = denoise_complementary(
            forward_fn,
            valid,
            grid,
            self.patch_size,
            tuple(x.shape[-3:]),
            K=K,
            generator=generator,
            keys=("clean", "noise", "recon"),
        )
        result["valid"] = valid
        result["grid"] = grid
        return result
