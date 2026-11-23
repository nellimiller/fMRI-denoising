"""U-Net style decoder with skip connections and asymmetric dual heads.

The decoder works per frame (batch and time are folded together) and receives

* the four Swin stage features (``s1 .. s4``; ``s4`` is the 1x1x1 bottleneck),
* a voxel-resolution convolutional stem of the always-visible anatomy
  (``static_anat, gm, wm, csf, brain``),

so unseen subjects keep their own gyral geometry. Transposed convolutions with
``kernel == stride`` are replaced by ``Upsample + Conv3d`` blocks to avoid the
non-overlapping block boundaries they produce.

Heads (asymmetric on purpose):

* **clean head**: a *per-frame* voxelwise function, i.e. free temporal degrees of
  freedom; the model afterwards gates it spatially and band-limits it.
* **physiological noise head**: outputs *static* coupling maps ``beta_j(v)`` computed
  from the time-averaged decoder features. Noise is ``sum_j beta_j(v) r_j(t)`` with
  fixed nuisance regressors, so it has no free temporal degrees of freedom.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class UpBlock(nn.Module):
    """``Upsample(x2) -> concat skip -> Conv3d -> GroupNorm -> GELU``."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(_groups(out_channels), out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        return self.act(self.norm(self.conv(torch.cat([x, skip], dim=1))))


class AnatomyStem(nn.Module):
    """Voxel-resolution convolutional stem of the visible anatomy channels."""

    def __init__(self, in_channels: int = 5, out_channels: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, anat: torch.Tensor) -> torch.Tensor:
        return self.net(anat)


class CascadingDecoder(nn.Module):
    """Multi-scale decoder producing the raw clean signal and noise coupling maps.

    Args:
        embed_dim: Base embedding dimension ``E`` of the Swin encoder (stage widths
            ``E, 2E, 4E, 8E``).
        patch_size: Spatial patch size of the encoder (token -> voxel upsampling factor).
        n_regressors: Number of nuisance regressors ``J`` (noise-head output maps).
        anat_chans: Channels of the anatomy stem input.
        stem_channels: Stem output channels.
        full_res_channels: Channels of the voxel-resolution refinement convolutions.
        skips: ``"full"`` (default) uses the stage skips and the voxel stem; ``"none"``
            zeroes them so the decoder only sees the 1x1x1 bottleneck (ablation that
            mimics the baseline's global-bottleneck topology).
        free_noise_head: Ablation: replace the static ``beta_j(v)`` noise head by a free
            per-frame voxelwise head (symmetric to the clean head, no regressor subspace).
    """

    def __init__(
        self,
        embed_dim: int = 96,
        patch_size: int = 14,
        n_regressors: int = 5,
        anat_chans: int = 5,
        stem_channels: int = 8,
        full_res_channels: int = 24,
        skips: str = "full",
        free_noise_head: bool = False,
    ) -> None:
        super().__init__()
        if skips not in {"full", "none"}:
            raise ValueError(f"skips must be 'full' or 'none', got {skips!r}")
        self.skips = skips
        self.free_noise_head = free_noise_head
        self.patch_size = patch_size
        e = embed_dim

        self.stem = AnatomyStem(anat_chans, stem_channels)

        self.up3 = UpBlock(8 * e, 4 * e, 4 * e)  # 1^3 -> 2^3, skip s3
        self.up2 = UpBlock(4 * e, 2 * e, 2 * e)  # 2^3 -> 4^3, skip s2
        self.up1 = UpBlock(2 * e, e, e)  # 4^3 -> 8^3, skip s1

        mid = full_res_channels
        self.reduce = nn.Conv3d(e, mid, kernel_size=1)
        self.refine = nn.Sequential(
            nn.Conv3d(mid + stem_channels, full_res_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(full_res_channels, full_res_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Asymmetric heads.
        self.clean_head = nn.Sequential(
            nn.Conv3d(full_res_channels, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(16, 1, kernel_size=1),
        )
        self.noise_head = nn.Sequential(
            nn.Conv3d(full_res_channels, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(16, 1 if free_noise_head else n_regressors, kernel_size=1),
        )

    def forward(
        self,
        features: dict[str, torch.Tensor],
        anat: torch.Tensor,
        T: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode encoder features.

        Args:
            features: ``s1..s4`` as ``(B, C, D, H, W, T)`` Swin feature maps.
            anat: Padded anatomy ``(B, C_a, X, Y, Z)`` (visible in every tube).
            T: Number of frames.

        Returns:
            ``raw_clean`` ``(B, 1, T, X, Y, Z)`` and ``beta`` ``(B, J, X, Y, Z)`` (or, with
            ``free_noise_head``, the free noise estimate ``(B, 1, T, X, Y, Z)``).
        """
        B = anat.shape[0]

        def to_frames(feature: torch.Tensor) -> torch.Tensor:
            return rearrange(feature, "b c d h w t -> (b t) c h w d")

        def skip(name: str) -> torch.Tensor:
            feature = to_frames(features[name])
            return torch.zeros_like(feature) if self.skips == "none" else feature

        x = to_frames(features["s4"])
        x = self.up3(x, skip("s3"))
        x = self.up2(x, skip("s2"))
        x = self.up1(x, skip("s1"))

        full = anat.shape[-3:]
        x = F.interpolate(self.reduce(x), size=full, mode="trilinear", align_corners=False)
        stem = self.stem(anat).repeat_interleave(T, dim=0)  # same anatomy for every frame
        if self.skips == "none":
            stem = torch.zeros_like(stem)
        x = self.refine(torch.cat([x, stem], dim=1))  # ((b t), C, X, Y, Z)

        clean = self.clean_head(x)
        clean = rearrange(clean, "(b t) c x y z -> b c t x y z", b=B, t=T)

        if self.free_noise_head:
            noise = self.noise_head(x)
            return clean, rearrange(noise, "(b t) c x y z -> b c t x y z", b=B, t=T)

        static_features = rearrange(x, "(b t) c x y z -> b t c x y z", b=B, t=T).mean(dim=1)
        beta = self.noise_head(static_features)
        return clean, beta
