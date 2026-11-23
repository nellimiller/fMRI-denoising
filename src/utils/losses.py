"""Identifiable objective for the asymmetric dual-head decomposition.

In percent-signal-change space, inside the parenchymal brain mask::

    X~(v,t) = S(v,t) + N(v,t) + e(v,t),      N(v,t) = sum_j beta_j(v) r_j(t)

with ``S`` the band-limited, GM-gated clean signal and ``r_j`` nuisance regressors
from visible eroded WM/CSF voxels. The loss is

    L = w_rec  * L_rec                         masked-voxel reconstruction
      + w_tis  * L_tissue                      clean suppression in eroded WM u CSF
      + w_band * L_band                        out-of-band clean energy in GM
      + w_ort  * L_ortho                       squared correlation clean vs. regressors

where

    L_rec    = sum_{v in M, t} e(v,t)^2 / (|M| T),   M = hidden voxels inside the brain
    L_tissue = mean_{v in erode(WM) u erode(CSF), t} S_pre(v,t)^2
    L_band   = mean_{v in GM} ||(I - B) S_pre(v,.)||^2 / T
    L_ortho  = mean_{v in GM, j} rho(S(v,.), r_j)^2

``S_pre`` is the clean head *before* gating and band-limiting: those architectural
constraints make penalties on the final ``S`` vanish identically, which would leave the
head weights unconstrained. With ``hard_bandpass=False`` (ablation) ``S_pre == S``.

There is no DC-dominated unnormalised FFT term and no variance-of-masked-tensor
``flat`` term.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from ..data.channels import BRAIN, CSF, DYN, GM, WM
    from .spectral import band_weights, apply_band
    from .tissue import erode_mask
except ImportError:
    from data.channels import BRAIN, CSF, DYN, GM, WM
    from utils.spectral import apply_band, band_weights
    from utils.tissue import erode_mask


def _masked_mean(values: torch.Tensor, weight: torch.Tensor, extra: int = 1) -> torch.Tensor:
    """``sum(values * weight) / (sum(weight) * extra)`` with an empty-mask guard (returns 0)."""
    denominator = weight.sum() * extra
    return (values * weight).sum() / denominator.clamp_min(1.0)


class IdentifiableDenoisingLoss(nn.Module):
    """Masked-token reconstruction plus tissue, spectral and orthogonality priors.

    Args:
        rec_weight, tissue_weight, band_weight, ortho_weight: Term weights.
        erosion_iterations: Erosion of WM/CSF for the tissue penalty (match the model).
        band_low, band_high: Pass band in Hz (match the model).
        eps: Numerical floor for correlations.
    """

    def __init__(
        self,
        rec_weight: float = 1.0,
        tissue_weight: float = 1.0,
        band_weight: float = 1.0,
        ortho_weight: float = 0.1,
        erosion_iterations: int = 1,
        band_low: float = 0.01,
        band_high: float = 0.15,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.rec_weight = rec_weight
        self.tissue_weight = tissue_weight
        self.band_weight = band_weight
        self.ortho_weight = ortho_weight
        self.erosion_iterations = erosion_iterations
        self.band_low = band_low
        self.band_high = band_high
        self.eps = eps

    def forward(
        self,
        out: dict[str, torch.Tensor],
        x: torch.Tensor,
        tr: torch.Tensor | float = 1.0,
        prior_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Compute all loss terms.

        Args:
            out: Model output dict (``clean``, ``noise``, ``recon``, ``clean_pre``,
                ``regressors``, ``voxel_mask``).
            x: Input sample ``[B, 6, T, X, Y, Z]``.
            tr: Repetition time(s) in seconds.
            prior_scale: Multiplier in ``[0, 1]`` on the three prior terms (tissue, band,
                ortho) for warm-up; ``rec`` is never scaled. ``1`` is the specified objective.

        Returns:
            Dict with ``total``, ``rec``, ``tissue``, ``band``, ``ortho`` and detached
            diagnostics (``clean_var_frac``, ``noise_var_frac``, ``resid_var_frac``: shares
            of the target variance in masked GM voxels).
        """
        target = x[:, DYN : DYN + 1].float()
        B, _, T = target.shape[:3]
        brain = (x[:, BRAIN : BRAIN + 1, 0] > 0)  # [B,1,X,Y,Z]
        gm = (x[:, GM : GM + 1, 0] > 0) & brain
        wm = (x[:, WM : WM + 1, 0] > 0) & brain
        csf = (x[:, CSF : CSF + 1, 0] > 0) & brain

        clean = out["clean"].float()
        noise = out["noise"].float()
        clean_pre = out["clean_pre"].float()
        regressors = out["regressors"].float()  # [B,J,T]

        # 1. Masked-token reconstruction (hidden voxels inside the brain only).
        residual = target - clean - noise
        hidden = (out["voxel_mask"] & brain).float().unsqueeze(2)  # [B,1,1,X,Y,Z]
        rec = _masked_mean(residual.square(), hidden, T)

        # 2. Clean suppression inside 1-voxel-eroded WM u CSF.
        tissue_region = (
            erode_mask(wm, self.erosion_iterations) | erode_mask(csf, self.erosion_iterations)
        ).float().unsqueeze(2)
        tissue = _masked_mean(clean_pre.square(), tissue_region, T)

        # 3. Out-of-band energy of the (pre-constraint) clean head in GM.
        if isinstance(tr, torch.Tensor) and tr.ndim > 0 and tr.shape[0] != B:
            raise ValueError("tr must be scalar or have one entry per sample.")
        weights = band_weights(T, tr, self.band_low, self.band_high, device=target.device)
        if weights.shape[0] == 1 and B > 1:
            weights = weights.expand(B, -1)
        out_of_band = clean_pre - apply_band(clean_pre, weights, time_dim=2)
        gm_t = gm.float().unsqueeze(2)
        band = _masked_mean(out_of_band.square(), gm_t, T)

        # 4. Squared Pearson correlation between the final clean signal and every regressor.
        centered = clean - clean.mean(dim=2, keepdim=True)  # [B,1,T,X,Y,Z]
        r = regressors - regressors.mean(dim=-1, keepdim=True)
        r = r / r.norm(dim=-1, keepdim=True).clamp_min(self.eps)  # unit-norm rows
        dot = torch.einsum("bctxyz,bjt->bjxyz", centered, r)  # [B,J,X,Y,Z]
        norm = centered.norm(dim=2).clamp_min(self.eps)  # [B,1,X,Y,Z]
        rho_sq = (dot / norm).square()
        active = (r.norm(dim=-1) > 0).float()[:, :, None, None, None]  # drop degenerate regressors
        weight = gm.float() * active  # [B,J,X,Y,Z]
        ortho = _masked_mean(rho_sq, weight)

        total = self.rec_weight * rec + prior_scale * (
            self.tissue_weight * tissue + self.band_weight * band + self.ortho_weight * ortho
        )

        with torch.no_grad():
            region = (gm & out["voxel_mask"]).float().unsqueeze(2)
            target_var = _masked_mean((target - target.mean(2, keepdim=True)).square(), region, T)

            def share(signal: torch.Tensor) -> torch.Tensor:
                centered_signal = signal - signal.mean(2, keepdim=True)
                return _masked_mean(centered_signal.square(), region, T) / target_var.clamp_min(self.eps)

            diagnostics = {
                "clean_var_frac": share(clean),
                "noise_var_frac": share(noise),
                "resid_var_frac": share(residual),
            }

        return {
            "total": total,
            "rec": rec,
            "tissue": tissue,
            "band": band,
            "ortho": ortho,
            **diagnostics,
        }
