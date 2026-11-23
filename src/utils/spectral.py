"""Differentiable temporal Fourier filtering and spectral diagnostics.

All transforms run in float32 (cuFFT has no half-precision support for arbitrary
lengths such as the 40-frame window) and are differentiable.
"""

from __future__ import annotations

import torch

_EDGE_TOL = 1e-6  # bins sitting exactly on a band edge (e.g. 6/40 = 0.15 Hz) are inside


def band_weights(
    n_frames: int,
    tr: torch.Tensor | float,
    f_low: float = 0.01,
    f_high: float = 0.15,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Binary rFFT pass-band ``(B, F)`` for windows of ``n_frames`` sampled every ``tr`` seconds.

    DC is always excluded (``f_low > 0``). Frequencies are ``k / (n_frames * tr)`` Hz.
    """
    tr = torch.as_tensor(tr, dtype=torch.float32, device=device).reshape(-1)
    freqs = torch.arange(n_frames // 2 + 1, device=tr.device, dtype=torch.float32)
    freqs = freqs[None, :] / (n_frames * tr[:, None])
    weights = (freqs >= f_low - _EDGE_TOL) & (freqs <= f_high + _EDGE_TOL)
    weights[:, 0] = False
    return weights.float()


def apply_band(x: torch.Tensor, weights: torch.Tensor, time_dim: int = 2) -> torch.Tensor:
    """Multiply the rFFT of ``x`` along ``time_dim`` by ``weights`` ``(B, F)`` and invert.

    Args:
        x: Tensor ``(B, ..., T, ...)`` with batch first.
        weights: Per-sample frequency weights ``(B, F)`` with ``F = T // 2 + 1``.
    """
    n_frames = x.shape[time_dim]
    shape = [1] * x.ndim
    shape[0], shape[time_dim] = x.shape[0], weights.shape[1]
    with torch.autocast(device_type=x.device.type, enabled=False):
        spectrum = torch.fft.rfft(x.float(), dim=time_dim)
        spectrum = spectrum * weights.to(spectrum.device).reshape(shape)
        return torch.fft.irfft(spectrum, n=n_frames, dim=time_dim)


def bandpass(
    x: torch.Tensor,
    tr: torch.Tensor | float,
    f_low: float = 0.01,
    f_high: float = 0.15,
    time_dim: int = 2,
) -> torch.Tensor:
    """Bandpass operator ``B`` (``f_low`` to ``f_high`` Hz) along ``time_dim``."""
    weights = band_weights(x.shape[time_dim], tr, f_low, f_high, device=x.device)
    if weights.shape[0] == 1 and x.shape[0] > 1:
        weights = weights.expand(x.shape[0], -1)
    return apply_band(x, weights, time_dim)


def band_power_fraction(
    x: torch.Tensor,
    tr: float,
    f_low: float = 0.01,
    f_high: float = 0.15,
) -> torch.Tensor:
    """Fraction of non-DC power inside ``[f_low, f_high]`` Hz, per column.

    Args:
        x: ``(T, N)`` time courses.

    Returns:
        ``(N,)`` fractions; columns without non-DC power return ``nan``.
    """
    n_frames = x.shape[0]
    with torch.autocast(device_type=x.device.type, enabled=False):
        power = torch.fft.rfft(x.float() - x.float().mean(0, keepdim=True), dim=0).abs().square()
    weights = band_weights(n_frames, tr, f_low, f_high, device=x.device)[0]
    total = power[1:].sum(0)
    inside = (power * weights[:, None]).sum(0)
    return torch.where(total > 1e-12, inside / total.clamp_min(1e-12), torch.full_like(total, float("nan")))
