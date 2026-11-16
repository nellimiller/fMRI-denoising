"""Temporal diagnostics for observed and candidate-clean BOLD signals."""

from __future__ import annotations

import torch
from torch import Tensor


def _static_mask(mask: Tensor, bold: Tensor, name: str) -> Tensor:
    if not torch.isfinite(mask).all():
        raise ValueError(f"{name} mask contains non-finite values.")
    if mask.ndim == 4:
        if mask.shape[0] != bold.shape[0]:
            raise ValueError(f"{name} mask time dimension does not match BOLD.")
        if not torch.equal(mask, mask[:1].expand_as(mask)):
            raise ValueError(f"{name} mask must be static across time.")
        mask = mask[0]
    elif mask.ndim != 3:
        raise ValueError(
            f"{name} mask must have shape [X,Y,Z] or [T,X,Y,Z], "
            f"got {tuple(mask.shape)}."
        )
    if tuple(mask.shape) != tuple(bold.shape[1:]):
        raise ValueError(
            f"{name} mask shape {tuple(mask.shape)} does not match "
            f"BOLD spatial shape {tuple(bold.shape[1:])}."
        )
    mask = mask.to(device=bold.device) > 0
    if not bool(mask.any()):
        raise ValueError(f"{name} mask is empty.")
    return mask


def _median_valid(values: Tensor, valid: Tensor) -> float | None:
    if not bool(valid.any()):
        return None
    return float(values[valid].median().item())


def _median_absolute_correlation(
    data: Tensor, trace: Tensor, eps: float
) -> float | None:
    voxel_traces = data.T
    centered_voxels = voxel_traces - voxel_traces.mean(dim=1, keepdim=True)
    centered_trace = trace - trace.mean()
    denominator = centered_voxels.norm(dim=1) * centered_trace.norm()
    valid = denominator > eps
    correlation = (centered_voxels * centered_trace).sum(dim=1) / denominator.clamp_min(
        eps
    )
    return _median_valid(correlation.abs(), valid)


def _high_frequency_fraction(data: Tensor, eps: float) -> float | None:
    centered = data - data.mean(dim=0, keepdim=True)
    power = torch.fft.rfft(centered.float(), dim=0).abs().square()
    frequencies = torch.fft.rfftfreq(
        data.shape[0],
        d=1.0,
        device=data.device,
    )
    nonzero = frequencies > 0
    high = frequencies >= 0.25
    total_power = power[nonzero].sum(dim=0)
    high_power = power[high].sum(dim=0)
    fraction = high_power / total_power.clamp_min(eps)
    return _median_valid(fraction, total_power > eps)


def temporal_metrics(
    bold: Tensor,
    clean: Tensor,
    gm_mask: Tensor,
    wm_mask: Tensor,
    csf_mask: Tensor,
) -> dict[str, float | None]:
    """Measure temporal changes in GM clean estimates against observed BOLD.

    Args:
        bold: Observed BOLD with shape ``[T, X, Y, Z]``.
        clean: Candidate-clean signal with the same shape.
        gm_mask: Static GM mask ``[X,Y,Z]`` or repeated ``[T,X,Y,Z]``.
        wm_mask: Static WM mask with the same accepted shapes.
        csf_mask: Static CSF mask with the same accepted shapes.

    Returns:
        Median per-GM-voxel temporal standard-deviation and first-difference
        ratios, temporal correlation with the observed signal, high-frequency
        power fractions, and absolute correlations with observed WM/CSF mean
        traces, before and after cleaning.

    High-frequency power uses the upper half of the discrete positive
    frequency range (cycles/frame); it is a descriptive proxy, not a
    physiological noise label. WM/CSF correlations use traces from the same
    window, so they are descriptive and not an independent validation.
    """
    if bold.ndim != 4 or clean.shape != bold.shape:
        raise ValueError(
            f"Expected matching BOLD and clean tensors [T,X,Y,Z], got "
            f"{tuple(bold.shape)} and {tuple(clean.shape)}."
        )
    if bold.shape[0] < 2:
        raise ValueError("At least two time points are required.")
    if not bold.is_floating_point() or not clean.is_floating_point():
        raise TypeError("BOLD and clean must be floating-point tensors.")
    if not torch.isfinite(bold).all() or not torch.isfinite(clean).all():
        raise ValueError("BOLD and clean must contain only finite values.")

    gm = _static_mask(gm_mask, bold, "GM")
    wm = _static_mask(wm_mask, bold, "WM")
    csf = _static_mask(csf_mask, bold, "CSF")

    observed_gm = bold[:, gm].float()
    clean_gm = clean[:, gm].float()
    eps = torch.finfo(torch.float32).eps

    observed_std = observed_gm.std(dim=0, unbiased=False)
    clean_std = clean_gm.std(dim=0, unbiased=False)
    std_ratio = clean_std / observed_std.clamp_min(eps)

    observed_diff = observed_gm[1:] - observed_gm[:-1]
    clean_diff = clean_gm[1:] - clean_gm[:-1]
    observed_diff_rms = observed_diff.square().mean(dim=0).sqrt()
    clean_diff_rms = clean_diff.square().mean(dim=0).sqrt()
    diff_ratio = clean_diff_rms / observed_diff_rms.clamp_min(eps)

    wm_trace = bold[:, wm].float().mean(dim=1)
    csf_trace = bold[:, csf].float().mean(dim=1)
    valid_std = observed_std > eps
    valid_diff = observed_diff_rms > eps

    return {
        "gm_temporal_std_ratio_median": _median_valid(std_ratio, valid_std),
        "gm_first_difference_rms_ratio_median": _median_valid(diff_ratio, valid_diff),
        "gm_temporal_correlation_with_input_median": _median_valid(
            _temporal_correlations(clean_gm, observed_gm, eps),
            valid_std,
        ),
        "gm_high_frequency_fraction_input_median": _high_frequency_fraction(
            observed_gm, eps
        ),
        "gm_high_frequency_fraction_clean_median": _high_frequency_fraction(
            clean_gm, eps
        ),
        "gm_abs_corr_wm_input_median": _median_absolute_correlation(
            observed_gm, wm_trace, eps
        ),
        "gm_abs_corr_wm_clean_median": _median_absolute_correlation(
            clean_gm, wm_trace, eps
        ),
        "gm_abs_corr_csf_input_median": _median_absolute_correlation(
            observed_gm, csf_trace, eps
        ),
        "gm_abs_corr_csf_clean_median": _median_absolute_correlation(
            clean_gm, csf_trace, eps
        ),
    }


def _temporal_correlations(left: Tensor, right: Tensor, eps: float) -> Tensor:
    centered_left = left - left.mean(dim=0, keepdim=True)
    centered_right = right - right.mean(dim=0, keepdim=True)
    denominator = centered_left.norm(dim=0) * centered_right.norm(dim=0)
    return (centered_left * centered_right).sum(dim=0) / denominator.clamp_min(eps)
