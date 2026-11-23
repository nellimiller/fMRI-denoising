"""Temporal diagnostics for observed and candidate-clean BOLD signals."""

from __future__ import annotations

import torch
from torch import Tensor

try:
    from .spectral import band_power_fraction
    from .tissue import erode_mask
except ImportError:
    from utils.spectral import band_power_fraction
    from utils.tissue import erode_mask


def _static_mask(
    mask: Tensor, bold: Tensor, name: str, require_nonempty: bool = True
) -> Tensor:
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
    if require_nonempty and not bool(mask.any()):
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


def gm_patch_traces(
    data: Tensor, gm_mask: Tensor, patch_size: int = 3
) -> tuple[Tensor, Tensor]:
    """Return GM-masked means for non-overlapping contiguous 3D patches."""
    if data.ndim != 4:
        raise ValueError(f"Expected data [T,X,Y,Z], got {tuple(data.shape)}.")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    gm = _static_mask(gm_mask, data, "GM")
    traces: list[Tensor] = []
    anchors: list[tuple[int, int, int]] = []
    x_size, y_size, z_size = gm.shape
    for x in range(0, x_size, patch_size):
        for y in range(0, y_size, patch_size):
            for z in range(0, z_size, patch_size):
                patch = gm[
                    x : min(x + patch_size, x_size),
                    y : min(y + patch_size, y_size),
                    z : min(z + patch_size, z_size),
                ]
                if not bool(patch.any()):
                    continue
                values = data[
                    :,
                    x : min(x + patch_size, x_size),
                    y : min(y + patch_size, y_size),
                    z : min(z + patch_size, z_size),
                ][:, patch]
                traces.append(values.float().mean(1))
                anchors.append((x, y, z))
    if not traces:
        raise ValueError("GM mask contains no valid patches.")
    anchor_tensor = torch.tensor(anchors, device=data.device, dtype=torch.int64)
    return torch.stack(traces, dim=1), anchor_tensor


def gm_local_patch_trace(
    data: Tensor, gm_mask: Tensor, anchor: tuple[int, int, int], patch_size: int = 3
) -> Tensor:
    """Average a GM-masked cubic neighborhood centered on an anchor voxel."""
    if data.ndim != 4:
        raise ValueError(f"Expected data [T,X,Y,Z], got {tuple(data.shape)}.")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    gm = _static_mask(gm_mask, data, "GM")
    radius = patch_size // 2
    x, y, z = anchor
    x_start, x_stop = max(0, x - radius), min(gm.shape[0], x - radius + patch_size)
    y_start, y_stop = max(0, y - radius), min(gm.shape[1], y - radius + patch_size)
    z_start, z_stop = max(0, z - radius), min(gm.shape[2], z - radius + patch_size)
    patch_mask = gm[x_start:x_stop, y_start:y_stop, z_start:z_stop]
    if not bool(patch_mask.any()):
        raise ValueError("Selected temporal patch contains no GM voxels.")
    return data[:, x_start:x_stop, y_start:y_stop, z_start:z_stop][:, patch_mask].float().mean(1)


def _temporal_correlations(left: Tensor, right: Tensor, eps: float) -> Tensor:
    centered_left = left - left.mean(dim=0, keepdim=True)
    centered_right = right - right.mean(dim=0, keepdim=True)
    denominator = centered_left.norm(dim=0) * centered_right.norm(dim=0)
    return (centered_left * centered_right).sum(dim=0) / denominator.clamp_min(eps)


def _median_finite(values: Tensor) -> float | None:
    values = values[torch.isfinite(values)]
    return float(values.median().item()) if values.numel() else None


def dynamic_metrics(
    dyn: Tensor,
    clean: Tensor,
    noise: Tensor,
    gm_mask: Tensor,
    tr: float,
    band: tuple[float, float] = (0.01, 0.15),
    eps: float = 1e-8,
    temporal_patch_size: int = 3,
    wm_mask: Tensor | None = None,
    csf_mask: Tensor | None = None,
    erosion_iterations: int = 1,
) -> dict[str, float | None]:
    """fMRI-specific diagnostics of a ``dyn = clean + noise + e`` decomposition inside GM.

    All inputs live in percent-signal-change (PSC) space, so ``tSNR = 100 / SD_t`` for a
    series whose baseline is 100 %. ``e`` is the residual ``dyn - clean - noise``.

    Args:
        dyn: Observed drift-removed PSC ``[T, X, Y, Z]``.
        clean: Estimated clean BOLD ``[T, X, Y, Z]``.
        noise: Estimated physiological component ``[T, X, Y, Z]``.
        gm_mask: Static GM mask ``[X, Y, Z]``.
        tr: Repetition time in seconds.
        band: Hemodynamic pass band in Hz.

    Returns:
        Median per-voxel de-meaned Pearson correlations with the input, tSNR before and
        after denoising (clean estimate; and input minus physiological component),
        variance shares (their remainder ``cross_frac`` is the covariance between
        components), fraction of non-DC power inside ``band`` per component, and the
        held-out explained variance ``r2_gm``. tSNR gains are descriptive: removing
        signal also raises tSNR.
    """
    if not (dyn.shape == clean.shape == noise.shape and dyn.ndim == 4):
        raise ValueError("dyn, clean and noise must share the shape [T, X, Y, Z].")
    gm = gm_mask > 0
    if not bool(gm.any()):
        raise ValueError("GM mask is empty.")

    x = dyn[:, gm].float()
    s = clean[:, gm].float()
    n = noise[:, gm].float()
    e = x - s - n

    def correlation(a: Tensor, b: Tensor) -> Tensor:
        a = a - a.mean(0, keepdim=True)
        b = b - b.mean(0, keepdim=True)
        denominator = a.norm(dim=0) * b.norm(dim=0)
        return torch.where(
            denominator > eps, (a * b).sum(0) / denominator.clamp_min(eps), torch.full_like(denominator, float("nan"))
        )

    def sd(a: Tensor) -> Tensor:
        return a.std(dim=0, unbiased=False)

    def tsnr(a: Tensor) -> Tensor:
        deviation = sd(a)
        return torch.where(deviation > eps, 100.0 / deviation.clamp_min(eps), torch.full_like(deviation, float("nan")))

    tsnr_input = tsnr(x)
    tsnr_clean = tsnr(s)
    tsnr_phys_removed = tsnr(x - n)
    total_var = x.var(dim=0, unbiased=False).sum().clamp_min(eps)
    shares = {
        "clean": s.var(dim=0, unbiased=False).sum() / total_var,
        "noise": n.var(dim=0, unbiased=False).sum() / total_var,
        "resid": e.var(dim=0, unbiased=False).sum() / total_var,
    }
    centred_x = x - x.mean(0, keepdim=True)
    energy = centred_x.square().sum().clamp_min(eps)

    patch_input, _ = gm_patch_traces(dyn, gm, temporal_patch_size)
    patch_clean, _ = gm_patch_traces(clean, gm, temporal_patch_size)
    patch_tsnr_input = tsnr(patch_input)
    patch_tsnr_clean = tsnr(patch_clean)
    clean_var = s.var(dim=0, unbiased=False)
    input_var = x.var(dim=0, unbiased=False)

    left = e[:-1] - e[:-1].mean(0, keepdim=True)
    right = e[1:] - e[1:].mean(0, keepdim=True)
    residual_denominator = left.norm(0) * right.norm(0)
    residual_valid = residual_denominator > eps
    residual_corr = (left * right).sum(0) / residual_denominator.clamp_min(eps)

    spatial_grad_ratio = None
    if all(size > 1 for size in clean.shape[1:]):
        input_grad = torch.gradient(dyn.float(), dim=(1, 2, 3))
        clean_grad = torch.gradient(clean.float(), dim=(1, 2, 3))
        input_norm = torch.stack(input_grad).square().sum(0).sqrt()
        clean_norm = torch.stack(clean_grad).square().sum(0).sqrt()
        input_mean = input_norm[:, gm].mean()
        clean_mean = clean_norm[:, gm].mean()
        spatial_grad_ratio = float(clean_mean / input_mean.clamp_min(eps))

    leakage_ratio = None
    if wm_mask is not None and csf_mask is not None:
        wm = _static_mask(wm_mask, dyn, "WM", require_nonempty=False)
        csf = _static_mask(csf_mask, dyn, "CSF", require_nonempty=False)
        wm_csf = torch.logical_or(
            erode_mask(wm.unsqueeze(0).unsqueeze(0), erosion_iterations)[0, 0],
            erode_mask(csf.unsqueeze(0).unsqueeze(0), erosion_iterations)[0, 0],
        )
        if bool(wm_csf.any()):
            leakage_ratio = float(
                clean[:, wm_csf].float().var(dim=0, unbiased=False).mean()
                / clean_var.mean().clamp_min(1e-8)
            )

    metrics: dict[str, float | None] = {
        "gm_corr_clean_input_median": _median_finite(correlation(s, x)),
        "gm_corr_recon_input_median": _median_finite(correlation(s + n, x)),
        "gm_corr_noise_input_median": _median_finite(correlation(n, x)),
        "gm_tsnr_input_median": _median_finite(tsnr_input),
        "gm_tsnr_clean_median": _median_finite(tsnr_clean),
        "gm_tsnr_phys_removed_median": _median_finite(tsnr_phys_removed),
        "gm_tsnr_gain_clean_median": _median_finite(tsnr_clean / tsnr_input),
        "gm_tsnr_gain_phys_removed_median": _median_finite(tsnr_phys_removed / tsnr_input),
        "var_clean_frac": float(shares["clean"]),
        "var_noise_frac": float(shares["noise"]),
        "var_resid_frac": float(shares["resid"]),
        "var_cross_frac": float(1.0 - sum(shares.values())),
        "r2_gm": float(1.0 - (e - e.mean(0, keepdim=True)).square().sum() / energy),
        "clean_active_voxel_frac": float((clean_var > 0.05 * input_var).float().mean()),
        "wm_csf_leakage_ratio": leakage_ratio,
        "gm_spatial_grad_ratio": spatial_grad_ratio,
        "residual_lag1_acf": _median_valid(residual_corr, residual_valid),
        "gm_patch_corr_clean_input_median": _median_finite(
            correlation(patch_clean, patch_input)
        ),
        "gm_patch_tsnr_before": _median_finite(patch_tsnr_input),
        "gm_patch_tsnr_after": _median_finite(patch_tsnr_clean),
        "gm_patch_band_power_frac": _median_finite(
            band_power_fraction(patch_clean, tr, band[0], band[1])
        ),
    }
    for name, series in (("input", x), ("clean", s), ("noise", n), ("resid", e)):
        metrics[f"band_power_frac_{name}_median"] = _median_finite(
            band_power_fraction(series, tr, band[0], band[1])
        )
    return metrics
