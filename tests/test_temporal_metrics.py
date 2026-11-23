from __future__ import annotations

import pytest
import torch

from utils.region_guided import region_guided_projection
from utils.temporal_metrics import dynamic_metrics, temporal_metrics


def _synthetic_regions() -> tuple[torch.Tensor, ...]:
    wm_trace = torch.tensor([-1.0, -1.0, 1.0, 1.0])
    csf_trace = torch.tensor([-1.0, 1.0, 1.0, -1.0])
    gm_signal = torch.tensor([1.0, -1.0, 1.0, -1.0])
    bold = torch.zeros(4, 1, 1, 6)
    bold[:, 0, 0, 0] = gm_signal + 2.0 * wm_trace - 0.5 * csf_trace
    bold[:, 0, 0, 1] = 2.0 * gm_signal - wm_trace + 3.0 * csf_trace
    bold[:, 0, 0, 2:4] = wm_trace[:, None]
    bold[:, 0, 0, 4:6] = csf_trace[:, None]
    gm = torch.tensor([[[1, 1, 0, 0, 0, 0]]], dtype=torch.float32)
    wm = torch.tensor([[[0, 0, 1, 1, 0, 0]]], dtype=torch.float32)
    csf = torch.tensor([[[0, 0, 0, 0, 1, 1]]], dtype=torch.float32)
    return bold, gm, wm, csf


def test_temporal_metrics_detect_projection_and_preserved_variation() -> None:
    bold, gm, wm, csf = _synthetic_regions()
    result = region_guided_projection(
        bold.unsqueeze(0),
        gm.unsqueeze(0),
        wm.unsqueeze(0),
        csf.unsqueeze(0),
    )

    metrics = temporal_metrics(bold, result.clean[0], gm, wm, csf)

    assert metrics["gm_abs_corr_wm_clean_median"] < 1e-5
    assert metrics["gm_abs_corr_csf_clean_median"] < 1e-5
    assert metrics["gm_temporal_std_ratio_median"] < 1.0
    assert metrics["gm_temporal_correlation_with_input_median"] is not None
    assert metrics["gm_high_frequency_fraction_input_median"] is not None


def test_temporal_metrics_reject_time_varying_masks() -> None:
    bold, gm, wm, csf = _synthetic_regions()
    varying_gm = gm.unsqueeze(0).expand(4, 1, 1, 6).clone()
    varying_gm[1, 0, 0, 0] = 0

    with pytest.raises(ValueError, match="GM mask must be static across time"):
        temporal_metrics(bold, bold, varying_gm, wm, csf)


def test_dynamic_metrics_decomposition_shares_tsnr_and_band_power() -> None:
    T = 40
    t = torch.arange(T).float()
    gm = torch.ones(2, 2, 2)
    signal = torch.sin(2 * torch.pi * 0.075 * t)  # bin-aligned (3 cycles), in the 0.01-0.15 Hz band
    nuisance = torch.sin(2 * torch.pi * 0.30 * t)  # outside the band
    clean = signal[:, None, None, None].expand(T, 2, 2, 2).clone()
    noise = 0.5 * nuisance[:, None, None, None].expand(T, 2, 2, 2).clone()
    resid = 0.1 * torch.cos(2 * torch.pi * 0.2 * t)[:, None, None, None].expand(T, 2, 2, 2)
    dyn = clean + noise + resid

    m = dynamic_metrics(dyn, clean, noise, gm, tr=1.0)

    expected_r2 = 1.0 - float(resid[:, 0, 0, 0].var(unbiased=False) / dyn[:, 0, 0, 0].var(unbiased=False))
    assert m["r2_gm"] == pytest.approx(expected_r2, abs=1e-5)
    reference_corr = float(torch.corrcoef(torch.stack([clean[:, 0, 0, 0], dyn[:, 0, 0, 0]]))[0, 1])
    assert m["gm_corr_clean_input_median"] == pytest.approx(reference_corr, abs=1e-5)
    assert 0.85 < reference_corr < 0.95
    assert m["var_clean_frac"] + m["var_noise_frac"] + m["var_resid_frac"] + m["var_cross_frac"] == pytest.approx(1.0)
    assert m["var_clean_frac"] > 0.7 and m["var_noise_frac"] < 0.25
    assert m["band_power_frac_clean_median"] > 0.99 and m["band_power_frac_noise_median"] < 0.01
    assert m["gm_tsnr_phys_removed_median"] > m["gm_tsnr_input_median"] * 0.99
    assert m["gm_tsnr_clean_median"] == pytest.approx(100.0 / float(signal.std(unbiased=False)), rel=1e-4)


def test_dynamic_metrics_validates_inputs() -> None:
    x = torch.randn(8, 2, 2, 2)
    with pytest.raises(ValueError):
        dynamic_metrics(x, x[:4], x, torch.ones(2, 2, 2), 1.0)
    with pytest.raises(ValueError):
        dynamic_metrics(x, x, x, torch.zeros(2, 2, 2), 1.0)


def test_dynamic_metrics_adds_patch_and_lightweight_diagnostics() -> None:
    T, shape = 20, (3, 3, 3)
    signal = torch.sin(2 * torch.pi * 0.1 * torch.arange(T).float())
    clean = signal[:, None, None, None].expand(T, *shape).clone()
    dyn = clean + 0.1 * torch.randn(T, *shape, generator=torch.Generator().manual_seed(2))
    noise = torch.zeros_like(clean)
    gm = torch.ones(shape, dtype=torch.bool)
    wm = torch.zeros(shape, dtype=torch.bool)
    csf = torch.zeros(shape, dtype=torch.bool)
    wm[0, 0, 0] = True

    metrics = dynamic_metrics(
        dyn,
        clean,
        noise,
        gm,
        tr=1.0,
        temporal_patch_size=3,
        wm_mask=wm,
        csf_mask=csf,
        erosion_iterations=0,
    )

    assert metrics["gm_patch_corr_clean_input_median"] > 0.99
    assert metrics["clean_active_voxel_frac"] == pytest.approx(1.0)
    assert metrics["gm_patch_tsnr_before"] is not None
    assert metrics["gm_patch_tsnr_after"] is not None
    assert metrics["gm_patch_band_power_frac"] > 0.99
    assert metrics["wm_csf_leakage_ratio"] is not None
    assert metrics["gm_spatial_grad_ratio"] is not None
    assert metrics["residual_lag1_acf"] is not None
