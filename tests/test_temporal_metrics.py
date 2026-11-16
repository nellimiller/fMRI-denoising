from __future__ import annotations

import pytest
import torch

from utils.region_guided import region_guided_projection
from utils.temporal_metrics import temporal_metrics


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
