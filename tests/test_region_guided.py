from __future__ import annotations

import pytest
import torch

from utils.region_guided import region_guided_projection


def _synthetic_regions() -> tuple[torch.Tensor, ...]:
    wm_trace = torch.tensor([-1.0, -1.0, 1.0, 1.0])
    csf_trace = torch.tensor([-1.0, 1.0, 1.0, -1.0])
    gm_signal = torch.tensor([1.0, -1.0, 1.0, -1.0])

    bold = torch.zeros(1, 4, 1, 1, 6)
    bold[0, :, 0, 0, 0] = gm_signal + 2.0 * wm_trace - 0.5 * csf_trace
    bold[0, :, 0, 0, 1] = 2.0 * gm_signal - wm_trace + 3.0 * csf_trace
    bold[0, :, 0, 0, 2:4] = wm_trace[:, None]
    bold[0, :, 0, 0, 4:6] = csf_trace[:, None]

    gm = torch.tensor([[[[1, 1, 0, 0, 0, 0]]]], dtype=torch.float32)
    wm = torch.tensor([[[[0, 0, 1, 1, 0, 0]]]], dtype=torch.float32)
    csf = torch.tensor([[[[0, 0, 0, 0, 1, 1]]]], dtype=torch.float32)
    expected_clean = torch.zeros_like(bold)
    expected_clean[0, :, 0, 0, 0] = gm_signal
    expected_clean[0, :, 0, 0, 1] = 2.0 * gm_signal
    expected_clean[0, :, 0, 0, 2:4] = wm_trace[:, None]
    expected_clean[0, :, 0, 0, 4:6] = csf_trace[:, None]
    return bold, gm, wm, csf, expected_clean


def test_projection_removes_region_mean_components_from_gm_only() -> None:
    bold, gm, wm, csf, expected_clean = _synthetic_regions()

    result = region_guided_projection(bold, gm, wm, csf)

    torch.testing.assert_close(result.clean, expected_clean, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(result.reconstruction, bold, atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(result.noise[..., 2:]) == 0
    assert result.nuisance_traces.shape == (1, 4, 2)

    residual_gm = result.clean[0, :, 0, 0, :2]
    for trace in result.nuisance_traces[0].unbind(dim=-1):
        torch.testing.assert_close(
            residual_gm.T @ trace,
            torch.zeros(2),
            atol=1e-5,
            rtol=0,
        )


def test_projection_rejects_empty_tissue_masks() -> None:
    bold, gm, wm, csf, _ = _synthetic_regions()
    csf.zero_()

    with pytest.raises(ValueError, match="CSF mask is empty"):
        region_guided_projection(bold, gm, wm, csf)


def test_projection_rejects_time_varying_anatomical_masks() -> None:
    bold, gm, wm, csf, _ = _synthetic_regions()
    varying_gm = gm.expand(1, 4, 1, 1, 6).clone()
    varying_gm[:, 1, :, :, 0] = 0

    with pytest.raises(ValueError, match="GM mask must be static across time"):
        region_guided_projection(bold, varying_gm, wm, csf)
