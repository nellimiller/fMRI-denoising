from __future__ import annotations

import torch

from data.channels import BRAIN, CSF, DYN, GM, NUM_CHANNELS, STATIC, WM
from evaluate_nsd import (
    METRIC_NAMES,
    _aggregate,
    _json_safe,
    _sample_spatiotemporal_shape,
    _save_visualization,
    _select_evaluation_indices,
    _select_visual_indices,
    _window_metrics,
)


def _sample(T: int = 16, shape=(10, 12, 8)) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    sample = torch.zeros(NUM_CHANNELS, T, *shape)
    brain = torch.zeros(*shape)
    brain[1:-1, 1:-1, 1:-1] = 1
    gm = brain.clone()
    gm[3:7, 3:9, 2:6] = 0
    wm = brain - gm
    t = torch.arange(T).float()
    signal = torch.sin(2 * torch.pi * 0.06 * t)
    sample[DYN] = (signal[:, None, None, None] * gm + 0.3 * torch.randn(T, *shape, generator=g)) * brain
    for channel, m in ((STATIC, torch.randn(*shape, generator=g) * brain), (GM, gm), (WM, wm), (CSF, torch.zeros(*shape)), (BRAIN, brain)):
        sample[channel] = m
    return sample


def test_sample_shape_uses_time_dimension_not_channel_count() -> None:
    sample = torch.empty(NUM_CHANNELS, 40, 8, 16, 24)

    assert _sample_spatiotemporal_shape(sample) == (40, 8, 16, 24)


def test_aggregate_groups_windows_by_split_subject_and_run() -> None:
    rows = []
    for run, r2 in (("run1", 0.2), ("run1", 0.4), ("run2", 0.9)):
        row = {"split": "test", "subject": "subj08", "run": run}
        row.update({name: None for name in METRIC_NAMES})
        row["r2_gm"] = r2
        rows.append(row)

    runs = _aggregate(rows, ("split", "subject", "run"))

    assert len(runs) == 2
    assert runs[0]["num_windows"] == 2
    assert abs(runs[0]["r2_gm"] - 0.3) < 1e-12
    assert runs[1]["r2_gm"] == 0.9
    assert runs[0]["gm_tsnr_input_median"] is None


def test_visual_and_evaluation_windows_are_spread_across_each_subject() -> None:
    entries = [{"subject": "subj08", "run": f"run{i}", "start": i} for i in range(10)] + [
        {"subject": "subj07", "run": "run1", "start": 0}
    ]

    assert _select_visual_indices(entries, examples_per_subject=3) == {0, 4, 9, 10}
    assert _select_visual_indices(entries, 3, subjects={"subj08"}) == {0, 4, 9}
    assert _select_evaluation_indices(entries, {"subj08"}, 0) == set(range(10))
    assert _select_evaluation_indices(entries, {"subj08"}, 2) == {0, 9}


def test_window_metrics_perfect_estimate_scores_r2_one() -> None:
    sample = _sample()
    dyn = sample[DYN]
    clean, noise = dyn.clone(), torch.zeros_like(dyn)

    metrics = _window_metrics(sample, clean, noise, tr=1.0)

    assert set(METRIC_NAMES) <= set(metrics)
    assert abs(metrics["r2_brain"] - 1.0) < 1e-6 and abs(metrics["mse_brain"]) < 1e-10
    assert abs(metrics["gm_corr_clean_input_median"] - 1.0) < 1e-5
    assert abs(metrics["var_clean_frac"] - 1.0) < 1e-5 and abs(metrics["var_resid_frac"]) < 1e-8


def test_json_safe_replaces_non_finite_numbers() -> None:
    assert _json_safe({"a": float("nan"), "b": [1.0, float("inf")], "c": 2}) == {"a": None, "b": [1.0, None], "c": 2}


def test_visualization_saves_components_masks_traces_and_spectra(tmp_path) -> None:
    sample = _sample()
    dyn = sample[DYN]
    clean, noise = dyn * 0.6, dyn * 0.1
    output_path = tmp_path / "example.png"
    entry = {"subject": "subj08", "run": "run1", "start": 0, "stop": 16}

    _save_visualization(output_path, sample, clean, noise, entry, tr=1.0)

    assert output_path.is_file() and output_path.stat().st_size > 0
