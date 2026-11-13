from __future__ import annotations

import torch

from evaluate_nsd import (
    METRIC_NAMES,
    _aggregate,
    _masked_voxel_grid,
    _save_visualization,
    _select_visual_indices,
)


def test_masked_voxel_grid_expands_tube_mask_and_crops_padding() -> None:
    shape = (2, 3, 3, 3)
    token_mask = torch.zeros(2 * 8**3)
    token_mask[0] = 1

    voxels = _masked_voxel_grid(token_mask, shape, patch_size=2)

    assert voxels.shape == shape
    assert int(voxels.sum()) == 16
    assert torch.equal(voxels[0], voxels[1])


def test_aggregate_groups_windows_by_subject_and_run() -> None:
    rows = []
    for run, mse in (("run1", 1.0), ("run1", 3.0), ("run2", 5.0)):
        row = {"subject": "subj08", "run": run}
        row.update({name: None for name in METRIC_NAMES})
        row["mse_full"] = mse
        rows.append(row)

    runs = _aggregate(rows, ("subject", "run"))

    assert len(runs) == 2
    assert runs[0]["num_windows"] == 2
    assert runs[0]["mse_full"] == 2.0
    assert runs[1]["mse_full"] == 5.0


def test_visual_examples_are_spread_across_each_subject() -> None:
    entries = [
        {"subject": "subj08", "run": f"run{index}", "start": index}
        for index in range(10)
    ] + [{"subject": "subj07", "run": "run1", "start": 0}]

    indices = _select_visual_indices(entries, examples_per_subject=3)
    selected_subject_indices = _select_visual_indices(
        entries,
        examples_per_subject=3,
        subjects={"subj08"},
    )

    assert indices == {0, 4, 9, 10}
    assert selected_subject_indices == {0, 4, 9}


def test_visualization_saves_bold_heads_and_masks_panel(tmp_path) -> None:
    sample = torch.zeros(4, 2, 4, 4, 2)
    sample[0] = torch.randn(2, 4, 4, 2)
    sample[1, :, 1:3, 1:3, :] = 1
    prediction = sample[0] * 0.8
    clean = sample[0] * 0.7
    noise = prediction - clean
    output_path = tmp_path / "example.png"
    entry = {"subject": "subj08", "run": "run1", "start": 0, "stop": 2}

    _save_visualization(output_path, sample, prediction, clean, noise, entry)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
