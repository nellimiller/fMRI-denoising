"""Phase 1 checks: parenchymal brain mask, PSC target, 6-channel sample, dynamic-only masking."""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest
import torch

from data.brain_mask import build_brain_mask
from data.channels import BRAIN, CSF, DYN, GM, NUM_CHANNELS, STATIC, WM
from data.nsd_dataset import NSDDataset, compute_psc_window, linear_detrend_coefficients
from data_preprocessing.nsd import preprocess as pp
from models.patch_embed import fMRIPatchEmbed4D
from utils.masking import (
    apply_swin_mask,
    complementary_tube_masks,
    denoise_complementary,
    sample_tube_mask,
    token_mask_to_voxel_mask,
)

SHAPE = (24, 28, 20)
N_FRAMES = 60
TR = 1.3


def _make_raw_nsd(root, subject="subj01"):
    """Synthetic NSD raw tree: concentric tissues inside an ellipsoidal 'head'."""
    grid = np.indices(SHAPE).astype(np.float32)
    center = np.array([(s - 1) / 2 for s in SHAPE], dtype=np.float32).reshape(3, 1, 1, 1)
    scale = np.array([11, 13, 9], dtype=np.float32).reshape(3, 1, 1, 1)
    r = np.sqrt((((grid - center) / scale) ** 2).sum(0))
    aseg = np.zeros(SHAPE, dtype=np.int16)
    aseg[r < 0.80] = 3  # cortex (GM)
    aseg[r < 0.55] = 2  # white matter
    aseg[r < 0.20] = 4  # ventricle (CSF)
    aseg[(r < 0.40) & (grid[0] > center[0, 0, 0] + 1)] = 77  # WM-hypointensity
    head = (r < 0.95).astype(np.float32)  # whole-head mask, includes a scalp shell
    affine = np.diag([1.8, 1.8, 1.8, 1.0])

    base = np.zeros(SHAPE, dtype=np.float32)
    base[aseg == 3] = 500.0
    base[(aseg == 2) | (aseg == 77)] = 400.0
    base[aseg == 4] = 700.0
    base[(head > 0) & (aseg == 0)] = 150.0  # scalp/skull
    t = np.arange(N_FRAMES, dtype=np.float32)
    rng = np.random.RandomState(0)
    gm = aseg == 3
    signal = 0.01 * np.sin(2 * np.pi * 0.06 * t)  # 1% amplitude, inside HRF band
    drift = 0.002 * (t - t.mean())  # 0.2% per frame linear drift
    series = base[..., None] * (1.0 + drift)
    series[gm] *= 1.0 + signal
    series = series + rng.randn(*series.shape).astype(np.float32) * 0.5
    series = series * head[..., None]  # exact zeros outside the head

    ppdata = root / "ppdata" / subject / "func1pt8mm"
    ts_dir = root / "timeseries" / subject / "func1pt8mm" / "timeseries"
    ppdata.mkdir(parents=True)
    ts_dir.mkdir(parents=True)
    nib.save(nib.Nifti1Image(aseg, affine), str(ppdata / "aseg.nii.gz"))
    nib.save(nib.Nifti1Image(head, affine), str(ppdata / "brainmask.nii.gz"))
    img = nib.Nifti1Image(series.astype(np.float32), affine)
    img.header.set_zooms((1.8, 1.8, 1.8, TR))
    nib.save(img, str(ts_dir / "timeseries_nsdimagery_run01.nii.gz"))
    return aseg, head, series, signal


@pytest.fixture()
def processed(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "proc"
    aseg, head, series, signal = _make_raw_nsd(raw)
    manifest = pp.preprocess_subject(raw, out, "subj01", window_length=40, stride=20)
    return out, manifest, aseg, head, series, signal


# ----------------------------------------------------------------------------- brain mask
def test_label_77_is_white_matter_not_grey_matter() -> None:
    assert 77 not in pp.GM_LABELS
    assert 77 in pp.WM_LABELS


def test_brain_mask_fills_sulci_removes_islands_and_stays_inside_head() -> None:
    mask = np.zeros((30, 30, 20), dtype=bool)
    mask[5:25, 5:25, 4:16] = True
    mask[14:16, 5:25, 8:10] = False  # sulcal slit, closing/filling must bridge it
    mask[1:3, 1:3, 1:3] = True  # disconnected island (e.g. eye), must be dropped
    head = np.ones_like(mask)
    head[:, :, 15:] = False  # scalp/air above the head mask

    brain = build_brain_mask(mask, head_mask=head, closing_iterations=3, connectivity=2)

    assert brain[15, 15, 9], "sulcal slit should be filled"
    assert not brain[1:3, 1:3, 1:3].any(), "disconnected island must be removed"
    assert not brain[:, :, 15:].any(), "result must be bounded by the head mask"
    assert build_brain_mask(np.zeros((4, 4, 4))).sum() == 0


def test_brain_mask_excludes_scalp_shell(processed) -> None:
    out, _, aseg, head, _, _ = processed
    brain = nib.load(str(out / "subj01" / "anat" / "brain.nii.gz")).get_fdata() > 0
    assert brain.sum() < (head > 0).sum()
    assert not brain[(head > 0) & (aseg == 0) & (np.indices(SHAPE)[0] == 0)].any()
    tissue = (aseg > 0) & (head > 0)
    # Leaving out the largest-component pruning, the parenchyma covers the tissue union.
    assert (brain & tissue).sum() >= 0.99 * tissue.sum()


def test_manifest_has_brain_path_tr_and_schema(processed) -> None:
    out, manifest_path, *_ = processed
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 2
    assert manifest["tr"] == pytest.approx(TR, abs=1e-6)
    entry = manifest["entries"][0]
    assert entry["brain_path"].endswith("brain.nii.gz")
    assert entry["tr"] == pytest.approx(TR, abs=1e-6)
    assert "pixdim4" in entry["tr_source"]
    assert "brain" in manifest["mask_stats"] and "head" in manifest["mask_stats"]


# ----------------------------------------------------------------------------- PSC target
def test_linear_detrend_matches_polyfit() -> None:
    rng = np.random.RandomState(1)
    run = rng.randn(3, 4, 5, 57).astype(np.float32) + np.arange(57, dtype=np.float32) * 0.3 + 7
    mean, slope = linear_detrend_coefficients(run, chunk=16)
    t = np.arange(57) - 28.0
    expected_slope = np.array([np.polyfit(t, run[1, 2, 3], 1)[0]])
    assert slope[1, 2, 3] == pytest.approx(expected_slope[0], rel=1e-4)
    assert mean[1, 2, 3] == pytest.approx(run[1, 2, 3].mean(), rel=1e-5)


def test_psc_window_recovers_known_signal_and_zero_background() -> None:
    T = 40
    t = np.arange(T, dtype=np.float32)
    brain = np.zeros((6, 6, 6), dtype=bool)
    brain[1:5, 1:5, 1:5] = True
    baseline = np.where(brain, 400.0, 0.0).astype(np.float32)
    signal = 0.01 * np.sin(2 * np.pi * 0.075 * t)  # 1% amplitude
    window = baseline[..., None] * (1 + signal) + 3.0 * (t - t.mean()) * brain[..., None]
    trend = baseline[..., None] + 3.0 * (t - t.mean()) * brain[..., None]

    dyn, static = compute_psc_window(window, brain, baseline, trend)

    assert (dyn[~brain] == 0).all() and (static[~brain] == 0).all()
    expected = 100 * signal - (100 * signal).mean()
    np.testing.assert_allclose(dyn[2, 2, 2], expected, atol=1e-3)
    np.testing.assert_allclose(dyn.mean(-1)[brain], 0, atol=1e-4)  # window mean removed

    rng = np.random.RandomState(0)
    varied = (baseline + rng.rand(*baseline.shape) * 100 * brain)[..., None] * np.ones(T)
    _, static = compute_psc_window(varied, brain, varied[..., 0], varied)
    assert static[brain].mean() == pytest.approx(0, abs=1e-5)
    assert static[brain].std() == pytest.approx(1, abs=1e-4)


def test_dataset_returns_six_channels_with_expected_statistics(processed) -> None:
    out, _, aseg, head, series, signal = processed
    ds = NSDDataset(out, time_frames=40)
    sample = ds[0]
    x = sample["x"]
    assert x.shape == (NUM_CHANNELS, 40, *SHAPE)
    assert float(sample["tr"]) == pytest.approx(TR, abs=1e-6)
    assert sample["subject"] == "subj01" and sample["start"] == 0

    brain = x[BRAIN, 0] > 0
    assert (x[DYN][:, ~brain] == 0).all(), "dyn must be exactly 0 outside the brain"
    assert (x[STATIC][:, ~brain] == 0).all()
    assert x[STATIC, 0][brain].mean().abs() < 1e-5
    assert x[STATIC, 0][brain].std(unbiased=False) == pytest.approx(1, abs=1e-4)
    for channel in (STATIC, GM, WM, CSF, BRAIN):
        assert torch.equal(x[channel, 0], x[channel, -1]), "anatomy channels are time-invariant"
    # Tissue masks are subsets of the brain, and WM-hypointensity voxels are WM.
    for channel in (GM, WM, CSF):
        assert ((x[channel, 0] > 0) & ~brain).sum() == 0
    assert (x[WM, 0] > 0)[aseg == 77].all()

    # Linear drift (0.2%/frame) is removed at run level: GM dyn tracks the 1% sinusoid.
    gm = x[GM, 0] > 0
    gm_mean_trace = x[DYN][:, gm].mean(1).numpy()
    target = 100 * signal[:40]
    target = target - target.mean()
    corr = np.corrcoef(gm_mean_trace, target)[0, 1]
    assert corr > 0.99
    assert gm_mean_trace.std() == pytest.approx(target.std(), rel=0.1)


def test_dataset_fallback_builds_brain_mask_without_brain_path(processed) -> None:
    out, manifest_path, *_ = processed
    reference = NSDDataset(out, time_frames=40)[0]["x"]
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["entries"]:
        entry.pop("brain_path")
        entry.pop("tr")
    manifest_path.write_text(json.dumps(manifest))
    (out / "subj01" / "anat" / "brain.nii.gz").unlink()

    fallback = NSDDataset(out, time_frames=40)[0]
    assert float(fallback["tr"]) == pytest.approx(TR, abs=1e-6)  # read from NIfTI header
    brain_fb = fallback["x"][BRAIN, 0] > 0
    brain_ref = reference[BRAIN, 0] > 0
    union = (fallback["x"][[GM, WM, CSF], 0] > 0).any(0)
    assert brain_fb.sum() > 0 and (brain_fb | ~union).all()  # covers the tissue union
    assert (brain_fb & brain_ref).sum() >= 0.95 * brain_ref.sum()


def test_dataset_tr_override_and_crop(processed) -> None:
    out, *_ = processed
    ds = NSDDataset(out, time_frames=40, crop_size=(16, 16, 12), tr_override=2.0)
    sample = ds[0]
    assert sample["x"].shape == (NUM_CHANNELS, 40, 16, 16, 12)
    assert float(sample["tr"]) == 2.0
    brain = sample["x"][BRAIN, 0] > 0
    assert (sample["x"][DYN][:, ~brain] == 0).all()


# ----------------------------------------------------------------------------- masking
def test_patch_embed_valid_cells_cover_only_brain_patches() -> None:
    embed = fMRIPatchEmbed4D(patch_size=4, embed_dim=8, merge_stages=1)
    dyn = torch.zeros(1, 1, 2, 16, 16, 16)
    anat = torch.zeros(1, 5, 16, 16, 16)
    brain = torch.zeros(1, 1, 16, 16, 16)
    brain[..., 5, 5, 5] = 1  # one voxel -> exactly one valid patch cell (1, 1, 1)
    dyn_tokens, anat_tokens, grid, valid = embed(dyn, anat, brain)
    assert grid == (4, 4, 4)
    assert dyn_tokens.shape == (1, 2 * 64, 8) and anat_tokens.shape == (1, 64, 8)
    assert valid.sum() == 1 and valid[0, 1 * 16 + 1 * 4 + 1]


def test_patch_embed_padding_is_pooling_compatible() -> None:
    embed = fMRIPatchEmbed4D(patch_size=14, embed_dim=8, merge_stages=3)
    assert embed.padded_shape((81, 104, 83)) == (112, 112, 112)
    assert embed.pad(torch.zeros(1, 1, 81, 104, 83)).shape[-3:] == (112, 112, 112)


def test_tube_mask_has_fixed_count_ignores_invalid_cells_and_repeats_over_time() -> None:
    B, S, T = 3, 40, 5
    valid = torch.zeros(B, S, dtype=torch.bool)
    valid[:, :24] = True
    g = torch.Generator().manual_seed(0)
    mask = sample_tube_mask(valid, 0.75, g)
    assert (mask.sum(1) == 18).all() and not mask[:, 24:].any()

    tokens = torch.randn(B, T * S, 6)
    token = torch.full((1, 1, 6), 7.0)
    x_masked, flat = apply_swin_mask(tokens, token, 0.75, True, T, valid=valid, generator=g)
    flat = flat.view(B, T, S)
    assert torch.equal(flat[:, 0], flat[:, -1]), "tubes are identical across frames"
    assert (flat[:, :, 24:] == 0).all(), "padding-only cells are never masked"
    hidden = flat.bool().unsqueeze(-1).expand_as(x_masked.view(B, T, S, 6))
    assert (x_masked.view(B, T, S, 6)[hidden] == 7.0).all()
    assert torch.equal(x_masked.view(B, T, S, 6)[~hidden], tokens.view(B, T, S, 6)[~hidden])


def test_complementary_masks_partition_valid_cells_and_keep_training_ratio() -> None:
    B, S, K = 2, 512, 4
    valid = torch.zeros(B, S, dtype=torch.bool)
    valid[:, :288] = True
    masks = complementary_tube_masks(B, S, K, valid, torch.Generator().manual_seed(3))
    visible = (~masks) & valid
    assert torch.equal(visible.sum(0), valid.long()), "each valid cell is visible in exactly 1 pass"
    for k in range(K):
        ratio = (masks[k] & valid).sum() / valid.sum()
        assert float(ratio) == pytest.approx(0.75, abs=1e-6)
    assert not (masks & ~valid.unsqueeze(0)).any()


def test_denoise_complementary_averages_only_masked_passes() -> None:
    B, K, P = 1, 4, 4
    grid, shape = (2, 2, 2), (8, 8, 8)
    valid = torch.ones(B, 8, dtype=torch.bool)
    seen_masks: list[torch.Tensor] = []

    def forward_fn(token_mask):
        seen_masks.append(token_mask.clone())
        # Return the pass index everywhere: the average over a voxel's masked passes
        # must equal the mean of the indices of the passes in which it was hidden.
        value = float(len(seen_masks) - 1)
        full = torch.full((B, 1, 3, *shape), value)
        return {"clean": full, "noise": -full}

    out = denoise_complementary(forward_fn, valid, grid, P, shape, K=K, generator=torch.Generator().manual_seed(0))
    assert len(seen_masks) == K and (out["count"] == K - 1).all()
    stacked = torch.stack(seen_masks)  # (K, B, S)
    for cell in range(8):
        hidden_in = torch.nonzero(stacked[:, 0, cell]).flatten().float()
        assert len(hidden_in) == K - 1
        x, y, z = np.unravel_index(cell, grid)
        voxel = out["clean"][0, 0, 0, x * P, y * P, z * P]
        assert float(voxel) == pytest.approx(float(hidden_in.mean()))
        assert float(out["noise"][0, 0, 0, x * P, y * P, z * P]) == pytest.approx(-float(hidden_in.mean()))


def test_token_mask_to_voxel_mask_crops_padding() -> None:
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[0, 0] = True
    voxels = token_mask_to_voxel_mask(mask, (2, 2, 2), 4, (6, 7, 5))
    assert voxels.shape == (1, 1, 6, 7, 5)
    assert int(voxels.sum()) == 4 * 4 * 4


def test_run_plus_window_detrend_removes_residual_in_window_drift(processed) -> None:
    out, *_ = processed
    t = torch.arange(40, dtype=torch.float32) - 19.5

    def drift_share(mode: str) -> float:
        x = NSDDataset(out, time_frames=40, detrend=mode)[0]["x"]
        g = x[DYN][:, x[GM, 0] > 0]
        beta = (g * t[:, None]).sum(0) / (t**2).sum()
        return float(((beta * t[:, None]) ** 2).sum() / (g**2).sum())

    assert drift_share("run+window") < 1e-6
    assert drift_share("run+window") <= drift_share("run")
    with pytest.raises(ValueError):
        NSDDataset(out, detrend="bogus")
