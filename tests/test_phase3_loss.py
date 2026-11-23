"""Phase 3 checks: asymmetric heads' utilities and the identifiable loss."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import ndimage as ndi

from data.channels import BRAIN, CSF, DYN, GM, NUM_CHANNELS, STATIC, WM
from models.mae_core import HierarchicalSwinMAE
from utils.losses import IdentifiableDenoisingLoss
from utils.spectral import band_power_fraction, band_weights, bandpass
from utils.tissue import erode_mask, nuisance_regressors, soft_support

T = 40
SHAPE = (16, 18, 14)


# ------------------------------------------------------------------ utilities
def test_erode_mask_matches_scipy_binary_erosion() -> None:
    rng = np.random.RandomState(0)
    blob = ndi.binary_dilation(rng.rand(14, 15, 12) > 0.97, iterations=3)
    for iterations in (1, 2):
        expected = ndi.binary_erosion(blob, iterations=iterations)
        got = erode_mask(torch.from_numpy(blob)[None, None], iterations)[0, 0].numpy()
        assert np.array_equal(got, expected)


def test_soft_support_is_a_partial_volume_ramp() -> None:
    mask = torch.zeros(1, 1, 7, 7, 7)
    mask[..., 2:5, 2:5, 2:5] = 1
    soft = soft_support(mask)
    assert soft.min() >= 0 and soft.max() <= 1
    assert float(soft[0, 0, 3, 3, 3]) == pytest.approx(1.0)
    assert 0 < float(soft[0, 0, 1, 3, 3]) < 1  # leaks one voxel outside


def test_bandpass_keeps_hrf_bins_and_removes_the_rest() -> None:
    t = torch.arange(T).float()
    x = torch.zeros(1, 1, T, 1, 1, 1)
    parts = {
        0.05: torch.sin(2 * np.pi * 0.05 * t),  # in band
        0.10: torch.sin(2 * np.pi * 0.10 * t + 0.3),  # in band
        0.30: torch.sin(2 * np.pi * 0.30 * t),  # out of band
    }
    x[0, 0, :, 0, 0, 0] = sum(parts.values()) + 2.0  # + DC
    y = bandpass(x, 1.0, 0.01, 0.15)
    expected = parts[0.05] + parts[0.10]
    assert torch.allclose(y[0, 0, :, 0, 0, 0], expected, atol=1e-4)
    y.sum().backward if y.requires_grad else None
    weights = band_weights(T, 1.0, 0.01, 0.15)[0]
    assert weights[0] == 0 and weights[6] == 1 and weights[7] == 0  # 0.15 Hz edge included
    # TR changes the bins: at TR = 2 s the bin spacing halves.
    assert band_weights(T, 2.0, 0.01, 0.15)[0].sum() == 12


def test_bandpass_is_differentiable_and_per_sample_tr() -> None:
    x = torch.randn(2, 1, T, 2, 2, 2, requires_grad=True)
    y = bandpass(x, torch.tensor([1.0, 2.0]), 0.01, 0.15)
    y.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    fraction = band_power_fraction(torch.randn(T, 5), 1.0)
    assert fraction.shape == (5,) and ((fraction >= 0) & (fraction <= 1)).all()


def _regressor_scene(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(T).float()
    wm_signal = torch.sin(2 * np.pi * 0.04 * t)
    csf_signal = torch.sign(torch.sin(2 * np.pi * 0.11 * t))
    dyn = torch.randn(1, 1, T, *SHAPE, generator=g) * 0.2
    wm = torch.zeros(1, 1, *SHAPE)
    csf = torch.zeros(1, 1, *SHAPE)
    wm[..., 1:9, 1:12, 1:12] = 1
    csf[..., 10:15, 1:12, 1:12] = 1
    dyn += wm.unsqueeze(2) * wm_signal[None, None, :, None, None, None]
    dyn += csf.unsqueeze(2) * csf_signal[None, None, :, None, None, None]
    return dyn, wm, csf, wm_signal, csf_signal


def test_regressors_are_unit_variance_and_recover_planted_nuisance() -> None:
    dyn, wm, csf, wm_signal, csf_signal = _regressor_scene()
    visible = torch.ones(1, 1, *SHAPE, dtype=torch.bool)
    r = nuisance_regressors(dyn, wm, csf, visible, n_comp=3)
    assert r.shape == (1, 5, T)
    assert torch.allclose(r.mean(-1), torch.zeros(1, 5), atol=1e-5)
    assert torch.allclose(r.square().mean(-1), torch.ones(1, 5), atol=1e-4)
    corr_wm = torch.corrcoef(torch.stack([r[0, 0], wm_signal]))[0, 1]
    corr_csf = torch.corrcoef(torch.stack([r[0, 1], csf_signal]))[0, 1]
    assert corr_wm.abs() > 0.95 and corr_csf.abs() > 0.95
    # aCompCor components are mutually orthogonal
    pcs = r[0, 2:]
    gram = pcs @ pcs.T / T
    assert torch.allclose(gram, torch.eye(3), atol=1e-3)


def test_regressors_use_only_visible_voxels() -> None:
    dyn, wm, csf, *_ = _regressor_scene()
    visible = torch.zeros(1, 1, *SHAPE, dtype=torch.bool)
    visible[..., :8, :, :] = True
    reference = nuisance_regressors(dyn, wm, csf, visible, n_comp=3)
    tampered = dyn.clone()
    tampered[:, :, :, 8:] += 100 * torch.randn_like(tampered[:, :, :, 8:])  # hidden voxels
    again = nuisance_regressors(tampered, wm, csf, visible, n_comp=3)
    assert torch.allclose(reference, again, atol=1e-5)
    empty = nuisance_regressors(dyn, wm, csf, torch.zeros_like(visible), n_comp=3)
    assert torch.count_nonzero(empty) == 0  # degenerate -> zeros, no NaN


# ------------------------------------------------------------------ loss terms
def _scene_output(clean, noise, regressors, hidden=None, clean_pre=None):
    ones = torch.ones(1, 1, *SHAPE, dtype=torch.bool) if hidden is None else hidden
    return {
        "clean": clean,
        "noise": noise,
        "recon": clean + noise,
        "clean_pre": clean if clean_pre is None else clean_pre,
        "regressors": regressors,
        "voxel_mask": ones,
    }


def _anatomy_scene():
    """Sample with GM shell, WM block and CSF block; planted signal / nuisance / noise."""
    g = torch.Generator().manual_seed(1)
    t = torch.arange(T).float()
    brain = torch.zeros(*SHAPE)
    brain[1:15, 1:17, 1:13] = 1
    wm = torch.zeros(*SHAPE)
    wm[3:8, 3:15, 3:11] = 1
    csf = torch.zeros(*SHAPE)
    csf[9:13, 3:15, 3:11] = 1
    gm = brain - wm - csf
    gm = gm.clamp(0, 1)
    neural = torch.sin(2 * np.pi * 0.06 * t + 0.4)  # in band
    nuis1 = torch.sin(2 * np.pi * 0.03 * t + 1.0)
    nuis2 = torch.sin(2 * np.pi * 0.12 * t)
    nuis = torch.stack([nuis1, nuis2])  # (2, T)
    nuis = (nuis - nuis.mean(-1, keepdim=True)) / nuis.std(-1, keepdim=True, unbiased=False)
    dyn = torch.zeros(T, *SHAPE)
    dyn += gm * neural[:, None, None, None] * 1.0
    # shared nuisance appears in WM/CSF strongly and in GM with random weights
    beta = torch.rand(2, *SHAPE, generator=g) * 1.0
    dyn += torch.einsum("jxyz,jt->txyz", beta, nuis) * brain
    dyn += torch.randn(T, *SHAPE, generator=g) * 0.1 * brain
    sample = torch.zeros(1, NUM_CHANNELS, T, *SHAPE)
    sample[0, DYN] = dyn
    for channel, m in ((STATIC, brain), (GM, gm), (WM, wm), (CSF, csf), (BRAIN, brain)):
        sample[0, channel] = m
    return sample, nuis[None], neural, gm, brain


def _canonical_solutions():
    sample, r, neural, gm, brain = _anatomy_scene()
    dyn = sample[:, DYN : DYN + 1]
    pinv = torch.linalg.pinv(r[0].T)  # (2, T)
    d = dyn[0, 0]  # (T, X, Y, Z)
    beta = torch.einsum("jt,txyz->jxyz", pinv, d)
    nphys = torch.einsum("jxyz,jt->txyz", beta, r[0]) * brain
    cleaned = d - nphys
    from utils.spectral import bandpass as bp

    s1 = bp(cleaned[None, None], 1.0)[0, 0] * gm  # intended: band-passed, nuisance projected out, GM only
    return sample, r, {
        "S1 intended": (s1, nphys),
        "S2 identity": (d.clone(), torch.zeros_like(d)),
        "S3 all-in-noise": (torch.zeros_like(d), nphys),
        "S4 nothing": (torch.zeros_like(d), torch.zeros_like(d)),
    }


def test_loss_ranks_intended_decomposition_first_by_order_one_margins() -> None:
    sample, r, solutions = _canonical_solutions()
    loss = IdentifiableDenoisingLoss()
    values = {}
    for name, (clean, noise) in solutions.items():
        out = _scene_output(clean[None, None], noise[None, None], r)
        values[name] = {k: float(v) for k, v in loss(out, sample, 1.0).items()}
    totals = {k: v["total"] for k, v in values.items()}
    intended = totals["S1 intended"]
    assert intended < min(v for k, v in totals.items() if k != "S1 intended")
    # margins are O(1) of the reconstruction scale, not O(1e-4)
    assert min(v for k, v in totals.items() if k != "S1 intended") - intended > 0.05 * totals["S4 nothing"]
    assert values["S2 identity"]["tissue"] > 0.05 and values["S2 identity"]["band"] >= 0
    assert values["S1 intended"]["tissue"] < 1e-6 and values["S1 intended"]["band"] < 1e-6
    assert values["S1 intended"]["ortho"] < values["S2 identity"]["ortho"]
    assert values["S3 all-in-noise"]["rec"] > values["S1 intended"]["rec"]


def test_loss_terms_have_exact_closed_forms() -> None:
    sample, r, _ = _canonical_solutions()
    loss = IdentifiableDenoisingLoss(ortho_weight=0.0, tissue_weight=0.0, band_weight=0.0)
    zeros = torch.zeros(1, 1, T, *SHAPE)
    out = _scene_output(zeros, zeros, r)
    brain = sample[:, BRAIN : BRAIN + 1, 0] > 0
    expected = (sample[:, DYN : DYN + 1][:, :, :, brain[0, 0]]).square().mean()
    assert float(loss(out, sample, 1.0)["rec"]) == pytest.approx(float(expected), rel=1e-5)

    # rec counts hidden voxels only: perturbing the residual in visible voxels changes nothing
    hidden = torch.zeros(1, 1, *SHAPE, dtype=torch.bool)
    hidden[..., :8, :, :] = True
    base = loss(_scene_output(zeros, zeros, r, hidden), sample, 1.0)["rec"]
    noisy = zeros.clone()
    noisy[..., 8:, :, :] = 5.0
    assert float(loss(_scene_output(noisy, zeros, r, hidden), sample, 1.0)["rec"]) == pytest.approx(float(base))

    # tissue: constant clean_pre c inside eroded WM/CSF gives exactly c^2
    c = zeros.clone()
    c[:] = 2.0
    only_tissue = IdentifiableDenoisingLoss(rec_weight=0, band_weight=0, ortho_weight=0)
    assert float(only_tissue(_scene_output(c, zeros, r), sample, 1.0)["tissue"]) == pytest.approx(4.0, rel=1e-5)

    # band: a pure out-of-band (0.3 Hz) clean_pre in GM -> mean square amplitude 0.5 * A^2
    t = torch.arange(T).float()
    oob = (torch.sin(2 * np.pi * 0.3 * t)[None, None, :, None, None, None] * torch.ones(1, 1, T, *SHAPE)) * 1.0
    only_band = IdentifiableDenoisingLoss(rec_weight=0, tissue_weight=0, ortho_weight=0)
    assert float(only_band(_scene_output(oob, zeros, r), sample, 1.0)["band"]) == pytest.approx(0.5, rel=1e-3)
    inband = torch.sin(2 * np.pi * 0.05 * t)[None, None, :, None, None, None] * torch.ones(1, 1, T, *SHAPE)
    assert float(only_band(_scene_output(inband, zeros, r), sample, 1.0)["band"]) < 1e-6


def test_ortho_is_a_true_squared_correlation_bounded_in_unit_interval() -> None:
    sample, r, _ = _canonical_solutions()
    only = IdentifiableDenoisingLoss(rec_weight=0, tissue_weight=0, band_weight=0, ortho_weight=1.0)
    zeros = torch.zeros(1, 1, T, *SHAPE)
    # clean proportional to a regressor everywhere -> rho^2 = 1 for that regressor, 0 for the other
    # regressors only if they are orthogonal; use two orthogonal sinusoids from the scene.
    aligned = r[0, 0][None, None, :, None, None, None] * 3.0 * torch.ones(1, 1, T, *SHAPE)
    value = float(only(_scene_output(aligned, zeros, r), sample, 1.0)["ortho"])
    rho_other = float(torch.corrcoef(r[0])[0, 1] ** 2)
    assert value == pytest.approx((1.0 + rho_other) / 2, abs=1e-3)  # mean over the 2 regressors
    assert 0.0 <= value <= 1.0
    scaled = float(only(_scene_output(aligned * 10, zeros, r), sample, 1.0)["ortho"])
    assert scaled == pytest.approx(value, rel=1e-4), "scale invariance (correlation, not covariance)"
    orthogonal = torch.cos(2 * np.pi * 0.2 * torch.arange(T).float())[None, None, :, None, None, None] * torch.ones(1, 1, T, *SHAPE)
    assert float(only(_scene_output(orthogonal, zeros, r), sample, 1.0)["ortho"]) < 0.05


def test_loss_has_no_fft_flat_or_structural_terms() -> None:
    sample, r, solutions = _canonical_solutions()
    clean, noise = solutions["S1 intended"]
    out = IdentifiableDenoisingLoss()(_scene_output(clean[None, None], noise[None, None], r), sample, 1.0)
    assert {"total", "rec", "tissue", "band", "ortho"} <= set(out)
    assert not {"fft", "flat", "structural_total", "masked_mse", "reconstruction"} & set(out)
    assert all(torch.isfinite(v) for v in out.values())


# ------------------------------------------------------------------ model + loss
def _tiny_model(**kw):
    torch.manual_seed(0)
    return HierarchicalSwinMAE(patch_size=4, embed_dim=24, mask_ratio=0.75, **kw)


def _random_sample():
    sample, *_ = _anatomy_scene()
    return sample[..., :16, :16, :12].contiguous()


def test_architectural_constraints_make_final_clean_penalties_vanish_but_not_pre() -> None:
    """Why the penalties act on the head output *before* gating / filtering."""
    model = _tiny_model().eval()
    x = _random_sample()
    with torch.no_grad():
        out = model(x, tr=1.0)
    region = erode_mask(x[:, WM : WM + 1, 0] > 0, 1) | erode_mask(x[:, CSF : CSF + 1, 0] > 0, 1)
    pre = out["clean_pre"][:, :, :, region[0, 0]].square().mean()
    final = out["clean"][:, :, :, region[0, 0]].square().mean()
    assert float(pre) > 1e-6
    assert float(final) < 0.05 * float(pre)  # gate suppresses the head almost entirely there


def test_end_to_end_loss_backward_and_ablation_flag() -> None:
    x = _random_sample()
    for hard in (True, False):
        model = _tiny_model(hard_bandpass=hard).train()
        out = model(x, tr=1.0)
        losses = IdentifiableDenoisingLoss()(out, x, 1.0)
        losses["total"].backward()
        grads = [p.grad for p in model.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert float(losses["rec"].detach()) > 0
        if not hard:
            assert torch.equal(out["clean"], out["clean_pre"])
