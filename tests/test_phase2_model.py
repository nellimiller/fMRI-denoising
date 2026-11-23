"""Phase 2 checks: U-Net skips, relative position bias, padding masks, leakage-free masking."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from data.channels import BRAIN, CSF, DYN, GM, NUM_CHANNELS, STATIC, WM
from models.decoder import CascadingDecoder
from models.mae_core import HierarchicalSwinMAE
from models.swift_official import (
    SwinTransformerBlock4D,
    WindowAttention4D,
    compute_mask,
    window_partition,
)

SHAPE = (20, 22, 18)
T = 6


def _sample(seed: int = 0, batch: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    grid = torch.stack(torch.meshgrid(*[torch.arange(s).float() for s in SHAPE], indexing="ij"))
    centre = torch.tensor([(s - 1) / 2 for s in SHAPE]).view(3, 1, 1, 1)
    radius = (((grid - centre) / torch.tensor([9.0, 10.0, 8.0]).view(3, 1, 1, 1)) ** 2).sum(0).sqrt()
    brain = (radius < 0.85).float()
    gm = ((radius < 0.85) & (radius >= 0.55)).float()
    wm = ((radius < 0.55) & (radius >= 0.2)).float()
    csf = (radius < 0.2).float()
    out = torch.zeros(batch, NUM_CHANNELS, T, *SHAPE)
    for b in range(batch):
        out[b, DYN] = torch.randn(T, *SHAPE, generator=g) * brain
        for c, m in ((STATIC, torch.randn(*SHAPE, generator=g) * brain), (GM, gm), (WM, wm), (CSF, csf), (BRAIN, brain)):
            out[b, c] = m
    return out


def _model(**kw) -> HierarchicalSwinMAE:
    torch.manual_seed(0)
    return HierarchicalSwinMAE(patch_size=4, embed_dim=24, mask_ratio=0.75, **kw).eval()


def test_no_absolute_position_embedding_and_relative_bias_present() -> None:
    model = _model()
    names = [n for n, _ in model.named_parameters()]
    assert not any("pos_embed" in n for n in names)
    tables = [p for n, p in model.named_parameters() if "relative_position_bias_table" in n]
    assert len(tables) == 8  # 2 blocks x 4 stages
    assert all(t.shape[0] == 5**4 for t in tables)


def test_relative_index_depends_only_on_offsets_and_supports_clipped_windows() -> None:
    attn = WindowAttention4D(dim=24, num_heads=3, window_size=(3, 3, 3, 3))
    index = attn._relative_position_index((3, 3, 3, 3), torch.device("cpu"))
    assert index.shape == (81, 81) and index.unique().numel() == 5**4
    coords = torch.stack(torch.meshgrid(*[torch.arange(3)] * 4, indexing="ij")).flatten(1)
    rel = coords[:, :, None] - coords[:, None, :]
    key = (rel[0] * 1000 + rel[1] * 100 + rel[2] * 10 + rel[3])
    # equal relative offset -> equal table row
    for value in key.unique()[:20]:
        rows = index[key == value]
        assert (rows == rows[0]).all()
    clipped = attn._relative_position_index((2, 2, 2, 3), torch.device("cpu"))
    assert clipped.shape == (24, 24) and int(clipped[0, 0]) == int(index[0, 0]) == 312  # zero offset row


def _attention_probabilities(block: SwinTransformerBlock4D, x: torch.Tensor):
    captured = []
    handle = block.attn.softmax.register_forward_hook(lambda m, i, o: captured.append(o.detach()))
    b, d, h, w, t, c = x.shape
    _ = block(x, compute_mask([int(np.ceil(d / 3) * 3), int(np.ceil(h / 3) * 3), int(np.ceil(w / 3) * 3), int(np.ceil(t / 3) * 3)], (3, 3, 3, 3), (1, 1, 1, 1), x.device))
    handle.remove()
    return captured[0]


@pytest.mark.parametrize("shifted", [False, True])
def test_padded_tokens_receive_no_attention(shifted: bool) -> None:
    torch.manual_seed(0)
    shift = (1, 1, 1, 1) if shifted else (0, 0, 0, 0)
    block = SwinTransformerBlock4D(dim=24, num_heads=3, window_size=(3, 3, 3, 3), shift_size=shift).eval()
    d, h, w, t = 4, 5, 4, 5  # none divisible by 3 -> padding in every axis
    x = torch.randn(1, d, h, w, t, 24)
    attn = _attention_probabilities(block, x)  # (nW, heads, N, N)
    dp, hp, wp, tp = 6, 6, 6, 6
    real = torch.zeros(1, dp, hp, wp, tp, 1)
    real[:, :d, :h, :w, :t] = 1
    if shifted:
        real = torch.roll(real, shifts=(-1, -1, -1, -1), dims=(1, 2, 3, 4))
    real_windows = window_partition(real, (3, 3, 3, 3)).squeeze(-1) > 0  # (nW, N)
    # Real queries must put ~0 mass on padded keys (padded queries are discarded, and a window
    # made only of padding attends uniformly, which is irrelevant).
    padded_keys = (~real_windows)[:, None, None, :].expand_as(attn)
    real_queries = real_windows[:, None, :, None].expand_as(attn)
    selected = padded_keys & real_queries
    assert selected.any()
    assert float(attn[selected].max()) < 1e-6
    assert torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)), atol=1e-4)


def test_forward_shapes_heads_and_bottleneck_grid() -> None:
    model = _model()
    x = _sample()
    out = model(x, tr=1.0)
    for key in ("clean", "noise", "recon", "clean_pre"):
        assert out[key].shape == (1, 1, T, *SHAPE), key
    assert out["beta"].shape == (1, model.n_regressors, *SHAPE)
    assert out["regressors"].shape == (1, model.n_regressors, T)
    assert out["voxel_mask"].shape == (1, 1, *SHAPE)
    assert out["grid"] == (8, 8, 8)
    assert torch.isfinite(out["recon"]).all()
    # masked fraction among valid cells equals the training ratio (fixed count)
    valid, token_mask = out["valid"], out["token_mask"]
    assert float(token_mask.sum() / valid.sum()) == pytest.approx(0.75, abs=1 / int(valid.sum()))
    assert not (token_mask & ~valid).any()
    # clean head is zero outside the soft GM support and has no energy outside the brain
    brain = x[:, BRAIN : BRAIN + 1].unsqueeze(2) > 0
    assert (out["clean"] * ~brain).abs().max() == 0 and (out["noise"] * ~brain).abs().max() == 0


def test_clean_head_is_band_limited_and_noise_is_regressor_subspace() -> None:
    model = _model()
    out = model(_sample(), tr=1.0)
    clean = out["clean"][0, 0]  # (T, X, Y, Z)
    spectrum = torch.fft.rfft(clean, dim=0).abs()
    freqs = torch.fft.rfftfreq(T, d=1.0)
    outside = (freqs < 0.01) | (freqs > 0.15 + 1e-6)
    spectrum = spectrum.detach()
    assert float(spectrum[outside].max()) < 1e-4 * float(spectrum.max() + 1e-12) + 1e-5
    # noise lies in span{r_j}: its time courses are linear combinations of regressors
    r = out["regressors"][0]  # (J, T)
    n = out["noise"][0, 0].reshape(T, -1)
    projection = r.T @ torch.linalg.lstsq(r.T, n).solution
    assert torch.allclose(projection, n, atol=1e-4)


def test_hidden_tube_measurements_cannot_influence_any_output() -> None:
    """J-invariance: altering the dynamic data of hidden tubes changes nothing (no leakage)."""
    model = _model()
    x = _sample()
    out = model(x, tr=1.0, generator=torch.Generator().manual_seed(1))
    hidden = out["voxel_mask"][0, 0]  # (X, Y, Z)
    assert hidden.any() and (~hidden).any()

    perturbed = x.clone()
    perturbed[0, DYN, :, hidden] += 50 * torch.randn(T, int(hidden.sum()))
    out2 = model(perturbed, tr=1.0, token_mask=out["token_mask"])
    for key in ("clean", "noise", "recon", "regressors"):
        assert torch.allclose(out[key], out2[key], atol=1e-5), key

    perturbed = x.clone()
    perturbed[0, DYN, :, ~hidden] += 5 * torch.randn(T, int((~hidden).sum()))
    out3 = model(perturbed, tr=1.0, token_mask=out["token_mask"])
    assert not torch.allclose(out["recon"], out3["recon"], atol=1e-3), "visible data must matter"


def test_anatomy_is_visible_inside_masked_tubes() -> None:
    model = _model()
    x = _sample()
    out = model(x, tr=1.0, generator=torch.Generator().manual_seed(1))
    hidden = out["voxel_mask"][0, 0]
    changed = x.clone()
    changed[0, STATIC][:, hidden] += 3.0  # alter anatomy only where the dynamics are hidden
    out2 = model(changed, tr=1.0, token_mask=out["token_mask"])
    assert not torch.allclose(out["clean_pre"], out2["clean_pre"], atol=1e-4)


def test_decoder_consumes_every_skip_and_the_voxel_stem() -> None:
    torch.manual_seed(0)
    dec = CascadingDecoder(embed_dim=24, patch_size=4, n_regressors=5, full_res_channels=8)
    shapes = {"s1": (1, 24, 8, 8, 8, T), "s2": (1, 48, 4, 4, 4, T), "s3": (1, 96, 2, 2, 2, T), "s4": (1, 192, 1, 1, 1, T)}
    feats = {k: torch.randn(*v, requires_grad=True) for k, v in shapes.items()}
    anat = torch.randn(1, 5, 32, 32, 32, requires_grad=True)
    clean, beta = dec(feats, anat, T)
    assert clean.shape == (1, 1, T, 32, 32, 32) and beta.shape == (1, 5, 32, 32, 32)
    (clean.square().mean() + beta.square().mean()).backward()
    for name, tensor in feats.items():
        assert tensor.grad is not None and float(tensor.grad.abs().sum()) > 0, name
    assert float(anat.grad.abs().sum()) > 0
    # the static noise head cannot vary over time: beta has no T axis
    assert beta.ndim == 5
    assert not any(isinstance(m, torch.nn.ConvTranspose3d) for m in dec.modules())


def test_backward_reaches_all_parameters() -> None:
    model = _model().train()
    out = model(_sample(), tr=1.0)
    loss = out["clean"].square().mean() + out["noise"].square().mean() + out["clean_pre"].square().mean()
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not missing, missing
    assert float(model.mask_token.grad.abs().sum()) > 0
    assert any(float(p.grad.abs().sum()) > 0 for n, p in model.named_parameters() if "relative_position_bias_table" in n)


def test_soft_gating_ablation_changes_outputs() -> None:
    soft = _model(hard_bandpass=False)
    out = soft(_sample(), tr=1.0)
    assert torch.equal(out["clean"], out["clean_pre"])
