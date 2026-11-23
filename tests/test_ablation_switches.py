"""Checks for the ablation switches (decoder skips, free noise head, inference modes)."""

from __future__ import annotations

import argparse

import lightning as L
import torch

import evaluate_nsd
from data.channels import DYN
from data_preprocessing.nsd import preprocess as pp
from models.decoder import CascadingDecoder
from models.lightning_model import fMRIDenoisingLightning
from models.mae_core import HierarchicalSwinMAE
from tests.test_phase1_data import _make_raw_nsd
from tests.test_phase2_model import SHAPE, T, _sample


def _model(**kw) -> HierarchicalSwinMAE:
    torch.manual_seed(0)
    return HierarchicalSwinMAE(patch_size=4, embed_dim=24, **kw).eval()


def test_decoder_skips_none_ignores_stage_features_and_voxel_stem() -> None:
    torch.manual_seed(0)
    decoder = CascadingDecoder(embed_dim=24, patch_size=4, n_regressors=5, full_res_channels=8, skips="none").eval()
    shapes = {"s1": (1, 24, 8, 8, 8, T), "s2": (1, 48, 4, 4, 4, T), "s3": (1, 96, 2, 2, 2, T), "s4": (1, 192, 1, 1, 1, T)}
    features = {k: torch.randn(*v) for k, v in shapes.items()}
    anat_a, anat_b = torch.randn(1, 5, 32, 32, 32), torch.randn(1, 5, 32, 32, 32)
    clean_a, beta_a = decoder(features, anat_a, T)
    changed = dict(features, s1=torch.randn(*shapes["s1"]), s2=torch.randn(*shapes["s2"]), s3=torch.randn(*shapes["s3"]))
    clean_b, beta_b = decoder(changed, anat_b, T)
    assert torch.allclose(clean_a, clean_b) and torch.allclose(beta_a, beta_b)
    changed_bottleneck = dict(features, s4=torch.randn(*shapes["s4"]))
    assert not torch.allclose(clean_a, decoder(changed_bottleneck, anat_a, T)[0])
    try:
        CascadingDecoder(embed_dim=24, patch_size=4, skips="bogus")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("invalid skips value must be rejected")


def test_free_noise_head_leaves_the_regressor_subspace() -> None:
    x = _sample()
    for free in (False, True):
        out = _model(free_noise_head=free)(x, 1.0, generator=torch.Generator().manual_seed(0))
        r = out["regressors"][0]
        n = out["noise"][0, 0].reshape(T, -1)
        residual = n - r.T @ torch.linalg.lstsq(r.T, n).solution
        if free:
            assert float(residual.abs().max()) > 1e-4
        else:
            assert float(residual.abs().max()) < 1e-4
        assert out["noise"].shape == (1, 1, T, *SHAPE)


def test_single_pass_and_unmasked_inference_leak_visible_data_while_k_pass_does_not() -> None:
    model, x = _model(), _sample()
    cell = (2, 2, 2)
    region = (slice(None), slice(None), slice(None)) + tuple(slice(c * 4, (c + 1) * 4) for c in cell)
    perturbed = x.clone()
    perturbed[0, DYN][(slice(None),) + region[3:]] += 30.0

    unmasked_a = model.denoise_single(x, 1.0, mask_ratio=0.0)
    unmasked_b = model.denoise_single(perturbed, 1.0, mask_ratio=0.0)
    assert float((unmasked_a["recon"][region] - unmasked_b["recon"][region]).abs().max()) > 1e-3
    assert float(unmasked_a["voxel_mask"].sum()) == 0

    g = lambda: torch.Generator().manual_seed(4)  # noqa: E731
    kp_a, kp_b = model.denoise_complementary(x, 1.0, K=4, generator=g()), model.denoise_complementary(perturbed, 1.0, K=4, generator=g())
    assert float((kp_a["recon"][region] - kp_b["recon"][region]).abs().max()) < 1e-5


def test_evaluate_nsd_supports_every_inference_mode(tmp_path) -> None:
    raw, processed = tmp_path / "raw", tmp_path / "proc"
    for subject in ("subj01", "subj02"):
        _make_raw_nsd(raw, subject)
        pp.preprocess_subject(raw, processed, subject, window_length=40, stride=20)
    lit = fMRIDenoisingLightning(patch_size=4, embed_dim=24, full_res_channels=8, n_comp=2, decoder_skips="none", free_noise_head=True)
    trainer = L.Trainer(max_epochs=1, accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
                        enable_progress_bar=False, enable_model_summary=False)
    trainer.strategy.connect(lit)
    ckpt = tmp_path / "m.ckpt"
    trainer.save_checkpoint(ckpt)
    reloaded = fMRIDenoisingLightning.load_from_checkpoint(str(ckpt))
    assert reloaded.hparams.decoder_skips == "none" and reloaded.hparams.free_noise_head is True

    r2 = {}
    for mode in ("kpass", "single_masked", "unmasked"):
        summary = evaluate_nsd.run_evaluation(
            argparse.Namespace(
                data_dir=str(processed), checkpoint=str(ckpt), subjects=["subj02"], validation_subjects=[], train_visual_subjects=[],
                output_dir=str(tmp_path / mode), time_frames=40, crop_size=None, num_workers=0, K=4, mask_repeats=1, seed=1,
                max_windows_per_subject=1, visual_examples_per_subject=0, detrend="run", tr=None, amp=False, inference_mode=mode,
            )
        )
        assert summary["inference_mode"] == mode
        r2[mode] = summary["splits"]["test"]["subject_mean"]["r2_gm"]
        assert r2[mode] is not None


def test_prior_warmup_scales_only_the_prior_terms() -> None:
    from utils.losses import IdentifiableDenoisingLoss

    x = _sample()
    model = _model()
    out = model(x, 1.0, generator=torch.Generator().manual_seed(0))
    loss = IdentifiableDenoisingLoss()
    full = loss(out, x, 1.0, prior_scale=1.0)
    off = loss(out, x, 1.0, prior_scale=0.0)
    half = loss(out, x, 1.0, prior_scale=0.5)
    assert torch.allclose(off["total"], full["rec"])  # rec only, never scaled
    expected_half = full["rec"] + 0.5 * (full["total"] - full["rec"])
    assert torch.allclose(half["total"], expected_half, atol=1e-6)
    for key in ("rec", "tissue", "band", "ortho"):
        assert torch.allclose(full[key], off[key])  # individual terms are reported unscaled


def test_lightning_logs_prior_scale_during_warmup() -> None:
    from data.synthetic import SyntheticNSDDataset
    from tests.test_synthetic import SyntheticModule

    ds = SyntheticNSDDataset(n_subjects=3, time_frames=8, shape=(16, 16, 8), seed=3)
    model = fMRIDenoisingLightning(patch_size=4, embed_dim=24, full_res_channels=8, n_comp=2, prior_warmup_steps=10)
    trainer = L.Trainer(max_epochs=1, accelerator="cpu", devices=1, limit_train_batches=2, limit_val_batches=1,
                        logger=False, enable_checkpointing=False, precision=32, enable_progress_bar=False,
                        enable_model_summary=False)
    trainer.fit(model, datamodule=SyntheticModule(ds))
    assert 0.0 <= float(trainer.callback_metrics["prior_scale"]) < 1.0
