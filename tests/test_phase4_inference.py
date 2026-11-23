"""Phase 4 checks: J-invariant K-pass inference, evaluation scripts and the training driver."""

from __future__ import annotations

import argparse
import csv
import json
import sys

import lightning as L
import numpy as np
import pytest
import torch

import evaluate_nsd
import evaluate_temporal
import train_nsd
from data.channels import DYN
from data_preprocessing.nsd import preprocess as pp
from models.lightning_model import fMRIDenoisingLightning
from models.mae_core import HierarchicalSwinMAE
from tests.test_phase1_data import _make_raw_nsd
from tests.test_phase2_model import SHAPE as MODEL_SHAPE
from tests.test_phase2_model import T as MODEL_T
from tests.test_phase2_model import _sample


# ------------------------------------------------------------------ K-pass inference
def _model() -> HierarchicalSwinMAE:
    torch.manual_seed(0)
    return HierarchicalSwinMAE(patch_size=4, embed_dim=24, mask_ratio=0.75).eval()


def _cell_slices(cell: tuple[int, int, int], patch: int = 4):
    return tuple(slice(c * patch, (c + 1) * patch) for c in cell)


def test_k_pass_estimate_is_exactly_j_invariant_per_tube() -> None:
    """Perturbing a tube's own data changes no estimate *inside* that tube (K-pass), but does elsewhere."""
    model, x = _model(), _sample()
    seed = 7
    reference = model.denoise_complementary(x, 1.0, K=4, generator=torch.Generator().manual_seed(seed))
    counts = reference["count"]
    assert (counts[counts > 0] == 3).all()

    cell = (2, 2, 2)  # a cell well inside the brain
    region = _cell_slices(cell)
    perturbed = x.clone()
    perturbed[0, DYN][(slice(None), *region)] += 40.0 * torch.randn(MODEL_T, 4, 4, 4)
    again = model.denoise_complementary(perturbed, 1.0, K=4, generator=torch.Generator().manual_seed(seed))

    for key in ("clean", "noise", "recon"):
        inside = (slice(None), slice(None), slice(None), *region)
        assert torch.allclose(reference[key][inside], again[key][inside], atol=1e-5), key
    outside_changed = (reference["recon"] - again["recon"]).abs()
    outside_changed[(slice(None), slice(None), slice(None), *region)] = 0
    assert float(outside_changed.max()) > 1e-4, "neighbouring tubes must see the perturbation"


def test_single_pass_leaks_visible_tubes_which_k_pass_removes() -> None:
    model, x = _model(), _sample()
    out = model(x, 1.0, generator=torch.Generator().manual_seed(3))
    visible_cells = (~out["token_mask"] & out["valid"])[0].nonzero().flatten()
    assert len(visible_cells) > 0
    grid = out["grid"]
    cell = tuple(int(v) for v in np.unravel_index(int(visible_cells[0]), grid))
    region = _cell_slices(cell)
    perturbed = x.clone()
    perturbed[0, DYN][(slice(None), *region)] += 40.0
    inside = (slice(None), slice(None), slice(None), *tuple(slice(r.start, min(r.stop, s)) for r, s in zip(region, MODEL_SHAPE)))
    changed = model(perturbed, 1.0, token_mask=out["token_mask"])
    assert float((changed["recon"][inside] - out["recon"][inside]).abs().max()) > 1e-3


def test_k_pass_covers_every_valid_voxel_exactly_k_minus_one_times() -> None:
    model, x = _model(), _sample()
    result = model.denoise_complementary(x, 1.0, K=4, generator=torch.Generator().manual_seed(1))
    brain = x[:, 5:6, 0] > 0
    counts = result["count"][:, :, 0]
    assert (counts[brain] == 3).all()
    assert result["clean"].shape == (1, 1, MODEL_T, *MODEL_SHAPE)
    assert torch.isfinite(result["recon"]).all()
    assert torch.allclose(result["recon"], result["clean"] + result["noise"], atol=1e-5)


# ------------------------------------------------------------------ end-to-end scripts
@pytest.fixture(scope="module")
def nsd_world(tmp_path_factory):
    base = tmp_path_factory.mktemp("nsd_e2e")
    raw, processed = base / "raw", base / "processed"
    for subject in ("subj01", "subj02", "subj03"):
        _make_raw_nsd(raw, subject)
        pp.preprocess_subject(raw, processed, subject, window_length=40, stride=20)
    # a tiny checkpoint produced by a real (1 step) fit on synthetic data
    from data.synthetic import SyntheticNSDDataset
    from tests.test_synthetic import SyntheticModule

    model = fMRIDenoisingLightning(patch_size=4, embed_dim=24, full_res_channels=8, n_comp=2)
    trainer = L.Trainer(
        max_epochs=1, accelerator="cpu", devices=1, limit_train_batches=1, limit_val_batches=0,
        logger=False, enable_checkpointing=False, precision=32, enable_progress_bar=False,
    )
    dm = SyntheticModule(SyntheticNSDDataset(n_subjects=3, time_frames=40, shape=(24, 28, 20), seed=0))
    trainer.fit(model, datamodule=dm)
    checkpoint = base / "tiny.ckpt"
    trainer.save_checkpoint(checkpoint)
    return base, processed, checkpoint


def _eval_namespace(processed, checkpoint, out_dir, **overrides) -> argparse.Namespace:
    values = dict(
        data_dir=str(processed), checkpoint=str(checkpoint), subjects=["subj03"],
        validation_subjects=["subj02"], train_visual_subjects=["subj01"], output_dir=str(out_dir),
        time_frames=40, crop_size=None, num_workers=0, K=4, mask_repeats=1, seed=1234,
        max_windows_per_subject=1, visual_examples_per_subject=1, detrend="run", tr=None, amp=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_evaluate_nsd_end_to_end_uses_k_pass_for_every_split(nsd_world) -> None:
    base, processed, checkpoint = nsd_world
    out_dir = base / "eval"
    summary = evaluate_nsd.run_evaluation(_eval_namespace(processed, checkpoint, out_dir))

    assert summary["subjects_by_split"] == {"train": ["subj01"], "validation": ["subj02"], "test": ["subj03"]}
    assert summary["K"] == 4 and set(summary["splits"]) == {"validation", "test"}
    for split in ("train", "validation", "test"):
        assert list((out_dir / "visualizations" / split).glob("*.png")), split
    rows = list(csv.DictReader((out_dir / "metrics_by_window.csv").open()))
    assert {r["split"] for r in rows} == {"validation", "test"}
    assert all(r["K"] == "4" for r in rows)
    for name in ("r2_gm", "gm_tsnr_input_median", "var_clean_frac", "band_power_frac_input_median"):
        assert name in rows[0] and np.isfinite(float(rows[0][name]))
    for name in ("metrics_by_run.csv", "metrics_by_subject.csv", "summary.json"):
        assert (out_dir / name).is_file()
    json.loads((out_dir / "summary.json").read_text())  # strict JSON (no NaN)


def test_evaluate_nsd_rejects_subject_in_two_splits(nsd_world) -> None:
    base, processed, checkpoint = nsd_world
    with pytest.raises(ValueError, match="assigned to both"):
        evaluate_nsd.run_evaluation(
            _eval_namespace(processed, checkpoint, base / "bad", subjects=["subj02"], validation_subjects=["subj02"])
        )


def test_evaluate_temporal_compares_model_with_projection_baseline(nsd_world, monkeypatch) -> None:
    base, processed, checkpoint = nsd_world
    out_dir = base / "temporal"
    monkeypatch.setattr(
        sys, "argv",
        ["evaluate_temporal", "--data_dir", str(processed), "--checkpoint", str(checkpoint),
         "--subjects", "subj03", "--output_dir", str(out_dir), "--max_windows_per_subject", "1",
         "--plots_per_subject", "1"],
    )
    evaluate_temporal.main()
    summary = json.loads((out_dir / "temporal_summary.json").read_text())
    assert set(summary["methods"]) == {"model", "wm_csf_projection"}
    model_metrics = summary["methods"]["model"]
    for name in ("gm_corr_clean_input_median", "gm_tsnr_input_median", "var_clean_frac", "band_power_frac_clean_median"):
        assert name in model_metrics
    assert list((out_dir / "temporal_plots").glob("*.png"))


def test_train_driver_writes_hparams_and_runs_final_evaluation(nsd_world) -> None:
    base, processed, _ = nsd_world
    ckpt_dir = base / "run" / "checkpoints"
    train_nsd.main(
        [
            "--data_dir", str(processed), "--subjects", "subj01", "subj02", "--validation_subjects", "subj02",
            "--test_subjects", "subj03", "--epochs", "1", "--num_workers", "0", "--precision", "32",
            "--patch_size", "4", "--embed_dim", "24", "--full_res_channels", "8", "--n_comp", "2",
            "--checkpoint_dir", str(ckpt_dir), "--log_dir", str(base / "run" / "logs"),
            "--visualization_dir", str(base / "run" / "vis"), "--final_eval_windows_per_subject", "1",
        ]
    )
    record = json.loads((ckpt_dir / "hparams.json").read_text())
    assert record["split"] == {
        "train_subjects": ["subj01"], "validation_subjects": ["subj02"], "test_subjects": ["subj03"],
    }
    assert record["best_checkpoint"].endswith(".ckpt") and record["best_val_total"] is not None
    assert record["dataset"]["tr_seconds"] == [1.3]
    assert set(record["final_evaluation"]["splits"]) == {"validation", "test"}
    assert (ckpt_dir / "final_evaluation" / "summary.json").is_file()
    assert list((base / "run" / "vis").glob("validation_epoch_*.png"))
