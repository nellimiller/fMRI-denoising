"""End-to-end verification of the remediated pipeline on real NSD data (e.g. subj01).

Run from the repository root::

    uv run python tools/verify_pipeline_subj01.py \
        --raw_root "C:/fMRI Denoising/NSD/nsd_data" --work_dir /tmp/nsd_verify \
        --subject subj01 --phases 1 2 3 4 --report verification_report.json

Phases (each prints PASS/FAIL checks and numbers, and is stored in the JSON report):

1. preprocessing (brain mask, TR provenance, label 77), PSC target, 6-channel tensor;
2. U-Net skip decoder, relative position bias, padding masks, leakage-free masking;
3. asymmetric dual head forward/backward and identifiable-loss ordering on real data;
4. K=4 complementary J-invariant inference, dynamic metrics, the evaluation script.

The cluster configuration (patch 14, embed 96, 112^3 padded grid) is checked with a no-grad
full-resolution forward pass; training-style checks use a real-data crop with a smaller
patch / model so that they fit a CPU workstation. One subject only: nothing here measures
generalisation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.channels import BRAIN, CSF, DYN, GM, NUM_CHANNELS, STATIC, WM  # noqa: E402
from data.nsd_dataset import NSDDataset  # noqa: E402
from data_preprocessing.nsd import preprocess as pp  # noqa: E402

REPORT: dict = {}
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


def phase1(args: argparse.Namespace) -> NSDDataset:
    import nibabel as nib

    print("== Phase 1: brain mask, PSC target, 6-channel conditioning ==")
    out_root = Path(args.work_dir) / "nsd_processed"
    manifest_path = out_root / args.subject / "manifest.json"
    if args.force or not manifest_path.is_file():
        start = time.time()
        pp.preprocess_subject(
            Path(args.raw_root), out_root, args.subject,
            window_length=args.time_frames, stride=args.stride, tr_override=args.tr,
        )
        print(f"  preprocessing took {time.time() - start:.1f}s")
    manifest = json.loads(manifest_path.read_text())
    stats = manifest["mask_stats"]
    entry = manifest["entries"][0]

    raw_subject = Path(args.raw_root) / "ppdata" / args.subject / "func1pt8mm"
    aseg = np.rint(nib.load(str(raw_subject / "aseg.nii.gz")).get_fdata()).astype(int)
    head = nib.load(str(raw_subject / "brainmask.nii.gz")).get_fdata() > 0
    old_gm = np.isin(aseg, list(pp.GM_LABELS | {77})) & head
    new_gm = np.isin(aseg, list(pp.GM_LABELS)) & head
    n77 = int(((aseg == 77) & head).sum())
    brain = nib.load(str(out_root / args.subject / "anat" / "brain.nii.gz")).get_fdata() > 0
    union = sum(stats[k]["voxels"] for k in ("gm", "wm", "csf"))
    REPORT["phase1"] = {
        "head_voxels": stats["head"]["voxels"],
        "tissue_voxels": {k: stats[k]["voxels"] for k in ("gm", "wm", "csf")},
        "brain_voxels": stats["brain"]["voxels"],
        "brain_fraction_of_fov": stats["brain"]["fraction"],
        "label77_voxels": n77,
        "tr_header": entry["tr"],
        "tr_source": entry["tr_source"],
    }
    print(f"  head mask {stats['head']['voxels']} vox | GM/WM/CSF {[stats[k]['voxels'] for k in ('gm','wm','csf')]} "
          f"| brain {stats['brain']['voxels']} ({100 * stats['brain']['fraction']:.1f}% of FOV)")
    check("label 77 removed from GM", 77 not in pp.GM_LABELS and int(old_gm.sum() - new_gm.sum()) == n77,
          f"{n77} voxels of label 77 now WM")
    check("brain mask strictly inside head mask", bool((brain & ~head).sum() == 0))
    check("brain mask smaller than head (scalp excluded)", brain.sum() < 0.75 * head.sum(),
          f"{brain.sum()} vs {head.sum()} head voxels")
    tissue_union = new_gm | (np.isin(aseg, list(pp.WM_LABELS | pp.CSF_LABELS)) & head)
    check("brain mask covers >=95% of tissue labels", int((brain & tissue_union).sum()) >= 0.95 * union)
    print(f"  TR from header: {entry['tr']} s [{entry['tr_source']}] (cannot confirm acquisition TR: header units unknown)")

    dataset = NSDDataset(out_root, subjects=[args.subject], time_frames=args.time_frames)
    start = time.time()
    sample = dataset[0]
    first_load = time.time() - start
    x = sample["x"]
    brain_t = x[BRAIN, 0] > 0
    gm_t = x[GM, 0] > 0
    wm_t = x[WM, 0] > 0
    csf_t = x[CSF, 0] > 0
    dyn = x[DYN]
    REPORT["phase1"].update(
        {
            "sample_shape": list(x.shape),
            "dyn_gm_sd_psc": float(dyn[:, gm_t].std()),
            "dyn_wm_sd_psc": float(dyn[:, wm_t].std()),
            "dyn_csf_sd_psc": float(dyn[:, csf_t].std()),
            "static_brain_mean": float(x[STATIC, 0][brain_t].mean()),
            "static_brain_sd": float(x[STATIC, 0][brain_t].std(unbiased=False)),
            "first_item_seconds": first_load,
        }
    )
    check("6-channel tensor", x.shape[0] == NUM_CHANNELS, f"shape {tuple(x.shape)}")
    check("tr stored in sample", float(sample["tr"]) > 0, f"tr={float(sample['tr'])}")
    check("dyn exactly 0 outside brain", bool((dyn[:, ~brain_t] == 0).all()))
    check("static_anat exactly 0 outside brain", bool((x[STATIC][:, ~brain_t] == 0).all()))
    check("static_anat z-scored in brain", abs(REPORT["phase1"]["static_brain_mean"]) < 1e-4
          and abs(REPORT["phase1"]["static_brain_sd"] - 1) < 1e-4)
    check("dyn window mean ~0", float(dyn[:, brain_t].mean(0).abs().max()) < 1e-3)
    sd = REPORT["phase1"]["dyn_gm_sd_psc"]
    check("GM PSC SD in plausible 7T range (0.5-6 %)", 0.5 < sd < 6.0, f"{sd:.2f}%")
    print(f"  PSC SD: GM {sd:.2f}% | WM {REPORT['phase1']['dyn_wm_sd_psc']:.2f}% | CSF {REPORT['phase1']['dyn_csf_sd_psc']:.2f}%")

    t = torch.arange(args.time_frames, dtype=torch.float32) - (args.time_frames - 1) / 2

    def drift_fraction(d: torch.Tensor) -> float:
        g = d[:, gm_t]
        beta = (g * t[:, None]).sum(0) / (t**2).sum()
        return float(((beta * t[:, None]) ** 2).sum() / (g**2).sum())

    for mode in ("window", "run+window"):
        d = NSDDataset(out_root, subjects=[args.subject], time_frames=args.time_frames, detrend=mode)[0]["x"][DYN]
        REPORT["phase1"][f"gm_linear_drift_fraction_{mode}"] = drift_fraction(d)
    REPORT["phase1"]["gm_linear_drift_fraction_run"] = drift_fraction(dyn)
    print(f"  in-window linear-drift share of GM variance: run-level {drift_fraction(dyn):.3f} | "
          f"window {REPORT['phase1']['gm_linear_drift_fraction_window']:.3f} | "
          f"run+window {REPORT['phase1']['gm_linear_drift_fraction_run+window']:.3f}")
    return dataset


def _model_dataset(args: argparse.Namespace, frames: int, crop: tuple[int, int, int] | None) -> NSDDataset:
    """Dataset with its own window length (and optional crop) for CPU-sized model checks."""
    root = Path(args.work_dir) / f"nsd_processed_t{frames}"
    if not (root / args.subject / "manifest.json").is_file():
        pp.preprocess_subject(
            Path(args.raw_root), root, args.subject, window_length=frames, stride=frames, tr_override=args.tr
        )
    return NSDDataset(root, subjects=[args.subject], time_frames=frames, crop_size=crop)


def _batch(item: dict) -> tuple[torch.Tensor, torch.Tensor]:
    return item["x"].unsqueeze(0), item["tr"].reshape(1)


def _small_model(args: argparse.Namespace, **kw):
    from models.mae_core import HierarchicalSwinMAE

    return HierarchicalSwinMAE(
        patch_size=args.patch_size, embed_dim=args.embed_dim, full_res_channels=args.full_res_channels, n_comp=3, **kw
    )


def phase2(args: argparse.Namespace) -> None:
    from models.mae_core import HierarchicalSwinMAE

    print("\n== Phase 2: U-Net decoder, relative position bias, padding masks ==")
    report: dict = {}
    # (a) cluster configuration (patch 14, embed 96): structure + full-resolution no-grad forward
    torch.manual_seed(0)
    full = HierarchicalSwinMAE(patch_size=14, embed_dim=96).eval()

    def count(*names: str) -> float:
        return sum(p.numel() for n in names for p in getattr(full, n).parameters()) / 1e6

    counts = {
        "total_M": sum(p.numel() for p in full.parameters()) / 1e6,
        "patch_embed_M": count("patch_embed"),
        "encoder_M": count("stage1", "stage2", "stage3", "stage4", "merge1", "merge2", "merge3"),
        "decoder_M": count("decoder"),
    }
    report["cluster_config_params"] = counts
    print("  cluster config params (M): " + ", ".join(f"{k}={v:.2f}" for k, v in counts.items())
          + "  (baseline: 53.20 total incl. 14.40 pos_embed)")
    check("no absolute pos_embed parameter", not any("pos_embed" in n for n, _ in full.named_parameters()))
    check("no ConvTranspose3d in model", not any(isinstance(m, torch.nn.ConvTranspose3d) for m in full.modules()))
    check("8 relative-position-bias tables (2 blocks x 4 stages)",
          sum("relative_position_bias_table" in n for n, _ in full.named_parameters()) == 8)

    dataset = _model_dataset(args, 2, None)
    x, tr = _batch(dataset[0])
    shapes: dict[str, tuple] = {}
    for name in ("patch_embed", "stage1", "stage2", "stage3", "stage4"):
        getattr(full, name).register_forward_hook(
            lambda m, i, o, name=name: shapes.__setitem__(name, tuple(o[0].shape) if isinstance(o, tuple) else tuple(o.shape))
        )
    start = time.time()
    with torch.no_grad():
        out = full(x, tr, generator=torch.Generator().manual_seed(0))
    elapsed = time.time() - start
    report["full_res_forward"] = {"input": list(x.shape), "stage_shapes": {k: list(v) for k, v in shapes.items()}, "seconds": elapsed}
    print(f"  full-res forward (T=2, 81x104x83 -> padded 112^3) took {elapsed:.1f}s; stage shapes: {shapes}")
    check("full-resolution forward runs and is finite", all(bool(torch.isfinite(out[k]).all()) for k in ("clean", "noise", "recon")))
    check("output cropped to input spatial shape", tuple(out["clean"].shape[-3:]) == tuple(x.shape[-3:]))
    check("skip features exist at 8^3, 4^3, 2^3, 1^3 tokens",
          [shapes[k][2:5] for k in ("stage1", "stage2", "stage3", "stage4")] == [(8, 8, 8), (4, 4, 4), (2, 2, 2), (1, 1, 1)])
    valid = out["valid"]
    report["valid_cells"] = {"valid": int(valid.sum()), "total": int(valid.numel()), "padding_or_empty": int((~valid).sum())}
    print(f"  valid (brain-containing) cells: {int(valid.sum())}/{valid.numel()} -> {int((~valid).sum())} padding/empty cells excluded from masking")
    check("padding-only cells are never masked", not bool((out["token_mask"] & ~valid).any()))
    del full, out

    # (b) CPU-sized model on a real crop: gradients through skips, leakage-free masking
    crop = tuple(args.crop)
    ds = _model_dataset(args, args.model_frames, crop)
    x, tr = _batch(ds[0])
    torch.manual_seed(0)
    model = _small_model(args)
    model.train()
    out = model(x, tr, generator=torch.Generator().manual_seed(1))
    loss = out["clean"].square().mean() + out["noise"].square().mean() + out["clean_pre"].square().mean()
    loss.backward()
    grad_norm = {
        n: float(sum(p.grad.abs().sum() for p in getattr(model, n).parameters() if p.grad is not None))
        for n in ("stage1", "stage2", "stage3", "stage4")
    }
    stem_grad = float(sum(p.grad.abs().sum() for p in model.decoder.stem.parameters() if p.grad is not None))
    check("gradients reach every encoder stage (skips + bottleneck)", all(v > 0 for v in grad_norm.values()),
          str({k: round(v, 3) for k, v in grad_norm.items()}))
    check("gradients reach the voxel-resolution anatomy stem", stem_grad > 0, f"{stem_grad:.3f}")
    model.eval()
    with torch.no_grad():
        reference = model(x, tr, generator=torch.Generator().manual_seed(2))
        hidden = reference["voxel_mask"][0, 0]
        perturbed = x.clone()
        perturbed[0, DYN][:, hidden] += 25.0
        again = model(perturbed, tr, token_mask=reference["token_mask"])
    leak = max(float((reference[k] - again[k]).abs().max()) for k in ("clean", "noise", "recon", "regressors"))
    report["hidden_tube_perturbation_max_abs_change"] = leak
    check("changing every hidden tube's measurements alters no output (no leakage)", leak < 1e-5, f"max |change| = {leak:.2e}")
    ratio = float(reference["token_mask"].sum() / reference["valid"].sum())
    check("fixed-count masking at ratio 0.75", abs(ratio - 0.75) < 0.01, f"{ratio:.3f}")
    REPORT["phase2"] = report


def phase3(args: argparse.Namespace) -> None:
    from utils.losses import IdentifiableDenoisingLoss
    from utils.spectral import bandpass
    from utils.tissue import nuisance_regressors

    print("\n== Phase 3: asymmetric heads and identifiable loss (real subj01) ==")
    report: dict = {}
    # (a) loss ordering on a real 40-frame window (brain-centred crop keeps RAM small)
    ds40 = _model_dataset(args, 40, (72, 88, 72))
    item = ds40[0]
    x40 = item["x"].unsqueeze(0)
    tr = item["tr"].reshape(1)
    dyn = x40[:, DYN : DYN + 1]
    brain = x40[:, BRAIN : BRAIN + 1, 0] > 0
    gm = x40[:, GM : GM + 1, 0] > 0
    r = nuisance_regressors(dyn, x40[:, WM : WM + 1, 0], x40[:, CSF : CSF + 1, 0], torch.ones_like(brain), n_comp=3)
    gm_mean = dyn[0, 0][:, gm[0, 0]].mean(1)
    report["regressors_corr_with_gm_mean"] = [float(torch.corrcoef(torch.stack([r[0, j], gm_mean]))[0, 1]) for j in range(r.shape[1])]
    print(f"  crop {tuple(x40.shape[-3:])}, regressor-vs-GM-mean correlations: {[round(v, 2) for v in report['regressors_corr_with_gm_mean']]}")
    check("regressors are unit variance", bool(torch.allclose(r.square().mean(-1), torch.ones_like(r.square().mean(-1)), atol=1e-3)))

    d = dyn[0, 0]  # (T, X, Y, Z)
    pinv = torch.linalg.pinv(r[0].T)  # (J, T)
    beta = torch.einsum("jt,txyz->jxyz", pinv, d)
    n_phys = torch.einsum("jxyz,jt->txyz", beta, r[0]) * brain[0, 0]
    s_intended = bandpass((d - n_phys)[None, None], tr, 0.01, 0.15)[0, 0] * gm[0, 0]
    solutions = {
        "S1 intended (band-passed, nuisance-projected, GM only)": (s_intended, n_phys),
        "S2 identity (clean = observed)": (d.clone(), torch.zeros_like(d)),
        "S3 all-in-noise (clean = 0, noise = LS projection)": (torch.zeros_like(d), n_phys),
        "S4 nothing (clean = noise = 0)": (torch.zeros_like(d), torch.zeros_like(d)),
    }
    loss_fn = IdentifiableDenoisingLoss()
    rows = {}
    for name, (c, n) in solutions.items():
        out = {
            "clean": c[None, None], "noise": n[None, None], "recon": (c + n)[None, None], "clean_pre": c[None, None],
            "regressors": r, "voxel_mask": torch.ones_like(brain)[:, None],
        }
        rows[name] = {k: float(v) for k, v in loss_fn(out, x40, tr).items() if k in ("total", "rec", "tissue", "band", "ortho")}
        print(f"  {name:58s} " + " ".join(f"{k}={v:8.3f}" for k, v in rows[name].items()))
    report["loss_ordering_real_window"] = rows
    totals = {k: v["total"] for k, v in rows.items()}
    first = next(iter(totals))
    others = [v for k, v in totals.items() if k != first]
    check("intended decomposition has the lowest loss on real data", totals[first] < min(others),
          f"{totals[first]:.3f} vs next {min(others):.3f} (margin {min(others) - totals[first]:.3f})")
    check("margin is order-one of the reconstruction scale (not ~1e-4)", min(others) - totals[first] > 0.05 * totals[first])
    del x40, dyn, d, beta, n_phys, s_intended, solutions

    # (b) trainable model on a real crop: forward/backward, band-limit, noise subspace
    ds = _model_dataset(args, args.model_frames, tuple(args.crop))
    x, tr = _batch(ds[0])
    torch.manual_seed(0)
    model = _small_model(args)
    model.train()
    out = model(x, tr, generator=torch.Generator().manual_seed(1))
    losses = IdentifiableDenoisingLoss()(out, x, tr)
    losses["total"].backward()
    finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    report["untrained_losses"] = {k: float(v.detach()) for k, v in losses.items()}
    print("  untrained real-crop losses: " + " ".join(f"{k}={float(v):.3f}" for k, v in losses.items()))
    check("backward gives finite gradients for every parameter", finite)
    with torch.no_grad():
        spectrum = torch.fft.rfft(out["clean"][0, 0].detach(), dim=0).abs()
        freqs = torch.fft.rfftfreq(args.model_frames, d=float(tr[0]))
        outside = (freqs < 0.01) | (freqs > 0.15 + 1e-6)
        leak = float(spectrum[outside].max() / spectrum.max().clamp_min(1e-12))
        noise_flat = out["noise"][0, 0].detach().reshape(args.model_frames, -1)
        rj = out["regressors"][0]
        residual = noise_flat - rj.T @ torch.linalg.lstsq(rj.T, noise_flat).solution
        subspace_err = float(residual.abs().max())
    check("clean head spectrum has no energy outside 0.01-0.15 Hz", leak < 1e-4, f"relative leak {leak:.1e}")
    check("noise lies in the span of the visible-voxel regressors", subspace_err < 1e-4, f"max residual {subspace_err:.1e}")

    # (c) short overfit sanity run (not a result): the objective is learnable and the heads behave
    if args.overfit_steps > 0:
        torch.manual_seed(0)
        model = _small_model(args)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
        history = []
        print(f"  overfit sanity run: {args.overfit_steps} steps on one real {tuple(args.crop)} window (T={args.model_frames})")
        for step in range(args.overfit_steps):
            out = model(x, tr)
            losses = loss_fn(out, x, tr)
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            history.append({k: float(v.detach()) for k, v in losses.items()})
            if step % max(1, args.overfit_steps // 5) == 0 or step == args.overfit_steps - 1:
                print(f"    step {step:3d} " + " ".join(
                    f"{k}={history[-1][k]:.3f}" for k in ("total", "rec", "tissue", "band", "ortho", "clean_var_frac", "noise_var_frac")))
        report["overfit_history"] = history
        first_rec = float(np.mean([h["rec"] for h in history[:3]]))
        last_rec = float(np.mean([h["rec"] for h in history[-3:]]))
        check("masked reconstruction loss decreases during the overfit sanity run", last_rec < first_rec, f"{first_rec:.3f} -> {last_rec:.3f}")
        torch.save(model.state_dict(), Path(args.work_dir) / "overfit_state.pt")
    REPORT["phase3"] = report


def phase4(args: argparse.Namespace) -> None:
    import argparse as ap

    import lightning as L

    import evaluate_nsd
    from models.lightning_model import fMRIDenoisingLightning
    from utils.temporal_metrics import dynamic_metrics

    print("\n== Phase 4: J-invariant complementary K-pass inference (real subj01) ==")
    report: dict = {}
    ds = _model_dataset(args, args.model_frames, tuple(args.crop))
    x, tr = _batch(ds[0])
    torch.manual_seed(0)
    model = _small_model(args).eval()
    state = Path(args.work_dir) / "overfit_state.pt"
    if state.is_file():
        model.load_state_dict(torch.load(state))
        print("  using the weights from the Phase 3 overfit sanity run")
    start = time.time()
    result = model.denoise_complementary(x, tr, K=4, generator=torch.Generator().manual_seed(5))
    elapsed = time.time() - start
    brain = x[:, BRAIN : BRAIN + 1, 0] > 0
    counts = result["count"][:, :, 0]
    report["k4_seconds"] = elapsed
    check("every brain voxel is hidden in exactly K-1 = 3 passes", bool((counts[brain] == 3).all()))
    check("K-pass outputs are finite", all(bool(torch.isfinite(result[k]).all()) for k in ("clean", "noise", "recon")))

    cells = np.argwhere(result["valid"][0].reshape(result["grid"]).numpy())
    cell = tuple(int(v) for v in cells[len(cells) // 2])
    region = tuple(slice(c * args.patch_size, (c + 1) * args.patch_size) for c in cell)
    perturbed = x.clone()
    perturbed[0, DYN][(slice(None), *region)] += 25.0
    again = model.denoise_complementary(perturbed, tr, K=4, generator=torch.Generator().manual_seed(5))
    inside = (slice(None), slice(None), slice(None), *region)
    inside_change = float((result["recon"][inside] - again["recon"][inside]).abs().max())
    diff = (result["recon"] - again["recon"]).abs()
    diff[inside] = 0
    outside_change = float(diff.max())
    report["k_pass_perturbation"] = {"inside_tube_max_change": inside_change, "other_tubes_max_change": outside_change}
    check("a tube's own measurements never influence its K-pass estimate", inside_change < 1e-5,
          f"inside {inside_change:.2e}, neighbours {outside_change:.2e}")
    check("neighbouring tubes do see the perturbation (context is used)", outside_change > 1e-4)

    sample = x[0]
    gm = sample[GM, 0] > 0
    metrics = dynamic_metrics(sample[DYN], result["clean"][0, 0], result["noise"][0, 0], gm, float(tr[0]))
    report["dynamic_metrics_overfit_model"] = metrics
    print("  dynamic metrics: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items() if v is not None))
    check("all dynamic metrics are finite", all(v is not None and np.isfinite(v) for v in metrics.values()))
    print(f"  K=4 inference on the {tuple(sample.shape[-3:])} crop, T={sample.shape[1]}: {elapsed:.1f}s")

    # Evaluation script on the real NIfTI window(s). One subject only: checks code paths, not generalisation.
    lit = fMRIDenoisingLightning(
        patch_size=args.patch_size, embed_dim=args.embed_dim, full_res_channels=args.full_res_channels, n_comp=3
    )
    if state.is_file():
        lit.model.load_state_dict(torch.load(state))
    ckpt = Path(args.work_dir) / "subj01_check.ckpt"
    trainer = L.Trainer(max_epochs=1, accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
                        enable_progress_bar=False, enable_model_summary=False)
    trainer.strategy.connect(lit)
    trainer.save_checkpoint(ckpt)
    summary = evaluate_nsd.run_evaluation(
        ap.Namespace(
            data_dir=str(Path(args.work_dir) / f"nsd_processed_t{args.model_frames}"), checkpoint=str(ckpt),
            subjects=[args.subject], validation_subjects=[], train_visual_subjects=[],
            output_dir=str(Path(args.work_dir) / "eval_subj01"), time_frames=args.model_frames, crop_size=tuple(args.crop),
            num_workers=0, K=4, mask_repeats=1, seed=1234, max_windows_per_subject=2, visual_examples_per_subject=1,
            detrend="run", tr=None, amp=False,
        ),
        model=lit,
    )
    report["evaluation_summary"] = summary["splits"]
    check("evaluate_nsd ran with K-pass inference on real NIfTI", "test" in summary["splits"])
    figures = list((Path(args.work_dir) / "eval_subj01" / "visualizations").rglob("*.png"))
    check("evaluation figure written", len(figures) > 0, str(figures[0]) if figures else "")
    REPORT["phase4"] = report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--subject", default="subj01")
    parser.add_argument("--time_frames", type=int, default=40)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--tr", type=float, default=None)
    parser.add_argument("--phases", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--crop", nargs=3, type=int, default=[56, 56, 56], help="Crop for CPU-sized model phases")
    parser.add_argument("--patch_size", type=int, default=7, help="Patch size for CPU-sized model phases")
    parser.add_argument("--embed_dim", type=int, default=24)
    parser.add_argument("--full_res_channels", type=int, default=8)
    parser.add_argument("--model_frames", type=int, default=16, help="Window length for CPU-sized model phases")
    parser.add_argument("--overfit_steps", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    if 1 in args.phases:
        phase1(args)
    for number, runner in ((2, phase2), (3, phase3), (4, phase4)):
        if number in args.phases:
            runner(args)
    if args.report:
        Path(args.report).write_text(json.dumps(REPORT, indent=2), encoding="utf-8")
    print(f"\n{'ALL CHECKS PASSED' if not FAILURES else 'FAILED: ' + ', '.join(FAILURES)}")
    raise SystemExit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
