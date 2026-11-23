"""Evaluate a trained NSD denoising model on validation and test subjects.

Every window is denoised with J-invariant complementary K-pass inference (K = 4 keeps
the 75 % training mask ratio; each voxel is averaged over the K - 1 passes in which it was
hidden), identically for the ``train`` (visual only), ``validation`` and ``test`` splits.

Run from the repository root::

    uv run python -m src.evaluate_nsd --data_dir /path/to/nsd_processed \
      --checkpoint /path/to/best.ckpt --subjects subj08 --validation_subjects subj07 \
      --train_visual_subjects subj01 --output_dir /path/to/evaluation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

try:
    from .data.channels import BRAIN, CSF, DYN, GM, STATIC, WM
    from .data.nsd_dataset import NSDDataset
    from .models.lightning_model import fMRIDenoisingLightning
    from .utils.temporal_metrics import dynamic_metrics
except ImportError:
    from data.channels import BRAIN, CSF, DYN, GM, STATIC, WM
    from data.nsd_dataset import NSDDataset
    from models.lightning_model import fMRIDenoisingLightning
    from utils.temporal_metrics import dynamic_metrics


METRIC_NAMES = (
    "mse_brain",
    "r2_brain",
    "r2_gm",
    "gm_corr_clean_input_median",
    "gm_corr_recon_input_median",
    "gm_corr_noise_input_median",
    "gm_tsnr_input_median",
    "gm_tsnr_clean_median",
    "gm_tsnr_phys_removed_median",
    "gm_tsnr_gain_clean_median",
    "gm_tsnr_gain_phys_removed_median",
    "var_clean_frac",
    "var_noise_frac",
    "var_resid_frac",
    "var_cross_frac",
    "band_power_frac_input_median",
    "band_power_frac_clean_median",
    "band_power_frac_noise_median",
    "band_power_frac_resid_median",
)

SPLIT_ORDER = ("train", "validation", "test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True, help="Processed NSD root containing subject manifests")
    parser.add_argument("--checkpoint", required=True, help="Lightning checkpoint to evaluate")
    parser.add_argument("--subjects", nargs="*", default=[], help="Test subject IDs (metrics + visuals)")
    parser.add_argument(
        "--validation_subjects", nargs="*", default=[], help="Validation subject IDs (metrics + visuals)"
    )
    parser.add_argument(
        "--train_visual_subjects",
        nargs="*",
        default=[],
        help="Training subjects to visualize only; excluded from evaluation metrics",
    )
    parser.add_argument("--output_dir", required=True, help="Directory for CSV, JSON and figures")
    parser.add_argument("--time_frames", type=int, default=40)
    parser.add_argument("--crop_size", nargs=3, type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--K", type=int, default=4, help="Complementary masks per window (hidden ratio (K-1)/K)")
    parser.add_argument("--mask_repeats", type=int, default=1, help="Independent K-partitions averaged per window")
    parser.add_argument("--seed", type=int, default=1234, help="Base seed for the tube partitions")
    parser.add_argument("--max_windows_per_subject", type=int, default=0, help="Evenly spaced windows; 0 = all")
    parser.add_argument("--visual_examples_per_subject", type=int, default=3, help="0 disables figures")
    parser.add_argument("--detrend", default="run", choices=["run", "window", "run+window", "none"])
    parser.add_argument("--tr", type=float, default=None, help="Repetition time override (seconds)")
    parser.add_argument("--amp", action="store_true", help="Run inference under fp16 autocast")
    parser.add_argument(
        "--inference_mode",
        default="kpass",
        choices=["kpass", "single_masked", "unmasked"],
        help=(
            "kpass: J-invariant complementary K-pass (default). Ablations: single_masked = one random "
            "tube mask at the training ratio, unmasked = no masking; both let visible voxels see their "
            "own data (not J-invariant, optimistic fits)."
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _sample_spatiotemporal_shape(sample: torch.Tensor) -> tuple[int, int, int, int]:
    if sample.ndim != 5:
        raise ValueError(f"Expected sample [C, T, X, Y, Z], got shape {tuple(sample.shape)}.")
    _, time, x_size, y_size, z_size = sample.shape
    return time, x_size, y_size, z_size


def _spread(indices: list[int], count: int) -> list[int]:
    if count <= 0 or count >= len(indices):
        return list(indices)
    if count == 1:
        return [indices[len(indices) // 2]]
    return [indices[round(position * (len(indices) - 1) / (count - 1))] for position in range(count)]


def _select_visual_indices(
    entries: list[dict[str, Any]],
    examples_per_subject: int,
    subjects: set[str] | None = None,
) -> set[int]:
    indices_by_subject: dict[str, list[int]] = defaultdict(list)
    for index, entry in enumerate(entries):
        subject = str(entry["subject"])
        if subjects is None or subject in subjects:
            indices_by_subject[subject].append(index)

    selected: set[int] = set()
    for indices in indices_by_subject.values():
        selected.update(_spread(indices, min(examples_per_subject, len(indices))))
    return selected


def _select_evaluation_indices(
    entries: list[dict[str, Any]], subjects: set[str], max_windows_per_subject: int
) -> set[int]:
    by_subject: dict[str, list[int]] = defaultdict(list)
    for index, entry in enumerate(entries):
        if str(entry["subject"]) in subjects:
            by_subject[str(entry["subject"])].append(index)
    selected: set[int] = set()
    for indices in by_subject.values():
        selected.update(_spread(indices, max_windows_per_subject) if max_windows_per_subject else indices)
    return selected


def _save_visualization(
    output_path: Path,
    sample: torch.Tensor,
    clean: torch.Tensor,
    noise: torch.Tensor,
    entry: dict[str, Any],
    tr: float = 1.0,
) -> None:
    """Save anatomy, observed PSC, the three components, tissue labels, traces and spectra.

    Args:
        sample: ``[6, T, X, Y, Z]`` input.
        clean, noise: ``[T, X, Y, Z]`` J-invariant estimates.
    """
    sample = sample.detach().cpu()
    clean = clean.detach().float().cpu()
    noise = noise.detach().float().cpu()
    dyn = sample[DYN]
    residual = dyn - clean - noise
    brain = sample[BRAIN, 0] > 0
    gm = sample[GM, 0] > 0
    time_index = dyn.shape[0] // 2
    slice_scores = brain.sum(dim=(0, 1))
    slice_index = int(slice_scores.argmax()) if bool(slice_scores.any()) else dyn.shape[-1] // 2

    inside = dyn[:, brain] if bool(brain.any()) else dyn.flatten()
    flat = inside.abs().flatten()
    flat = flat[:: max(1, flat.numel() // 2_000_000)]  # torch.quantile has an input-size cap
    limit = max(float(torch.quantile(flat, 0.99)), 1e-6)

    labels = torch.zeros_like(brain, dtype=torch.float32)
    labels[sample[GM, 0] > 0] = 1
    labels[sample[WM, 0] > 0] = 2
    labels[sample[CSF, 0] > 0] = 3

    def plane(volume: torch.Tensor) -> torch.Tensor:
        return volume[time_index, :, :, slice_index] if volume.ndim == 4 else volume[:, :, slice_index]

    image_panels = [
        ("Static anatomy (z)", sample[STATIC, 0][:, :, slice_index], "gray", -2.5, 2.5),
        ("Observed PSC (%)", plane(dyn), "coolwarm", -limit, limit),
        ("Clean S (K-pass)", plane(clean), "coolwarm", -limit, limit),
        ("Physiological N", plane(noise), "coolwarm", -limit, limit),
        ("Residual e", plane(residual), "coolwarm", -limit, limit),
    ]

    figure, axes = plt.subplots(2, 4, figsize=(17, 8))
    flat_axes = list(axes.flat)
    for axis, (title, image, cmap, lower, upper) in zip(flat_axes, image_panels):
        axis.imshow(image.numpy(), cmap=cmap, vmin=lower, vmax=upper)
        axis.set_title(title)
        axis.axis("off")
    flat_axes[5].imshow(
        labels[:, :, slice_index].numpy(),
        cmap=ListedColormap(["black", "#2e7d32", "#1565c0", "#7b1fa2"]),
        vmin=0,
        vmax=3,
    )
    flat_axes[5].set_title("GM / WM / CSF")
    flat_axes[5].axis("off")

    seconds = torch.arange(dyn.shape[0]).float() * tr
    gm_dyn, gm_clean, gm_noise, gm_resid = (v[:, gm] for v in (dyn, clean, noise, residual))
    if gm_dyn.shape[1]:
        voxel = int(gm_dyn.std(dim=0, unbiased=False).argsort()[gm_dyn.shape[1] // 2])
        trace_axis = flat_axes[6]
        trace_axis.plot(seconds, gm_dyn[:, voxel], label="observed")
        trace_axis.plot(seconds, gm_clean[:, voxel], label="clean S")
        trace_axis.plot(seconds, gm_noise[:, voxel], label="phys. N")
        trace_axis.set_title("Median-variance GM voxel")
        trace_axis.set_xlabel("Time (s)")
        trace_axis.set_ylabel("PSC (%)")
        trace_axis.legend(fontsize=7)

        spectrum_axis = flat_axes[7]
        for label, series in (
            ("observed", gm_dyn),
            ("clean S", gm_clean),
            ("phys. N", gm_noise),
            ("residual e", gm_resid),
        ):
            centered = series - series.mean(0, keepdim=True)
            power = torch.fft.rfft(centered, dim=0).abs().square().mean(1)[1:]
            frequency = torch.fft.rfftfreq(series.shape[0], d=tr)[1:]
            spectrum_axis.plot(frequency, power / power.sum().clamp_min(1e-12), label=label)
        spectrum_axis.axvspan(0.01, 0.15, color="grey", alpha=0.15)
        spectrum_axis.set_title("GM power spectrum (band shaded)")
        spectrum_axis.set_xlabel("Frequency (Hz)")
        spectrum_axis.legend(fontsize=7)

    figure.suptitle(
        f"{entry['subject']} | {entry['run']} | frames {entry['start']}:{entry['stop']} | "
        f"t={time_index}, z={slice_index}"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def _window_metrics(
    sample: torch.Tensor, clean: torch.Tensor, noise: torch.Tensor, tr: float
) -> dict[str, float | None]:
    """Held-out fit and dynamic diagnostics for one window (all tensors on one device)."""
    dyn = sample[DYN]
    brain = sample[BRAIN, 0] > 0
    gm = sample[GM, 0] > 0
    residual = (dyn - clean - noise)[:, brain]
    energy = dyn[:, brain].square().sum().clamp_min(1e-12)
    metrics: dict[str, float | None] = {
        "mse_brain": float(residual.square().mean()),
        "r2_brain": float(1.0 - residual.square().sum() / energy),
    }
    metrics.update(dynamic_metrics(dyn, clean, noise, gm, tr))
    return metrics


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name in METRIC_NAMES:
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        result[name] = statistics.mean(values) if values else None
    return result


def _aggregate(rows: list[dict[str, Any]], group_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[field]) for field in group_fields)].append(row)

    results = []
    for key, group_rows in sorted(groups.items()):
        aggregate: dict[str, Any] = dict(zip(group_fields, key))
        aggregate["num_windows"] = len(group_rows)
        aggregate.update(_mean_metrics(group_rows))
        results.append(aggregate)
    return results


def _json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats by ``None`` so strict JSON can be written."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty metrics file: {path}")
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_evaluation(args: argparse.Namespace, model: fMRIDenoisingLightning | None = None) -> dict[str, Any]:
    """Evaluate ``args.checkpoint`` and write CSV / JSON / PNG outputs; returns the summary dict."""
    if args.time_frames <= 0 or args.mask_repeats <= 0 or args.K < 2:
        raise ValueError("--time_frames and --mask_repeats must be positive and --K >= 2.")
    if args.visual_examples_per_subject < 0 or args.max_windows_per_subject < 0:
        raise ValueError("Counts cannot be negative.")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = fMRIDenoisingLightning.load_from_checkpoint(str(checkpoint_path), map_location=device)
    model.to(device).eval()

    roles: dict[str, str] = {}
    for role, subjects in (
        ("train", args.train_visual_subjects),
        ("validation", args.validation_subjects),
        ("test", args.subjects),
    ):
        for subject in subjects:
            if subject in roles:
                raise ValueError(f"Subject {subject} is assigned to both {roles[subject]} and {role}.")
            roles[subject] = role
    if not roles:
        raise ValueError("Provide at least one of --subjects / --validation_subjects / --train_visual_subjects.")

    dataset = NSDDataset(
        args.data_dir,
        subjects=sorted(roles),
        time_frames=args.time_frames,
        crop_size=tuple(args.crop_size) if args.crop_size else None,
        detrend=args.detrend,
        tr_override=args.tr,
    )
    loaded = {str(entry["subject"]) for entry in dataset.entries}
    missing = set(roles) - loaded
    if missing:
        raise ValueError(f"Requested subjects have no dataset windows: {sorted(missing)}")

    evaluated = {subject for subject, role in roles.items() if role in {"validation", "test"}}
    evaluation_indices = _select_evaluation_indices(dataset.entries, evaluated, args.max_windows_per_subject)
    visual_indices = (
        _select_visual_indices(dataset.entries, args.visual_examples_per_subject, set(roles))
        if args.visual_examples_per_subject
        else set()
    )
    selected = sorted(evaluation_indices | visual_indices)

    output_dir = Path(args.output_dir)
    visualization_dir = output_dir / "visualizations"
    if visual_indices:
        for role in set(roles.values()):
            (visualization_dir / role).mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        Subset(dataset, selected),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    window_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for position, batch in enumerate(loader):
            index = selected[position]
            entry = dataset.entries[index]
            subject = str(entry["subject"])
            role = roles[subject]
            x = batch["x"].to(device, non_blocking=True)
            tr = batch["tr"].to(device)

            clean_sum = noise_sum = None
            for repeat in range(args.mask_repeats):
                generator = torch.Generator(device=device).manual_seed(args.seed + index * args.mask_repeats + repeat)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
                    mode = getattr(args, "inference_mode", "kpass")
                    if mode == "kpass":
                        result = model.denoise_complementary(x, tr, K=args.K, generator=generator)
                    else:
                        result = model.denoise_single(
                            x, tr, mask_ratio=0.0 if mode == "unmasked" else None, generator=generator
                        )
                clean_sum = result["clean"] if clean_sum is None else clean_sum + result["clean"]
                noise_sum = result["noise"] if noise_sum is None else noise_sum + result["noise"]
            clean = (clean_sum / args.mask_repeats)[0, 0]
            noise = (noise_sum / args.mask_repeats)[0, 0]
            sample = x[0]

            saved = False
            if index in visual_indices:
                filename = (
                    f"{_safe_filename(subject)}_{_safe_filename(str(entry['run']))}_"
                    f"{int(entry['start']):06d}-{int(entry['stop']):06d}.png"
                )
                _save_visualization(
                    visualization_dir / role / filename, sample, clean, noise, entry, float(tr[0])
                )
                saved = True

            if index in evaluation_indices:
                row: dict[str, Any] = {
                    "split": role,
                    "subject": subject,
                    "run": str(entry["run"]),
                    "start": int(entry["start"]),
                    "stop": int(entry["stop"]),
                    "tr": float(tr[0]),
                    "K": args.K,
                    "inference_mode": getattr(args, "inference_mode", "kpass"),
                }
                row.update(_window_metrics(sample, clean, noise, float(tr[0])))
                window_rows.append(row)
            print(
                f"[{role}] {subject} {entry['run']} frames {entry['start']}:{entry['stop']} "
                f"({position + 1}/{len(selected)})" + ("; saved visualization" if saved else "")
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_splits: dict[str, Any] = {}
    run_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    if window_rows:
        run_rows = _aggregate(window_rows, ("split", "subject", "run"))
        by_subject: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in run_rows:
            by_subject[(row["split"], row["subject"])].append(row)
        for (split, subject), rows in sorted(by_subject.items()):
            summary = {
                "split": split,
                "subject": subject,
                "num_runs": len(rows),
                "num_windows": sum(int(r["num_windows"]) for r in rows),
            }
            summary.update(_mean_metrics(rows))
            subject_rows.append(summary)
        for split in SPLIT_ORDER:
            rows = [r for r in subject_rows if r["split"] == split]
            if rows:
                summary_splits[split] = {
                    "subjects": sorted(r["subject"] for r in rows),
                    "num_windows": sum(int(r["num_windows"]) for r in rows),
                    "subject_mean": _mean_metrics(rows),
                }
        _write_csv(output_dir / "metrics_by_window.csv", window_rows)
        _write_csv(output_dir / "metrics_by_run.csv", run_rows)
        _write_csv(output_dir / "metrics_by_subject.csv", subject_rows)

    summary_json = {
        "checkpoint": str(checkpoint_path),
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "subjects_by_split": {
            split: sorted(s for s, role in roles.items() if role == split) for split in SPLIT_ORDER
        },
        "time_frames": args.time_frames,
        "crop_size": args.crop_size,
        "K": args.K,
        "inference_mode": getattr(args, "inference_mode", "kpass"),
        "mask_repeats": args.mask_repeats,
        "seed": args.seed,
        "detrend": args.detrend,
        "tr_override": args.tr,
        "visualizations_dir": str(visualization_dir.resolve()) if visual_indices else None,
        "num_windows": len(window_rows),
        "splits": summary_splits,
        "aggregation": (
            "Window metrics -> run means -> subject means (equal weight per run) -> split means "
            "(equal weight per subject)."
        ),
        "interpretation": (
            "Inference is J-invariant: each voxel's estimate is averaged over the K-1 passes in which "
            "its tube was hidden, so r2 / mse are held-out fits of the observed drift-removed PSC. "
            "There is no clean-fMRI ground truth; tSNR gains and variance shares are descriptive "
            "(removing signal also raises tSNR) and are not proof of improved denoising."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_json, indent=2, allow_nan=False, default=lambda o: None) + "\n", encoding="utf-8"
    )
    print(f"Saved evaluation metrics to {output_dir}")
    return summary_json


def main() -> None:
    run_evaluation(parse_args())


if __name__ == "__main__":
    main()
