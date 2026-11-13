"""Evaluate a trained NSD denoising model on held-out subjects.

Run from the repository root:
    uv run python -m src.evaluate_nsd --data_dir /path/to/nsd_processed \
      --checkpoint /path/to/best.ckpt --subjects subj08 --output_dir /path/to/evaluation
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

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

try:
    from .data.nsd_dataset import NSDDataset
    from .models.lightning_model import fMRIDenoisingLightning
except ImportError:
    from data.nsd_dataset import NSDDataset
    from models.lightning_model import fMRIDenoisingLightning


METRIC_NAMES = (
    "mse_full",
    "mae_full",
    "rmse_full",
    "pearson_full",
    "mse_masked",
    "mae_masked",
    "rmse_masked",
    "pearson_masked",
    "mse_gm",
    "mse_wm",
    "mse_csf",
    "clean_energy",
    "noise_energy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, help="Processed NSD root containing subject manifests")
    parser.add_argument("--checkpoint", required=True, help="Lightning checkpoint to evaluate")
    parser.add_argument("--subjects", nargs="+", required=True, help="Held-out subject IDs")
    parser.add_argument(
        "--train_visual_subjects",
        nargs="*",
        default=[],
        help="Training subjects to visualize only; excluded from evaluation metrics",
    )
    parser.add_argument(
        "--validation_visual_subjects",
        nargs="*",
        default=[],
        help="Validation subjects to visualize only; excluded from evaluation metrics",
    )
    parser.add_argument("--output_dir", required=True, help="Directory for CSV and JSON metrics")
    parser.add_argument("--time_frames", type=int, default=40)
    parser.add_argument("--crop_size", nargs=3, type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mask_repeats", type=int, default=1, help="Number of seeded mask draws per window")
    parser.add_argument("--seed", type=int, default=1234, help="Base seed for deterministic mask draws")
    parser.add_argument(
        "--visual_examples_per_subject",
        type=int,
        default=3,
        help="Save this many example panels per subject; use 0 to disable",
    )
    return parser.parse_args()


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float | None:
    if left.numel() < 2:
        return None
    left = left.float() - left.float().mean()
    right = right.float() - right.float().mean()
    denominator = left.norm() * right.norm()
    if denominator <= 0:
        return None
    return float((torch.dot(left, right) / denominator).item())


def _masked_voxel_grid(
    token_mask: torch.Tensor,
    shape: tuple[int, int, int, int],
    patch_size: int,
) -> torch.Tensor:
    """Expand the model's spatial tube mask to [T, X, Y, Z] voxels."""
    time, x_size, y_size, z_size = shape
    grid_multiple = 8
    grid = tuple(
        math.ceil(math.ceil(size / patch_size) / grid_multiple) * grid_multiple
        for size in (x_size, y_size, z_size)
    )
    expected_tokens = time * math.prod(grid)
    if token_mask.numel() != expected_tokens:
        raise ValueError(
            f"Model returned {token_mask.numel()} mask tokens, expected {expected_tokens} "
            f"for input shape {shape} and patch size {patch_size}."
        )
    mask = token_mask.reshape(time, *grid)
    mask = mask.repeat_interleave(patch_size, dim=1)
    mask = mask.repeat_interleave(patch_size, dim=2)
    mask = mask.repeat_interleave(patch_size, dim=3)
    return mask[:time, :x_size, :y_size, :z_size].bool()


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


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
        count = min(examples_per_subject, len(indices))
        if count == 1:
            selected.add(indices[len(indices) // 2])
        elif count > 1:
            selected.update(
                indices[round(position * (len(indices) - 1) / (count - 1))]
                for position in range(count)
            )
    return selected


def _save_visualization(
    output_path: Path,
    sample: torch.Tensor,
    reconstruction: torch.Tensor,
    clean: torch.Tensor,
    noise: torch.Tensor,
    entry: dict[str, Any],
) -> None:
    """Save a representative slice showing BOLD, model heads, and tissue masks."""
    input_bold = sample[0].detach().cpu()
    reconstruction = reconstruction.detach().cpu()
    clean = clean.detach().cpu()
    noise = noise.detach().cpu()
    time_index = input_bold.shape[0] // 2
    tissue_by_slice = sample[1:, 0].detach().cpu().sum(dim=(0, 1, 2))
    slice_index = (
        int(tissue_by_slice.argmax().item())
        if bool(tissue_by_slice.any())
        else input_bold.shape[-1] // 2
    )

    input_slice = input_bold[time_index, :, :, slice_index]
    reconstruction_slice = reconstruction[time_index, :, :, slice_index]
    clean_slice = clean[time_index, :, :, slice_index]
    noise_slice = noise[time_index, :, :, slice_index]
    residual_slice = input_slice - reconstruction_slice
    brain_mask = sample[1:, 0].detach().cpu().amax(dim=0).bool()
    intensity_values = input_bold[:, brain_mask] if bool(brain_mask.any()) else input_bold.flatten()
    intensity_range = torch.quantile(
        intensity_values.flatten(),
        torch.tensor([0.01, 0.99]),
    )
    vmin, vmax = (float(value) for value in intensity_range)
    if vmin >= vmax:
        vmax = vmin + 1e-6
    residual_limit = max(float(torch.quantile(residual_slice.abs().flatten(), 0.99)), 1e-6)

    panels = [
        ("Observed BOLD (input)", input_slice, "gray", vmin, vmax),
        ("Clean head (estimate)", clean_slice, "gray", vmin, vmax),
        ("Reconstruction", reconstruction_slice, "gray", vmin, vmax),
        ("Noise head", noise_slice, "gray", vmin, vmax),
        ("Input - reconstruction", residual_slice, "coolwarm", -residual_limit, residual_limit),
        ("GM mask", sample[1, 0, :, :, slice_index].detach().cpu(), "Greens", 0, 1),
        ("WM mask", sample[2, 0, :, :, slice_index].detach().cpu(), "Blues", 0, 1),
        ("CSF mask", sample[3, 0, :, :, slice_index].detach().cpu(), "Purples", 0, 1),
    ]

    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    for axis, (title, image, cmap, lower, upper) in zip(axes.flat, panels):
        axis.imshow(image.numpy(), cmap=cmap, vmin=lower, vmax=upper)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(
        f"{entry['subject']} | {entry['run']} | frames "
        f"{entry['start']}:{entry['stop']} | t={time_index}, z={slice_index}"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def _selected_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    selected: torch.Tensor,
    prefix: str,
) -> dict[str, float | None]:
    if not bool(selected.any()):
        return {
            f"mse_{prefix}": None,
            f"mae_{prefix}": None,
            f"rmse_{prefix}": None,
            f"pearson_{prefix}": None,
        }
    pred_values = prediction[selected].float()
    target_values = target[selected].float()
    mse = torch.mean((pred_values - target_values).square())
    return {
        f"mse_{prefix}": float(mse.item()),
        f"mae_{prefix}": float(torch.mean((pred_values - target_values).abs()).item()),
        f"rmse_{prefix}": float(torch.sqrt(mse).item()),
        f"pearson_{prefix}": _pearson(pred_values, target_values),
    }


def _evaluate_draw(
    prediction: torch.Tensor,
    clean: torch.Tensor,
    noise: torch.Tensor,
    sample: torch.Tensor,
    voxel_mask: torch.Tensor,
) -> dict[str, float | None]:
    target = sample[0]
    metrics = _selected_metrics(prediction, target, torch.ones_like(voxel_mask), "full")
    metrics.update(_selected_metrics(prediction, target, voxel_mask, "masked"))

    for channel, region in ((1, "gm"), (2, "wm"), (3, "csf")):
        tissue_mask = sample[channel] > 0
        metrics[f"mse_{region}"] = (
            float(torch.mean((prediction[tissue_mask] - target[tissue_mask]).square()).item())
            if bool(tissue_mask.any())
            else None
        )

    metrics["clean_energy"] = float(clean.float().square().mean().item())
    metrics["noise_energy"] = float(noise.float().square().mean().item())
    return metrics


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name in METRIC_NAMES:
        values = [float(row[name]) for row in rows if row[name] is not None]
        result[name] = statistics.mean(values) if values else None
    return result


def _aggregate(
    rows: list[dict[str, Any]],
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty metrics file: {path}")
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _seed_mask_draw(seed: int, device: torch.device) -> None:
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.manual_seed(seed)
    else:
        torch.manual_seed(seed)


def main() -> None:
    args = parse_args()
    if args.time_frames <= 0 or args.mask_repeats <= 0:
        raise ValueError("--time_frames and --mask_repeats must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num_workers cannot be negative.")
    if args.visual_examples_per_subject < 0:
        raise ValueError("--visual_examples_per_subject cannot be negative.")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fMRIDenoisingLightning.load_from_checkpoint(
        str(checkpoint_path),
        map_location=device,
    )
    model.to(device).eval()
    patch_size = int(model.hparams.patch_size)

    evaluation_subjects = set(args.subjects)
    train_visual_subjects = set(args.train_visual_subjects)
    validation_visual_subjects = set(args.validation_visual_subjects)
    if evaluation_subjects & train_visual_subjects:
        raise ValueError("Test subjects cannot also be listed as training visualization subjects.")
    if evaluation_subjects & validation_visual_subjects:
        raise ValueError("Test subjects cannot also be listed as validation visualization subjects.")
    if train_visual_subjects & validation_visual_subjects:
        raise ValueError("A subject cannot be both a training and validation visualization subject.")

    all_subjects = evaluation_subjects | train_visual_subjects | validation_visual_subjects
    dataset = NSDDataset(
        args.data_dir,
        subjects=sorted(all_subjects),
        time_frames=args.time_frames,
        crop_size=tuple(args.crop_size) if args.crop_size else None,
    )
    requested_subjects = all_subjects
    loaded_subjects = {str(entry["subject"]) for entry in dataset.entries}
    missing_subjects = requested_subjects - loaded_subjects
    if missing_subjects:
        raise ValueError(f"Requested subjects have no dataset windows: {sorted(missing_subjects)}")

    visual_roles = {subject: "test" for subject in evaluation_subjects}
    visual_roles.update({subject: "train" for subject in train_visual_subjects})
    visual_roles.update({subject: "validation" for subject in validation_visual_subjects})
    visual_subjects = set(visual_roles) if args.visual_examples_per_subject else set()
    evaluation_indices = {
        index
        for index, entry in enumerate(dataset.entries)
        if str(entry["subject"]) in evaluation_subjects
    }
    visual_indices = _select_visual_indices(
        dataset.entries,
        args.visual_examples_per_subject,
        visual_subjects,
    )
    selected_indices = sorted(evaluation_indices | visual_indices)

    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    output_dir = Path(args.output_dir)
    visualization_dir = output_dir / "visualizations"
    if visual_subjects:
        for role in set(visual_roles.values()):
            (visualization_dir / role).mkdir(parents=True, exist_ok=True)

    window_rows: list[dict[str, Any]] = []
    visualized_per_subject: dict[str, int] = defaultdict(int)
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            index = selected_indices[batch_index]
            entry = dataset.entries[index]
            subject = str(entry["subject"])
            sample = batch[0].to(device, non_blocking=device.type == "cuda")
            draw_rows = []
            saved_visualization = False
            draw_count = args.mask_repeats if subject in evaluation_subjects else 1
            for draw in range(draw_count):
                _seed_mask_draw(args.seed + index * args.mask_repeats + draw, device)
                prediction, clean, noise, token_mask = model(
                    sample.unsqueeze(0),
                    tube_masking=True,
                )
                time, _, x_size, y_size, z_size = sample.shape
                prediction = prediction[0, 0, :time, :x_size, :y_size, :z_size]
                clean = clean[0, 0, :time, :x_size, :y_size, :z_size]
                noise = noise[0, 0, :time, :x_size, :y_size, :z_size]
                voxel_mask = _masked_voxel_grid(
                    token_mask[0],
                    (time, x_size, y_size, z_size),
                    patch_size,
                )
                if subject in evaluation_subjects:
                    draw_rows.append(_evaluate_draw(prediction, clean, noise, sample, voxel_mask))

                if (
                    draw == 0
                    and index in visual_indices
                ):
                    filename = (
                        f"{_safe_filename(subject)}_{_safe_filename(str(entry['run']))}_"
                        f"{int(entry['start']):06d}-{int(entry['stop']):06d}.png"
                    )
                    _save_visualization(
                        visualization_dir / visual_roles[subject] / filename,
                        sample,
                        prediction,
                        clean,
                        noise,
                        entry,
                    )
                    visualized_per_subject[subject] += 1
                    saved_visualization = True

            if subject not in evaluation_subjects:
                print(
                    f"Visualized {visual_roles[subject]} subject {subject} "
                    f"({batch_index + 1}/{len(selected_indices)})"
                    + ("; saved visualization" if saved_visualization else "")
                )
                continue

            row: dict[str, Any] = {
                "subject": subject,
                "run": str(entry["run"]),
                "start": int(entry["start"]),
                "stop": int(entry["stop"]),
                "mask_repeats": args.mask_repeats,
            }
            row.update(_mean_metrics(draw_rows))
            window_rows.append(row)
            print(
                f"Evaluated {row['subject']} {row['run']} "
                f"frames {row['start']}:{row['stop']} "
                f"({batch_index + 1}/{len(selected_indices)})"
                + ("; saved visualization" if saved_visualization else "")
            )

    run_rows = _aggregate(window_rows, ("subject", "run"))
    subject_runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        subject_runs[str(row["subject"])].append(row)
    subject_rows = []
    for subject, rows in sorted(subject_runs.items()):
        summary: dict[str, Any] = {
            "subject": subject,
            "num_runs": len(rows),
            "num_windows": sum(int(row["num_windows"]) for row in rows),
        }
        summary.update(_mean_metrics(rows))
        subject_rows.append(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "metrics_by_window.csv", window_rows)
    _write_csv(output_dir / "metrics_by_run.csv", run_rows)
    _write_csv(output_dir / "metrics_by_subject.csv", subject_rows)

    summary_json = {
        "checkpoint": str(checkpoint_path),
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "subjects": sorted(evaluation_subjects),
        "visualization_subjects_by_split": {
            role: sorted(subject for subject, subject_role in visual_roles.items() if subject_role == role)
            for role in ("train", "validation", "test")
        },
        "time_frames": args.time_frames,
        "crop_size": args.crop_size,
        "patch_size": patch_size,
        "mask_repeats": args.mask_repeats,
        "seed": args.seed,
        "visualizations_dir": str(visualization_dir.resolve()) if args.visual_examples_per_subject else None,
        "visual_examples_per_subject": args.visual_examples_per_subject,
        "visualizations_saved": sum(visualized_per_subject.values()),
        "num_windows": len(window_rows),
        "aggregation": (
            "Window metrics average mask draws; run metrics average windows; "
            "subject metrics average runs equally."
        ),
        "interpretation": (
            "Metrics compare reconstruction with observed, per-window-normalized BOLD. "
            "They measure reconstruction/consistency, not denoising accuracy, because "
            "the evaluation data has no clean-fMRI ground truth."
        ),
        "overall_subject_mean": _mean_metrics(subject_rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_json, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved evaluation metrics to {output_dir}")


if __name__ == "__main__":
    main()
