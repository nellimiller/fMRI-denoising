"""Compare temporal behavior of a trained clean head and WM/CSF projection.

Run from the repository root:
    uv run python -m src.evaluate_temporal --data_dir /path/to/nsd_processed \
      --checkpoint /path/to/best.ckpt --subjects subj08 \
      --output_dir /path/to/temporal_evaluation
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import Tensor

try:
    from .data.nsd_dataset import NSDDataset
    from .models.lightning_model import fMRIDenoisingLightning
    from .utils.region_guided import region_guided_projection
    from .utils.temporal_metrics import temporal_metrics
except ImportError:
    from data.nsd_dataset import NSDDataset
    from models.lightning_model import fMRIDenoisingLightning
    from utils.region_guided import region_guided_projection
    from utils.temporal_metrics import temporal_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, help="Processed NSD root")
    parser.add_argument("--checkpoint", required=True, help="Lightning checkpoint")
    parser.add_argument(
        "--subjects", nargs="+", required=True, help="Subjects to evaluate"
    )
    parser.add_argument(
        "--output_dir", required=True, help="CSV, JSON, and plot output directory"
    )
    parser.add_argument("--time_frames", type=int, default=40)
    parser.add_argument("--crop_size", nargs=3, type=int, default=None)
    parser.add_argument(
        "--max_windows_per_subject",
        type=int,
        default=12,
        help="Evenly-spaced windows per subject; use 0 for all windows",
    )
    parser.add_argument("--plots_per_subject", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def _spread_indices(indices: list[int], count: int) -> list[int]:
    if count <= 0 or count >= len(indices):
        return indices
    if count == 1:
        return [indices[len(indices) // 2]]
    return [
        indices[round(position * (len(indices) - 1) / (count - 1))]
        for position in range(count)
    ]


def _select_indices(
    entries: list[dict[str, Any]],
    subjects: set[str],
    max_windows_per_subject: int,
) -> tuple[list[int], set[int]]:
    indices_by_subject: dict[str, list[int]] = defaultdict(list)
    for index, entry in enumerate(entries):
        subject = str(entry["subject"])
        if subject in subjects:
            indices_by_subject[subject].append(index)

    selected: list[int] = []
    plots: set[int] = set()
    for subject in sorted(subjects):
        subject_indices = indices_by_subject.get(subject, [])
        if not subject_indices:
            raise ValueError(f"Subject {subject} has no dataset windows.")
        if max_windows_per_subject:
            subject_indices = _spread_indices(subject_indices, max_windows_per_subject)
        selected.extend(subject_indices)
        plots.update(_spread_indices(subject_indices, min(3, len(subject_indices))))
    return selected, plots


def _seed_mask(seed: int, device: torch.device) -> None:
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.manual_seed(seed)
    else:
        torch.manual_seed(seed)


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _mean_power(signal: Tensor) -> tuple[Tensor, Tensor]:
    centered = signal.float() - signal.float().mean(dim=0, keepdim=True)
    power = torch.fft.rfft(centered, dim=0).abs().square()
    frequencies = torch.fft.rfftfreq(signal.shape[0], d=1.0, device=signal.device)
    power = power[1:]
    frequencies = frequencies[1:]
    power = power / power.sum(dim=0, keepdim=True).clamp_min(
        torch.finfo(power.dtype).eps
    )
    return frequencies.cpu(), power.mean(dim=1).cpu()


def _save_temporal_plot(
    path: Path,
    sample: Tensor,
    model_clean: Tensor,
    projection_clean: Tensor,
    entry: dict[str, Any],
) -> None:
    bold = sample[0]
    gm_mask = sample[1, 0] > 0
    wm_mask = sample[2, 0] > 0
    csf_mask = sample[3, 0] > 0
    gm_coordinates = gm_mask.nonzero(as_tuple=False)
    gm_values = bold[:, gm_mask]
    if gm_coordinates.numel() == 0:
        raise ValueError(f"GM mask is empty for {entry['subject']} {entry['run']}.")
    voxel_index = int(
        gm_values.std(dim=0, unbiased=False).argsort()[len(gm_coordinates) // 2]
    )
    x_index, y_index, z_index = (int(value) for value in gm_coordinates[voxel_index])

    traces = {
        "Observed GM voxel": bold[:, x_index, y_index, z_index],
        "Model clean": model_clean[:, x_index, y_index, z_index],
        "WM mean": bold[:, wm_mask].mean(dim=1),
        "CSF mean": bold[:, csf_mask].mean(dim=1),
    }

    figure, axes = plt.subplots(1, 2, figsize=(13, 4))
    frames = range(bold.shape[0])
    for label, trace in traces.items():
        centered = trace - trace.mean()
        axes[0].plot(frames, centered.numpy(), label=label, linewidth=1.3)
    axes[0].plot(
        frames,
        (
            projection_clean[:, x_index, y_index, z_index]
            - projection_clean[:, x_index, y_index, z_index].mean()
        ).numpy(),
        label="WM/CSF projection",
        linewidth=1.3,
    )
    axes[0].set_title(f"Centered traces at GM voxel ({x_index}, {y_index}, {z_index})")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Centered normalized BOLD")
    axes[0].legend(fontsize=8)

    for label, signal in (
        ("Observed GM", bold[:, gm_mask]),
        ("Model clean GM", model_clean[:, gm_mask]),
        ("WM/CSF projection GM", projection_clean[:, gm_mask]),
    ):
        frequency, power = _mean_power(signal)
        axes[1].plot(frequency.numpy(), power.numpy(), label=label, linewidth=1.3)
    axes[1].set_title("Mean per-voxel GM power spectrum")
    axes[1].set_xlabel("Frequency (cycles/frame; no TR applied)")
    axes[1].set_ylabel("Fraction of non-DC power")
    axes[1].legend(fontsize=8)

    figure.suptitle(
        f"{entry['subject']} | {entry['run']} | frames {entry['start']}:{entry['stop']}"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty temporal metrics file: {path}")
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    metric_names = [
        name
        for name in rows[0]
        if name not in {"subject", "run", "start", "stop", "method"}
    ]
    summary: dict[str, dict[str, float | None]] = {}
    for method in sorted({str(row["method"]) for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        summary[method] = {}
        for name in metric_names:
            values = [float(row[name]) for row in method_rows if row[name] is not None]
            summary[method][name] = statistics.mean(values) if values else None
    return summary


def main() -> None:
    args = parse_args()
    if args.time_frames < 2:
        raise ValueError("--time_frames must be at least 2.")
    if args.max_windows_per_subject < 0:
        raise ValueError("--max_windows_per_subject cannot be negative.")
    if args.plots_per_subject < 0:
        raise ValueError("--plots_per_subject cannot be negative.")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fMRIDenoisingLightning.load_from_checkpoint(
        str(checkpoint),
        map_location=device,
    )
    model.to(device).eval()

    subjects = set(args.subjects)
    dataset = NSDDataset(
        args.data_dir,
        subjects=sorted(subjects),
        time_frames=args.time_frames,
        crop_size=tuple(args.crop_size) if args.crop_size else None,
    )
    loaded_subjects = {str(entry["subject"]) for entry in dataset.entries}
    missing_subjects = subjects - loaded_subjects
    if missing_subjects:
        raise ValueError(
            f"Requested subjects have no windows: {sorted(missing_subjects)}"
        )

    selected_indices, plot_indices = _select_indices(
        dataset.entries,
        subjects,
        args.max_windows_per_subject,
    )
    if args.plots_per_subject == 0:
        plot_indices = set()
    else:
        plot_indices = {
            index
            for subject in subjects
            for index in _spread_indices(
                [
                    selected_index
                    for selected_index in selected_indices
                    if str(dataset.entries[selected_index]["subject"]) == subject
                ],
                args.plots_per_subject,
            )
        }

    output_dir = Path(args.output_dir)
    plot_dir = output_dir / "temporal_plots"
    if plot_indices:
        plot_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.inference_mode():
        for selected_position, index in enumerate(selected_indices):
            entry = dataset.entries[index]
            sample = dataset[index].to(device)
            time, x_size, y_size, z_size = sample[0].shape
            seed = args.seed + index
            devices = (
                [
                    device.index
                    if device.index is not None
                    else torch.cuda.current_device()
                ]
                if device.type == "cuda"
                else []
            )
            with torch.random.fork_rng(devices=devices):
                _seed_mask(seed, device)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    _, clean, _, _ = model(sample.unsqueeze(0), tube_masking=True)
            model_clean = clean[0, 0, :time, :x_size, :y_size, :z_size].float()

            projection = region_guided_projection(
                sample[0].unsqueeze(0),
                sample[1].unsqueeze(0),
                sample[2].unsqueeze(0),
                sample[3].unsqueeze(0),
            )
            estimates = {
                "model_clean": model_clean,
                "wm_csf_projection": projection.clean[0],
            }
            sample_cpu = sample.detach().cpu()
            for method, estimate in estimates.items():
                metrics = temporal_metrics(
                    sample[0],
                    estimate,
                    sample[1],
                    sample[2],
                    sample[3],
                )
                row = {
                    "subject": str(entry["subject"]),
                    "run": str(entry["run"]),
                    "start": int(entry["start"]),
                    "stop": int(entry["stop"]),
                    "method": method,
                }
                row.update(metrics)
                rows.append(row)

            if index in plot_indices:
                filename = (
                    f"{_safe_filename(str(entry['subject']))}_"
                    f"{_safe_filename(str(entry['run']))}_"
                    f"{int(entry['start']):06d}-{int(entry['stop']):06d}.png"
                )
                _save_temporal_plot(
                    plot_dir / filename,
                    sample_cpu,
                    model_clean.detach().cpu(),
                    projection.clean[0].detach().cpu(),
                    entry,
                )
            print(
                f"Temporal evaluation {entry['subject']} {entry['run']} "
                f"{entry['start']}:{entry['stop']} "
                f"({selected_position + 1}/{len(selected_indices)})"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "temporal_metrics_by_window.csv", rows)
    summary = {
        "checkpoint": str(checkpoint),
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "subjects": sorted(subjects),
        "time_frames": args.time_frames,
        "windows_per_subject": args.max_windows_per_subject or "all",
        "num_windows": len(selected_indices),
        "methods": _summarize(rows),
        "interpretation": (
            "Temporal diagnostics compare the learned clean head and the direct "
            "WM/CSF regional-mean projection. There is no clean-fMRI ground truth. "
            "WM/CSF correlations use traces from the same window and are descriptive, "
            "not independent validation. Reduced correlation or high-frequency power "
            "does not by itself prove improved denoising or preserved neural signal."
        ),
        "plots_dir": str(plot_dir.resolve()) if plot_indices else None,
    }
    (output_dir / "temporal_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved temporal evaluation to {output_dir}")


if __name__ == "__main__":
    main()
