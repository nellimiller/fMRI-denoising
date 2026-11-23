"""Compare the model's K-pass clean estimate with the WM/CSF projection baseline.

Run from the repository root::

    uv run python -m src.evaluate_temporal --data_dir /path/to/nsd_processed \
      --checkpoint /path/to/best.ckpt --subjects subj08 \
      --output_dir /path/to/temporal_evaluation

Both estimates are scored on the same fMRI-specific diagnostics (see
:func:`utils.temporal_metrics.dynamic_metrics`). The model estimate is J-invariant
(complementary K-pass inference); the baseline is a deterministic regression of
1-voxel-eroded WM / CSF mean traces from GM voxels. There is no clean-fMRI ground
truth: correlations with WM/CSF traces and tSNR gains are descriptive only.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torch import Tensor  # noqa: E402

try:
    from .data.channels import CSF, DYN, GM, WM
    from .data.nsd_dataset import NSDDataset
    from .models.lightning_model import fMRIDenoisingLightning
    from .utils.region_guided import region_guided_projection
    from .utils.temporal_metrics import dynamic_metrics, temporal_metrics
    from .utils.tissue import erode_mask
except ImportError:
    from data.channels import CSF, DYN, GM, WM
    from data.nsd_dataset import NSDDataset
    from models.lightning_model import fMRIDenoisingLightning
    from utils.region_guided import region_guided_projection
    from utils.temporal_metrics import dynamic_metrics, temporal_metrics
    from utils.tissue import erode_mask


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True, help="Processed NSD root")
    parser.add_argument("--checkpoint", required=True, help="Lightning checkpoint")
    parser.add_argument("--subjects", nargs="+", required=True, help="Subjects to evaluate")
    parser.add_argument("--output_dir", required=True, help="CSV, JSON, and plot output directory")
    parser.add_argument("--time_frames", type=int, default=40)
    parser.add_argument("--crop_size", nargs=3, type=int, default=None)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument(
        "--max_windows_per_subject",
        type=int,
        default=12,
        help="Evenly-spaced windows per subject; use 0 for all windows",
    )
    parser.add_argument("--plots_per_subject", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--detrend", default="run", choices=["run", "window", "run+window", "none"])
    parser.add_argument("--tr", type=float, default=None, help="Repetition time override (seconds)")
    parser.add_argument("--amp", action="store_true", help="Run inference under fp16 autocast")
    return parser.parse_args(argv)


def _spread_indices(indices: list[int], count: int) -> list[int]:
    if count <= 0 or count >= len(indices):
        return indices
    if count == 1:
        return [indices[len(indices) // 2]]
    return [indices[round(position * (len(indices) - 1) / (count - 1))] for position in range(count)]


def _select_indices(
    entries: list[dict[str, Any]], subjects: set[str], max_windows_per_subject: int, plots_per_subject: int
) -> tuple[list[int], set[int]]:
    by_subject: dict[str, list[int]] = defaultdict(list)
    for index, entry in enumerate(entries):
        if str(entry["subject"]) in subjects:
            by_subject[str(entry["subject"])].append(index)

    selected: list[int] = []
    plots: set[int] = set()
    for subject in sorted(subjects):
        indices = by_subject.get(subject, [])
        if not indices:
            raise ValueError(f"Subject {subject} has no dataset windows.")
        if max_windows_per_subject:
            indices = _spread_indices(indices, max_windows_per_subject)
        selected.extend(indices)
        if plots_per_subject:
            plots.update(_spread_indices(indices, min(plots_per_subject, len(indices))))
    return selected, plots


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _save_temporal_plot(
    path: Path,
    sample: Tensor,
    estimates: dict[str, tuple[Tensor, Tensor]],
    tr: float,
    entry: dict[str, Any],
) -> None:
    """Traces of one median-variance GM voxel, tissue traces and GM power spectra."""
    dyn = sample[DYN]
    gm_mask, wm_mask, csf_mask = (sample[c, 0] > 0 for c in (GM, WM, CSF))
    gm_coordinates = gm_mask.nonzero(as_tuple=False)
    if gm_coordinates.numel() == 0:
        raise ValueError(f"GM mask is empty for {entry['subject']} {entry['run']}.")
    gm_values = dyn[:, gm_mask]
    voxel = int(gm_values.std(dim=0, unbiased=False).argsort()[len(gm_coordinates) // 2])
    x, y, z = (int(v) for v in gm_coordinates[voxel])
    seconds = torch.arange(dyn.shape[0]).float() * tr

    figure, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(seconds, dyn[:, x, y, z], label="Observed GM voxel", linewidth=1.3)
    for name, (clean, noise) in estimates.items():
        axes[0].plot(seconds, clean[:, x, y, z], label=f"{name}: clean", linewidth=1.3)
        if name == "model":
            axes[0].plot(seconds, noise[:, x, y, z], label="model: phys. noise", linewidth=1.0, linestyle=":")
    axes[0].plot(seconds, dyn[:, wm_mask].mean(1), label="WM mean", linewidth=1.0, alpha=0.7)
    axes[0].plot(seconds, dyn[:, csf_mask].mean(1), label="CSF mean", linewidth=1.0, alpha=0.7)
    axes[0].set_title(f"GM voxel ({x}, {y}, {z}) traces")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Percent signal change")
    axes[0].legend(fontsize=7)

    spectra = {"Observed GM": gm_values}
    spectra.update({f"{name} clean GM": clean[:, gm_mask] for name, (clean, _) in estimates.items()})
    frequency = torch.fft.rfftfreq(dyn.shape[0], d=tr)[1:]
    for label, signal in spectra.items():
        centered = signal.float() - signal.float().mean(0, keepdim=True)
        power = torch.fft.rfft(centered, dim=0).abs().square()[1:]
        power = power / power.sum(0, keepdim=True).clamp_min(1e-12)
        axes[1].plot(frequency, power.mean(1), label=label, linewidth=1.3)
    axes[1].axvspan(0.01, 0.15, color="grey", alpha=0.15)
    axes[1].set_title("Mean per-voxel GM power spectrum")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Fraction of non-DC power")
    axes[1].legend(fontsize=7)

    figure.suptitle(f"{entry['subject']} | {entry['run']} | frames {entry['start']}:{entry['stop']}")
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
    metric_names = [n for n in rows[0] if n not in {"subject", "run", "start", "stop", "method"}]
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
    if args.time_frames < 2 or args.K < 2:
        raise ValueError("--time_frames must be >= 2 and --K >= 2.")
    if args.max_windows_per_subject < 0 or args.plots_per_subject < 0:
        raise ValueError("Counts cannot be negative.")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fMRIDenoisingLightning.load_from_checkpoint(str(checkpoint), map_location=device)
    model.to(device).eval()
    erosion = int(model.hparams.erosion_iterations)

    subjects = set(args.subjects)
    dataset = NSDDataset(
        args.data_dir,
        subjects=sorted(subjects),
        time_frames=args.time_frames,
        crop_size=tuple(args.crop_size) if args.crop_size else None,
        detrend=args.detrend,
        tr_override=args.tr,
    )
    loaded = {str(entry["subject"]) for entry in dataset.entries}
    if subjects - loaded:
        raise ValueError(f"Requested subjects have no windows: {sorted(subjects - loaded)}")

    selected, plot_indices = _select_indices(
        dataset.entries, subjects, args.max_windows_per_subject, args.plots_per_subject
    )
    output_dir = Path(args.output_dir)
    plot_dir = output_dir / "temporal_plots"
    if plot_indices:
        plot_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for position, index in enumerate(selected):
            entry = dataset.entries[index]
            item = dataset[index]
            x = item["x"].unsqueeze(0).to(device)
            tr = item["tr"].reshape(1).to(device)
            tr_value = float(tr[0])
            generator = torch.Generator(device=device).manual_seed(args.seed + index)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
                result = model.denoise_complementary(x, tr, K=args.K, generator=generator)
            sample = x[0]
            dyn = sample[DYN]
            wm_e = erode_mask(sample[WM : WM + 1, 0].unsqueeze(0) > 0, erosion)[0, 0]
            csf_e = erode_mask(sample[CSF : CSF + 1, 0].unsqueeze(0) > 0, erosion)[0, 0]
            projection = region_guided_projection(
                dyn.unsqueeze(0), sample[GM].unsqueeze(0) > 0, wm_e.unsqueeze(0), csf_e.unsqueeze(0)
            )
            estimates = {
                "model": (result["clean"][0, 0].float(), result["noise"][0, 0].float()),
                "wm_csf_projection": (projection.clean[0], projection.noise[0]),
            }
            for method, (clean, noise) in estimates.items():
                row: dict[str, Any] = {
                    "subject": str(entry["subject"]),
                    "run": str(entry["run"]),
                    "start": int(entry["start"]),
                    "stop": int(entry["stop"]),
                    "method": method,
                }
                row.update(dynamic_metrics(dyn, clean, noise, sample[GM, 0], tr_value))
                row.update(temporal_metrics(dyn, clean, sample[GM], sample[WM], sample[CSF]))
                rows.append(row)

            if index in plot_indices:
                filename = (
                    f"{_safe_filename(str(entry['subject']))}_{_safe_filename(str(entry['run']))}_"
                    f"{int(entry['start']):06d}-{int(entry['stop']):06d}.png"
                )
                _save_temporal_plot(
                    plot_dir / filename,
                    sample.cpu(),
                    {k: (c.cpu(), n.cpu()) for k, (c, n) in estimates.items()},
                    tr_value,
                    entry,
                )
            print(
                f"Temporal evaluation {entry['subject']} {entry['run']} "
                f"{entry['start']}:{entry['stop']} ({position + 1}/{len(selected)})"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "temporal_metrics_by_window.csv", rows)
    summary = {
        "checkpoint": str(checkpoint),
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "subjects": sorted(subjects),
        "time_frames": args.time_frames,
        "K": args.K,
        "windows_per_subject": args.max_windows_per_subject or "all",
        "num_windows": len(selected),
        "methods": _summarize(rows),
        "interpretation": (
            "Model clean/noise are J-invariant K-pass estimates; the baseline regresses 1-voxel-eroded "
            "WM/CSF mean traces from GM. There is no clean-fMRI ground truth. Correlations with WM/CSF "
            "traces use traces from the same window and are descriptive; a tSNR gain is also obtained by "
            "removing genuine signal and is not proof of improved denoising."
        ),
        "plots_dir": str(plot_dir.resolve()) if plot_indices else None,
    }

    def clean_json(value: Any) -> Any:
        if isinstance(value, float) and value != value:
            return None
        if isinstance(value, dict):
            return {k: clean_json(v) for k, v in value.items()}
        return value

    (output_dir / "temporal_summary.json").write_text(
        json.dumps(clean_json(summary), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Saved temporal evaluation to {output_dir}")


if __name__ == "__main__":
    main()
