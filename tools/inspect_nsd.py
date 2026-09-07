"""Inspect one processed NSD window and save diagnostic visualizations.

Example:
    uv run python tools/inspect_nsd.py \
      --processed-root "C:\\fMRI Denoising\\NSD\\nsd_processed" \
      --subject subj01 \
      --entry 0 \
      --output-dir "C:\\fMRI Denoising\\NSD\\inspection_subj01"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def _load_entry(processed_root: Path, subject: str, entry_index: int) -> dict:
    manifest_path = processed_root / subject / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["entries"]
    if entry_index < 0 or entry_index >= len(entries):
        raise IndexError(f"entry must be between 0 and {len(entries) - 1}")
    return entries[entry_index]


def _load_window(entry: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    bold_image = nib.load(entry["bold_path"])
    start, stop = int(entry["start"]), int(entry["stop"])
    bold = bold_image.slicer[:, :, :, start:stop].get_fdata(dtype=np.float32)
    masks = {
        "GM": nib.load(entry["gm_path"]).get_fdata(dtype=np.float32) > 0,
        "WM": nib.load(entry["wm_path"]).get_fdata(dtype=np.float32) > 0,
        "CSF": nib.load(entry["csf_path"]).get_fdata(dtype=np.float32) > 0,
    }
    for name, mask in masks.items():
        if mask.shape != bold.shape[:3]:
            raise ValueError(f"{name} shape {mask.shape} does not match BOLD {bold.shape[:3]}")
    return bold, masks


def _save_slices(bold: np.ndarray, masks: dict[str, np.ndarray], output: Path, frame: int) -> None:
    mean_volume = bold.mean(axis=3)
    x, y, z = (size // 2 for size in mean_volume.shape)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    images = [
        ("BOLD mean", mean_volume),
        ("GM", masks["GM"]),
        ("WM", masks["WM"]),
        ("CSF", masks["CSF"]),
    ]
    for row, axis in enumerate(("sagittal", "coronal", "axial")):
        for col, (name, volume) in enumerate(images):
            if row == 0:
                image = volume[x, :, :].T
            elif row == 1:
                image = volume[:, y, :].T
            else:
                image = volume[:, :, z].T
            axes[row, col].imshow(image, cmap="gray", origin="lower")
            axes[row, col].set_title(f"{name} ({axis})")
            axes[row, col].axis("off")
    fig.suptitle(f"NSD diagnostic slices, frame {frame}")
    fig.tight_layout()
    fig.savefig(output / "spatial_slices.png", dpi=150)
    plt.close(fig)


def _save_frame_and_timeseries(
    bold: np.ndarray,
    masks: dict[str, np.ndarray],
    output: Path,
    frame: int,
) -> None:
    x, y, z = (size // 2 for size in bold.shape[:3])
    frame_image = bold[:, :, :, frame]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for axis, (name, volume) in zip(
        axes,
        [("BOLD", frame_image), ("GM", masks["GM"]), ("WM", masks["WM"]), ("CSF", masks["CSF"])],
    ):
        image = volume[:, :, z].T
        axis.imshow(image, cmap="gray", origin="lower")
        axis.set_title(f"{name}, axial z={z}")
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output / "frame_and_masks.png", dpi=150)
    plt.close(fig)

    tissue_means = {
        name: bold[mask].reshape(-1, bold.shape[3]).mean(axis=0)
        for name, mask in masks.items()
        if mask.any()
    }
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(tissue_means["GM"], label="GM")
    axis.plot(tissue_means["WM"], label="WM")
    axis.plot(tissue_means["CSF"], label="CSF")
    axis.axvline(frame, color="black", linestyle="--", linewidth=1, label="selected frame")
    axis.set(title="Mean BOLD signal by tissue mask", xlabel="Frame", ylabel="Signal")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "tissue_timeseries.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--subject", default="subj01")
    parser.add_argument("--entry", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    entry = _load_entry(args.processed_root, args.subject, args.entry)
    bold, masks = _load_window(entry)
    if args.frame < 0 or args.frame >= bold.shape[3]:
        raise IndexError(f"frame must be between 0 and {bold.shape[3] - 1}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    finite = bool(np.isfinite(bold).all())
    summary = {
        "subject": entry["subject"],
        "run": entry["run"],
        "start": entry["start"],
        "stop": entry["stop"],
        "shape_XYZT": list(bold.shape),
        "voxel_sizes": [float(v) for v in nib.load(entry["bold_path"]).header.get_zooms()[:3]],
        "finite": finite,
        "bold_min": float(bold.min()),
        "bold_max": float(bold.max()),
        "bold_mean": float(bold.mean()),
        "bold_std": float(bold.std()),
        "mask_voxels": {name: int(mask.sum()) for name, mask in masks.items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _save_slices(bold, masks, args.output_dir, args.frame)
    _save_frame_and_timeseries(bold, masks, args.output_dir, args.frame)
    print(json.dumps(summary, indent=2))
    print(f"Saved visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
