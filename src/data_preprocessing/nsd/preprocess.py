"""Prepare NSD BOLD runs and tissue masks without modifying raw data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np


# Map standard FreeSurfer aseg indices to structural tissue classes
GM_LABELS = {
    3, 8, 10, 11, 12, 13, 17, 18, 26, 42, 47, 49, 50, 51, 52, 53, 54, 58,
    77,
}
GM_LABELS.update(range(1001, 1036))
GM_LABELS.update(range(2001, 2036))

WM_LABELS = {2, 7, 16, 41, 46}
WM_LABELS.update(range(251, 256))

CSF_LABELS = {4, 5, 14, 15, 24, 31, 43, 44, 63, 72}


def _paths_for_subject(raw_root: Path, subject: str) -> tuple[Path, Path, Path]:
    """Resolve and validate required raw data paths for a subject."""
    subject_root = raw_root / "ppdata" / subject / "func1pt8mm"
    aseg = subject_root / "aseg.nii.gz"
    brainmask = subject_root / "brainmask.nii.gz"
    timeseries_root = raw_root / "timeseries" / subject / "func1pt8mm" / "timeseries"

    if not aseg.is_file():
        raise FileNotFoundError(f"Missing segmentation: {aseg}")
    if not brainmask.is_file():
        raise FileNotFoundError(f"Missing brain mask: {brainmask}")
    if not timeseries_root.is_dir():
        raise FileNotFoundError(f"Missing timeseries directory: {timeseries_root}")
    
    return aseg, brainmask, timeseries_root


def _validate_geometry(reference: nib.Nifti1Image, image: nib.Nifti1Image, name: str) -> None:
    """Ensure spatial dimensions and affine matrices match the reference exactly."""
    reference_shape = reference.shape[:3]
    image_shape = image.shape[:3]

    if reference_shape != image_shape:
        raise ValueError(
            f"{name} spatial shape {image_shape} does not match reference shape {reference_shape}"
        )
    if not np.allclose(reference.affine, image.affine, atol=1e-4):
        raise ValueError(f"{name} affine does not match the BOLD affine")


def _write_mask(
    path: Path, source: nib.Nifti1Image, labels: Iterable[int], brainmask: np.ndarray
) -> dict:
    """Filter FreeSurfer segmentations into a binary tissue mask and save to disk.
    
    Args:
        path: Output file path.
        source: Loaded FreeSurfer segmentation image (aseg).
        labels: Set of integer labels belonging to the target tissue.
        brainmask: Binary array of the brain area.

    Returns:
        Dictionary containing voxel counts and fractional volume statistics.
    """
    aseg_data = np.rint(source.get_fdata(dtype=np.float32)).astype(np.int16)

    # Isolate target tissue and intersect with brainmask to strip skull/background
    mask = np.isin(aseg_data, list(labels)) & (brainmask > 0)

    if not np.any(mask):
        raise ValueError(f"Generated empty tissue mask: {path.name}")
    
    image = nib.Nifti1Image(mask.astype(np.float32), source.affine, source.header.copy())
    image.set_data_dtype(np.float32)
    nib.save(image, str(path))

    return {"voxels": int(mask.sum()), "fraction": float(mask.mean())}


def preprocess_subject(
    raw_root: Path,
    output_root: Path,
    subject: str,
    window_length: int,
    stride: int,
) -> Path:
    """Generate anatomical masks and a JSON manifest mapping temporal BOLD windows.
    
    Args:
        raw_root: Path to the raw NSD dataset root.
        output_root: Path to save preprocessed masks and manifests.
        subject: Subject identifier (e.g., 'subj01').
        window_length: Number of timeframes per temporal window.
        stride: Step size for the sliding window.
    
    Returns:
        Path to the generated 'manifest.json' file.
    """
    if window_length <= 0 or stride <= 0:
        raise ValueError("window_length and stride must be positive")

    aseg_path, brainmask_path, timeseries_root = _paths_for_subject(raw_root, subject)
    aseg_image = nib.load(str(aseg_path))
    brainmask_image = nib.load(str(brainmask_path))

    # Critical for masked Autoencoders: ensure spatial priors align perfectly
    _validate_geometry(aseg_image, brainmask_image, "brainmask")
    brainmask = brainmask_image.get_fdata(dtype=np.float32)

    subject_output = output_root / subject
    anat_output = subject_output / "anat"
    anat_output.mkdir(parents=True, exist_ok=True)

    mask_stats = {
        "gm": _write_mask(anat_output / "gm.nii.gz", aseg_image, GM_LABELS, brainmask),
        "wm": _write_mask(anat_output / "wm.nii.gz", aseg_image, WM_LABELS, brainmask),
        "csf": _write_mask(anat_output / "csf.nii.gz", aseg_image, CSF_LABELS, brainmask),
    }

    runs = sorted(timeseries_root.glob("timeseries_nsdimagery_run*.nii.gz"))
    if not runs:
        raise FileNotFoundError(f"No NSD imagery runs found in {timeseries_root}")

    entries = []

    def portable_path(path: Path) -> str:
        return os.path.relpath(path.resolve(), output_root.resolve()).replace("\\", "/")

    for run_path in runs:
        bold_image = nib.load(str(run_path))
        if len(bold_image.shape) != 4:
            raise ValueError(f"BOLD run is not 4D: {run_path}")
        
        _validate_geometry(bold_image, aseg_image, "aseg")
        timepoints = int(bold_image.shape[3])

        if timepoints < window_length:
            continue

        # Create overlapping temporal chunks (sliding window) to multiply training samples
        for start in range(0, timepoints - window_length + 1, stride):
            entries.append(
                {
                    "subject": subject,
                    "run": run_path.stem.removesuffix(".nii"),
                    "bold_path": portable_path(run_path),
                    "gm_path": portable_path(anat_output / "gm.nii.gz"),
                    "wm_path": portable_path(anat_output / "wm.nii.gz"),
                    "csf_path": portable_path(anat_output / "csf.nii.gz"),
                    "start": start,
                    "stop": start + window_length,
                    "timepoints": timepoints,
                }
            )

    if not entries:
        raise ValueError("No temporal windows were generated; check window_length")
    
    # The manifest acts as a lightweight map, enabling PyTorch DataLoader
    # to fetch specific time chunks without holding full 7T NIfTI runs in RAM.
    manifest = {
        "schema_version": 1,
        "subject": subject,
        "window_length": window_length,
        "stride": stride,
        "bold_axis_order": "XYZT",
        "spatial_shape": list(aseg_image.shape),
        "voxel_sizes": [float(value) for value in aseg_image.header.get_zooms()[:3]],
        "mask_stats": mask_stats,
        "entries": entries,
    }

    manifest_path = subject_output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--window-length", type=int, default=40)
    parser.add_argument("--stride", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    """CLI execution entry point."""
    args = parse_args()
    manifest = preprocess_subject(
        raw_root=args.raw_root,
        output_root=args.output_root,
        subject=args.subject,
        window_length=args.window_length,
        stride=args.stride,
    )
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
