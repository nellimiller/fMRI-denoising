"""Prepare NSD BOLD runs and tissue masks without modifying raw data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np

try:
    from ...data.brain_mask import build_brain_mask
except ImportError:
    from data.brain_mask import build_brain_mask


# Map standard FreeSurfer aseg indices to structural tissue classes.
# Label 77 (WM-hypointensities) is white matter pathology, not grey matter, so it
# lives in WM_LABELS.
GM_LABELS = {
    3, 8, 10, 11, 12, 13, 17, 18, 26, 42, 47, 49, 50, 51, 52, 53, 54, 58,
}
GM_LABELS.update(range(1001, 1036))
GM_LABELS.update(range(2001, 2036))

WM_LABELS = {2, 7, 16, 41, 46, 77}
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


def _tissue_mask(source: nib.Nifti1Image, labels: Iterable[int], brainmask: np.ndarray) -> np.ndarray:
    """Return the boolean tissue mask for ``labels`` restricted to the head mask."""
    aseg_data = np.rint(source.get_fdata(dtype=np.float32)).astype(np.int16)

    # Isolate target tissue and intersect with the NSD head mask to strip background
    return np.isin(aseg_data, list(labels)) & (brainmask > 0)


def _save_mask(path: Path, source: nib.Nifti1Image, mask: np.ndarray) -> dict:
    """Save a boolean mask as float32 NIfTI and return voxel-count statistics."""
    if not np.any(mask):
        raise ValueError(f"Generated empty mask: {path.name}")

    image = nib.Nifti1Image(mask.astype(np.float32), source.affine, source.header.copy())
    image.set_data_dtype(np.float32)
    nib.save(image, str(path))

    return {"voxels": int(mask.sum()), "fraction": float(mask.mean())}


def _write_mask(
    path: Path, source: nib.Nifti1Image, labels: Iterable[int], brainmask: np.ndarray
) -> dict:
    """Filter FreeSurfer segmentations into a binary tissue mask and save to disk.

    Args:
        path: Output file path.
        source: Loaded FreeSurfer segmentation image (aseg).
        labels: Set of integer labels belonging to the target tissue.
        brainmask: Whole-head mask (NSD ``brainmask.nii.gz``).

    Returns:
        Dictionary containing voxel counts and fractional volume statistics.
    """
    return _save_mask(path, source, _tissue_mask(source, labels, brainmask))


def read_tr(image: nib.Nifti1Image) -> tuple[float, str]:
    """Read the repetition time from a 4D NIfTI header (``pixdim[4]``).

    Returns ``(tr, provenance)``. NSD time series carry ``xyzt_units = unknown`` and
    no slice timing, so the header value is reported as read and must be
    cross-checked against the NSD data manual (or overridden with ``--tr``).
    """
    zooms = image.header.get_zooms()
    if len(zooms) < 4 or not np.isfinite(zooms[3]) or float(zooms[3]) <= 0:
        raise ValueError("BOLD header has no positive pixdim[4]; pass an explicit TR.")
    units = image.header.get_xyzt_units()[1]
    return round(float(zooms[3]), 6), f"nifti_pixdim4(time_units={units})"


def preprocess_subject(
    raw_root: Path,
    output_root: Path,
    subject: str,
    window_length: int,
    stride: int,
    tr_override: float | None = None,
) -> Path:
    """Generate anatomical masks and a JSON manifest mapping temporal BOLD windows.

    Writes ``anat/{gm,wm,csf,brain}.nii.gz``. ``brain`` is the parenchymal mask built
    by :func:`data.brain_mask.build_brain_mask` (NSD's own ``brainmask.nii.gz`` is a
    whole-head mask and is only used to bound it).

    Args:
        raw_root: Path to the raw NSD dataset root.
        output_root: Path to save preprocessed masks and manifests.
        subject: Subject identifier (e.g., 'subj01').
        window_length: Number of timeframes per temporal window.
        stride: Step size for the sliding window.
        tr_override: Repetition time in seconds; replaces the NIfTI header value.

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

    gm_mask = _tissue_mask(aseg_image, GM_LABELS, brainmask)
    wm_mask = _tissue_mask(aseg_image, WM_LABELS, brainmask)
    csf_mask = _tissue_mask(aseg_image, CSF_LABELS, brainmask)
    brain_mask = build_brain_mask(
        gm_mask | wm_mask | csf_mask,
        head_mask=brainmask,
        closing_iterations=3,
        connectivity=2,
    )
    # Tissue classes are subsets of the parenchyma (largest-component pruning may drop islands).
    gm_mask, wm_mask, csf_mask = (m & brain_mask for m in (gm_mask, wm_mask, csf_mask))

    mask_stats = {
        "gm": _save_mask(anat_output / "gm.nii.gz", aseg_image, gm_mask),
        "wm": _save_mask(anat_output / "wm.nii.gz", aseg_image, wm_mask),
        "csf": _save_mask(anat_output / "csf.nii.gz", aseg_image, csf_mask),
        "brain": _save_mask(anat_output / "brain.nii.gz", aseg_image, brain_mask),
        "head": {"voxels": int((brainmask > 0).sum()), "fraction": float((brainmask > 0).mean())},
    }

    runs = sorted(timeseries_root.glob("timeseries_nsdimagery_run*.nii.gz"))
    if not runs:
        raise FileNotFoundError(f"No NSD imagery runs found in {timeseries_root}")

    entries = []
    tr_values: list[float] = []

    def portable_path(path: Path) -> str:
        return os.path.relpath(path.resolve(), output_root.resolve()).replace("\\", "/")

    for run_path in runs:
        bold_image = nib.load(str(run_path))
        if len(bold_image.shape) != 4:
            raise ValueError(f"BOLD run is not 4D: {run_path}")
        
        _validate_geometry(bold_image, aseg_image, "aseg")
        timepoints = int(bold_image.shape[3])
        if tr_override is not None:
            tr, tr_source = float(tr_override), "cli_override"
        else:
            tr, tr_source = read_tr(bold_image)
        tr_values.append(tr)

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
                    "brain_path": portable_path(anat_output / "brain.nii.gz"),
                    "tr": tr,
                    "tr_source": tr_source,
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
        "schema_version": 2,
        "subject": subject,
        "tr": float(np.median(tr_values)),
        "tr_consistent_across_runs": bool(np.allclose(tr_values, tr_values[0])),
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
    parser.add_argument(
        "--tr",
        type=float,
        default=None,
        help="Repetition time in seconds; overrides the NIfTI header (pixdim[4]).",
    )
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
        tr_override=args.tr,
    )
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
