"""Dataset loaders and parsing utilities for the Natural Scenes Dataset (NSD).

Each sample is a dictionary::

    {"x": Tensor[6, T, X, Y, Z], "tr": Tensor[], "subject": str, "run": str,
     "start": int, "stop": int}

with ``x`` channels ``[dyn, static_anat, gm, wm, csf, brain]`` (see
:mod:`data.channels`):

* ``dyn``: voxelwise percent signal change (PSC) after linear detrending of the
  full run, window-mean removed, exactly ``0`` outside the brain mask;
* ``static_anat``: temporal-mean image of the window, z-scored over brain voxels
  only (``0`` outside);
* ``gm``, ``wm``, ``csf``, ``brain``: static binary masks broadcast over time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .brain_mask import build_brain_mask
    from .channels import NUM_CHANNELS
except ImportError:
    from data.brain_mask import build_brain_mask
    from data.channels import NUM_CHANNELS


def linear_detrend_coefficients(
    run: np.ndarray, chunk: int = 40
) -> tuple[np.ndarray, np.ndarray]:
    """Per-voxel mean and per-frame linear slope of a full run.

    Uses the closed form ``slope = sum_t x_t (t - t_mean) / sum_t (t - t_mean)^2`` so
    ``x_t ~ mean + slope * (t - (N - 1) / 2)``.

    Args:
        run: Array ``[X, Y, Z, N]``.
        chunk: Frames processed at a time (bounds float32 temporary memory).

    Returns:
        ``(mean, slope)``, each ``[X, Y, Z]`` float32.
    """
    n_frames = run.shape[-1]
    t_center = (n_frames - 1) / 2.0
    total = np.zeros(run.shape[:3], dtype=np.float64)
    cross = np.zeros(run.shape[:3], dtype=np.float64)
    for begin in range(0, n_frames, chunk):
        end = min(begin + chunk, n_frames)
        block = np.asarray(run[..., begin:end], dtype=np.float32)
        t_c = (np.arange(begin, end, dtype=np.float32) - t_center)
        total += block.sum(axis=-1, dtype=np.float64)
        cross += (block * t_c).sum(axis=-1, dtype=np.float64)
    denominator = float(((np.arange(n_frames) - t_center) ** 2).sum())
    slope = cross / denominator if denominator > 0 else np.zeros_like(cross)
    return (total / n_frames).astype(np.float32), slope.astype(np.float32)


def compute_psc_window(
    window: np.ndarray,
    brain: np.ndarray,
    baseline: np.ndarray,
    trend: np.ndarray,
    floor_fraction: float = 0.25,
    window_detrend: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a raw window to the dynamic PSC channel and the static anatomy channel.

    Args:
        window: Raw intensities ``[X, Y, Z, T]``.
        brain: Boolean parenchymal mask ``[X, Y, Z]``.
        baseline: Per-voxel PSC baseline (run mean) ``[X, Y, Z]``.
        trend: Per-voxel fitted linear drift evaluated on the window, ``[X, Y, Z, T]``
            (includes the baseline level, i.e. the full fitted line).
        floor_fraction: Baseline floor as a fraction of the brain-median baseline;
            prevents division by near-zero (signal-dropout) voxels.
        window_detrend: Additionally remove the per-voxel linear slope of the window
            (after the supplied ``trend``), i.e. run-level + window-level detrending.

    Returns:
        ``(dyn, static)``: ``dyn`` is ``[X, Y, Z, T]`` PSC with zero temporal mean
        per voxel and ``0`` outside ``brain``; ``static`` is ``[X, Y, Z]`` z-scored
        inside ``brain`` and ``0`` outside.
    """
    if not brain.any():
        raise ValueError("Brain mask is empty.")
    floor = floor_fraction * float(np.median(baseline[brain]))
    denominator = np.maximum(baseline, max(floor, 1e-6))[..., None]
    dyn = 100.0 * (window - trend) / denominator
    dyn = dyn - dyn.mean(axis=-1, keepdims=True)
    if window_detrend:
        t_c = np.arange(dyn.shape[-1], dtype=np.float32) - (dyn.shape[-1] - 1) / 2.0
        slope = (dyn * t_c).sum(axis=-1, keepdims=True) / float((t_c**2).sum())
        dyn = dyn - slope * t_c
    dyn = dyn * brain[..., None]

    mean_image = window.mean(axis=-1)
    inside = mean_image[brain]
    spread = float(inside.std())
    static = np.where(brain, (mean_image - inside.mean()) / max(spread, 1e-8), 0.0)
    return dyn.astype(np.float32), static.astype(np.float32)


class NSDDataset(Dataset):
    """Manifest-backed Natural Scenes Dataset (NSD) loader.

    The NSD preprocessing step writes one ``manifest.json`` per subject. Each
    manifest entry identifies a BOLD run and a temporal window.

    Args:
        nsd_root: Path to the processed NSD dataset root.
        subjects: Optional list of subject IDs to load. Defaults to all.
        time_frames: Expected sequence length. Defaults to 40.
        crop_size: Optional ``(X, Y, Z)`` spatial dimensions for center cropping.
        detrend: ``"run"`` removes the full-run linear trend (default),
            ``"window"`` fits it per window, ``"run+window"`` does both,
            ``"none"`` only removes the window mean.
        tr_override: Repetition time in seconds that replaces the NIfTI header value.
        psc_floor_fraction: Baseline floor for PSC, as a fraction of the brain median.

    Attributes:
        entries (list[dict]): Loaded manifest entries defining temporal windows.
    """

    def __init__(
        self,
        nsd_root: str | Path,
        subjects: Sequence[str] | None = None,
        time_frames: int | None = 40,
        crop_size: tuple[int, int, int] | None = None,
        detrend: str = "run",
        tr_override: float | None = None,
        psc_floor_fraction: float = 0.25,
    ) -> None:
        """Initialize the dataset and parse subject manifests."""
        if detrend not in {"run", "window", "run+window", "none"}:
            raise ValueError(
                f"detrend must be 'run', 'window', 'run+window' or 'none', got {detrend!r}"
            )
        self.nsd_root = Path(nsd_root)
        self.time_frames = time_frames
        self.crop_size = crop_size
        self.detrend = detrend
        self.tr_override = tr_override
        self.psc_floor_fraction = psc_floor_fraction

        if not self.nsd_root.is_dir():
            raise FileNotFoundError(f"NSD processed root does not exist: {self.nsd_root}")

        requested = set(subjects) if subjects is not None else None
        self.entries: list[dict[str, Any]] = []

        # Load JSON manifests to map temporal windows without loading all BOLD data into memory
        manifest_paths = sorted(self.nsd_root.glob("*/manifest.json"))

        if not manifest_paths:
            raise FileNotFoundError(
                f"No subject manifests found under {self.nsd_root}. "
                "Run data_preprocessing.nsd.preprocess first."
            )

        for manifest_path in manifest_paths:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            subject = str(manifest.get("subject", manifest_path.parent.name))

            if requested is not None and subject not in requested:
                continue

            manifest_window = int(manifest["window_length"])

            # Flatten all temporal windows across all runs into a single indexable list
            for entry in manifest["entries"]:
                item = dict(entry)
                item["_manifest_window_length"] = manifest_window
                item["_manifest_path"] = str(manifest_path)
                self.entries.append(item)

        if not self.entries:
            raise ValueError(f"No manifest entries found for subjects={subjects!r}")

        # Static per-subject volumes and per-run trend maps are cached to avoid repeated I/O.
        self._mask_cache: dict[tuple[str, str, str], np.ndarray] = {}
        self._brain_cache: dict[tuple[str, tuple[int, ...]], np.ndarray] = {}
        self._trend_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        """Return the total number of temporal windows across all subjects."""
        return len(self.entries)

    @staticmethod
    def _load_image(path: Path) -> nib.Nifti1Image:
        """Load a NIfTI image from disk."""
        if not path.is_file():
            raise FileNotFoundError(f"Missing NIfTI file referenced by manifest: {path}")
        return nib.load(str(path))

    def _resolve_entry_path(self, entry: dict[str, Any], key: str) -> Path:
        """Resolve a manifest path, supporting portable and legacy manifests."""
        raw_path = Path(str(entry[key]))
        if raw_path.is_absolute() and raw_path.is_file():
            return raw_path
        manifest_path = Path(str(entry["_manifest_path"]))
        candidate = (manifest_path.parent.parent / raw_path).resolve()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"Manifest path for {key!r} could not be resolved: {entry[key]!r}. "
            f"Checked {candidate}."
        )

    @staticmethod
    def _validate_spatial_geometry(
        bold_image: nib.Nifti1Image,
        mask_image: nib.Nifti1Image,
        mask_name: str,
    ) -> None:
        """Ensure the mask matches the spatial dimensions and affine of the BOLD run."""
        # Critical for Masked Autoencoders: misaligned spatial priors will corrupt denoising
        if bold_image.shape[:3] != mask_image.shape[:3]:
            raise ValueError(
                f"{mask_name} shape {mask_image.shape[:3]} does not match "
                f"BOLD shape {bold_image.shape[:3]}"
            )
        if not np.allclose(bold_image.affine, mask_image.affine, atol=1e-4):
            raise ValueError(f"{mask_name} affine does not match the BOLD affine")

    def _load_mask(
        self,
        path: Path,
        bold_image: nib.Nifti1Image,
        mask_name: str,
    ) -> np.ndarray:
        """Load, validate and cache an anatomical mask"""
        cache_key = (str(path), mask_name, str(tuple(bold_image.shape[:3])))

        # Only read from disk if mask not already in RAM
        if cache_key not in self._mask_cache:
            image = self._load_image(path)
            self._validate_spatial_geometry(bold_image, image, mask_name)
            data = image.get_fdata(dtype=np.float32)
            if data.ndim != 3:
                raise ValueError(f"{mask_name} must be 3D, got shape {data.shape}")

            # Binarize mask probabilities (background=0, tissue=1)
            self._mask_cache[cache_key] = (data > 0).astype(np.float32)

        return self._mask_cache[cache_key]

    def _load_brain(
        self,
        entry: dict[str, Any],
        bold_image: nib.Nifti1Image,
        gm: np.ndarray,
        wm: np.ndarray,
        csf: np.ndarray,
    ) -> np.ndarray:
        """Return the parenchymal brain mask as a bool array.

        Uses ``brain_path`` from the manifest when present; otherwise builds the mask
        on the fly from the tissue union (manifests written before ``brain.nii.gz``
        existed).
        """
        if "brain_path" in entry:
            try:
                path = self._resolve_entry_path(entry, "brain_path")
            except FileNotFoundError:
                path = None
        else:
            path = None

        if path is not None:
            return self._load_mask(path, bold_image, "brain mask") > 0

        key = (str(entry["_manifest_path"]), tuple(bold_image.shape[:3]))
        if key not in self._brain_cache:
            self._brain_cache[key] = build_brain_mask(
                (gm + wm + csf) > 0, head_mask=None, closing_iterations=3, connectivity=2
            )
        return self._brain_cache[key]

    def _resolve_tr(self, entry: dict[str, Any], bold_image: nib.Nifti1Image) -> float:
        """Repetition time in seconds: override > manifest entry > NIfTI header."""
        if self.tr_override is not None:
            return float(self.tr_override)
        if "tr" in entry:
            return float(entry["tr"])
        zooms = bold_image.header.get_zooms()
        if len(zooms) < 4 or float(zooms[3]) <= 0:
            raise ValueError("No TR in manifest or NIfTI header; pass tr_override.")
        return round(float(zooms[3]), 6)

    def _run_trend(self, bold_path: Path, bold_image: nib.Nifti1Image):
        """Cached per-voxel ``(mean, slope)`` of the whole run."""
        key = str(bold_path)
        if key not in self._trend_cache:
            run = bold_image.get_fdata(dtype=np.float32, caching="unchanged")
            self._trend_cache[key] = linear_detrend_coefficients(run)
            del run
        return self._trend_cache[key]

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load a single fMRI temporal window and corresponding tissue masks.

        Args:
            index: Manifest entry index.

        Returns:
            A dict with ``x`` of shape ``(6, T, X, Y, Z)`` (``[dyn, static_anat, gm, wm,
            csf, brain]``), scalar ``tr`` (seconds), and ``subject``/``run``/``start``/
            ``stop`` metadata.
        """
        entry = self.entries[index]
        bold_path = self._resolve_entry_path(entry, "bold_path")
        bold_image = self._load_image(bold_path)

        if len(bold_image.shape) != 4:
            raise ValueError(f"BOLD run must be 4D, got shape {bold_image.shape}")

        start = int(entry["start"])
        stop = int(entry["stop"])
        available = int(bold_image.shape[3])

        if start < 0 or stop > available or start >= stop:
            raise IndexError(
                f"Invalid temporal window [{start}, {stop}) for BOLD run with {available} frames"
            )

        # Lazy load only the specific temporal window to save memory
        window = bold_image.slicer[:, :, :, start:stop].get_fdata(dtype=np.float32)

        if self.time_frames is not None and window.shape[3] != self.time_frames:
            raise ValueError(
                f"Manifest window has {window.shape[3]} frames, expected {self.time_frames}"
            )

        if not np.isfinite(window).all():
            raise ValueError(f"BOLD window contains non-finite values: {entry['bold_path']}")

        gm = self._load_mask(self._resolve_entry_path(entry, "gm_path"), bold_image, "GM mask")
        wm = self._load_mask(self._resolve_entry_path(entry, "wm_path"), bold_image, "WM mask")
        csf = self._load_mask(self._resolve_entry_path(entry, "csf_path"), bold_image, "CSF mask")
        brain = self._load_brain(entry, bold_image, gm, wm, csf)
        brain_f = brain.astype(np.float32)
        gm, wm, csf = gm * brain_f, wm * brain_f, csf * brain_f

        # Linear drift: full-run fit when available (default), else per window.
        n_frames = window.shape[3]
        if self.detrend in {"run", "run+window"}:
            run_mean, run_slope = self._run_trend(bold_path, bold_image)
            t_c = np.arange(start, stop, dtype=np.float32) - (available - 1) / 2.0
            trend = run_mean[..., None] + run_slope[..., None] * t_c
            baseline = run_mean
        elif self.detrend == "window":
            t_c = np.arange(n_frames, dtype=np.float32) - (n_frames - 1) / 2.0
            baseline = window.mean(axis=-1)
            slope = (window * t_c).sum(axis=-1) / float((t_c**2).sum())
            trend = baseline[..., None] + slope[..., None] * t_c
        else:
            baseline = window.mean(axis=-1)
            trend = baseline[..., None]

        dyn, static = compute_psc_window(
            window,
            brain,
            baseline,
            trend,
            floor_fraction=self.psc_floor_fraction,
            window_detrend=self.detrend == "run+window",
        )

        # [T, X, Y, Z] / [C, X, Y, Z]
        dyn = np.transpose(dyn, (3, 0, 1, 2))
        anatomy = np.stack([static, gm, wm, csf, brain_f], axis=0)

        if self.crop_size is not None:
            dyn, anatomy = self._center_crop(dyn, anatomy, self.crop_size, brain)

        x = np.empty((NUM_CHANNELS, dyn.shape[0], *dyn.shape[1:]), dtype=np.float32)
        x[0] = dyn
        x[1:] = anatomy[:, None]  # time-invariant channels broadcast over T

        return {
            "x": torch.from_numpy(x),
            "tr": torch.tensor(self._resolve_tr(entry, bold_image), dtype=torch.float32),
            "subject": str(entry["subject"]),
            "run": str(entry["run"]),
            "start": start,
            "stop": stop,
        }

    @staticmethod
    def _center_crop(
        dyn: np.ndarray,
        anatomy: np.ndarray,
        crop_size: tuple[int, int, int],
        brain: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Crop around the brain centroid, keeping every channel aligned.

        Args:
            dyn: Dynamic channel ``(T, X, Y, Z)``.
            anatomy: Static channels ``(C, X, Y, Z)``.
            crop_size: Target spatial dimensions ``(X, Y, Z)``.
            brain: Mask used for the centroid; defaults to the union of the
                tissue channels ``anatomy[1:4]``.

        Returns:
            The cropped ``(dyn, anatomy)``.
        """
        if len(crop_size) != 3 or any(size <= 0 for size in crop_size):
            raise ValueError("crop_size must contain three positive dimensions")

        spatial_shape = np.asarray(anatomy.shape[1:], dtype=int)
        requested = np.asarray(crop_size, dtype=int)

        if np.any(requested > spatial_shape):
            raise ValueError(
                f"crop_size {tuple(requested)} exceeds volume shape "
                f"{tuple(spatial_shape)}"
            )

        # Identify brain matter to crop out empty scanner background
        tissue = brain if brain is not None else (anatomy[1:4].max(axis=0) > 0)
        coordinates = np.argwhere(tissue)

        if coordinates.size:
            center = coordinates.mean(axis=0).round().astype(int)
        else:
            center = spatial_shape // 2

        start = center - requested // 2
        start = np.maximum(start, 0)
        start = np.minimum(start, spatial_shape - requested)
        stop = start + requested
        slices = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))

        return dyn[(slice(None), *slices)], anatomy[(slice(None), *slices)]


if __name__ == "__main__":
    """CLI execution entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="NSD Dataset validator and inspector.")
    parser.add_argument("root", help="Processed NSD root containing subject manifests")
    args = parser.parse_args()

    dataset = NSDDataset(args.root)
    sample = dataset[0]
    print(f"Found {len(dataset)} temporal windows")
    print(f"Example tensor shape: {tuple(sample['x'].shape)}; TR={float(sample['tr'])} s")
