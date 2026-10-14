"""Dataset loaders and parsing utilities for the Natural Scenes Dataset (NSD)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
import torch
from einops import repeat
from torch.utils.data import Dataset

class NSDDataset(Dataset):
    """Manifest-backed Natural Scenes Dataset (NSD) loader.

    The NSD preprocessing step writes one `manifest.json` per subject. Each
    manifest entry identifies a BOLD run and a temporal window.

    Args:
        nsd_root: Path to the processed NSD dataset root.
        subjects: Optional list of subject IDs to load. Defaults to all.
        time_frames: Expected sequence length. Defaults to 40.
        normalize_per_window: If True, applies Z-score normalization per window.
        crop_size: Optional (X, Y, Z) spatial dimensions for center cropping.

    Attributes:
        entries (list[dict]): Loaded manifest entries defining temporal windows.
    """

    def __init__(
        self,
        nsd_root: str | Path,
        subjects: Sequence[str] | None = None,
        time_frames: int | None = 40,
        normalize_per_window: bool = True,
        crop_size: tuple[int, int, int] | None = None,
    ) -> None:
        """Initialize the dataset and parse subject manifests."""
        self.nsd_root = Path(nsd_root)
        self.time_frames = time_frames
        self.normalize_per_window = normalize_per_window
        self.crop_size = crop_size

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

        # Cache static structural masks to avoid repeated disk I/O bottlenecks
        self._mask_cache: dict[tuple[str, str, str], np.ndarray] = {}

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

    def __getitem__(self, index: int) -> torch.Tensor:
        """Load a single fMRI temporal window and corresponding tissue masks.

        Args:
            index: Manifest entry index.

        Returns:
            A tensor of shape `(4, T, X, Y, Z)` where channels correspond to 
            normalized BOLD, GM, WM, and CSF masks.
        """
        entry = self.entries[index]
        bold_image = self._load_image(self._resolve_entry_path(entry, "bold_path"))

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
        bold_data = bold_image.slicer[:, :, :, start:stop].get_fdata(dtype=np.float32)

        if self.time_frames is not None and bold_data.shape[3] != self.time_frames:
            raise ValueError(
                f"Manifest window has {bold_data.shape[3]} frames, expected {self.time_frames}"
            )
        
        if not np.isfinite(bold_data).all():
            raise ValueError(f"BOLD window contains non-finite values: {entry['bold_path']}")

        # Standardize features for stable transformer gradients
        if self.normalize_per_window:
            mean = float(bold_data.mean())
            std = float(bold_data.std())
            bold_data = (bold_data - mean) / max(std, 1e-8)

        # Fetch tissue masks (hits cache after the first run)
        gm = self._load_mask(self._resolve_entry_path(entry, "gm_path"), bold_image, "GM mask")
        wm = self._load_mask(self._resolve_entry_path(entry, "wm_path"), bold_image, "WM mask")
        csf = self._load_mask(self._resolve_entry_path(entry, "csf_path"), bold_image, "CSF mask")

        # Convert to [Time, X, Y, Z]
        bold_time_first = np.transpose(bold_data, (3, 0, 1, 2))
        masks = np.stack([gm, wm, csf], axis=0)

        if self.crop_size is not None:
            bold_time_first, masks = self._center_crop(
                bold_time_first, masks, self.crop_size
            )

        # Broadcast static 3D masks to match sequence length T
        masks_time = repeat(masks, "c x y z -> c t x y z", t=bold_time_first.shape[0])

        # Final concatenation: [BOLD, GM, WM, CSF] -> shape (4, T, X, Y, Z)
        return torch.from_numpy(
            np.concatenate([bold_time_first[None, ...], masks_time], axis=0).astype(np.float32)
        )

    @staticmethod
    def _center_crop(
        bold: np.ndarray,
        masks: np.ndarray,
        crop_size: tuple[int, int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Crop around the union of tissue masks, padding when needed.
        
        Args:
            bold: BOLD sequence array of shape (T, X, Y, Z).
            masks: 4D stacked tissue masks array of shape (C, X, Y, Z).
            crop_size: Target spatial dimensions (X, Y, Z).
        
        Returns:
            A tuple containing the cropped BOLD array and masks array.
        
        """
        if len(crop_size) != 3 or any(size <= 0 for size in crop_size):
            raise ValueError("crop_size must contain three positive dimensions")

        spatial_shape = np.asarray(masks.shape[1:], dtype=int)
        requested = np.asarray(crop_size, dtype=int)

        if np.any(requested > spatial_shape):
            raise ValueError(
                f"crop_size {tuple(requested)} exceeds volume shape "
                f"{tuple(spatial_shape)}"
            )

        # Identify brain matter to crop out empty scanner background
        # This drastically reduces sequence length for self-attention layers
        tissue = masks.max(axis=0) > 0
        coordinates = np.argwhere(tissue)

        if coordinates.size:
            center = coordinates.mean(axis=0).round().astype(int)
        else:
            center = spatial_shape // 2

        # Calculate bounding box coordinates
        start = center - requested // 2
        start = np.maximum(start, 0)
        start = np.minimum(start, spatial_shape - requested)
        stop = start + requested
        slices = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))

        return bold[(slice(None), *slices)], masks[(slice(None), *slices)]


if __name__ == "__main__":
    """CLI execution entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="NSD Dataset validator and inspector.")
    parser.add_argument("root", help="Processed NSD root containing subject manifests")
    args = parser.parse_args()

    dataset = NSDDataset(args.root)
    print(f"Found {len(dataset)} temporal windows")
    print(f"Example tensor shape: {tuple(dataset[0].shape)}")