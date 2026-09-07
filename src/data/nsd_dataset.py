"""Manifest-backed Natural Scenes Dataset loader.

The NSD preprocessing step writes one ``manifest.json`` per subject. Each
manifest entry identifies a BOLD run and a temporal window. This loader
returns one window as ``[4, T, X, Y, Z]``:

    channel 0: normalized BOLD
    channel 1: GM mask
    channel 2: WM mask
    channel 3: CSF mask
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import nibabel as nib
import numpy as np
import torch
from einops import repeat
from torch.utils.data import Dataset


class NSDDataset(Dataset):
    """Load temporal windows described by NSD preprocessing manifests."""

    def __init__(
        self,
        nsd_root: str | Path,
        subjects: Optional[Sequence[str]] = None,
        time_frames: Optional[int] = 40,
        normalize_per_window: bool = True,
        crop_size: Optional[tuple[int, int, int]] = None,
    ) -> None:
        self.nsd_root = Path(nsd_root)
        self.time_frames = time_frames
        self.normalize_per_window = normalize_per_window
        self.crop_size = crop_size
        if not self.nsd_root.is_dir():
            raise FileNotFoundError(f"NSD processed root does not exist: {self.nsd_root}")

        requested = set(subjects) if subjects is not None else None
        self.entries: list[dict[str, Any]] = []
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
            for entry in manifest["entries"]:
                item = dict(entry)
                item["_manifest_window_length"] = manifest_window
                item["_manifest_path"] = str(manifest_path)
                self.entries.append(item)

        if not self.entries:
            raise ValueError(f"No manifest entries found for subjects={subjects!r}")

        self._mask_cache: dict[tuple[str, str, str], np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def _load_image(path: Path) -> nib.Nifti1Image:
        if not path.is_file():
            raise FileNotFoundError(f"Missing NIfTI file referenced by manifest: {path}")
        return nib.load(str(path))

    @staticmethod
    def _validate_spatial_geometry(
        bold_image: nib.Nifti1Image,
        mask_image: nib.Nifti1Image,
        mask_name: str,
    ) -> None:
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
        cache_key = (str(path), mask_name, str(tuple(bold_image.shape[:3])))
        if cache_key not in self._mask_cache:
            image = self._load_image(path)
            self._validate_spatial_geometry(bold_image, image, mask_name)
            data = image.get_fdata(dtype=np.float32)
            if data.ndim != 3:
                raise ValueError(f"{mask_name} must be 3D, got shape {data.shape}")
            self._mask_cache[cache_key] = (data > 0).astype(np.float32)
        return self._mask_cache[cache_key]

    def __getitem__(self, index: int) -> torch.Tensor:
        entry = self.entries[index]
        bold_image = self._load_image(Path(entry["bold_path"]))
        if len(bold_image.shape) != 4:
            raise ValueError(f"BOLD run must be 4D, got shape {bold_image.shape}")

        start = int(entry["start"])
        stop = int(entry["stop"])
        available = int(bold_image.shape[3])
        if start < 0 or stop > available or start >= stop:
            raise IndexError(
                f"Invalid temporal window [{start}, {stop}) for BOLD run with {available} frames"
            )

        bold_data = bold_image.slicer[:, :, :, start:stop].get_fdata(dtype=np.float32)
        if self.time_frames is not None and bold_data.shape[3] != self.time_frames:
            raise ValueError(
                f"Manifest window has {bold_data.shape[3]} frames, expected {self.time_frames}"
            )
        if not np.isfinite(bold_data).all():
            raise ValueError(f"BOLD window contains non-finite values: {entry['bold_path']}")

        if self.normalize_per_window:
            mean = float(bold_data.mean())
            std = float(bold_data.std())
            bold_data = (bold_data - mean) / max(std, 1e-8)

        gm = self._load_mask(Path(entry["gm_path"]), bold_image, "GM mask")
        wm = self._load_mask(Path(entry["wm_path"]), bold_image, "WM mask")
        csf = self._load_mask(Path(entry["csf_path"]), bold_image, "CSF mask")

        bold_time_first = np.transpose(bold_data, (3, 0, 1, 2))
        masks = np.stack([gm, wm, csf], axis=0)
        if self.crop_size is not None:
            bold_time_first, masks = self._center_crop(
                bold_time_first, masks, self.crop_size
            )
        masks_time = repeat(masks, "c x y z -> c t x y z", t=bold_time_first.shape[0])
        return torch.from_numpy(
            np.concatenate([bold_time_first[None, ...], masks_time], axis=0).astype(np.float32)
        )

    @staticmethod
    def _center_crop(
        bold: np.ndarray,
        masks: np.ndarray,
        crop_size: tuple[int, int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Crop around the union of tissue masks, padding when needed."""
        if len(crop_size) != 3 or any(size <= 0 for size in crop_size):
            raise ValueError("crop_size must contain three positive dimensions")

        spatial_shape = np.asarray(masks.shape[1:], dtype=int)
        requested = np.asarray(crop_size, dtype=int)
        if np.any(requested > spatial_shape):
            raise ValueError(
                f"crop_size {tuple(requested)} exceeds volume shape "
                f"{tuple(spatial_shape)}"
            )

        tissue = masks.max(axis=0) > 0
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
        return bold[(slice(None), *slices)], masks[(slice(None), *slices)]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Processed NSD root containing subject manifests")
    args = parser.parse_args()
    dataset = NSDDataset(args.root)
    print(f"Found {len(dataset)} temporal windows")
    print(f"Example tensor shape: {tuple(dataset[0].shape)}")
