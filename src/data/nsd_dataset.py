"""
NSD Dataset loader

This Dataset expects a directory with subject subfolders. Each subject folder should contain:
- one or more 4D BOLD NIfTI files (e.g. *_bold.nii.gz or bold.nii.gz)
- corresponding anatomical masks (gm.nii.gz, wm.nii.gz, csf.nii.gz) located either in the same subject folder or in an `anat/` subfolder

Return format (torch.Tensor):
- final_tensor: shape [4, T, X, Y, Z]
  channel 0: BOLD (T, X, Y, Z)
  channel 1: GM mask (T, X, Y, Z)
  channel 2: WM mask (T, X, Y, Z)
  channel 3: CSF mask (T, X, Y, Z)

The loader will z-score normalize the BOLD volume globally, pad/truncate to `time_frames` and duplicate static masks across time.

Usage:
    from data.nsd_dataset import NSDDataset
    ds = NSDDataset('/path/to/nsd_root', time_frames=40)

"""
from __future__ import annotations

import os
import fnmatch
from typing import List, Optional, Tuple

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from einops import repeat


def _find_file(folder: str, patterns: List[str]) -> Optional[str]:
    for root, _, files in os.walk(folder):
        for p in patterns:
            for f in fnmatch.filter(files, p):
                return os.path.join(root, f)
    return None


class NSDDataset(Dataset):
    def __init__(
        self,
        nsd_root: str,
        subjects: Optional[List[str]] = None,
        time_frames: int = 40,
        bold_patterns: Optional[List[str]] = None,
        gm_name: str = "gm.nii.gz",
        wm_name: str = "wm.nii.gz",
        csf_name: str = "csf.nii.gz",
    ):
        """Create an NSD-style dataset.

        Args:
            nsd_root: root folder containing subject subfolders
            subjects: optional list of subject folder names to include. If None, all directories are used.
            time_frames: enforced temporal length (pads or truncates to this length)
            bold_patterns: list of glob patterns to match bold files (default tries common names)
            gm_name/wm_name/csf_name: expected mask filenames (searched recursively under the subject folder)
        """
        self.nsd_root = nsd_root
        self.time_frames = int(time_frames)

        if bold_patterns is None:
            self.bold_patterns = ["*bold*.nii.gz", "*bold*.nii", "*BOLD*.nii.gz"]
        else:
            self.bold_patterns = bold_patterns

        if subjects is None:
            # list directories only
            self.subjects = [d for d in os.listdir(nsd_root) if os.path.isdir(os.path.join(nsd_root, d))]
            self.subjects.sort()
        else:
            self.subjects = subjects

        self.gm_name = gm_name
        self.wm_name = wm_name
        self.csf_name = csf_name

        if len(self.subjects) == 0:
            raise ValueError(f"No subject folders found in {nsd_root}")

    def __len__(self) -> int:
        return len(self.subjects)

    def _load_nifti(self, path: str) -> np.ndarray:
        img = nib.load(path)
        data = img.get_fdata(dtype=np.float32)
        return data

    def __getitem__(self, idx: int) -> torch.Tensor:
        subject_folder = os.path.join(self.nsd_root, self.subjects[idx])

        # 1) Find a bold file
        bold_path = _find_file(subject_folder, self.bold_patterns)
        if bold_path is None:
            raise FileNotFoundError(f"No BOLD file found under subject {subject_folder} with patterns {self.bold_patterns}")

        bold_data = self._load_nifti(bold_path)  # shape [X, Y, Z, T] or [T, X, Y, Z]

        # Normalize orientation: ensure time is axis 3 if nibabel returned (X,Y,Z,T)
        if bold_data.ndim == 4 and bold_data.shape[3] >= 2:
            # assume shape [X, Y, Z, T]
            bold_data = np.transpose(bold_data, (3, 0, 1, 2))
        elif bold_data.ndim != 4:
            raise ValueError(f"Unsupported BOLD nifti shape: {bold_data.shape}")

        # Global z-score normalize the entire 4D volume
        mean = np.mean(bold_data)
        std = np.std(bold_data) + 1e-8
        bold_data = (bold_data - mean) / std

        # Enforce temporal length
        if bold_data.shape[0] >= self.time_frames:
            bold_data = bold_data[: self.time_frames]
        else:
            pad_len = self.time_frames - bold_data.shape[0]
            bold_data = np.pad(bold_data, ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode="constant", constant_values=0.0)

        # 2) Find anatomical masks (try anat/ first then anywhere under subject)
        gm_path = _find_file(subject_folder, [self.gm_name])
        wm_path = _find_file(subject_folder, [self.wm_name])
        csf_path = _find_file(subject_folder, [self.csf_name])

        if gm_path is None or wm_path is None or csf_path is None:
            # if any mask missing, still allow but create empty masks with zeros and warn
            missing = [name for name, p in [("GM", gm_path), ("WM", wm_path), ("CSF", csf_path)] if p is None]
            print(f"Warning: missing masks for subject {self.subjects[idx]}: {missing}. Using zeros for missing masks.")

        def _maybe_load_mask(p, exemplar_shape: Tuple[int, int, int]):
            if p is None:
                return np.zeros(exemplar_shape, dtype=np.float32)
            data = np.asarray(nib.load(p).get_fdata(dtype=np.float32))
            # if mask has different orientation, attempt to reshape or crop/pad
            if data.shape != exemplar_shape:
                # try transpose if time dimension present
                if data.ndim == 4:
                    data = np.squeeze(data)
                # naive crop/pad to exemplar
                out = np.zeros(exemplar_shape, dtype=np.float32)
                minx = min(out.shape[0], data.shape[0])
                miny = min(out.shape[1], data.shape[1])
                minz = min(out.shape[2], data.shape[2])
                out[:minx, :miny, :minz] = data[:minx, :miny, :minz]
                return out
            return data

        exemplar_shape = bold_data.shape[1:4]
        gm_data = _maybe_load_mask(gm_path, exemplar_shape)
        wm_data = _maybe_load_mask(wm_path, exemplar_shape)
        csf_data = _maybe_load_mask(csf_path, exemplar_shape)

        # Stack and duplicate masks across time: [3, T, X, Y, Z]
        masks_stacked = np.stack([gm_data, wm_data, csf_data], axis=0)
        masks_time = repeat(masks_stacked, "c x y z -> c t x y z", t=self.time_frames)

        # BOLD to tensor [1, T, X, Y, Z]
        bold_tensor = torch.from_numpy(bold_data).unsqueeze(0)
        masks_tensor = torch.from_numpy(masks_time)

        final_tensor = torch.cat([bold_tensor, masks_tensor], dim=0)
        # final_tensor shape: [4, T, X, Y, Z]

        return final_tensor


if __name__ == "__main__":
    # simple smoke test when run directly
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="NSD root folder containing subject subfolders")
    args = parser.parse_args()

    ds = NSDDataset(args.root, time_frames=40)
    print(f"Found {len(ds)} subjects. Example element shapes: {ds[0].shape}")