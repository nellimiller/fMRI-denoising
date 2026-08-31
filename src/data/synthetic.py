"""
Synthetic NSD-like Dataset for testing and CI.

Produces deterministic, reproducible synthetic 4D BOLD volumes and anatomy masks
so the training pipeline can be exercised without disk-based NIfTIs.

Output format matches NSDDataset: torch.Tensor shape [4, T, X, Y, Z]
Channels: [BOLD, GM, WM, CSF]
"""
from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from einops import repeat


class SyntheticNSDDataset(Dataset):
    def __init__(
        self,
        n_subjects: int = 8,
        time_frames: int = 40,
        shape: tuple[int, int, int] = (32, 32, 16),
        seed: Optional[int] = 1234,
    ):
        self.n_subjects = int(n_subjects)
        self.time_frames = int(time_frames)
        self.shape = tuple(shape)
        self.rng = np.random.RandomState(seed if seed is not None else random.randint(0, 2**31 - 1))

    def __len__(self):
        return self.n_subjects

    def _generate_bold(self):
        T, X, Y, Z = self.time_frames, *self.shape
        # Low-frequency BOLD-like signals with gaussian noise
        t = np.linspace(0, 2 * math.pi, T)
        base = np.sin(t * (1 + self.rng.rand()))[..., None, None, None]
        spatial = self.rng.randn(X, Y, Z) * 0.1
        bold = base + spatial
        bold = bold + 0.05 * self.rng.randn(T, X, Y, Z)
        # Global z-score
        bold = (bold - bold.mean()) / (bold.std() + 1e-8)
        return bold.astype(np.float32)

    def _generate_mask(self, kind: str):
        X, Y, Z = self.shape
        # Simple geometric masks to emulate GM/WM/CSF
        grid = np.indices((X, Y, Z)).astype(np.float32)
        cx, cy, cz = X // 2, Y // 2, Z // 2
        dist = np.sqrt((grid[0] - cx) ** 2 + (grid[1] - cy) ** 2 + (grid[2] - cz) ** 2)
        if kind == "gm":
            mask = (dist < min(X, Y) * 0.45).astype(np.float32)
        elif kind == "wm":
            mask = ((dist >= min(X, Y) * 0.45) & (dist < min(X, Y) * 0.6)).astype(np.float32)
        else:
            # csf: small central voxels (ventricle-like)
            mask = (dist < min(X, Y) * 0.12).astype(np.float32)
        return mask

    def __getitem__(self, idx: int):
        # ignore idx for deterministic per-subject variation
        bold = self._generate_bold()
        gm = self._generate_mask("gm")
        wm = self._generate_mask("wm")
        csf = self._generate_mask("csf")

        # duplicate masks across time: [3, T, X, Y, Z]
        masks = np.stack([gm, wm, csf], axis=0)
        masks_time = repeat(masks, "c x y z -> c t x y z", t=self.time_frames)

        bold_tensor = torch.from_numpy(bold).unsqueeze(0)  # [1, T, X, Y, Z]
        masks_tensor = torch.from_numpy(masks_time)
        final = torch.cat([bold_tensor, masks_tensor], dim=0).float()
        return final


if __name__ == "__main__":
    ds = SyntheticNSDDataset(n_subjects=4, time_frames=8, shape=(16, 16, 8))
    print(len(ds), ds[0].shape)
