"""Synthetic NSD-like Dataset for testing and continuous integration (CI).

Produces deterministic, reproducible synthetic 4D BOLD volumes and anatomy masks
so the training pipeline can be exercised without disk-based NIfTIs.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
from einops import repeat
from torch.utils.data import Dataset

class SyntheticNSDDataset(Dataset):
    """Generates synthetic fMRI data matching the NSDDataset output format.

    Args:
        n_subjects: Number of synthetic subjects to simulate. Defaults to 8.
        time_frames: Sequence length (T) for the synthetic BOLD run. Defaults to 40.
        shape: Spatial dimensions (X, Y, Z) of the synthetic volume.
        seed: Random seed for reproducible generation.
    """

    def __init__(
        self,
        n_subjects: int = 8,
        time_frames: int = 40,
        shape: tuple[int, int, int] = (32, 32, 16),
        seed: int | None = 1234,
    ) -> None:
        """Initialize the synthetic dataset generators."""
        self.n_subjects = int(n_subjects)
        self.time_frames = int(time_frames)
        self.shape = tuple(shape)

        # Initialize an isolated random state to guarantee CI reproducibility
        seed_value = seed if seed is not None else random.randint(0, 2**31 - 1)
        self.rng = np.random.RandomState(seed_value)

    def __len__(self) -> int:
        """Return the total number of synthetic subjects."""
        return self.n_subjects

    def _generate_bold(self, gm: np.ndarray, brain: np.ndarray) -> np.ndarray:
        """Simulate drift-free PSC: low-frequency GM sinusoid plus Gaussian noise inside the brain."""
        T, X, Y, Z = self.time_frames, *self.shape

        # Low-frequency temporal baseline (approximating hemodynamic fluctuations), ~1 % amplitude
        t = np.arange(T, dtype=np.float32)
        base = np.sin(2 * math.pi * 0.06 * t * (0.8 + 0.4 * self.rng.rand()) + 6 * self.rng.rand())
        dyn = base[:, None, None, None] * gm[None]
        dyn = dyn + 0.5 * self.rng.randn(T, X, Y, Z)
        dyn = dyn - dyn.mean(axis=0, keepdims=True)
        return (dyn * brain[None]).astype(np.float32)

    def _generate_mask(self, kind: str) -> np.ndarray:
        """Generate concentric geometric spheres to emulate tissue segmentations"""
        X, Y, Z = self.shape

        # Compute the Euclidean distance from the center of the volume for every voxel
        grid = np.indices((X, Y, Z)).astype(np.float32)
        cx, cy, cz = X // 2, Y // 2, Z // 2
        dist = np.sqrt((grid[0] - cx) ** 2 + (grid[1] - cy) ** 2 + (grid[2] - cz) ** 2)

        # Assign binary masks based on arbitrary radial thresholds
        if kind == "gm":
            mask = ((dist < min(X, Y) * 0.45) & (dist >= min(X, Y) * 0.30)).astype(np.float32)
        elif kind == "wm":
            mask = ((dist >= min(X, Y) * 0.12) & (dist < min(X, Y) * 0.30)).astype(np.float32)
        else:
            # csf: small central voxels to emulate ventricles
            mask = (dist < min(X, Y) * 0.12).astype(np.float32)

        return mask

    def __getitem__(self, idx: int) -> dict:
        """Generate a single synthetic fMRI window in the real 6-channel format.

        Args:
            idx: Subject index (ignored internally as generation relies on PRNG state).

        Returns:
            ``{"x": [6, T, X, Y, Z], "tr": scalar, "subject", "run", "start", "stop"}`` with
            channels ``[dyn, static_anat, gm, wm, csf, brain]``.
        """
        gm = self._generate_mask("gm")
        wm = self._generate_mask("wm")
        csf = self._generate_mask("csf")
        brain = np.clip(gm + wm + csf, 0, 1)

        dyn = self._generate_bold(gm, brain)
        static = (self.rng.randn(*self.shape) * 0.5 + 2.0 * gm + wm) * brain
        static = np.where(brain > 0, (static - static[brain > 0].mean()) / (static[brain > 0].std() + 1e-8), 0.0)

        anatomy = np.stack([static.astype(np.float32), gm, wm, csf, brain], axis=0)
        anatomy_time = repeat(anatomy, "c x y z -> c t x y z", t=self.time_frames)
        x = torch.cat([torch.from_numpy(dyn).unsqueeze(0), torch.from_numpy(anatomy_time)], dim=0).float()
        return {
            "x": x,
            "tr": torch.tensor(1.0),
            "subject": f"synthetic{idx:02d}",
            "run": "run01",
            "start": 0,
            "stop": self.time_frames,
        }


if __name__ == "__main__":
    """CLI execution entry point for manual testing."""
    ds = SyntheticNSDDataset(n_subjects=4, time_frames=8, shape=(16, 16, 8))
    print(f"Dataset length: {len(ds)}; sample shape {tuple(ds[0]['x'].shape)}")
    print(f"Example tensor shape: {tuple(ds[0].shape)}")