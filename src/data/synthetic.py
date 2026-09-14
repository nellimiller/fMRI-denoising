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

    def _generate_bold(self) -> np.ndarray:
        """Simulate a 4D BOLD signal using low-frequency sinusoids and Gaussian noise."""
        T, X, Y, Z = self.time_frames, *self.shape

        # Create a low-frequency temporal baseline (approximating hemodynamic fluctuations)
        t = np.linspace(0, 2 * math.pi, T)
        base = np.sin(t * (1 + self.rng.rand()))[..., None, None, None]
        
        # Add static spatial noise and dynamic spatio-temporal noise
        spatial = self.rng.randn(X, Y, Z) * 0.1
        bold = base + spatial
        bold = bold + 0.05 * self.rng.randn(T, X, Y, Z)
        
        # Global Z-Score normalization to match the real dataset pipeline
        bold = (bold - bold.mean()) / (bold.std() + 1e-8)
        return bold.astype(np.float32)

    def _generate_mask(self, kind: str) -> np.ndarray:
        """Generate concentric geometric spheres to emulate tissue segmentations"""
        X, Y, Z = self.shape

        # Compute the Euclidean distance from the center of the volume for every voxel
        grid = np.indices((X, Y, Z)).astype(np.float32)
        cx, cy, cz = X // 2, Y // 2, Z // 2
        dist = np.sqrt((grid[0] - cx) ** 2 + (grid[1] - cy) ** 2 + (grid[2] - cz) ** 2)
        
        # Assign binary masks based on arbitrary radial thresholds
        if kind == "gm":
            mask = (dist < min(X, Y) * 0.45).astype(np.float32)
        elif kind == "wm":
            mask = ((dist >= min(X, Y) * 0.45) & (dist < min(X, Y) * 0.6)).astype(np.float32)
        else:
            # csf: small central voxels to emulate ventricles
            mask = (dist < min(X, Y) * 0.12).astype(np.float32)
            
        return mask

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Generate a signle synthetic fMRI temporal window and corresponding tissue masks.
        
        Args:
            idx: Subject index (ignored internally as generation relies on PRNG state).
            
        Retrurns:
            A tensor of shape '(4, T, X, Y, Z)' where channels correspond to
            normalized BOLD, GM, WM, and CSF masks.
        """
        bold = self._generate_bold()
        gm = self._generate_mask("gm")
        wm = self._generate_mask("wm")
        csf = self._generate_mask("csf")

        # Broadcast static 3D anatomical masks across the time dimension T
        masks = np.stack([gm, wm, csf], axis=0)
        masks_time = repeat(masks, "c x y z -> c t x y z", t=self.time_frames)

        # Concatenate into the final [4, T, X, Y, Z] tensor
        bold_tensor = torch.from_numpy(bold).unsqueeze(0)
        masks_tensor = torch.from_numpy(masks_time)

        return torch.cat([bold_tensor, masks_tensor], dim=0).float()


if __name__ == "__main__":
    """CLI execution entry point for manual testing."""
    ds = SyntheticNSDDataset(n_subjects=4, time_frames=8, shape=(16, 16, 8))
    print(f"Dataset length: {len(ds)}")
    print(f"Example tensor shape: {tuple(ds[0].shape)}")