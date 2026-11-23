"""Tissue helpers: eroded masks, soft GM support and visible-voxel nuisance regressors.

The physiological noise subspace is spanned by unit-variance regressors ``r_j(t)``
estimated only from *visible* voxels of the 1-voxel-eroded WM and CSF masks:

* ``r_1`` eroded-WM mean, ``r_2`` eroded-CSF mean,
* ``r_3 ... r_{2+K}`` the top ``K`` aCompCor principal components of the
  variance-normalised eroded WM+CSF time courses.

Visible-only estimation keeps hidden voxels' own measurements out of the regressors
(J-invariance); the regressors carry no gradient.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _cross_kernel(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """6-neighbourhood structuring element (matches ``scipy.ndimage.binary_erosion`` default)."""
    kernel = torch.zeros(1, 1, 3, 3, 3, device=device, dtype=dtype)
    kernel[0, 0, 1, 1, 1] = 1
    for axis in (2, 3, 4):
        for offset in (0, 2):
            index = [0, 0, 1, 1, 1]
            index[axis] = offset
            kernel[tuple(index)] = 1
    return kernel


@torch.no_grad()
def erode_mask(mask: torch.Tensor, iterations: int = 1) -> torch.Tensor:
    """Binary erosion with the 6-neighbourhood; the volume border counts as background.

    Args:
        mask: ``(B, 1, X, Y, Z)`` bool or 0/1 float.

    Returns:
        Bool tensor of the same shape.
    """
    current = (mask > 0).float()
    kernel = _cross_kernel(current.device, current.dtype)
    for _ in range(iterations):
        current = (F.conv3d(current, kernel, padding=1) > 6.5).float()
    return current > 0


@torch.no_grad()
def soft_support(mask: torch.Tensor) -> torch.Tensor:
    """Soft partial-volume support in ``[0, 1]``: separable ``[1, 2, 1] / 4`` smoothing of a binary mask."""
    base = torch.tensor([1.0, 2.0, 1.0], device=mask.device, dtype=torch.float32) / 4.0
    kernel = base[:, None, None] * base[None, :, None] * base[None, None, :]
    return F.conv3d((mask > 0).float(), kernel[None, None], padding=1).clamp(0.0, 1.0)


@torch.no_grad()
def nuisance_regressors(
    dyn: torch.Tensor,
    wm: torch.Tensor,
    csf: torch.Tensor,
    visible: torch.Tensor,
    n_comp: int = 3,
    erosion_iterations: int = 1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Unit-variance nuisance regressors from visible, eroded WM/CSF voxels.

    Args:
        dyn: Dynamic PSC channel ``(B, 1, T, X, Y, Z)``.
        wm, csf: Masks ``(B, 1, X, Y, Z)``.
        visible: Bool ``(B, 1, X, Y, Z)``; voxels whose tube is not hidden.
        n_comp: Number of aCompCor components.
        erosion_iterations: Erosion applied to WM and CSF before use.

    Returns:
        ``(B, 2 + n_comp, T)``; each row is zero-mean with unit variance, or all
        zeros when its source region has no visible voxels / no variance.
    """
    B, _, T = dyn.shape[:3]
    data = dyn.float()
    wm_e = erode_mask(wm, erosion_iterations) & visible
    csf_e = erode_mask(csf, erosion_iterations) & visible
    union = (wm_e | csf_e).float()  # (B, 1, X, Y, Z)

    def masked_mean(region: torch.Tensor) -> torch.Tensor:
        weight = region.float()
        count = weight.sum(dim=(1, 2, 3, 4)).clamp_min(1.0)  # (B,)
        return (data * weight.unsqueeze(2)).sum(dim=(1, 3, 4, 5)) / count[:, None]  # (B, T)

    traces = [masked_mean(wm_e), masked_mean(csf_e)]

    if n_comp > 0:
        flat = data[:, 0].reshape(B, T, -1)  # (B, T, V)
        weight = union.reshape(B, 1, -1)  # (B, 1, V)
        # Per-voxel temporal centering and variance normalisation (aCompCor).
        mean = flat.mean(dim=1, keepdim=True)  # (B, 1, V) temporal mean of each voxel
        centered = (flat - mean) * weight
        variance = centered.square().sum(dim=1, keepdim=True) / T  # (B, 1, V)
        scale = torch.where(variance > eps, variance.sqrt().reciprocal(), torch.zeros_like(variance))
        normalised = centered * scale  # voxels without variance drop out
        gram = (normalised @ normalised.transpose(1, 2)).double()  # (B, T, T)
        values, vectors = torch.linalg.eigh(gram)  # ascending
        values, vectors = values.float(), vectors.float()
        k = min(n_comp, T - 1)
        top_values = values[:, -k:].flip(-1)  # (B, k)
        top_vectors = vectors[:, :, -k:].flip(-1)  # (B, T, k)
        components = top_vectors * top_values.clamp_min(0).sqrt().unsqueeze(1)
        for index in range(k):
            traces.append(components[:, :, index])
        for _ in range(n_comp - k):
            traces.append(torch.zeros_like(traces[0]))

    stacked = torch.stack(traces, dim=1)  # (B, J, T)
    stacked = stacked - stacked.mean(-1, keepdim=True)
    std = stacked.square().mean(-1, keepdim=True).sqrt()
    return torch.where(std > eps, stacked / std.clamp_min(eps), torch.zeros_like(stacked))
