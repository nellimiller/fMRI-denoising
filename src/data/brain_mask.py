"""Parenchymal brain-mask construction from tissue segmentations.

NSD's ``brainmask.nii.gz`` is a *whole-head* mask (skull and scalp are inside
it and BOLD is exactly zero outside it), so it cannot be used as a brain mask.
The parenchymal mask is built from the FreeSurfer tissue union instead:

1. ``(GM | WM | CSF)`` intersected with the head mask,
2. 3D morphological closing (``iterations=3``, connectivity 2),
3. largest connected component,
4. slice-wise 2D ``binary_fill_holes`` (sulcal CSF is unlabeled in aseg),
5. re-intersection with the head mask.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def build_brain_mask(
    tissue_union: np.ndarray,
    head_mask: np.ndarray | None = None,
    closing_iterations: int = 3,
    connectivity: int = 2,
    fill_axis: int = 2,
) -> np.ndarray:
    """Build a boolean parenchymal brain mask.

    Args:
        tissue_union: Boolean/0-1 array ``[X, Y, Z]``, union of GM, WM and CSF.
        head_mask: Optional whole-head mask (NSD ``brainmask.nii.gz``); the result
            never extends outside it.
        closing_iterations: Iterations of 3D binary closing.
        connectivity: Structuring-element connectivity for closing and labeling
            (``scipy.ndimage.generate_binary_structure`` rank-3 convention).
        fill_axis: Axis whose 2D slices are hole-filled (axial = Z for NSD).

    Returns:
        Boolean ``[X, Y, Z]`` mask. Empty input yields an empty mask.
    """
    union = np.asarray(tissue_union) > 0
    if union.ndim != 3:
        raise ValueError(f"tissue_union must be 3D, got shape {union.shape}")
    head = None if head_mask is None else np.asarray(head_mask) > 0
    if head is not None:
        if head.shape != union.shape:
            raise ValueError(f"head_mask shape {head.shape} != tissue shape {union.shape}")
        union = union & head
    if not union.any():
        return np.zeros_like(union, dtype=bool)

    structure = ndi.generate_binary_structure(3, connectivity)

    # Zero-pad so closing's erosion step does not eat voxels touching the array border.
    pad = closing_iterations + 1
    padded = np.pad(union, pad, mode="constant", constant_values=False)
    closed = ndi.binary_closing(padded, structure=structure, iterations=closing_iterations)
    closed = closed[pad:-pad, pad:-pad, pad:-pad]

    labels, count = ndi.label(closed, structure=structure)
    if count > 1:
        sizes = ndi.sum(closed, labels, index=np.arange(1, count + 1))
        closed = labels == (1 + int(np.argmax(sizes)))

    filled = np.moveaxis(closed, fill_axis, 0).copy()
    for index in range(filled.shape[0]):
        filled[index] = ndi.binary_fill_holes(filled[index])
    filled = np.moveaxis(filled, 0, fill_axis)

    if head is not None:
        filled = filled & head
    return filled.astype(bool)
