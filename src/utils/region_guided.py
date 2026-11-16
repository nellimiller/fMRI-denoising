"""WM/CSF-guided nuisance projection for BOLD time series.

This is a deterministic regional-projection baseline, not a neural network or
a full aCompCor implementation. It estimates two nuisance traces from the
observed WM and CSF signals and removes their least-squares projection from
GM voxels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RegionGuidedResult:
    """Projection outputs for one batch of BOLD runs or windows.

    All image tensors have shape ``[B, T, X, Y, Z]``. ``noise`` contains only
    the component removed from GM; outside GM it is zero. ``nuisance_traces``
    has shape ``[B, T, 2]`` and contains the centered, unit-RMS WM and CSF
    regional mean traces. ``coefficients`` has shape ``[B, 2, X, Y, Z]``.
    """

    clean: Tensor
    noise: Tensor
    reconstruction: Tensor
    nuisance_traces: Tensor
    coefficients: Tensor


def _static_mask(mask: Tensor, bold: Tensor, name: str) -> Tensor:
    """Normalize a tissue mask to boolean ``[B, X, Y, Z]``."""
    batch, time, x_size, y_size, z_size = bold.shape
    if not torch.isfinite(mask).all():
        raise ValueError(f"{name} mask contains non-finite values.")

    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    if mask.ndim == 4:
        mask = mask.unsqueeze(1)
    if mask.ndim != 5:
        raise ValueError(
            f"{name} mask must have shape [X,Y,Z], [B,X,Y,Z], "
            f"or [B,T,X,Y,Z]; got {tuple(mask.shape)}."
        )

    if mask.shape[0] not in (1, batch):
        raise ValueError(f"{name} mask batch dimension must be 1 or {batch}.")
    if mask.shape[1] not in (1, time):
        raise ValueError(f"{name} mask time dimension must be 1 or {time}.")
    if tuple(mask.shape[2:]) != (x_size, y_size, z_size):
        raise ValueError(
            f"{name} mask spatial shape {tuple(mask.shape[2:])} does not match "
            f"BOLD shape {(x_size, y_size, z_size)}."
        )

    mask = mask.to(device=bold.device) > 0
    if mask.shape[0] == 1 and batch != 1:
        mask = mask.expand(batch, *mask.shape[1:])
    if mask.shape[1] == time and not torch.equal(mask, mask[:, :1].expand_as(mask)):
        raise ValueError(f"{name} mask must be static across time.")

    return mask[:, 0]


def region_guided_projection(
    bold: Tensor,
    gm_mask: Tensor,
    wm_mask: Tensor,
    csf_mask: Tensor,
) -> RegionGuidedResult:
    """Remove WM/CSF regional-mean projections from GM voxel time courses.

    Args:
        bold: Finite BOLD data with shape ``[B, T, X, Y, Z]``.
        gm_mask: GM mask, either static ``[X,Y,Z]`` / ``[B,X,Y,Z]`` or
            repeated ``[B,T,X,Y,Z]``.
        wm_mask: WM mask with the same accepted shapes.
        csf_mask: CSF mask with the same accepted shapes.

    Returns:
        Clean and removed-component volumes, their sum, the two normalized
        nuisance traces, and voxelwise regression coefficients. The removed
        component is zero outside GM, so the clean output is unchanged there.

    Notes:
        Each nuisance trace is the spatial mean of the *observed* BOLD in its
        tissue mask, centered and scaled to unit RMS. For every GM voxel, the
        removed signal is its least-squares projection onto the span of those
        two traces. This removes any GM variation correlated with the traces,
        whether nuisance or genuine shared signal. Compute traces over full
        runs where possible; estimates from short overlapping windows can
        vary substantially.
    """
    if bold.ndim != 5:
        raise ValueError(f"Expected BOLD [B,T,X,Y,Z], got {tuple(bold.shape)}.")
    if not bold.is_floating_point():
        raise TypeError(f"BOLD must be floating point, got {bold.dtype}.")
    if bold.shape[1] < 2:
        raise ValueError("At least two time points are required for projection.")
    if not torch.isfinite(bold).all():
        raise ValueError("BOLD contains non-finite values.")

    gm = _static_mask(gm_mask, bold, "GM")
    wm = _static_mask(wm_mask, bold, "WM")
    csf = _static_mask(csf_mask, bold, "CSF")

    for name, mask in (("GM", gm), ("WM", wm), ("CSF", csf)):
        empty_samples = ~mask.flatten(1).any(dim=1)
        if bool(empty_samples.any()):
            indices = empty_samples.nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"{name} mask is empty for batch items {indices}.")

    work_dtype = torch.float64 if bold.dtype == torch.float64 else torch.float32
    bold_work = bold.to(dtype=work_dtype)
    tissue_masks = (wm, csf)
    traces = []
    for mask in tissue_masks:
        weights = mask.to(dtype=work_dtype)
        count = weights.sum(dim=(1, 2, 3)).clamp_min(1)
        trace = (bold_work * weights[:, None]).sum(dim=(2, 3, 4)) / count[:, None]
        trace = trace - trace.mean(dim=1, keepdim=True)
        rms = trace.square().mean(dim=1, keepdim=True).sqrt()
        traces.append(trace / rms.clamp_min(torch.finfo(work_dtype).eps))

    nuisance_traces = torch.stack(traces, dim=-1)
    design_pinv = torch.linalg.pinv(nuisance_traces)
    bold_flat = bold_work.flatten(start_dim=2)
    coefficients_flat = torch.bmm(design_pinv, bold_flat)
    projected_flat = torch.bmm(nuisance_traces, coefficients_flat)

    gm_flat = gm.flatten(start_dim=1).to(dtype=work_dtype)
    noise = (projected_flat * gm_flat[:, None, :]).reshape_as(bold_work)
    clean = bold_work - noise
    reconstruction = clean + noise
    coefficients = (coefficients_flat * gm_flat[:, None, :]).reshape(
        bold.shape[0], 2, *bold.shape[2:]
    )

    return RegionGuidedResult(
        clean=clean.to(dtype=bold.dtype),
        noise=noise.to(dtype=bold.dtype),
        reconstruction=reconstruction.to(dtype=bold.dtype),
        nuisance_traces=nuisance_traces,
        coefficients=coefficients.to(dtype=bold.dtype),
    )
