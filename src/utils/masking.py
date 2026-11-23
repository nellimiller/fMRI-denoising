"""Token masking and complementary K-pass inference for the Masked Autoencoder.

Training draws one random *tube* mask per sample: all ``T`` frames of a chosen
spatial patch are hidden together, over *valid* cells only (cells containing at
least one brain voxel; pure padding cells are never counted). The number of masked
cells is fixed, ``round(ratio * n_valid)``.

Inference uses ``K`` complementary masks whose *visible* sets partition the valid
cells, so every hidden-ratio equals ``(K - 1) / K`` (0.75 for ``K = 4``), and each
voxel is hidden in exactly ``K - 1`` passes. Averaging only those passes gives a
J-invariant estimate: a voxel's own measurements never enter its prediction.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import torch


def sample_tube_mask(
    valid: torch.Tensor,
    mask_ratio: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a fixed-count spatial tube mask over valid cells.

    Args:
        valid: Bool ``(B, S)``; cells that may be masked.
        mask_ratio: Fraction of *valid* cells to hide.
        generator: Optional RNG for reproducible draws.

    Returns:
        Bool ``(B, S)``, ``True`` = hidden. Invalid cells are always ``False``.
    """
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError("mask_ratio must be in [0, 1].")
    B, S = valid.shape
    scores = torch.rand(B, S, generator=generator, device=valid.device)
    scores = scores.masked_fill(~valid, 2.0)  # invalid cells sort last
    rank = scores.argsort(dim=1).argsort(dim=1)
    n_mask = torch.round(valid.sum(dim=1, keepdim=True).float() * mask_ratio).long()
    return (rank < n_mask) & valid


def complementary_tube_masks(
    B: int,
    S: int,
    K: int = 4,
    valid: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build ``K`` complementary tube masks.

    The visible sets are a disjoint, balanced partition of the valid cells.

    Returns:
        Bool ``(K, B, S)``; ``masks[k]`` is ``True`` where the tube is HIDDEN in
        pass ``k``. Invalid cells are never hidden (``False``).
    """
    if K < 2:
        raise ValueError("K must be >= 2.")
    if valid is None:
        valid = torch.ones(B, S, dtype=torch.bool, device=device)
    scores = torch.rand(B, S, generator=generator, device=valid.device)
    scores = scores.masked_fill(~valid, 2.0)
    group = scores.argsort(dim=1).argsort(dim=1) % K  # balanced group id per cell
    visible = torch.stack([(group == k) & valid for k in range(K)])
    return (~visible) & valid.unsqueeze(0)


def apply_swin_mask(
    x: torch.Tensor,
    mask_token: torch.Tensor,
    mask_ratio: float = 0.75,
    tube_masking: bool = False,
    T: int | None = None,
    valid: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace masked tokens with a learnable MASK token (sequence length is preserved).

    Args:
        x: Embedded dynamic tokens ``(B, L, E)`` ordered ``(t, spatial)``.
        mask_token: Learnable parameter ``(1, 1, E)``.
        mask_ratio: Fraction of valid cells (tube) or tokens to mask.
        tube_masking: Hide identical spatial cells across all frames.
        T: Number of frames (required for tube masking / ``token_mask``).
        valid: Bool ``(B, S)`` of maskable spatial cells (tube masking only).
        token_mask: Optional explicit spatial mask ``(B, S)`` bool; overrides sampling.
        generator: Optional RNG.

    Returns:
        ``(x_masked, mask)`` with ``mask`` float ``(B, L)``, 1 = masked.
    """
    B, L, E = x.shape

    if token_mask is not None or tube_masking:
        if T is None:
            raise ValueError("T (number of timepoints) must be provided for tube masking.")
        S = L // T
        if token_mask is None:
            if valid is None:
                valid = torch.ones(B, S, dtype=torch.bool, device=x.device)
            token_mask = sample_tube_mask(valid, mask_ratio, generator)
        elif token_mask.shape != (B, S):
            raise ValueError(f"token_mask must have shape {(B, S)}, got {tuple(token_mask.shape)}.")
        mask = token_mask.bool().unsqueeze(1).expand(B, T, S).reshape(B, L).float()
    else:
        noise = torch.rand(B, L, generator=generator, device=x.device)
        mask = (noise < mask_ratio).float()

    m = mask.unsqueeze(-1)
    x_masked = x * (1 - m) + m * mask_token.to(x.dtype)
    return x_masked, mask


def token_mask_to_voxel_mask(
    spatial_mask: torch.Tensor,
    grid: tuple[int, int, int],
    patch_size: int,
    shape: tuple[int, int, int],
) -> torch.Tensor:
    """Expand a spatial tube mask ``(B, S)`` to voxels ``(B, 1, X, Y, Z)`` (cropped to ``shape``)."""
    B = spatial_mask.shape[0]
    m = spatial_mask.reshape(B, 1, *grid)
    for axis in (2, 3, 4):
        m = m.repeat_interleave(patch_size, dim=axis)
    X, Y, Z = shape
    return m[:, :, :X, :Y, :Z]


@torch.no_grad()
def denoise_complementary(
    forward_fn: Callable[[torch.Tensor], Mapping[str, torch.Tensor]],
    valid: torch.Tensor,
    grid: tuple[int, int, int],
    patch_size: int,
    shape: tuple[int, int, int],
    K: int = 4,
    generator: torch.Generator | None = None,
    keys: Sequence[str] = ("clean", "noise"),
) -> dict[str, torch.Tensor]:
    """J-invariant full-volume inference with ``K`` complementary tube masks.

    Args:
        forward_fn: Runs the model with a spatial token mask ``(B, S)`` (``True`` =
            hidden) and returns tensors ``(B, 1, T, X, Y, Z)`` for every name in ``keys``.
        valid: Bool ``(B, S)`` of valid spatial cells.
        grid: Patch grid ``(H_p, W_p, D_p)``.
        patch_size: Patch edge length in voxels.
        shape: Output spatial shape ``(X, Y, Z)``.
        K: Number of passes; the training hidden ratio is ``(K - 1) / K``.
        generator: Optional RNG for the partition.
        keys: Output tensors to average.

    Returns:
        ``{key: tensor}`` averaged over the ``K - 1`` passes in which each voxel was
        hidden, plus ``"count"`` ``(B, 1, 1, X, Y, Z)`` = passes averaged per voxel
        (``K - 1`` inside valid cells, ``0`` elsewhere).
    """
    B, S = valid.shape
    masks = complementary_tube_masks(B, S, K, valid, generator, device=valid.device)
    totals: dict[str, torch.Tensor] = {}
    count: torch.Tensor | None = None
    for k in range(K):
        outputs = forward_fn(masks[k])
        weight = token_mask_to_voxel_mask(masks[k], grid, patch_size, shape).float().unsqueeze(2)
        for key in keys:
            value = outputs[key].float() * weight
            totals[key] = value if key not in totals else totals[key] + value
        count = weight if count is None else count + weight
    assert count is not None
    result = {key: total / count.clamp_min(1.0) for key, total in totals.items()}
    result["count"] = count
    return result
