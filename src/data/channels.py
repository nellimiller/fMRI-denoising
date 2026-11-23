"""Channel layout of the 6-channel NSD sample tensor ``[C, T, X, Y, Z]``.

Only the dynamic channel carries (noisy) temporal information. Every other
channel is time-invariant anatomy and is therefore safe to expose to the
encoder in all tubes, masked or not.
"""

from __future__ import annotations

DYN = 0  # detrended percent signal change (PSC), zero outside the brain
STATIC = 1  # window temporal-mean image, z-scored inside the brain
GM = 2
WM = 3
CSF = 4
BRAIN = 5  # parenchymal brain mask

NUM_CHANNELS = 6
CHANNEL_NAMES = ("dyn", "static_anat", "gm", "wm", "csf", "brain")

# Channels fed to the always-visible anatomy pathway (patch embedding + voxel stem).
ANATOMY_CHANNELS = (STATIC, GM, WM, CSF, BRAIN)
NUM_ANATOMY_CHANNELS = len(ANATOMY_CHANNELS)

# Position of BRAIN inside the anatomy sub-tensor ``x[:, ANATOMY_CHANNELS]``.
ANAT_BRAIN = ANATOMY_CHANNELS.index(BRAIN)
ANAT_GM = ANATOMY_CHANNELS.index(GM)
ANAT_WM = ANATOMY_CHANNELS.index(WM)
ANAT_CSF = ANATOMY_CHANNELS.index(CSF)
