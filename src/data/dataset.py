import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from einops import repeat

class fMRIDataset(Dataset):
    """PyTorch Dataset for loading fMRI data and tissue masks.

    Args:
        data_dir (str): Path to the directory containing subject folders.
        time_frames (int): Number of time frames to load from each BOLD run. Defaults to 40.

    Attributes:
        subjects (list[str]): Sorted list of subject folder names in `data_dir`.

    Notes:
        Expects a neuroimaging directory structured as:
        data_dir/
            sub-01/
                bold.nii.gz
                gm.nii.gz
                wm.nii.gz
                csf.nii.gz
    """

    def __init__(self, data_dir: str, time_frames: int = 40) -> None:
        """Initialize the dataset and scan for subjects."""
        self.data_dir = data_dir
        self.time_frames = time_frames
        
        self.subjects = [d for d in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, d))]
        self.subjects.sort()

    def __len__(self) -> int:
        """Return the total number of subjects."""
        return len(self.subjects)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load, normalize, and merge BOLD and anatomical masks for a subject.

        Args:
            idx: Subject index.

        Returns:
            torch.Tensor: Preprocessed fMRI data of shape `(4, T, X, Y, Z)`.
                Channel 0: Normalized BOLD
                Channel 1: Gray Matter (GM) mask
                Channel 2: White Matter (WM) mask
                Channel 3: Cerebrospinal Fluid (CSF) mask
        """
        subject_folder = os.path.join(self.data_dir, self.subjects[idx])
        
        # Load BOLD Signal
        bold_path = os.path.join(subject_folder, 'bold.nii.gz')
        bold_img = nib.load(bold_path) # image object with metadata
        bold_data = bold_img.get_fdata(dtype=np.float32)  # Shape: [X, Y, Z, T]

        ############ critical: notes.txt
        # Global Z-Score Normalization
        mean = np.mean(bold_data)
        std = np.std(bold_data) + 1e-8
        bold_data = (bold_data - mean) / std

        # Reorder dimensions to [T, X, Y, Z]
        bold_data = np.transpose(bold_data, (3, 0, 1, 2))
        
        # Truncate or zero-pad sequence to enforce exact sequence length (e.g., T=40)
        if bold_data.shape[0] >= self.time_frames:
            bold_data = bold_data[:self.time_frames]
        else:
            pad_len = self.time_frames - bold_data.shape[0]
            bold_data = np.pad(bold_data, ((0, pad_len), (0,0), (0,0), (0,0)), mode='constant')

        # Load and stack anatomical masks
        masks = []
        for mask_name in ['gm.nii.gz', 'wm.nii.gz', 'csf.nii.gz']:
            mask_path = os.path.join(subject_folder, mask_name)
            masks.append(nib.load(mask_path).get_fdata(dtype=np.float32))

        masks_stacked = np.stack(masks, axis=0) # Shape: [3, X, Y, Z]
        
        # Convert to tensors
        bold_tensor = torch.from_numpy(bold_data)
        masks_tensor = torch.from_numpy(masks_stacked)

        # Broadcast static 3D masks across all timeframes to match BOLD shape
        masks_time = repeat(masks_tensor, 'c x y z -> c t x y z', t=self.time_frames) # Shape: [3, T, X, Y, Z]
        bold_tensor = bold_tensor.unsqueeze(0)

        # Concatenate BOLD and masks along the channel dimension
        final_tensor = torch.cat([bold_tensor, masks_time], dim=0) 

        return final_tensor