# This file defines a PyTorch Dataset class that loads and preprocesses fMRI data for training the model.
# Built as placeholder code. It handels the 4D fMRI data and 3D masks of grey matter, white matter, csf; derived from the structural scan.

import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from einops import repeat

class fMRIDataset(Dataset):
    def __init__(self, data_dir, time_frames=40):
        """
        Expects a standard neuroimaging directory structure:
        data_dir/
            sub-01/
                bold.nii.gz
                gm.nii.gz
                wm.nii.gz
                csf.nii.gz
            sub-02/
                ...
        """
        self.data_dir = data_dir
        self.time_frames = time_frames
        
        # Scan directory for subject folders
        self.subjects = [d for d in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, d))]
        self.subjects.sort()

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_folder = os.path.join(self.data_dir, self.subjects[idx])
        
        # ==========================================
        #       LOAD AND PREP BOLD SIGNAL (4D)
        # ==========================================
        bold_path = os.path.join(subject_folder, 'bold.nii.gz')
        bold_img = nib.load(bold_path)
        # NiBabel loads fMRI as [X, Y, Z, Time]
        bold_data = bold_img.get_fdata(dtype=np.float32) 
        
        # Global Z-Score Normalization
        # Neural networks need values close to 0 to learn effectively
        mean = np.mean(bold_data)
        std = np.std(bold_data) + 1e-8
        bold_data = (bold_data - mean) / std

        # Transpose from [X, Y, Z, T] to [T, X, Y, Z]
        bold_data = np.transpose(bold_data, (3, 0, 1, 2))
        
        # Enforce exact sequence length (e.g., T=40)
        if bold_data.shape[0] >= self.time_frames:
            bold_data = bold_data[:self.time_frames]
        else:
            # Pad with zeros if the sequence is too short
            pad_len = self.time_frames - bold_data.shape[0]
            bold_data = np.pad(bold_data, ((0, pad_len), (0,0), (0,0), (0,0)), mode='constant')

        # ==========================================
        #    LOAD AND PREP ANATOMICAL MASKS (3D)
        # ==========================================
        masks = []
        for mask_name in ['gm.nii.gz', 'wm.nii.gz', 'csf.nii.gz']:
            mask_path = os.path.join(subject_folder, mask_name)
            # Anatomical masks are static 3D volumes: [X, Y, Z]
            mask_data = nib.load(mask_path).get_fdata(dtype=np.float32) 
            masks.append(mask_data)

        # Stack the 3 masks together: [3, X, Y, Z]
        masks_stacked = np.stack(masks, axis=0)
        
        # ==========================================
        #      MERGE INTO FINAL PYTORCH TENSOR
        # ==========================================
        bold_tensor = torch.from_numpy(bold_data)        # [T, X, Y, Z]
        masks_tensor = torch.from_numpy(masks_stacked)   # [3, X, Y, Z]

        # The model expects Time to be present in the masks too.
        # Using einops to instantly duplicate the static 3D masks across all 40 timeframes.
        # Resulting shape: [3, T, X, Y, Z]
        masks_time = repeat(masks_tensor, 'c x y z -> c t x y z', t=self.time_frames)
        
        # Reshape BOLD to [1, T, X, Y, Z] so it matches the mask dimensions
        bold_tensor = bold_tensor.unsqueeze(0)
        
        # Glue them together along the Channel dimension (dim=0)
        # Final shape: [4, T, X, Y, Z] -> (BOLD, GM, WM, CSF)
        final_tensor = torch.cat([bold_tensor, masks_time], dim=0)

        return final_tensor