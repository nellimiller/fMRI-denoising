### This file verifys the functionality of the files "patch_embed.py" and "masking.py" by running a simple forward pass with dummy data.

import torch
import torch.nn as nn
from models.patch_embed import fMRIPatchEmbed4D
from utils.masking import apply_swin_mask

import torch
from models.mae_core import HierarchicalSwinMAE as SwinMAE

def run_end_to_end_test():
    print("Initializing dummy fMRI tensor [Batch=1, Ch=4, T=2, X=210, Y=210, Z=114]...")
    dummy_fmri = torch.randn(1, 4, 2, 210, 210, 114)
    
    # Initialize the complete Autoencoder
    model = SwinMAE(patch_size=14, in_chans=4, embed_dim=96, mask_ratio=0.75)
    
    # Run the forward pass
    print("Running forward pass through Encoder-Decoder...")
    reconstruction, mask = model(dummy_fmri, tube_masking=False)
    
    print(f"\n--- Final Output ---")
    print(f"Original Input Shape: {dummy_fmri.shape}")
    print(f"Reconstructed Shape: {reconstruction.shape}")
    
    # The reconstruction will be padded to Z=126 -> slice it back to Z=114 
    # Dimensions: [Batch, Channel, Time, X, Y, Z]
    final_output = reconstruction[:, :, :, :210, :210, :114]
    
    print(f"Cropped Output Shape (Ready for Loss): {final_output.shape}")

    if dummy_fmri.shape == final_output.shape:
        print("SUCCESS: The Autoencoder architecture is structurally sound.")
    
    else:
        print("ERROR: Shape mismatch detected.")

if __name__ == "__main__":
    run_end_to_end_test()
