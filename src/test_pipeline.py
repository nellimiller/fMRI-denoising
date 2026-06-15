### This file verifys the functionality of the files in the models and utils folders by running a forward and backward pass through the entire pipeline using dummy data.
import torch
from models.mae_core import HierarchicalSwinMAE
from utils.losses import fMRIMaskedLoss

def run_pipeline_test():
    # 1. Initialize data (T=2 to save CPU RAM)
    print("Initializing dummy fMRI tensor...")
    #dummy_fmri = torch.randn(1, 4, 2, 210, 210, 114)
    # Using smaller dimensions to ensure the test runs on limited resources
    dummy_fmri = torch.randn(1, 4, 2, 42, 42, 28)

    # 2. Initialize Model and Loss Function
    model = HierarchicalSwinMAE(patch_size=14, in_chans=4, embed_dim=96, mask_ratio=0.75)
    loss_fn = fMRIMaskedLoss(patch_size=14, fft_weight=0.1)
    
    # 3. Forward Pass
    print("Running forward pass...")
    reconstruction, mask = model(dummy_fmri, tube_masking=True)
    
    # Cropping the padding off the reconstruction to match the raw target
    #final_output = reconstruction[:, :, :, :210, :210, :114]
    final_output = reconstruction[:, :, :, :42, :42, :28]
    
    # 4. Calculating Loss
    print("Calculating Masked MSE and FFT Loss...")
    total_loss, mse, fft = loss_fn(pred=final_output, target=dummy_fmri, mask=mask)
    
    print(f"\n--- Loss Output ---")
    print(f"Masked MSE Loss: {mse.item():.4f}")
    print(f"Temporal FFT Loss: {fft.item():.4f}")
    print(f"Total Loss: {total_loss.item():.4f}")
    
    # 5. Check backward pass (Gradient calculation)
    print("Testing backward pass (gradient flow)...")
    total_loss.backward()
    print("SUCCESS: The pipeline and loss engine are fully operational.")

if __name__ == "__main__":
    run_pipeline_test()
