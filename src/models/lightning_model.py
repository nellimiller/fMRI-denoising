# This file defines the PyTorch Lightning module for training the fMRI denoising model.

import lightning as L
import torch
import torch.optim as optim
from models.mae_core import HierarchicalSwinMAE
from utils.losses import fMRIMaskedLoss

class fMRIDenoisingLightning(L.LightningModule):
    def __init__(self,
                 in_chans=4,
                 patch_size=14,
                 embed_dim=96,
                 mask_ratio=0.75,
                 lr=1.5e-4, # Standard MAE base learning rate
                 weight_decay=0.05):
        super().__init__()
        # Saves all arguments to self.hparams and logs them automatically
        self.save_hyperparameters()

        # 1. Initialize the Core Architecture
        self.model = HierarchicalSwinMAE(
            patch_size=self.hparams.patch_size,
            in_chans=self.hparams.in_chans,
            embed_dim=self.hparams.embed_dim,
            mask_ratio=self.hparams.mask_ratio
        )

        # 2. Initialize the Dual-Engine Loss Function
        self.loss_fn = fMRIMaskedLoss(
            patch_size=self.hparams.patch_size, 
            fft_weight=0.1
        )

    def forward(self, x, tube_masking=True):
        # The standard forward pass
        return self.model(x, tube_masking=tube_masking)

    def training_step(self, batch, batch_idx):
        # Batch is expected to be your raw fMRI tensor: [B, C, T, X, Y, Z]
        x = batch
        
        # 1. Forward Pass (Generate Guess & Mask)
        pred, mask = self(x, tube_masking=True)
        
        # 2. Dynamic Cropping
        # We read the raw dimensions from the input batch and strip the padding 
        # off the prediction so the loss function can compare them perfectly.
        X, Y, Z = x.shape[3], x.shape[4], x.shape[5]
        pred_cropped = pred[:, :, :, :X, :Y, :Z]
        
        # 3. Calculate Loss
        total_loss, mse, fft = self.loss_fn(pred_cropped, target=x, mask=mask)
        
        # 4. Log Metrics (Automatically connects to Weights & Biases or TensorBoard)
        self.log('train_loss', total_loss, prog_bar=True, sync_dist=True)
        self.log('train_mse', mse, sync_dist=True)
        self.log('train_fft', fft, sync_dist=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        # Identical to training_step, but PyTorch Lightning automatically 
        # disables gradients to save VRAM and calculates validation metrics.
        x = batch
        pred, mask = self(x, tube_masking=True)
        
        X, Y, Z = x.shape[3], x.shape[4], x.shape[5]
        pred_cropped = pred[:, :, :, :X, :Y, :Z]
        
        total_loss, mse, fft = self.loss_fn(pred_cropped, target=x, mask=mask)
        
        self.log('val_loss', total_loss, prog_bar=True, sync_dist=True)
        self.log('val_mse', mse, sync_dist=True)
        self.log('val_fft', fft, sync_dist=True)
        
        return total_loss

    def configure_optimizers(self):
        # AdamW is the strict industry standard for Vision Transformers
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95) # Specific betas recommended for MAEs
        )
        
        # Transformers are highly unstable at the start of training.
        # Cosine Annealing slowly lowers the learning rate over time to 
        # help the network settle into a deep mathematical minimum.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=100, # Assuming 100 epochs
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }