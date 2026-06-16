# Main execution script for training the fMRI Denoising model using PyTorch Lightning and Weights & Biases for experiment tracking.
# This script sets up the logger, initializes the model, configures callbacks for checkpointing, and starts the training process.
# Contains placeholder code for the DataLoaders.

import os
import lightning as L
from torch.utils.data import DataLoader, random_split
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from models.lightning_model import fMRIDenoisingLightning
from data.dataset import fMRIDataset

class fMRIDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=1, time_frames=40, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.time_frames = time_frames
        self.num_workers = num_workers

    def setup(self, stage=None):
        # 1. Load the entire dataset
        full_dataset = fMRIDataset(self.data_dir, time_frames=self.time_frames)
        
        # 2. Automatically split into 80% Train, 20% Validation
        total_len = len(full_dataset)
        val_len = int(total_len * 0.2)
        train_len = total_len - val_len
        
        self.train_data, self.val_data = random_split(
            full_dataset, [train_len, val_len]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data, 
            batch_size=self.batch_size, 
            shuffle=True, # Randomize order for training
            num_workers=self.num_workers, # Use CPU cores to load data in background
            pin_memory=True # Speeds up CPU to GPU transfer
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data, 
            batch_size=self.batch_size, 
            shuffle=False, # Never shuffle validation data
            num_workers=self.num_workers,
            pin_memory=True
        )

def main():
    # =========================================================
    #                    CONFIGURATION
    # =========================================================
    # This path describes where the patient folders are.
    DATA_DIRECTORY = "C:/fMRI Denoising/dataset_dir" 
    
    # Critical: small Batch Size (1 or 2) for 7T 4D data 
    # to avoid Out-Of-Memory (OOM) errors on the GPU.
    BATCH_SIZE = 1 
    
    print("Initializing Data Pipeline...")
    datamodule = fMRIDataModule(
        data_dir=DATA_DIRECTORY,
        batch_size=BATCH_SIZE,
        time_frames=40, # Needs to be changes if the sequence length is different.
        num_workers=4
    )

    # =========================================================
    #          INITIALIZE ARCHITECTURE & LOGGING
    # =========================================================
    print("Initializing 4-Stage Hierarchical SwinMAE...")
    model = fMRIDenoisingLightning(
        in_chans=4, 
        patch_size=14, 
        embed_dim=96, 
        mask_ratio=0.75, 
        lr=1.5e-4
    )

    print("Connecting to Weights & Biases...")
    wandb_logger = WandbLogger(
        project="7T-fMRI-Denoising", 
        name="SwinMAE-4Stage-Baseline", 
        log_model="all" 
    )

    # Save a checkpoint every time the validation loss reaches a new low
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        dirpath='checkpoints/', # Saves locally just in case W&B drops connection
        filename='swinmae-{epoch:02d}-{val_loss:.4f}'
    )

    # =========================================================
    #                    TRAINER ENGINE
    # =========================================================
    trainer = L.Trainer(
        max_epochs=100,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        accelerator="auto", # Automatically detects GPU vs CPU
        devices="auto",
        precision="16-mixed", # Cuts VRAM usage in half
        log_every_n_steps=1 # Forces W&B to update on every batch
    )

    # =========================================================
    #                      IGNITION
    # =========================================================
    print("Starting Training Loop...")
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()