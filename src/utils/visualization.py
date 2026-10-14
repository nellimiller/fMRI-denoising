"""Validation visualizations for quick inspection of denoising experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import Callback


class ValidationVisualizationCallback(Callback):
    """Save a fixed validation reconstruction panel after each validation epoch."""

    def __init__(self, output_dir: str | Path, every_n_epochs: int = 1) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking or trainer.current_epoch % self.every_n_epochs:
            return

        dataloader = trainer.val_dataloaders
        if dataloader is None:
            return
        if isinstance(dataloader, (list, tuple)):
            dataloader = dataloader[0]
        dataset = dataloader.dataset
        if len(dataset) == 0:
            return

        batch = dataset[0]
        device = pl_module.device
        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            prediction, clean, noise, _ = pl_module(batch.unsqueeze(0).to(device))
        if was_training:
            pl_module.train()

        input_bold = batch[0].cpu()
        prediction = prediction[0, 0].detach().cpu()
        clean = clean[0, 0].detach().cpu()
        noise = noise[0, 0].detach().cpu()
        time_index = input_bold.shape[0] // 2
        spatial_slice = input_bold.shape[-1] // 2

        panels = [
            ("Input", input_bold[time_index, :, :, spatial_slice]),
            ("Reconstruction", prediction[time_index, :, :, spatial_slice]),
            ("Clean head", clean[time_index, :, :, spatial_slice]),
            ("Noise head", noise[time_index, :, :, spatial_slice]),
            ("Absolute error", (prediction - input_bold)[time_index, :, :, spatial_slice].abs()),
        ]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(1, len(panels), figsize=(18, 4))
        for axis, (title, image) in zip(axes, panels):
            axis.imshow(image.numpy(), cmap="gray")
            axis.set_title(title)
            axis.axis("off")
        figure.suptitle(
            f"epoch={trainer.current_epoch} time={time_index} z={spatial_slice}"
        )
        figure.tight_layout()
        figure.savefig(
            self.output_dir / f"validation_epoch_{trainer.current_epoch:04d}.png",
            dpi=120,
        )
        plt.close(figure)
