"""Validation visualizations for quick inspection of denoising experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import Callback


def crop_to_input(prediction: torch.Tensor, input_sample: torch.Tensor) -> torch.Tensor:
    """Crop padded predictions to a dataset sample's original spatial extent."""
    if prediction.ndim != 6:
        raise ValueError(f"Expected prediction [B, C, T, X, Y, Z], got {prediction.shape}.")
    if input_sample.ndim != 5:
        raise ValueError(f"Expected input sample [C, T, X, Y, Z], got {input_sample.shape}.")
    if prediction.shape[0] != 1 or prediction.shape[1] != 1:
        raise ValueError(
            "Visualization expects one sample and one predicted BOLD channel, "
            f"got shape {prediction.shape}."
        )
    if prediction.shape[2] != input_sample.shape[1]:
        raise ValueError(
            f"Prediction has {prediction.shape[2]} time points, "
            f"input has {input_sample.shape[1]}."
        )

    target_spatial = input_sample.shape[2:]
    if any(actual < target for actual, target in zip(prediction.shape[3:], target_spatial)):
        raise ValueError(
            f"Prediction spatial shape {prediction.shape[3:]} is smaller than "
            f"input shape {target_spatial}."
        )
    return prediction[:, :, :, : target_spatial[0], : target_spatial[1], : target_spatial[2]]


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
        try:
            with torch.no_grad():
                prediction, clean, noise, _ = pl_module(batch.unsqueeze(0).to(device))
        finally:
            pl_module.train(was_training)

        prediction = crop_to_input(prediction, batch)[0, 0].detach().cpu()
        clean = crop_to_input(clean, batch)[0, 0].detach().cpu()
        noise = crop_to_input(noise, batch)[0, 0].detach().cpu()
        input_bold = batch[0].cpu()
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
