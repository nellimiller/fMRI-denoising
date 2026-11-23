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
    """Save a fixed validation panel after each validation epoch.

    The panel uses the same J-invariant complementary K-pass inference as the
    evaluation scripts, so training-time figures and final evaluation figures are
    directly comparable.
    """

    def __init__(self, output_dir: str | Path, every_n_epochs: int = 1, K: int = 4, seed: int = 1234) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.every_n_epochs = every_n_epochs
        self.K = K
        self.seed = seed

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

        item = dataset[0]
        device = pl_module.device
        x = item["x"].unsqueeze(0).to(device)
        tr = item["tr"].reshape(1).to(device)
        was_training = pl_module.training
        pl_module.eval()
        try:
            with torch.no_grad():
                generator = torch.Generator(device=device).manual_seed(self.seed)
                result = pl_module.denoise_complementary(x, tr, K=self.K, generator=generator)
        finally:
            pl_module.train(was_training)

        clean = crop_to_input(result["clean"], item["x"])[0, 0].detach().cpu()
        noise = crop_to_input(result["noise"], item["x"])[0, 0].detach().cpu()
        dyn = item["x"][0].cpu()
        brain = item["x"][5, 0] > 0
        time_index = dyn.shape[0] // 2
        scores = brain.sum(dim=(0, 1))
        spatial_slice = int(scores.argmax()) if bool(scores.any()) else dyn.shape[-1] // 2
        limit = max(float(dyn[:, brain].abs().quantile(0.99)) if bool(brain.any()) else 1.0, 1e-6)

        panels = [
            ("Static anatomy", item["x"][1, 0, :, :, spatial_slice], "gray", -2.5, 2.5),
            ("Observed PSC", dyn[time_index, :, :, spatial_slice], "coolwarm", -limit, limit),
            ("Clean S", clean[time_index, :, :, spatial_slice], "coolwarm", -limit, limit),
            ("Physiological N", noise[time_index, :, :, spatial_slice], "coolwarm", -limit, limit),
            (
                "Residual e",
                (dyn - clean - noise)[time_index, :, :, spatial_slice],
                "coolwarm",
                -limit,
                limit,
            ),
        ]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(1, len(panels), figsize=(18, 4))
        for axis, (title, image, cmap, lower, upper) in zip(axes, panels):
            axis.imshow(image.numpy(), cmap=cmap, vmin=lower, vmax=upper)
            axis.set_title(title)
            axis.axis("off")
        figure.suptitle(
            f"epoch={trainer.current_epoch} time={time_index} z={spatial_slice} K={self.K}"
        )
        figure.tight_layout()
        figure.savefig(self.output_dir / f"validation_epoch_{trainer.current_epoch:04d}.png", dpi=120)
        plt.close(figure)
