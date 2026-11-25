"""Validation visualizations for quick inspection of denoising experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import Callback

try:
    from .temporal_metrics import gm_local_patch_trace
except ImportError:
    from utils.temporal_metrics import gm_local_patch_trace


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

    def __init__(
        self,
        output_dir: str | Path,
        every_n_epochs: int = 1,
        K: int = 4,
        seed: int = 1234,
        temporal_patch_size: int = 3,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.every_n_epochs = every_n_epochs
        self.K = K
        self.seed = seed
        self.temporal_patch_size = temporal_patch_size

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
        gm = item["x"][2, 0].cpu() > 0
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
        figure, axes = plt.subplots(1, len(panels) + 2, figsize=(24, 4))
        for axis, (title, image, cmap, lower, upper) in zip(axes, panels):
            axis.imshow(image.numpy(), cmap=cmap, vmin=lower, vmax=upper)
            axis.set_title(title)
            axis.axis("off")
        if bool(gm.any()):
            coordinates = gm.nonzero(as_tuple=False)
            gm_values = dyn[:, gm]
            voxel = int(gm_values.std(dim=0, unbiased=False).argsort()[len(coordinates) // 2])
            anchor = tuple(int(value) for value in coordinates[voxel])
            patch_input = gm_local_patch_trace(dyn, gm, anchor, self.temporal_patch_size)
            patch_clean = gm_local_patch_trace(clean, gm, anchor, self.temporal_patch_size)
            global_input = gm_values.mean(1)
            global_clean = clean[:, gm].mean(1)
            seconds = torch.arange(dyn.shape[0]).float() * float(item["tr"])
            axes[-2].plot(seconds, patch_input, label="observed GM patch")
            axes[-2].plot(seconds, patch_clean, label="clean GM patch")
            axes[-2].plot(seconds, global_input, linestyle="--", label="global GM observed")
            axes[-2].plot(seconds, global_clean, linestyle="--", label="global GM clean")
            axes[-2].set_title(f"GM-masked {self.temporal_patch_size}x patch")
            axes[-2].set_xlabel("Time (s)")
            axes[-2].legend(fontsize=6)
            frequency = torch.fft.rfftfreq(dyn.shape[0], d=float(item["tr"]))[1:]
            for label, trace in (
                ("observed patch", patch_input),
                ("clean patch", patch_clean),
                ("global GM observed", global_input),
                ("global GM clean", global_clean),
            ):
                centered = trace - trace.mean()
                power = torch.fft.rfft(centered, dim=0).abs().square()[1:]
                axes[-1].plot(frequency, power / power.sum().clamp_min(1e-12), label=label)
            axes[-1].axvspan(0.01, 0.15, color="grey", alpha=0.15)
            axes[-1].set_title("Patch/global GM PSD")
            axes[-1].set_xlabel("Frequency (Hz)")
            axes[-1].legend(fontsize=6)
        figure.suptitle(
            f"epoch={trainer.current_epoch} time={time_index} z={spatial_slice} K={self.K}"
        )
        figure.tight_layout()
        output_path = (
            self.output_dir
            / f"validation_ep{trainer.current_epoch:03d}_step{trainer.global_step:06d}.png"
        )
        figure.savefig(output_path, dpi=120)
        plt.close(figure)
        for logger in trainer.loggers:
            if logger.__class__.__name__ != "WandbLogger":
                continue
            import wandb

            logger.experiment.log(
                {
                    "validation/panel": wandb.Image(
                        str(output_path),
                        caption=f"epoch={trainer.current_epoch}, step={trainer.global_step}",
                    )
                },
                step=trainer.global_step,
            )
