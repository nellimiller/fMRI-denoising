"""Forward/backward smoke test of the full pipeline on dummy data.

Run with ``PYTHONPATH=src python src/test_pipeline.py``. It builds a synthetic 6-channel
sample (``[dyn, static_anat, gm, wm, csf, brain]``), runs one masked forward pass, the
identifiable loss, a backward pass and a K = 4 complementary J-invariant inference.
"""

from data.synthetic import SyntheticNSDDataset
from models.mae_core import HierarchicalSwinMAE
from utils.losses import IdentifiableDenoisingLoss


def run_pipeline_test() -> None:
    print("Initializing synthetic 6-channel sample...")
    sample = SyntheticNSDDataset(n_subjects=1, time_frames=8, shape=(32, 32, 24), seed=0)[0]
    x, tr = sample["x"].unsqueeze(0), sample["tr"].reshape(1)

    model = HierarchicalSwinMAE(patch_size=4, embed_dim=24, mask_ratio=0.75)
    loss_fn = IdentifiableDenoisingLoss()

    print("Running masked forward pass...")
    out = model(x, tr)
    losses = loss_fn(out, x, tr)
    print("\n--- Loss output ---")
    for name in ("total", "rec", "tissue", "band", "ortho"):
        print(f"{name:>7s}: {losses[name].item():.4f}")

    print("Testing backward pass (gradient flow)...")
    losses["total"].backward()
    assert all(p.grad is not None for p in model.parameters())

    print("Running K=4 complementary inference...")
    result = model.denoise_complementary(x, tr, K=4)
    assert (result["count"][result["count"] > 0] == 3).all()
    print("SUCCESS: model, loss and J-invariant inference are operational.")


if __name__ == "__main__":
    run_pipeline_test()
