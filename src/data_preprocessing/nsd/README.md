# NSD preprocessing

This package converts the downloaded NSD subject layout into a separate,
training-ready directory. It does not modify the raw NSD files.

The preprocessing command:

1. Finds `timeseries_nsdimagery_run*.nii.gz` files.
2. Validates that `aseg.nii.gz` and `brainmask.nii.gz` match each BOLD run.
3. Converts FreeSurfer-style segmentation labels into GM, WM, and CSF masks
   (label 77, WM-hypointensities, belongs to WM).
4. Builds a parenchymal `brain.nii.gz`: the GM/WM/CSF union restricted to NSD's
   `brainmask.nii.gz` (which is a **whole-head** mask including skull and scalp), 3-D
   closing (3 iterations, 18-neighbourhood), largest connected component, and slice-wise
   2-D hole filling so sulcal CSF is included.
5. Reads the repetition time from the BOLD header (`pixdim[4]`), or from `--tr`, and
   stores it with its provenance. NSD time series carry unknown time units and no slice
   timing, so verify the TR against the NSD documentation.
6. Writes the masks once per subject and copies no raw data.
7. Writes a JSON manifest (schema version 2) with run paths, `brain_path`, `tr` and
   temporal windows.

Example:

```powershell
uv run python -m data_preprocessing.nsd.preprocess `
  --raw-root "C:\fMRI Denoising\NSD\nsd_data" `
  --output-root "C:\fMRI Denoising\NSD\nsd_processed" `
  --subject subj01 `
  --window-length 40 `
  --stride 20
```

The expected output is:

```text
nsd_processed/
└── subj01/
    ├── anat/
    │   ├── gm.nii.gz
    │   ├── wm.nii.gz
    │   ├── csf.nii.gz
    │   └── brain.nii.gz
    └── manifest.json
```

The manifest references the raw BOLD files. This avoids duplicating large
4D files. The training dataset uses each manifest entry as one temporal window
rather than treating the entire subject as one sample. At load time
`NSDDataset` removes the full-run linear trend, converts the window to voxelwise
percent signal change inside `brain.nii.gz`, z-scores the window's temporal-mean
image inside the brain, and returns a dict with the 6-channel tensor
`[dyn, static_anat, gm, wm, csf, brain]` and the scalar `tr`. Manifests written by
the previous version (no `brain_path`) still load: the brain mask is rebuilt from the
tissue masks, but they keep the old tissue labels, so re-run preprocessing.

For a data-backed one-window overfit diagnostic, run from the repository root.
The `PYTHONPATH` setting allows the diagnostic script to use the project's
source-tree imports:

```powershell
$env:PYTHONPATH = "src"
uv run python src/utils/train_nsd_overfit.py `
  --data_dir "C:\fMRI Denoising\NSD\nsd_processed" `
  --output_dir "C:\fMRI Denoising\NSD\overfit_test" `
  --subject subj01 `
  --entry 0 `
  --crop_size 32 32 32 `
  --patch_size 4 --embed_dim 24 `
  --epochs 1 `
  --repeats 1
```

This intentionally repeats one temporal window. It checks data loading,
spatial dimensions, model forward/backward execution, and loss computation.
It is not a scientific training configuration and does not assess
generalization.

The label groups are explicit in `preprocess.py` and should be reviewed if a
different segmentation convention is used.
