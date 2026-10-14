# NSD preprocessing

This package converts the downloaded NSD subject layout into a separate,
training-ready directory. It does not modify the raw NSD files.

The preprocessing command:

1. Finds `timeseries_nsdimagery_run*.nii.gz` files.
2. Validates that `aseg.nii.gz` and `brainmask.nii.gz` match each BOLD run.
3. Converts FreeSurfer-style segmentation labels into GM, WM, and CSF masks.
4. Writes the masks once per subject and copies no raw data.
5. Writes a JSON manifest with run paths and temporal windows.

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
    │   └── csf.nii.gz
    └── manifest.json
```

The manifest references the raw BOLD files. This avoids duplicating large
4D files. The training dataset should use each manifest entry as one temporal
window rather than treating the entire subject as one sample.

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
  --epochs 1 `
  --repeats 1
```

This intentionally repeats one temporal window. It checks data loading,
spatial dimensions, model forward/backward execution, and loss computation.
It is not a scientific training configuration and does not assess
generalization.

The label groups are explicit in `preprocess.py` and should be reviewed if a
different segmentation convention is used.
