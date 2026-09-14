"""Structural MRI preprocessing and tissue mask alignment pipeline for fMRI denoising."""

import subprocess
from pathlib import Path

import dicom2nifti
import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img


def run_synthseg(t1_path: Path, synthseg_out_path: Path) -> None:
    """Execute the SynthSeg CLI tool to segment a T1 structural scan.

    Args:
        t1_path: Path to the input T1 NIfTI file.
        synthseg_out_path: Path where the segmented NIfTI output will be saved.
    """
    print(f"Running SynthSeg on {t1_path.name}...")
    subprocess.run(
        [
            "python", "-m", "synthseg.predict",
            "--i", str(t1_path),
            "--o", str(synthseg_out_path)
        ],
        check=True
    )

def extract_and_align_masks(
        synthseg_out_path: Path, bold_path: Path, subject_folder: Path
) -> None:
    """Isolate GM, WM, and CSF mass and align them to the BOLD tensor grid.

    Args:
        synthseg_out_path: Path to the SynthSeg segmentation output.
        bold_path: Path to the target BOLD NIfTI scan for spatial reference.
        subject_folder: Output directory to save the aligned binary masks.
    """
    print("Isolating tissues and aligning to BOLD grid...")
    
    seg_img = nib.load(str(synthseg_out_path))
    seg_data = seg_img.get_fdata()
    bold_img = nib.load(str(bold_path))

    # Standard FreeSurfer integer labels for the required tissue classes
    wm_labels = [2, 41]
    gm_labels = [3, 42, 8, 47, 10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54]
    csf_labels = [4, 43, 14, 15, 24]

    # Map labels to binary masks (1.0 for target tissue, 0.0 for background)
    masks = {
        "wm.nii.gz": np.isin(seg_data, wm_labels).astype(np.float32),
        "gm.nii.gz": np.isin(seg_data, gm_labels).astype(np.float32),
        "csf.nii.gz": np.isin(seg_data, csf_labels).astype(np.float32),
    }

    for mask_name, mask_array in masks.items():
        temp_mask_img = nib.Nifti1Image(mask_array, seg_img.affine, seg_img.header)
        
        # Resample the high-res mask to match the low-res functional BOLD grid.
        aligned_mask_img = resample_to_img(
            source_img=temp_mask_img,
            target_img=bold_img,
            interpolation="nearest"
        )
        
        final_path = subject_folder / mask_name
        nib.save(aligned_mask_img, str(final_path))
        print(f"Saved: {final_path}")

def preprocess_pipeline(subject_folder):
    """Convert DICOMs, segment anatomy, and extract aligned tissue masks for a subject.

    Args:
        subject_folder: Path to the individual subject's data directory.
    """
    print(f"\n--- Processing Subject: {subject_folder.name} ---")
    
    dicom_dir = subject_folder / "structural_dicoms"
    bold_path = subject_folder / "bold.nii.gz"
    t1_nifti_path = subject_folder / "t1_structural.nii.gz"
    synthseg_out_path = subject_folder / "synthseg_output.nii.gz"
    
    # Step 1: Convert raw structural DICOMs to a single NIfTI volume
    if not t1_nifti_path.exists():
        print("Converting structural DICOMs to NIfTI...")
        temp_nifti_dir = subject_folder / "temp_nifti"
        temp_nifti_dir.mkdir(exist_ok=True)
        
        dicom2nifti.convert_directory(
            str(dicom_dir), str(temp_nifti_dir), compression=True, reorient=True
        )
        
        # Rename the default generated file to expected standardized name
        converted_file = next(temp_nifti_dir.glob("*.nii.gz"))
        converted_file.rename(t1_nifti_path)
        temp_nifti_dir.rmdir()
    
    # Step 2: Run robust deep-learning-based segmentation (SynthSeg)
    if not synthseg_out_path.exists():
        run_synthseg(t1_nifti_path, synthseg_out_path)
    
    # Step 3: Extract tissue priors and project them into functional space
    extract_and_align_masks(synthseg_out_path, bold_path, subject_folder)

def main() -> None:
    """Execute the preprocessing pipeline across all subjects in the dataset."""
    data_directory = Path("C:/fMRI Denoising/dataset_dir")
    
    # Scan the dataset directory for subject folders
    subjects = [d for d in data_directory.iterdir() if d.is_dir()]
    
    for subject_folder in subjects:
        preprocess_pipeline(subject_folder)
        
    print("\n[SUCCESS] All anatomical masks generated and aligned. Ready for training.")


if __name__ == "__main__":
    main()