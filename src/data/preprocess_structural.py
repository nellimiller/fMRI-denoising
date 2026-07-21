import os
import subprocess
import numpy as np
import nibabel as nib
import dicom2nifti
from nilearn.image import resample_to_img

def run_synthseg(t1_path, synthseg_out_path):
    """
    Calls the SynthSeg command line tool to segment the T1 scan.
    Requires SynthSeg to be installed and accessible via CLI.
    """
    print("Running SynthSeg (this will utilize the GPU if available)...")
    # The standard command to run SynthSeg on a single image
    subprocess.run([
        "python", "-m", "synthseg.predict",
        "--i", t1_path,
        "--o", synthseg_out_path
    ], check=True)

def extract_and_align_masks(synthseg_out_path, bold_path, subject_folder):
    """
    Isolates specific integer labels into binary GM, WM, and CSF masks,
    then aligns them exactly to the preprocessed BOLD tensor.
    """
    print("Isolating tissues and aligning to BOLD grid...")
    
    # Load SynthSeg prediction and the target BOLD image
    seg_img = nib.load(synthseg_out_path)
    seg_data = seg_img.get_fdata()
    bold_img = nib.load(bold_path)

    # Standard FreeSurfer integer labels for the different tissues
    wm_labels = [2, 41]  # Left and Right Cerebral White Matter
    
    gm_labels = [
        3, 42,           # Cerebral Cortex
        8, 47,           # Cerebellar Cortex
        10, 49,          # Thalamus
        11, 50,          # Caudate
        12, 51,          # Putamen
        13, 52,          # Pallidum
        17, 53,          # Hippocampus
        18, 54           # Amygdala
    ]
    
    csf_labels = [
        4, 43,           # Lateral Ventricles
        14, 15,          # 3rd and 4th Ventricles
        24               # Extracerebral CSF
    ]

    # Create binary arrays based on the labels
    masks = {
        'wm.nii.gz': np.isin(seg_data, wm_labels).astype(np.float32),
        'gm.nii.gz': np.isin(seg_data, gm_labels).astype(np.float32),
        'csf.nii.gz': np.isin(seg_data, csf_labels).astype(np.float32)
    }

    # Save, resample, and overwrite each mask
    for mask_name, mask_array in masks.items():
        # Create a temporary NIfTI for the high-res binary mask
        temp_mask_img = nib.Nifti1Image(mask_array, seg_img.affine, seg_img.header)
        
        # Warp the high-res mask to match the BOLD grid exactly.
        # 'nearest' interpolation is critical to keep the mask strictly binary (0.0 or 1.0)
        aligned_mask_img = resample_to_img(
            source_img=temp_mask_img,
            target_img=bold_img,
            interpolation='nearest'
        )
        
        # Save the final aligned mask ready for dataset.py
        final_path = os.path.join(subject_folder, mask_name)
        nib.save(aligned_mask_img, final_path)
        print(f"Saved: {final_path}")

def preprocess_pipeline(subject_folder):
    print(f"\n--- Processing Subject: {subject_folder} ---")
    
    dicom_dir = os.path.join(subject_folder, "structural_dicoms")
    bold_path = os.path.join(subject_folder, "bold.nii.gz")
    t1_nifti_path = os.path.join(subject_folder, "t1_structural.nii.gz")
    synthseg_out_path = os.path.join(subject_folder, "synthseg_output.nii.gz")
    
    # 1. Convert structural DICOM to NIfTI
    if not os.path.exists(t1_nifti_path):
        print("Converting structural DICOMs to NIfTI...")
        temp_nifti_dir = os.path.join(subject_folder, "temp_nifti")
        os.makedirs(temp_nifti_dir, exist_ok=True)
        
        dicom2nifti.convert_directory(dicom_dir, temp_nifti_dir, compression=True, reorient=True)
        
        # Rename the resulting file to the standardized name
        converted_file = [f for f in os.listdir(temp_nifti_dir) if f.endswith('.nii.gz')][0]
        os.rename(os.path.join(temp_nifti_dir, converted_file), t1_nifti_path)
        os.rmdir(temp_nifti_dir)
    
    # 2. Run SynthSeg
    if not os.path.exists(synthseg_out_path):
        run_synthseg(t1_nifti_path, synthseg_out_path)
    
    # 3. Extract masks and align to the BOLD tensor
    extract_and_align_masks(synthseg_out_path, bold_path, subject_folder)

def main():
    # Target your dataset directory directly
    DATA_DIRECTORY = "C:/fMRI Denoising/dataset_dir"
    
    # Iterate through all subject folders
    subjects = [d for d in os.listdir(DATA_DIRECTORY) if os.path.isdir(os.path.join(DATA_DIRECTORY, d))]
    
    for sub in subjects:
        subject_folder = os.path.join(DATA_DIRECTORY, sub)
        preprocess_pipeline(subject_folder)
        
    print("\n[SUCCESS] All anatomical masks generated and aligned. Ready for training.")

if __name__ == "__main__":
    main()