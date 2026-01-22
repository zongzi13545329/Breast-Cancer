"""
Test script to visualize DICOM images before conversion
"""

import os
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_dicom_before_conversion(
    clinical_csv: str,
    metadata_csv: str,
    image_root: str,
    num_samples: int = 3,
    save_dir: str = "./dicom_visualization"
):
    """
    Visualize DICOM images before any conversion/preprocessing.
    
    Args:
        clinical_csv: Path to clinical CSV
        metadata_csv: Path to metadata CSV
        image_root: Root directory for DICOM files
        num_samples: Number of random samples to visualize
        save_dir: Directory to save visualization results
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    print("Loading data files...")
    clinical_df = pd.read_csv(clinical_csv, low_memory=False)
    metadata_df = pd.read_csv(metadata_csv, low_memory=False)
    
    # Get random exams
    unique_exams = clinical_df['acc_anon'].unique()
    np.random.seed(42)
    sample_exams = np.random.choice(unique_exams, size=min(num_samples, len(unique_exams)), replace=False)
    
    print(f"\nVisualizing {len(sample_exams)} random exams...")
    
    for exam_idx, exam_id in enumerate(sample_exams):
        print(f"\n{'='*70}")
        print(f"Exam {exam_idx + 1}/{len(sample_exams)}: {exam_id}")
        print('='*70)
        
        # Get metadata for this exam
        exam_metadata = metadata_df[metadata_df['acc_anon'] == exam_id]
        
        # Filter for 2D images
        exam_metadata = exam_metadata[exam_metadata['FinalImageType'].isin(['2D', 'C-view'])]
        
        # Prefer 2D over C-view
        if len(exam_metadata[exam_metadata['FinalImageType'] == '2D']) > 0:
            exam_metadata = exam_metadata[exam_metadata['FinalImageType'] == '2D']
        
        # Get clinical info
        exam_clinical = clinical_df[clinical_df['acc_anon'] == exam_id].iloc[0]
        
        print(f"Patient ID: {exam_clinical.get('empi_anon', 'N/A')}")
        print(f"Study Date: {exam_clinical.get('study_date_anon', 'N/A')}")
        if 'new_label' in exam_clinical:
            label_names = ['Low Risk', 'Medium Risk', 'High Risk']
            label = int(exam_clinical['new_label'])
            print(f"Label: {label} ({label_names[label]})")
        print(f"BI-RADS: {exam_clinical.get('asses', 'N/A')}")
        print(f"Density: {exam_clinical.get('tissueden', 'N/A')}")
        print(f"Race: {exam_clinical.get('RACE_DESC', 'N/A')}")
        print(f"Age: {exam_clinical.get('age', 'N/A')}")
        
        # Define views
        views = [
            ('L', 'CC', 'LCC'),
            ('R', 'CC', 'RCC'),
            ('L', 'MLO', 'LMLO'),
            ('R', 'MLO', 'RMLO')
        ]
        
        # Create figure
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Exam {exam_id} - Before and After Preprocessing', fontsize=16, y=0.98)
        
        for view_idx, (laterality, view_position, view_name) in enumerate(views):
            # Find matching image
            view_metadata = exam_metadata[
                (exam_metadata['ImageLateralityFinal'] == laterality) &
                (exam_metadata['ViewPosition'] == view_position)
            ]
            
            if len(view_metadata) == 0:
                # No image found
                axes[0, view_idx].text(0.5, 0.5, f'No {view_name} found', 
                                       ha='center', va='center', fontsize=12)
                axes[0, view_idx].axis('off')
                axes[1, view_idx].axis('off')
                continue
            
            # Get file path
            if len(view_metadata) > 1:
                non_special = view_metadata[view_metadata['spot_mag'] == '0']
                if len(non_special) > 0:
                    view_metadata = non_special
                view_metadata = view_metadata.iloc[0:1]
            
            row = view_metadata.iloc[0]
            
            # Construct file path
            if 'anon_dicom_path' in row and not pd.isna(row['anon_dicom_path']):
                file_path = os.path.join(image_root, row['anon_dicom_path'])
            else:
                axes[0, view_idx].text(0.5, 0.5, f'{view_name}: No path', 
                                       ha='center', va='center', fontsize=12)
                axes[0, view_idx].axis('off')
                axes[1, view_idx].axis('off')
                continue
            
            print(f"\n{view_name}:")
            print(f"  File: {file_path}")
            print(f"  Exists: {os.path.exists(file_path)}")
            
            if not os.path.exists(file_path):
                axes[0, view_idx].text(0.5, 0.5, f'{view_name}: File not found', 
                                       ha='center', va='center', fontsize=12)
                axes[0, view_idx].axis('off')
                axes[1, view_idx].axis('off')
                continue
            
            try:
                # Load DICOM
                dcm = pydicom.dcmread(file_path)
                img_original = dcm.pixel_array.astype(np.float32)
                
                print(f"  Original shape: {img_original.shape}")
                print(f"  Original dtype: {img_original.dtype}")
                print(f"  Original range: [{img_original.min():.1f}, {img_original.max():.1f}]")
                print(f"  PhotometricInterpretation: {getattr(dcm, 'PhotometricInterpretation', 'N/A')}")
                print(f"  PatientOrientation: {getattr(dcm, 'PatientOrientation', 'N/A')}")
                
                # Plot original image (before flipping)
                axes[0, view_idx].imshow(img_original, cmap='gray', aspect='auto')
                axes[0, view_idx].set_title(f'{view_name} - Original\n{img_original.shape}', fontsize=10)
                axes[0, view_idx].axis('off')
                
                # Apply flipping logic (from your dataset code)
                img_processed = img_original.copy()
                
                # Determine if should flip
                flip_horz = should_flip_horizontal(dcm, laterality, view_position)
                
                if flip_horz:
                    img_processed = np.fliplr(img_processed)
                    print(f"  Applied horizontal flip: YES")
                else:
                    print(f"  Applied horizontal flip: NO")
                
                # Plot processed image (after flipping)
                axes[1, view_idx].imshow(img_processed, cmap='gray', aspect='auto')
                axes[1, view_idx].set_title(f'{view_name} - After Flip\n{img_processed.shape}', fontsize=10)
                axes[1, view_idx].axis('off')
                
            except Exception as e:
                print(f"  Error: {e}")
                axes[0, view_idx].text(0.5, 0.5, f'{view_name}: Error loading', 
                                       ha='center', va='center', fontsize=12)
                axes[0, view_idx].axis('off')
                axes[1, view_idx].axis('off')
        
        # Adjust layout and save
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'exam_{exam_idx+1}_{exam_id}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to: {save_path}")
        plt.close()
    
    print(f"\n{'='*70}")
    print(f"✓ Visualization complete! Saved to: {save_dir}")
    print('='*70)


def should_flip_horizontal(dcm: pydicom.Dataset, laterality: str, view_position: str) -> bool:
    """
    Determine if image should be flipped horizontally.
    (Same logic as in your dataset code)
    """
    try:
        orientation = getattr(dcm, 'PatientOrientation', None)
        
        if orientation:
            # CC view
            if view_position == 'CC':
                if orientation[0] == 'P':
                    return True
            # MLO or ML views
            elif view_position in ['MLO', 'ML']:
                if orientation[0] == 'P':
                    return True
        else:
            # If no orientation tag, flip right breast images
            if laterality == 'R':
                return True
        
        return False
    
    except Exception:
        # Default: flip right breast
        return laterality == 'R'


def visualize_single_exam(
    exam_id: str,
    clinical_csv: str,
    metadata_csv: str,
    image_root: str,
    save_path: str = None
):
    """
    Visualize a specific exam by ID.
    
    Args:
        exam_id: Exam accession number
        clinical_csv: Path to clinical CSV
        metadata_csv: Path to metadata CSV
        image_root: Root directory for DICOM files
        save_path: Path to save the figure (optional)
    """
    # Load data
    clinical_df = pd.read_csv(clinical_csv, low_memory=False)
    metadata_df = pd.read_csv(metadata_csv, low_memory=False)
    
    # Check if exam exists
    if exam_id not in clinical_df['acc_anon'].values:
        print(f"Error: Exam {exam_id} not found in clinical data!")
        return
    
    print(f"Visualizing exam: {exam_id}")
    
    # Get metadata and clinical info
    exam_metadata = metadata_df[metadata_df['acc_anon'] == exam_id]
    exam_metadata = exam_metadata[exam_metadata['FinalImageType'].isin(['2D', 'C-view'])]
    
    if len(exam_metadata[exam_metadata['FinalImageType'] == '2D']) > 0:
        exam_metadata = exam_metadata[exam_metadata['FinalImageType'] == '2D']
    
    exam_clinical = clinical_df[clinical_df['acc_anon'] == exam_id].iloc[0]
    
    # Print info
    print(f"Patient: {exam_clinical.get('empi_anon', 'N/A')}")
    print(f"Date: {exam_clinical.get('study_date_anon', 'N/A')}")
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Exam {exam_id}', fontsize=16, y=0.98)
    
    views = [
        ('L', 'CC', 'LCC'),
        ('R', 'CC', 'RCC'),
        ('L', 'MLO', 'LMLO'),
        ('R', 'MLO', 'RMLO')
    ]
    
    for view_idx, (laterality, view_position, view_name) in enumerate(views):
        view_metadata = exam_metadata[
            (exam_metadata['ImageLateralityFinal'] == laterality) &
            (exam_metadata['ViewPosition'] == view_position)
        ]
        
        if len(view_metadata) == 0:
            axes[0, view_idx].text(0.5, 0.5, f'No {view_name}', 
                                   ha='center', va='center')
            axes[0, view_idx].axis('off')
            axes[1, view_idx].axis('off')
            continue
        
        if len(view_metadata) > 1:
            non_special = view_metadata[view_metadata['spot_mag'] == '0']
            if len(non_special) > 0:
                view_metadata = non_special
            view_metadata = view_metadata.iloc[0:1]
        
        row = view_metadata.iloc[0]
        
        if 'anon_dicom_path' in row and not pd.isna(row['anon_dicom_path']):
            file_path = os.path.join(image_root, row['anon_dicom_path'])
        else:
            axes[0, view_idx].text(0.5, 0.5, f'{view_name}: No path', 
                                   ha='center', va='center')
            axes[0, view_idx].axis('off')
            axes[1, view_idx].axis('off')
            continue
        
        if not os.path.exists(file_path):
            axes[0, view_idx].text(0.5, 0.5, f'{view_name}: Not found', 
                                   ha='center', va='center')
            axes[0, view_idx].axis('off')
            axes[1, view_idx].axis('off')
            continue
        
        try:
            dcm = pydicom.dcmread(file_path)
            img_original = dcm.pixel_array.astype(np.float32)
            
            # Original
            axes[0, view_idx].imshow(img_original, cmap='gray', aspect='auto')
            axes[0, view_idx].set_title(f'{view_name} - Original', fontsize=10)
            axes[0, view_idx].axis('off')
            
            # After flip
            img_processed = img_original.copy()
            if should_flip_horizontal(dcm, laterality, view_position):
                img_processed = np.fliplr(img_processed)
            
            axes[1, view_idx].imshow(img_processed, cmap='gray', aspect='auto')
            axes[1, view_idx].set_title(f'{view_name} - Processed', fontsize=10)
            axes[1, view_idx].axis('off')
            
        except Exception as e:
            print(f"Error loading {view_name}: {e}")
            axes[0, view_idx].axis('off')
            axes[1, view_idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == '__main__':
    # Configuration
    CLINICAL_CSV = "/projects/standard/lin01231/song0760/embed_recall_reduction/EMBED_OpenData_clinical_relabeled.csv"
    METADATA_CSV = "/projects/standard/lin01231/public/datasets/embed/tables/EMBED_OpenData_metadata.csv"
    IMAGE_ROOT = "/projects/standard/lin01231/public/datasets/embed/"
    
    print("="*70)
    print("DICOM Visualization Test")
    print("="*70)
    
    # Option 1: Visualize random samples
    visualize_dicom_before_conversion(
        clinical_csv=CLINICAL_CSV,
        metadata_csv=METADATA_CSV,
        image_root=IMAGE_ROOT,
        num_samples=5,  # Visualize 5 random exams
        save_dir="./dicom_viz_test"
    )
    
    # Option 2: Visualize a specific exam (uncomment and provide exam_id)
    # visualize_single_exam(
    #     exam_id="YOUR_EXAM_ID_HERE",
    #     clinical_csv=CLINICAL_CSV,
    #     metadata_csv=METADATA_CSV,
    #     image_root=IMAGE_ROOT,
    #     save_path="./specific_exam.png"
    # )