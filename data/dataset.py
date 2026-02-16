"""
EMBED Exam-Level Dataset for Recall Prediction (NYU-Compatible) - OPTIMIZED
===========================================================================
Task: Predict whether left/right breast needs recall based on screening mammogram
Labels: Binary per breast (0=no recall, 1=need recall)
Compatible with NYU pretrained weights for initialization

Key Features:
1. ✅ Separate label and image processing, then merge
2. ✅ Breast-specific labels from finding-level data
3. ✅ 3-class to binary conversion (0→no recall, 1/2→need recall)
4. ✅ NYU-style preprocessing with caching
5. ✅ 4-view structure: L-CC, L-MLO, R-CC, R-MLO

Author: Yiran (Optimized)
Date: 2025
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from collections import Counter
from typing import Optional, Dict, List, Tuple, Literal
import warnings
import cv2

warnings.filterwarnings('ignore')


# ============================================================================
# NYU-Style Preprocessing Functions
# ============================================================================

def crop_mammogram_nyu_style(image: np.ndarray, margin: int = 20) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop mammogram around breast tissue (NYU-style)."""
    if image.max() > 255:
        img_normalized = (image / image.max() * 255).astype(np.uint8)
    else:
        img_normalized = image.astype(np.uint8)
    
    threshold = max(10, int(img_normalized.mean() * 0.1))
    _, binary = cv2.threshold(img_normalized, threshold, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    if num_labels <= 1:
        H, W = image.shape
        return image, (0, H, 0, W)
    
    largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    x = stats[largest_component, cv2.CC_STAT_LEFT]
    y = stats[largest_component, cv2.CC_STAT_TOP]
    w = stats[largest_component, cv2.CC_STAT_WIDTH]
    h = stats[largest_component, cv2.CC_STAT_HEIGHT]
    
    H, W = image.shape
    top = max(0, y - margin)
    bottom = min(H, y + h + margin)
    left = max(0, x - margin)
    right = min(W, x + w + margin)
    
    cropped = image[top:bottom, left:right]
    window_location = (top, bottom, left, right)
    
    return cropped, window_location


def calculate_optimal_center(image: np.ndarray, target_size: Tuple[int, int]) -> Tuple[int, int]:
    """Calculate optimal center point for cropping window (NYU-style)."""
    H, W = image.shape
    target_H, target_W = target_size
    
    if H < target_H or W < target_W:
        center_y = max(target_H // 2, H // 2)
        center_x = max(target_W // 2, W // 2)
        return (center_y, center_x)
    
    threshold = max(10, int(image.mean() * 0.1)) if image.max() > 0 else 10
    tissue_mask = image > threshold
    
    if tissue_mask.any():
        y_coords, x_coords = np.where(tissue_mask)
        center_y = int(y_coords.mean())
        center_x = int(x_coords.mean())
    else:
        center_y = H // 2
        center_x = W // 2
    
    center_y = max(target_H // 2, min(H - target_H // 2, center_y))
    center_x = max(target_W // 2, min(W - target_W // 2, center_x))
    
    return (center_y, center_x)


def extract_crop_with_center(
    image: np.ndarray,
    center: Tuple[int, int],
    target_size: Tuple[int, int],
    padding_value: float = 0.0
) -> np.ndarray:
    """Extract crop from image using center point, with padding if needed."""
    H, W = image.shape
    target_H, target_W = target_size
    center_y, center_x = center
    
    half_H = target_H // 2
    half_W = target_W // 2
    
    top = center_y - half_H
    bottom = top + target_H
    left = center_x - half_W
    right = left + target_W
    
    pad_top = max(0, -top)
    pad_bottom = max(0, bottom - H)
    pad_left = max(0, -left)
    pad_right = max(0, right - W)
    
    top = max(0, top)
    bottom = min(H, bottom)
    left = max(0, left)
    right = min(W, right)
    
    cropped = image[top:bottom, left:right]
    
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        cropped = np.pad(
            cropped,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=padding_value
        )
    
    return cropped


# ============================================================================
# Balanced Batch Sampler
# ============================================================================

class BalancedBatchSampler(Sampler):
    """Ensures each batch has a target ratio of positive samples."""
    
    def __init__(
        self, 
        labels: np.ndarray, 
        batch_size: int, 
        positive_ratio: float = 0.3,
        drop_last: bool = False
    ):
        self.labels = labels
        self.batch_size = batch_size
        self.positive_ratio = positive_ratio
        self.drop_last = drop_last
        
        self.positive_idx = np.where(labels == 1)[0]
        self.negative_idx = np.where(labels == 0)[0]
        
        self.n_pos_per_batch = max(1, int(batch_size * positive_ratio))
        self.n_neg_per_batch = batch_size - self.n_pos_per_batch
        
        self.num_batches = len(self.positive_idx) // self.n_pos_per_batch
        
        print(f"\n✓ BalancedBatchSampler:")
        print(f"  Positive samples: {len(self.positive_idx):,}")
        print(f"  Negative samples: {len(self.negative_idx):,}")
        print(f"  Per batch: {self.n_pos_per_batch} pos + {self.n_neg_per_batch} neg")
        print(f"  Total batches: {self.num_batches}")
    
    def __iter__(self):
        pos_shuffled = np.random.permutation(self.positive_idx)
        
        for i in range(self.num_batches):
            pos_start = i * self.n_pos_per_batch
            pos_end = pos_start + self.n_pos_per_batch
            pos_batch = pos_shuffled[pos_start:pos_end]
            
            neg_batch = np.random.choice(
                self.negative_idx, 
                size=self.n_neg_per_batch, 
                replace=False
            )
            
            batch_indices = np.concatenate([pos_batch, neg_batch])
            np.random.shuffle(batch_indices)
            
            yield batch_indices.tolist()
    
    def __len__(self):
        return self.num_batches


# ============================================================================
# Main Dataset Class
# ============================================================================

class EMBEDRecallDataset(Dataset):
    """
    EMBED Dataset for Recall Prediction (NYU-Compatible) - OPTIMIZED.
    """
    
    def __init__(
        self,
        clinical_csv: str,
        metadata_csv: str,
        mode: Literal['train', 'val', 'test'] = 'train',
        transform=None,
        prior_time_window: Tuple[int, int] = (365, 1095),
        image_size: Tuple[int, int] = (2944, 1920),
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        verbose: bool = True,
        apply_nyu_preprocessing: bool = True,
        patient_filter: Optional[List[str]] = None  # ✅ 新增：直接传递患者列表
    ):
        self.clinical_csv = clinical_csv
        self.metadata_csv = metadata_csv
        self.mode = mode
        self.transform = transform
        self.prior_time_window = prior_time_window
        self.image_size = image_size
        self.cache_dir = cache_dir or os.path.dirname(clinical_csv)
        self.use_cache = use_cache
        self.verbose = verbose
        self.apply_nyu_preprocessing = apply_nyu_preprocessing
        self.patient_filter = patient_filter  # ✅ 保存患者过滤列表
        
        self.preprocessing_cache_dir = os.path.join(
            self.cache_dir, 
            f'preprocessed_{self.image_size[0]}x{self.image_size[1]}'
        )
        
        if verbose:
            print("\n" + "="*70)
            print("EMBED Recall Prediction Dataset (NYU-Compatible) - OPTIMIZED")
            print("="*70)
            print(f"✓ Task: Breast-level recall prediction")
            print(f"✓ Processing: Labels first → Images second → Merge")
            print(f"✓ Image size: {image_size}")
            print(f"✓ NYU preprocessing: {apply_nyu_preprocessing}")
            print(f"✓ Structure: L-CC, L-MLO, R-CC, R-MLO")
            if patient_filter is not None:
                print(f"✓ Patient filter: {len(patient_filter):,} patients")
        
        self._load_data()
        self._build_exam_samples()
        
        if self.apply_nyu_preprocessing:
            self._precompute_preprocessing()
        
        if self.verbose:
            self._print_statistics()
    
    def _load_data(self):
        """Load clinical and metadata files."""
        if self.verbose:
            print("\nLoading data...")
        
        dtype_dict = {'acc_anon': str, 'empi_anon': str}
        
        # Load clinical data (finding-level) - relatively small
        self.clinical_df = pd.read_csv(
            self.clinical_csv, 
            low_memory=False,
            dtype=dtype_dict
        )
        
        # Filter clinical by patient list
        if self.patient_filter is not None:
            patient_set = set(self.patient_filter)
            original_size = len(self.clinical_df)
            self.clinical_df = self.clinical_df[
                self.clinical_df['empi_anon'].isin(patient_set)
            ].copy()
            
            if self.verbose:
                print(f"  Filtered: {original_size:,} → {len(self.clinical_df):,} records")
                print(f"  Patients: {self.clinical_df['empi_anon'].nunique():,}")
        
        if 'new_label' not in self.clinical_df.columns:
            raise ValueError("'new_label' column not found in clinical CSV!")
        
        self.clinical_df['study_date_anon'] = pd.to_datetime(
            self.clinical_df['study_date_anon'], errors='coerce'
        )
        
        # =====================================================================
        # Load metadata in CHUNKS to avoid OOM
        # =====================================================================
        if self.verbose:
            print(f"  Loading metadata (chunked)...")
        
        if self.patient_filter is not None:
            # Chunked reading: only keep rows matching patient_filter
            patient_set = set(self.patient_filter)
            chunks = []
            chunk_size = 100_000
            
            for chunk in pd.read_csv(
                self.metadata_csv,
                low_memory=False,
                dtype=dtype_dict,
                chunksize=chunk_size
            ):
                filtered_chunk = chunk[chunk['empi_anon'].isin(patient_set)]
                if len(filtered_chunk) > 0:
                    chunks.append(filtered_chunk)
            
            if chunks:
                self.metadata_df = pd.concat(chunks, ignore_index=True)
            else:
                self.metadata_df = pd.DataFrame()
            
            if self.verbose:
                print(f"  Metadata filtered (chunked): {len(self.metadata_df):,} records")
        else:
            # No filter: load full (only when using full dataset)
            self.metadata_df = pd.read_csv(
                self.metadata_csv, 
                low_memory=False,
                dtype=dtype_dict
            )
        
        self.metadata_df['study_date_anon'] = pd.to_datetime(
            self.metadata_df.get('study_date_anon', pd.Series()), errors='coerce'
        )
        
        # Filter for 2D CC/MLO views
        if 'FinalImageType' in self.metadata_df.columns:
            self.metadata_df = self.metadata_df[
                self.metadata_df['FinalImageType'] == '2D'
            ].copy()
        
        if 'ViewPosition' in self.metadata_df.columns:
            self.metadata_df = self.metadata_df[
                self.metadata_df['ViewPosition'].isin(['CC', 'MLO'])
            ].copy()
        
        if 'png_path' not in self.metadata_df.columns:
            raise ValueError("Metadata must contain 'png_path'!")
        
        if 'png_exists' in self.metadata_df.columns:
            self.metadata_df = self.metadata_df[
                self.metadata_df['png_exists'] == True
            ].copy()
        
        if self.verbose:
            print(f"  Clinical: {len(self.clinical_df):,} finding-level records")
            print(f"  Clinical: {self.clinical_df['acc_anon'].nunique():,} unique exams")
            print(f"  Metadata: {len(self.metadata_df):,} 2D CC/MLO images")
            print(f"  Metadata: {self.metadata_df['acc_anon'].nunique():,} unique exams")
    
    # ========================================================================
    # ✅ OPTIMIZED: Step 1 - Generate Exam-Level Labels
    # ========================================================================
    
    def _generate_all_exam_labels(self) -> pd.DataFrame:
        """
        Generate exam-level labels from finding-level clinical data.
        
        Rules:
        1. For each exam, iterate through all findings
        2. Take the most severe label for each breast
        3. Convert 3-class to binary: 0→no recall, 1/2→need recall
        4. Take the most severe BI-RADS assessment across all findings
        
        Side handling:
        - 'L': Update left only
        - 'R': Update right only
        - 'B': Update both left and right
        - NaN or other: Update both left and right (conservative approach)
        
        Returns:
            DataFrame with columns: acc_anon, empi_anon, left_malignant, right_malignant, etc.
        """
        if self.verbose:
            print("\n[Step 1] Generating exam-level labels from findings...")
        
        # BI-RADS severity priority for taking the worst assessment
        birads_priority = {
            'A': 0, 'X': 0,   # Assessment incomplete
            'N': 1,            # Negative
            'B': 2,            # Benign
            'P': 3,            # Probably benign
            'S': 4, 'M': 4, 'K': 4  # Suspicious / Malignant / Known
        }
        
        exam_labels_list = []
        
        # Track side distribution for debugging
        side_counter = Counter()
        
        # Group by exam
        grouped = self.clinical_df.groupby('acc_anon')
        
        for exam_id, exam_findings in grouped:
            # Get patient ID and date (same for all findings in an exam)
            first_finding = exam_findings.iloc[0]
            patient_id = first_finding['empi_anon']
            study_date = first_finding['study_date_anon']
            
            # Initialize with lowest severity
            left_max_label = 0
            right_max_label = 0
            
            # Track worst BI-RADS across all findings
            max_birads_score = -1
            max_birads_raw = np.nan
            
            # Iterate through all findings for this exam
            for _, finding in exam_findings.iterrows():
                side = finding.get('side', np.nan)
                label = finding.get('new_label', np.nan)
                
                # Track worst BI-RADS assessment
                asses_val = finding.get('asses', np.nan)
                if not pd.isna(asses_val):
                    score = birads_priority.get(str(asses_val).strip().upper(), 0)
                    if score > max_birads_score:
                        max_birads_score = score
                        max_birads_raw = asses_val
                
                # Skip findings without a label
                if pd.isna(label):
                    continue
                
                label = int(label)
                
                # Track side distribution
                side_counter[str(side)] = side_counter.get(str(side), 0) + 1
                
                # Handle all side cases
                if pd.isna(side):
                    # Side is NaN → apply to both breasts (conservative)
                    left_max_label = max(left_max_label, label)
                    right_max_label = max(right_max_label, label)
                elif side == 'L':
                    # Left breast only
                    left_max_label = max(left_max_label, label)
                elif side == 'R':
                    # Right breast only
                    right_max_label = max(right_max_label, label)
                elif side == 'B':
                    # Bilateral → both breasts
                    left_max_label = max(left_max_label, label)
                    right_max_label = max(right_max_label, label)
                else:
                    # Unknown side value → apply to both breasts (conservative)
                    if self.verbose:
                        print(f"  ⚠️  Unknown side value: '{side}' for exam {exam_id}")
                    left_max_label = max(left_max_label, label)
                    right_max_label = max(right_max_label, label)
            
            # Convert 3-class to binary: label >= 1 means need recall
            left_recall = 1 if left_max_label >= 1 else 0
            right_recall = 1 if right_max_label >= 1 else 0
            
            exam_labels_list.append({
                'acc_anon': str(exam_id),
                'empi_anon': str(patient_id),
                'study_date_anon': study_date,
                
                # NYU-compatible binary labels
                'left_benign': 1 - left_recall,
                'left_malignant': left_recall,
                'right_benign': 1 - right_recall,
                'right_malignant': right_recall,
                
                # Raw 3-class labels for analysis
                'left_label_3class': left_max_label,
                'right_label_3class': right_max_label,
                
                # Metadata - use worst BI-RADS across all findings
                'asses': max_birads_raw,
                'ETHNICITY_DESC': first_finding.get('ETHNICITY_DESC', np.nan),
                'RACE_DESC': first_finding.get('RACE_DESC', np.nan),
                'age': first_finding.get('age', np.nan),
            })
        
        exam_labels_df = pd.DataFrame(exam_labels_list)
        
        if self.verbose:
            print(f"  ✓ Generated labels for {len(exam_labels_df):,} exams")
            
            # Show side distribution
            print(f"\n  Side distribution across all findings:")
            for side_val, count in sorted(side_counter.items()):
                print(f"    '{side_val}': {count:,}")
            
            # Show label distribution
            left_recall_count = exam_labels_df['left_malignant'].sum()
            right_recall_count = exam_labels_df['right_malignant'].sum()
            total = len(exam_labels_df)
            
            print(f"\n  Recall labels:")
            print(f"    Left breast needs recall: {left_recall_count:,} ({left_recall_count/total*100:.1f}%)")
            print(f"    Right breast needs recall: {right_recall_count:,} ({right_recall_count/total*100:.1f}%)")
            
            # Show BI-RADS distribution
            birads_dist = exam_labels_df['asses'].value_counts(dropna=False)
            print(f"\n  BI-RADS (worst per exam) distribution:")
            for val, count in birads_dist.items():
                print(f"    '{val}': {count:,} ({count/total*100:.1f}%)")
        
        return exam_labels_df
    
    # ========================================================================
    # ✅ OPTIMIZED: Step 2 - Organize Image Paths
    # ========================================================================
    
    def _organize_exam_images(self) -> pd.DataFrame:
        """
        Organize image paths from metadata by exam and view position.
        
        For each exam:
        1. Group images by laterality (L/R) and view (CC/MLO)
        2. Select one image per view (prefer standard FFDM over spot/mag)
        3. Store as dictionary: {L-CC: view_info, L-MLO: view_info, ...}
        
        Returns:
            DataFrame with columns: acc_anon, empi_anon, L-CC_view, L-MLO_view, etc.
        """
        if self.verbose:
            print("\n[Step 2] Organizing image paths from metadata...")
        
        exam_images_list = []
        
        # Group by exam
        grouped = self.metadata_df.groupby('acc_anon')
        
        for exam_id, exam_images in grouped:
            first_image = exam_images.iloc[0]
            patient_id = first_image.get('empi_anon', np.nan)
            study_date = first_image.get('study_date_anon', np.nan)
            
            # Organize images by view
            views_dict = {
                'L-CC': [],
                'L-MLO': [],
                'R-CC': [],
                'R-MLO': []
            }
            
            for _, img_row in exam_images.iterrows():
                laterality = img_row.get('ImageLateralityFinal', np.nan)
                view_pos = img_row.get('ViewPosition', np.nan)
                
                if laterality not in ['L', 'R'] or view_pos not in ['CC', 'MLO']:
                    continue
                
                view_key = f"{laterality}-{view_pos}"
                
                view_info = {
                    'png_path': img_row['png_path'],
                    'view_position': view_pos,
                    'laterality': laterality,
                    'spot_mag': img_row.get('spot_mag', '0'),
                    'density': img_row.get('tissueden', np.nan)
                }
                
                views_dict[view_key].append(view_info)
            
            # Select one image per view (prefer standard FFDM)
            selected_views = {}
            for view_key in ['L-CC', 'L-MLO', 'R-CC', 'R-MLO']:
                if len(views_dict[view_key]) > 0:
                    # Prefer spot_mag == '0' (standard FFDM)
                    standard = [v for v in views_dict[view_key] if v['spot_mag'] == '0']
                    if standard:
                        selected_views[f'{view_key}_view'] = standard[0]
                    else:
                        selected_views[f'{view_key}_view'] = views_dict[view_key][0]
                else:
                    selected_views[f'{view_key}_view'] = None
            
            exam_images_list.append({
                'acc_anon': str(exam_id),
                'empi_anon': str(patient_id),
                'study_date_anon': study_date,
                **selected_views
            })
        
        exam_images_df = pd.DataFrame(exam_images_list)
        
        if self.verbose:
            print(f"  ✓ Organized images for {len(exam_images_df):,} exams")
            
            # Show view availability
            for view_key in ['L-CC', 'L-MLO', 'R-CC', 'R-MLO']:
                col_name = f'{view_key}_view'
                available = exam_images_df[col_name].notna().sum()
                print(f"  {view_key} available: {available:,} ({available/len(exam_images_df)*100:.1f}%)")
        
        return exam_images_df
    
    # ========================================================================
    # ✅ OPTIMIZED: Step 3 - Merge Labels and Images
    # ========================================================================
    
    def _build_exam_samples(self):
        """Build exam-level samples by merging labels and images."""
        # ✅ Generate patient hash for cache validation
        import hashlib
        
        if self.patient_filter is not None:
            # Hash the patient list to create unique cache key
            patient_str = '_'.join(sorted(self.patient_filter))
            patient_hash = hashlib.md5(patient_str.encode()).hexdigest()[:8]
            cache_suffix = f'_patients_{patient_hash}'
        else:
            # Use a marker for full dataset
            patient_hash = 'full'
            cache_suffix = f'_patients_{patient_hash}'
        
        cache_file = os.path.join(
            self.cache_dir,
            f'recall_samples_{self.mode}_'
            f'{self.prior_time_window[0]}_{self.prior_time_window[1]}_'
            f'{self.image_size[0]}x{self.image_size[1]}'
            f'{cache_suffix}.pkl'
        )
        
        if self.use_cache and os.path.exists(cache_file):
            if self.verbose:
                print(f"\nLoading cached samples from {cache_file}")
            self.exam_samples = pd.read_pickle(cache_file)
            
            # ✅ Validate cache: check if patient count matches
            if self.patient_filter is not None:
                unique_patients = self.exam_samples['empi_anon'].nunique()
                expected_patients = len(set(self.patient_filter))
                
                if unique_patients != expected_patients:
                    if self.verbose:
                        print(f"  ⚠️  Cache mismatch: {unique_patients} vs {expected_patients} patients")
                        print(f"  Rebuilding samples...")
                    
                    self.exam_samples = self._create_exam_samples()
                    
                    if self.use_cache:
                        os.makedirs(self.cache_dir, exist_ok=True)
                        self.exam_samples.to_pickle(cache_file)
                        if self.verbose:
                            print(f"  ✓ Cached to: {cache_file}")
            else:
                # For full dataset, just trust the cache
                pass
        else:
            if self.verbose:
                print("\nBuilding exam-level samples...")
            self.exam_samples = self._create_exam_samples()
            
            if self.use_cache:
                os.makedirs(self.cache_dir, exist_ok=True)
                self.exam_samples.to_pickle(cache_file)
                if self.verbose:
                    print(f"  ✓ Cached to: {cache_file}")
        
        self.sample_list = self.exam_samples.to_dict('records')
        
        if self.verbose:
            print(f"  ✓ Total samples: {len(self.sample_list):,}")

    
    def _create_exam_samples(self) -> pd.DataFrame:
        """
        Create exam-level samples with optimized workflow.
        
        Workflow:
        1. Generate exam-level labels from clinical data
        2. Organize image paths from metadata
        3. Merge labels and images
        4. Find prior exams and build final samples
        """
        # Step 1: Generate labels
        exam_labels_df = self._generate_all_exam_labels()
        
        # Step 2: Organize images
        exam_images_df = self._organize_exam_images()
        
        # Step 3: Merge labels and images
        if self.verbose:
            print("\n[Step 3] Merging labels and images...")
        
        merged_exams = exam_labels_df.merge(
            exam_images_df,
            on=['empi_anon', 'acc_anon'],
            how='inner',
            suffixes=('_label', '_image')
        )
        
        # Handle date column conflicts
        if 'study_date_anon_label' in merged_exams.columns:
            merged_exams['study_date_anon'] = merged_exams['study_date_anon_label']
        elif 'study_date_anon_image' in merged_exams.columns:
            merged_exams['study_date_anon'] = merged_exams['study_date_anon_image']
        
        if self.verbose:
            print(f"  Exams with labels: {len(exam_labels_df):,}")
            print(f"  Exams with images: {len(exam_images_df):,}")
            print(f"  ✓ Merged exams: {len(merged_exams):,}")
        
        # Step 4: Build samples with prior exams
        if self.verbose:
            print("\n[Step 4] Building samples with prior exams...")
        
        samples = []
        dropped_incomplete = 0
        
        for _, exam_row in merged_exams.iterrows():
            patient_id = exam_row['empi_anon']
            exam_id = exam_row['acc_anon']
            current_date = exam_row['study_date_anon']
            
            # Extract current views
            current_views = {
                'L-CC': exam_row.get('L-CC_view'),
                'L-MLO': exam_row.get('L-MLO_view'),
                'R-CC': exam_row.get('R-CC_view'),
                'R-MLO': exam_row.get('R-MLO_view')
            }
            
            # Check completeness: need at least one view per laterality
            has_left = current_views['L-CC'] is not None or current_views['L-MLO'] is not None
            has_right = current_views['R-CC'] is not None or current_views['R-MLO'] is not None
            
            if not (has_left and has_right):
                dropped_incomplete += 1
                continue
            
            # Find prior exam within time window
            patient_exams = merged_exams[merged_exams['empi_anon'] == patient_id]
            prior_exam_row = self._find_prior_exam(patient_exams, current_date)
            
            # Extract prior views
            prior_views = {k: None for k in ['L-CC', 'L-MLO', 'R-CC', 'R-MLO']}
            
            if prior_exam_row is not None:
                prior_views = {
                    'L-CC': prior_exam_row.get('L-CC_view'),
                    'L-MLO': prior_exam_row.get('L-MLO_view'),
                    'R-CC': prior_exam_row.get('R-CC_view'),
                    'R-MLO': prior_exam_row.get('R-MLO_view')
                }
            
            # Count valid views
            n_current = sum(1 for v in current_views.values() if v is not None)
            n_prior = sum(1 for v in prior_views.values() if v is not None)
            
            sample = {
                'exam_id': f"{patient_id}_{exam_id}",
                'empi_anon': str(patient_id),
                'acc_anon': str(exam_id),
                'current_views': current_views,
                'prior_views': prior_views,
                'num_current_views': n_current,
                'num_prior_views': n_prior,
                
                # Labels
                'left_benign': exam_row['left_benign'],
                'left_malignant': exam_row['left_malignant'],
                'right_benign': exam_row['right_benign'],
                'right_malignant': exam_row['right_malignant'],
                'left_label_3class': exam_row['left_label_3class'],
                'right_label_3class': exam_row['right_label_3class'],
                
                # Metadata
                'birads': exam_row.get('asses', np.nan),
                'race': exam_row.get('RACE_DESC', exam_row.get('ETHNICITY_DESC', np.nan)),
                'age': exam_row.get('age', np.nan),
                'current_date': current_date,
                'prior_date': (
                    prior_exam_row['study_date_anon']
                    if prior_exam_row is not None else None
                ),
            }
            
            samples.append(sample)
        
        if self.verbose:
            print(f"  ✓ Created {len(samples):,} samples")
            print(f"  Dropped {dropped_incomplete:,} exams (incomplete views)")
        
        return pd.DataFrame(samples)
    
    def _find_prior_exam(
        self,
        patient_exams: pd.DataFrame,
        current_date: pd.Timestamp
    ) -> Optional[pd.Series]:
        """Find prior exam within time window."""
        if pd.isna(current_date):
            return None
        
        min_days, max_days = self.prior_time_window
        
        prior_exams = patient_exams[
            (patient_exams['study_date_anon'] < current_date) &
            (patient_exams['study_date_anon'] >= current_date - pd.Timedelta(days=max_days)) &
            (patient_exams['study_date_anon'] <= current_date - pd.Timedelta(days=min_days))
        ]
        
        if len(prior_exams) == 0:
            return None
        
        # Return the most recent prior exam
        most_recent_idx = prior_exams['study_date_anon'].idxmax()
        return prior_exams.loc[most_recent_idx]
    
    # ========================================================================
    # Preprocessing Cache Methods
    # ========================================================================
    
    def _precompute_preprocessing(self):
        """Precompute NYU preprocessing for all images and cache."""
        os.makedirs(self.preprocessing_cache_dir, exist_ok=True)
        
        if self.verbose:
            print(f"\nPrecomputing NYU preprocessing...")
        
        # Collect all unique image paths
        all_image_paths = set()
        for sample in self.sample_list:
            for view_key in ['L-CC', 'L-MLO', 'R-CC', 'R-MLO']:
                view_info = sample['current_views'].get(view_key)
                if view_info is not None and isinstance(view_info, dict):
                    all_image_paths.add(view_info['png_path'])
                
                prior_info = sample['prior_views'].get(view_key)
                if prior_info is not None and isinstance(prior_info, dict):
                    all_image_paths.add(prior_info['png_path'])
        
        if self.verbose:
            print(f"  Total unique images: {len(all_image_paths):,}")
        
        # Check which images need processing
        to_process = []
        for img_path in all_image_paths:
            if pd.isna(img_path) or not os.path.exists(img_path):
                continue
                
            cache_key = self._get_cache_key(img_path)
            cache_path = os.path.join(self.preprocessing_cache_dir, f"{cache_key}.npy")
            if not os.path.exists(cache_path):
                print(f"MISSING CACHE: {img_path}")
                to_process.append(img_path)
        
        if len(to_process) > 0:
            if self.verbose:
                print(f"  Processing {len(to_process):,} new images...")
            
            try:
                from tqdm import tqdm
                use_tqdm = True
            except ImportError:
                use_tqdm = False
            
            if use_tqdm:
                for img_path in tqdm(to_process, desc="  Preprocessing"):
                    self._preprocess_and_cache(img_path)
            else:
                for i, img_path in enumerate(to_process):
                    self._preprocess_and_cache(img_path)
                    if self.verbose and (i + 1) % 100 == 0:
                        print(f"    Processed {i+1}/{len(to_process)} images...")
        else:
            if self.verbose:
                print(f"  ✓ All {len(all_image_paths):,} images already cached!")
    
    def _get_cache_key(self, img_path: str) -> str:
        """Generate unique cache key from image path."""
        import hashlib
        return hashlib.md5(img_path.encode()).hexdigest()
    
    def _preprocess_and_cache(self, img_path: str):
        """Preprocess single image and save to cache."""
        try:
            # Load original image
            img = np.array(Image.open(img_path)).astype(np.float32)
            
            # NYU preprocessing pipeline
            cropped, _ = crop_mammogram_nyu_style(img, margin=20)
            center = calculate_optimal_center(cropped, self.image_size)
            final_crop = extract_crop_with_center(
                cropped, 
                center, 
                self.image_size,
                padding_value=0.0
            )
            
            # Normalize to [0, 1]
            if final_crop.max() > 0:
                final_crop = final_crop / final_crop.max()
            
            # Save to cache
            cache_key = self._get_cache_key(img_path)
            cache_path = os.path.join(self.preprocessing_cache_dir, f"{cache_key}.npy")
            np.save(cache_path, final_crop.astype(np.float32))
            
        except Exception as e:
            if self.verbose:
                print(f"\n    ⚠️  Warning: Failed to preprocess {img_path}")
                print(f"        Error: {e}")
            # Save placeholder to avoid reprocessing
            try:
                cache_key = self._get_cache_key(img_path)
                cache_path = os.path.join(self.preprocessing_cache_dir, f"{cache_key}.npy")
                placeholder = np.zeros(self.image_size, dtype=np.float32)
                np.save(cache_path, placeholder)
            except:
                pass
    
    # ========================================================================
    # Data Loading
    # ========================================================================
    
    def __len__(self) -> int:
        return len(self.sample_list)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get one exam sample with NYU-compatible structure.
        
        Returns:
            Dictionary with:
                - current_views: Dict[str, Tensor] - L-CC, L-MLO, R-CC, R-MLO [1, H, W]
                - prior_views: Same structure
                - current_mask: Dict[str, float] - indicates valid views
                - prior_mask: Same
                - labels: Dict with left_benign, left_malignant, right_benign, right_malignant
                - metadata: race, age, etc.
                - exam_info: exam_id, patient_id
        """
        sample = self.sample_list[idx]
        
        # Load current views
        current_imgs = {}
        current_mask = {}
        
        for view_key in ['L-CC', 'L-MLO', 'R-CC', 'R-MLO']:
            view_info = sample['current_views'][view_key]
            
            if view_info is not None and isinstance(view_info, dict):
                img = self._load_and_process_image(view_info['png_path'])
                current_imgs[view_key] = img
                current_mask[view_key] = 1.0
            else:
                current_imgs[view_key] = self._get_placeholder_tensor()
                current_mask[view_key] = 0.0
        
        # Load prior views
        prior_imgs = {}
        prior_mask = {}
        
        has_prior = sample['num_prior_views'] > 0
        
        for view_key in ['L-CC', 'L-MLO', 'R-CC', 'R-MLO']:
            prior_info = sample['prior_views'][view_key]
            
            if has_prior and prior_info is not None and isinstance(prior_info, dict):
                img = self._load_and_process_image(prior_info['png_path'])
                prior_imgs[view_key] = img
                prior_mask[view_key] = 1.0
            else:
                # Use current as prior if no prior exists
                if not has_prior:
                    prior_imgs[view_key] = current_imgs[view_key].clone()
                    prior_mask[view_key] = current_mask[view_key]
                else:
                    prior_imgs[view_key] = self._get_placeholder_tensor()
                    prior_mask[view_key] = 0.0
        
        result = {
            'current_views': current_imgs,
            'prior_views': prior_imgs,
            'current_mask': current_mask,
            'prior_mask': prior_mask,
            
            'labels': {
                # NYU-compatible binary labels [1] per breast
                'left_benign': torch.tensor([sample['left_benign']], dtype=torch.float32),
                'left_malignant': torch.tensor([sample['left_malignant']], dtype=torch.float32),
                'right_benign': torch.tensor([sample['right_benign']], dtype=torch.float32),
                'right_malignant': torch.tensor([sample['right_malignant']], dtype=torch.float32),
                
                # Raw 3-class labels for analysis
                'left_label_3class': torch.tensor(sample['left_label_3class'], dtype=torch.long),
                'right_label_3class': torch.tensor(sample['right_label_3class'], dtype=torch.long),
                
                'birads': torch.tensor(
                    self._encode_birads(sample['birads']),
                    dtype=torch.long
                ),
            },
            
            'metadata': {
                'race': torch.tensor(
                    self._encode_race(sample['race']),
                    dtype=torch.long
                ),
                'age': torch.tensor(
                    sample['age'] if not pd.isna(sample['age']) else -1.0,
                    dtype=torch.float32
                ),
                'num_current_views': torch.tensor(
                    sample['num_current_views'], dtype=torch.long
                ),
                'num_prior_views': torch.tensor(
                    sample['num_prior_views'], dtype=torch.long
                ),
            },
            
            'exam_info': {
                'exam_id': sample['exam_id'],
                'acc_anon': sample['acc_anon'],
                'patient_id': sample['empi_anon']
            }
        }
        
        return result
    
    def _load_and_process_image(self, png_path: str) -> torch.Tensor:
        """Load and process image (use cache if NYU preprocessing enabled)."""
        if pd.isna(png_path):
            return self._get_placeholder_tensor()
        
        try:
            if self.apply_nyu_preprocessing:
                cache_key = self._get_cache_key(png_path)
                cache_path = os.path.join(
                    self.preprocessing_cache_dir, 
                    f"{cache_key}.npy"
                )
                
                if os.path.exists(cache_path):
                    # npy存在，直接加载，不需要png
                    img = np.load(cache_path)
                elif os.path.exists(png_path):
                    # npy不存在但png在，on-the-fly处理
                    if self.verbose:
                        print(f"\n  ⚠️  Cache miss for {os.path.basename(png_path)}")
                    
                    img = np.array(Image.open(png_path)).astype(np.float32)
                    cropped, _ = crop_mammogram_nyu_style(img, margin=20)
                    center = calculate_optimal_center(cropped, self.image_size)
                    img = extract_crop_with_center(cropped, center, self.image_size)
                    
                    if img.max() > 0:
                        img = img / img.max()
                else:
                    # npy不存在，png也被删了
                    print(f"  ❌ MISSING both npy and png: {png_path}")
                    return self._get_placeholder_tensor()
            else:
                if not os.path.exists(png_path):
                    print(f"  ❌ MISSING png (no preprocessing): {png_path}")
                    return self._get_placeholder_tensor()
                
                img = np.array(Image.open(png_path)).astype(np.float32)
                img = self._resize_image(img, self.image_size)
                if img.max() > 0:
                    img = img / img.max()
            
            if self.transform:
                img = self.transform(img)
                if not isinstance(img, torch.Tensor):
                    img = torch.from_numpy(img).float()
            else:
                img = torch.from_numpy(img).float()
            
            if img.ndim == 2:
                img = img.unsqueeze(0)
            
            return img
            
        except Exception as e:
            if self.verbose:
                print(f"\n  ⚠️  Error loading {png_path}: {e}")
            return self._get_placeholder_tensor()
    
    def _get_placeholder_tensor(self) -> torch.Tensor:
        """Return zero-filled placeholder tensor [1, H, W]."""
        return torch.zeros(1, *self.image_size, dtype=torch.float32)
    
    def _resize_image(self, img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target size."""
        H, W = target_size
        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize((W, H), Image.BILINEAR)
        return np.array(img_pil).astype(np.float32)
    
    def _encode_race(self, race: str) -> int:
        """Encode race: 0=White, 1=Black, 2=Asian, 3=Other."""
        if pd.isna(race):
            return 3
        
        race = str(race).strip()
        race_mapping = {
            'Caucasian or White': 0,
            'White': 0,
            'African American or Black': 1,
            'Black': 1,
            'Asian': 2,
        }
        
        if race in race_mapping:
            return race_mapping[race]
        
        race_lower = race.lower()
        if 'white' in race_lower or 'caucasian' in race_lower:
            return 0
        elif 'black' in race_lower or 'african' in race_lower:
            return 1
        elif 'asian' in race_lower:
            return 2
        else:
            return 3
    
    def _encode_birads(self, birads: str) -> int:
        """Encode BI-RADS score."""
        birads_mapping = {
            'A': 0, 'X': 0,
            'N': 1,
            'B': 2,
            'P': 3,
            'S': 4, 'M': 4, 'K': 4
        }
        return birads_mapping.get(str(birads).upper(), 0)
    
    def _print_statistics(self):
        """Print dataset statistics."""
        print("\n" + "="*70)
        print(f"Dataset: {self.mode.upper()}")
        print("="*70)
        
        print(f"Total exams: {len(self.sample_list):,}")
        
        # Breast-specific recall statistics
        left_recall = sum(s['left_malignant'] for s in self.sample_list)
        right_recall = sum(s['right_malignant'] for s in self.sample_list)
        
        total = len(self.sample_list)
        
        print(f"\nRecall labels (breast-specific, binary):")
        print(f"  Left breast:")
        print(f"    No recall: {total - left_recall:,} ({(total-left_recall)/total*100:.1f}%)")
        print(f"    Need recall: {left_recall:,} ({left_recall/total*100:.1f}%)")
        
        print(f"  Right breast:")
        print(f"    No recall: {total - right_recall:,} ({(total-right_recall)/total*100:.1f}%)")
        print(f"    Need recall: {right_recall:,} ({right_recall/total*100:.1f}%)")
        
        # Exam-level statistics
        exam_recall = sum(
            1 for s in self.sample_list
            if s['left_malignant'] == 1 or s['right_malignant'] == 1
        )
        print(f"\n  Exam-level:")
        print(f"    Any side needs recall: {exam_recall:,} ({exam_recall/total*100:.1f}%)")
        print(f"    Both sides normal: {total - exam_recall:,} ({(total-exam_recall)/total*100:.1f}%)")
        if exam_recall > 0:
            print(f"    Imbalance ratio: {(total-exam_recall)/exam_recall:.1f}:1")
        
        # 3-class label distribution
        left_labels_3class = [s['left_label_3class'] for s in self.sample_list]
        right_labels_3class = [s['right_label_3class'] for s in self.sample_list]
        
        print(f"\nRaw 3-class label distribution:")
        label_names = {
            0: "Class 0 (BIRADS 1/2: No recall)",
            1: "Class 1 (BIRADS 3: Probably benign)",
            2: "Class 2 (BIRADS 4/5/6: Suspicious/Malignant)"
        }
        
        print(f"  Left breast:")
        for label in [0, 1, 2]:
            count = sum(1 for l in left_labels_3class if l == label)
            print(f"    {label_names[label]}: {count:,} ({count/total*100:.1f}%)")
        
        print(f"  Right breast:")
        for label in [0, 1, 2]:
            count = sum(1 for l in right_labels_3class if l == label)
            print(f"    {label_names[label]}: {count:,} ({count/total*100:.1f}%)")
        
        # View availability
        view_counts = {k: [] for k in ['L-CC', 'L-MLO', 'R-CC', 'R-MLO']}
        
        for sample in self.sample_list:
            for view_key in ['L-CC', 'L-MLO', 'R-CC', 'R-MLO']:
                view_info = sample['current_views'][view_key]
                if view_info is not None and isinstance(view_info, dict):
                    view_counts[view_key].append(1)
                else:
                    view_counts[view_key].append(0)
        
        print(f"\nView availability:")
        for view_key in ['L-CC', 'L-MLO', 'R-CC', 'R-MLO']:
            count = sum(view_counts[view_key])
            pct = count / total * 100
            print(f"  {view_key}: {count:,} ({pct:.1f}%)")
        
        # Complete 4-view exams
        complete_4view = sum(
            1 for s in self.sample_list
            if all(s['current_views'][k] is not None and isinstance(s['current_views'][k], dict) 
                   for k in ['L-CC', 'L-MLO', 'R-CC', 'R-MLO'])
        )
        print(f"\nComplete 4-view exams: {complete_4view:,} ({complete_4view/total*100:.1f}%)")
        
        # Prior exam availability
        has_prior = sum(1 for s in self.sample_list if s['num_prior_views'] > 0)
        print(f"With prior exam: {has_prior:,} ({has_prior/total*100:.1f}%)")
        
        print("="*70 + "\n")
    
    def get_sample_weights(self) -> np.ndarray:
        """Get per-sample weights for balanced sampling (exam-level)."""
        # Any side needs recall → positive
        binary_labels = np.array([
            1 if (s['left_malignant'] == 1 or s['right_malignant'] == 1) else 0
            for s in self.sample_list
        ])
        class_counts = np.bincount(binary_labels)
        class_weights = 1.0 / class_counts
        return class_weights[binary_labels]
    
    def get_binary_labels(self) -> np.ndarray:
        """Get binary labels for balanced sampling (exam-level)."""
        return np.array([
            1 if (s['left_malignant'] == 1 or s['right_malignant'] == 1) else 0
            for s in self.sample_list
        ])


# ============================================================================
# Collate Function
# ============================================================================

# 在 data/dataset.py 的 collate_fn 函数中，确保返回格式正确

def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for exam-level batches."""
    view_keys = ['L-CC', 'L-MLO', 'R-CC', 'R-MLO']
    
    current_views = {}
    prior_views = {}
    current_mask = {}
    prior_mask = {}
    
    for view_key in view_keys:
        current_views[view_key] = torch.stack([
            item['current_views'][view_key] for item in batch
        ])  # [B, 1, H, W]
        
        current_mask[view_key] = torch.tensor([
            item['current_mask'][view_key] for item in batch
        ], dtype=torch.float32)  # [B]
        
        prior_views[view_key] = torch.stack([
            item['prior_views'][view_key] for item in batch
        ])  # [B, 1, H, W]
        
        prior_mask[view_key] = torch.tensor([
            item['prior_mask'][view_key] for item in batch
        ], dtype=torch.float32)  # [B]
    
    result = {
        'current_views': current_views,  # ✅ 这个格式完美匹配NYU
        'prior_views': prior_views,
        'current_mask': current_mask,
        'prior_mask': prior_mask,
        
        'labels': {
            'left_malignant': torch.cat([item['labels']['left_malignant'] for item in batch]),
            'right_malignant': torch.cat([item['labels']['right_malignant'] for item in batch]),
            'left_label_3class': torch.stack([item['labels']['left_label_3class'] for item in batch]),
            'right_label_3class': torch.stack([item['labels']['right_label_3class'] for item in batch]),
            'birads': torch.stack([item['labels']['birads'] for item in batch]),
        },
        
        'metadata': {
            'race': torch.stack([item['metadata']['race'] for item in batch]),
            'age': torch.stack([item['metadata']['age'] for item in batch]),
            'num_current_views': torch.stack([item['metadata']['num_current_views'] for item in batch]),
            'num_prior_views': torch.stack([item['metadata']['num_prior_views'] for item in batch]),
        },
        
        'exam_info': [item['exam_info'] for item in batch]
    }
    
    return result


# ============================================================================
# Helper Functions
# ============================================================================

def create_data_loaders(args):
    """Create train/val/test data loaders with patient-level splitting."""
    print("\n" + "="*70)
    print("CREATING DATA LOADERS")
    print("="*70)
    
    df_full = pd.read_csv(
        args.clinical_csv, 
        low_memory=False, 
        dtype={'empi_anon': str, 'acc_anon': str}
    )
    
    print(f"  Total records: {len(df_full):,}")
    print(f"  Total patients: {df_full['empi_anon'].nunique():,}")
    
    all_patients = df_full['empi_anon'].unique()
    
    if args.sample_fraction < 1.0:
        print(f"\n  Sampling {args.sample_fraction*100:.1f}% of patients...")
        n_sample = int(len(all_patients) * args.sample_fraction)
        np.random.seed(args.random_seed)
        sampled_patients = np.random.choice(all_patients, size=n_sample, replace=False)
    else:
        sampled_patients = all_patients
    
    n_total = len(sampled_patients)
    n_train = int(n_total * args.train_split)
    n_val = int(n_total * args.val_split)
    
    np.random.seed(args.random_seed)
    shuffled_patients = np.random.permutation(sampled_patients)
    
    train_pts = shuffled_patients[:n_train]
    val_pts = shuffled_patients[n_train:n_train + n_val]
    test_pts = shuffled_patients[n_train + n_val:]
    
    print(f"\n  Patient split:")
    print(f"    Train: {len(train_pts):,}")
    print(f"    Val:   {len(val_pts):,}")
    print(f"    Test:  {len(test_pts):,}")
    
    # ✅ 直接传递患者列表，不再创建临时CSV文件
    print(f"\n  Creating datasets (no temp files)...")
    
    train_ds = EMBEDRecallDataset(
        clinical_csv=args.clinical_csv,  # ✅ 使用原始CSV
        metadata_csv=args.metadata_csv,
        mode='train',
        image_size=tuple(args.image_size),
        use_cache=args.use_cache,
        verbose=True,
        apply_nyu_preprocessing=args.apply_nyu_preprocessing,
        patient_filter=train_pts.tolist()  # ✅ 传递患者列表
    )
    
    val_ds = EMBEDRecallDataset(
        clinical_csv=args.clinical_csv,  # ✅ 使用原始CSV
        metadata_csv=args.metadata_csv,
        mode='val',
        image_size=tuple(args.image_size),
        use_cache=args.use_cache,
        verbose=True,
        apply_nyu_preprocessing=args.apply_nyu_preprocessing,
        patient_filter=val_pts.tolist()  # ✅ 传递患者列表
    )
    
    test_ds = EMBEDRecallDataset(
        clinical_csv=args.clinical_csv,  # ✅ 使用原始CSV
        metadata_csv=args.metadata_csv,
        mode='test',
        image_size=tuple(args.image_size),
        use_cache=args.use_cache,
        verbose=True,
        apply_nyu_preprocessing=args.apply_nyu_preprocessing,
        patient_filter=test_pts.tolist()  # ✅ 传递患者列表
    )
    
    print(f"\n  Creating data loaders...")
    
    if args.use_balanced_batch:
        train_labels = train_ds.get_binary_labels()
        train_sampler = BalancedBatchSampler(
            labels=train_labels,
            batch_size=args.batch_size,
            positive_ratio=args.positive_ratio
        )
        
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"\n  ✓ Data loaders created (no temp files)")
    print("="*70 + "\n")
    
    datasets = {
        'train': train_ds,
        'val': val_ds,
        'test': test_ds
        # ✅ 移除 'temp_dir'，不再需要临时目录
    }
    
    return train_loader, val_loader, test_loader, datasets


# ============================================================================
# Testing
# ============================================================================






def diagnose_data_loading(clinical_csv, metadata_csv, patient_filter=None):
    """
    诊断数据加载过程中每一步的样本数量变化
    """
    print("\n" + "="*80)
    print("EMBED DATASET LOADING DIAGNOSIS")
    print("="*80)
    
    # ============================================================================
    # Step 0: Load Raw Data
    # ============================================================================
    print("\n[Step 0] Loading raw data...")
    
    dtype_dict = {'acc_anon': str, 'empi_anon': str}
    
    clinical_df = pd.read_csv(clinical_csv, low_memory=False, dtype=dtype_dict)
    metadata_df = pd.read_csv(metadata_csv, low_memory=False, dtype=dtype_dict)
    
    print(f"  ✓ Clinical records (finding-level): {len(clinical_df):,}")
    print(f"  ✓ Clinical unique exams: {clinical_df['acc_anon'].nunique():,}")
    print(f"  ✓ Clinical unique patients: {clinical_df['empi_anon'].nunique():,}")
    
    print(f"\n  ✓ Metadata records (image-level): {len(metadata_df):,}")
    print(f"  ✓ Metadata unique exams: {metadata_df['acc_anon'].nunique():,}")
    print(f"  ✓ Metadata unique patients: {metadata_df['empi_anon'].nunique():,}")
    
    # ============================================================================
    # Step 0.5: Apply Patient Filter (if provided)
    # ============================================================================
    if patient_filter is not None:
        print(f"\n[Step 0.5] Applying patient filter ({len(patient_filter):,} patients)...")
        
        patient_set = set(patient_filter)
        
        clinical_before = len(clinical_df)
        clinical_df = clinical_df[clinical_df['empi_anon'].isin(patient_set)].copy()
        print(f"  Clinical: {clinical_before:,} → {len(clinical_df):,} records")
        print(f"  Clinical: {clinical_df['acc_anon'].nunique():,} unique exams")
        print(f"  Clinical: {clinical_df['empi_anon'].nunique():,} unique patients")
        
        metadata_before = len(metadata_df)
        metadata_df = metadata_df[metadata_df['empi_anon'].isin(patient_set)].copy()
        print(f"  Metadata: {metadata_before:,} → {len(metadata_df):,} records")
        print(f"  Metadata: {metadata_df['acc_anon'].nunique():,} unique exams")
        print(f"  Metadata: {metadata_df['empi_anon'].nunique():,} unique patients")
    
    # ============================================================================
    # Step 1: Check 'new_label' existence and distribution
    # ============================================================================
    print("\n[Step 1] Checking 'new_label' field...")
    
    if 'new_label' not in clinical_df.columns:
        print("  ❌ ERROR: 'new_label' column not found!")
        print(f"  Available columns: {list(clinical_df.columns)}")
        return
    
    print(f"  ✓ 'new_label' field exists")
    
    # Check for NaN values
    label_na_count = clinical_df['new_label'].isna().sum()
    print(f"  NaN values in 'new_label': {label_na_count:,} ({label_na_count/len(clinical_df)*100:.2f}%)")
    
    # Distribution of new_label
    label_dist = clinical_df['new_label'].value_counts().sort_index()
    print(f"\n  'new_label' distribution:")
    for label, count in label_dist.items():
        print(f"    Label {label}: {count:,} ({count/len(clinical_df)*100:.1f}%)")
    
    # ============================================================================
    # Step 2: Check 'side' field distribution
    # ============================================================================
    print("\n[Step 2] Checking 'side' field distribution...")
    
    if 'side' not in clinical_df.columns:
        print("  ⚠️  WARNING: 'side' column not found!")
    else:
        side_dist = clinical_df['side'].value_counts(dropna=False)
        print(f"  'side' distribution:")
        for side, count in side_dist.items():
            print(f"    '{side}': {count:,} ({count/len(clinical_df)*100:.1f}%)")
    
    # ============================================================================
    # Step 3: Generate Exam-Level Labels
    # ============================================================================
    print("\n[Step 3] Generating exam-level labels...")
    
    exam_labels_list = []
    side_counter = Counter()
    
    clinical_df['study_date_anon'] = pd.to_datetime(
        clinical_df['study_date_anon'], errors='coerce'
    )
    
    grouped = clinical_df.groupby('acc_anon')
    
    for exam_id, exam_findings in grouped:
        first_finding = exam_findings.iloc[0]
        patient_id = first_finding['empi_anon']
        study_date = first_finding['study_date_anon']
        
        left_max_label = 0
        right_max_label = 0
        
        for _, finding in exam_findings.iterrows():
            side = finding.get('side', np.nan)
            label = finding.get('new_label', np.nan)
            
            # 只跳过label为NaN的
            if pd.isna(label):
                continue
            
            label = int(label)
            side_counter[str(side)] += 1
            
            # 处理所有side情况
            if pd.isna(side):
                left_max_label = max(left_max_label, label)
                right_max_label = max(right_max_label, label)
            elif side == 'L':
                left_max_label = max(left_max_label, label)
            elif side == 'R':
                right_max_label = max(right_max_label, label)
            elif side == 'B':
                left_max_label = max(left_max_label, label)
                right_max_label = max(right_max_label, label)
            else:
                # Unknown side → apply to both (conservative)
                left_max_label = max(left_max_label, label)
                right_max_label = max(right_max_label, label)
        
        left_recall = 1 if left_max_label >= 1 else 0
        right_recall = 1 if right_max_label >= 1 else 0
        
        exam_labels_list.append({
            'acc_anon': str(exam_id),
            'empi_anon': str(patient_id),
            'study_date_anon': study_date,
            'left_benign': 1 - left_recall,
            'left_malignant': left_recall,
            'right_benign': 1 - right_recall,
            'right_malignant': right_recall,
            'left_label_3class': left_max_label,
            'right_label_3class': right_max_label,
        })
    
    exam_labels_df = pd.DataFrame(exam_labels_list)
    
    print(f"  ✓ Generated labels for {len(exam_labels_df):,} exams")
    print(f"\n  Side distribution in findings:")
    for side_val, count in sorted(side_counter.items()):
        print(f"    '{side_val}': {count:,}")
    
    left_recall_count = exam_labels_df['left_malignant'].sum()
    right_recall_count = exam_labels_df['right_malignant'].sum()
    total = len(exam_labels_df)
    
    print(f"\n  Recall labels:")
    print(f"    Left needs recall: {left_recall_count:,} ({left_recall_count/total*100:.1f}%)")
    print(f"    Right needs recall: {right_recall_count:,} ({right_recall_count/total*100:.1f}%)")
    
    # ============================================================================
    # Step 4: Filter Metadata for 2D CC/MLO views
    # ============================================================================
    print("\n[Step 4] Filtering metadata for 2D CC/MLO views...")
    
    metadata_before = len(metadata_df)
    
    if 'FinalImageType' in metadata_df.columns:
        metadata_df = metadata_df[metadata_df['FinalImageType'] == '2D'].copy()
        print(f"  After FinalImageType='2D': {len(metadata_df):,} images")
    
    if 'ViewPosition' in metadata_df.columns:
        metadata_df = metadata_df[
            metadata_df['ViewPosition'].isin(['CC', 'MLO'])
        ].copy()
        print(f"  After ViewPosition in [CC, MLO]: {len(metadata_df):,} images")
    
    if 'png_exists' in metadata_df.columns:
        metadata_df = metadata_df[metadata_df['png_exists'] == True].copy()
        print(f"  After png_exists=True: {len(metadata_df):,} images")
    
    print(f"  ✓ Filtered: {metadata_before:,} → {len(metadata_df):,} images")
    print(f"  ✓ Unique exams in filtered metadata: {metadata_df['acc_anon'].nunique():,}")
    
    # ============================================================================
    # Step 5: Organize Image Paths by Exam
    # ============================================================================
    print("\n[Step 5] Organizing image paths by exam...")
    
    exam_images_list = []
    
    metadata_df['study_date_anon'] = pd.to_datetime(
        metadata_df.get('study_date_anon', pd.Series()), errors='coerce'
    )
    
    grouped = metadata_df.groupby('acc_anon')
    
    for exam_id, exam_images in grouped:
        first_image = exam_images.iloc[0]
        patient_id = first_image.get('empi_anon', np.nan)
        study_date = first_image.get('study_date_anon', np.nan)
        
        views_dict = {
            'L-CC': [],
            'L-MLO': [],
            'R-CC': [],
            'R-MLO': []
        }
        
        for _, img_row in exam_images.iterrows():
            laterality = img_row.get('ImageLateralityFinal', np.nan)
            view_pos = img_row.get('ViewPosition', np.nan)
            
            if laterality not in ['L', 'R'] or view_pos not in ['CC', 'MLO']:
                continue
            
            view_key = f"{laterality}-{view_pos}"
            
            view_info = {
                'png_path': img_row.get('png_path', np.nan),
                'view_position': view_pos,
                'laterality': laterality,
                'spot_mag': img_row.get('spot_mag', '0'),
            }
            
            views_dict[view_key].append(view_info)
        
        selected_views = {}
        for view_key in ['L-CC', 'L-MLO', 'R-CC', 'R-MLO']:
            if len(views_dict[view_key]) > 0:
                standard = [v for v in views_dict[view_key] if v['spot_mag'] == '0']
                if standard:
                    selected_views[f'{view_key}_view'] = standard[0]
                else:
                    selected_views[f'{view_key}_view'] = views_dict[view_key][0]
            else:
                selected_views[f'{view_key}_view'] = None
        
        exam_images_list.append({
            'acc_anon': str(exam_id),
            'empi_anon': str(patient_id),
            'study_date_anon': study_date,
            **selected_views
        })
    
    exam_images_df = pd.DataFrame(exam_images_list)
    
    print(f"  ✓ Organized images for {len(exam_images_df):,} exams")
    
    for view_key in ['L-CC', 'L-MLO', 'R-CC', 'R-MLO']:
        col_name = f'{view_key}_view'
        available = exam_images_df[col_name].notna().sum()
        print(f"  {view_key} available: {available:,} ({available/len(exam_images_df)*100:.1f}%)")
    
    # ============================================================================
    # Step 6: Merge Labels and Images - CRITICAL STEP
    # ============================================================================
    print("\n[Step 6] Merging labels and images...")
    
    print(f"  Before merge:")
    print(f"    Exams with labels: {len(exam_labels_df):,}")
    print(f"    Exams with images: {len(exam_images_df):,}")
    
    # Check overlap
    labels_exams = set(exam_labels_df['acc_anon'])
    images_exams = set(exam_images_df['acc_anon'])
    
    overlap = labels_exams & images_exams
    only_labels = labels_exams - images_exams
    only_images = images_exams - labels_exams
    
    print(f"\n  Exam overlap analysis:")
    print(f"    Exams in both: {len(overlap):,}")
    print(f"    Only in labels: {len(only_labels):,}")
    print(f"    Only in images: {len(only_images):,}")
    
    merged_exams = exam_labels_df.merge(
        exam_images_df,
        on=['empi_anon', 'acc_anon'],
        how='inner',
        suffixes=('_label', '_image')
    )
    
    if 'study_date_anon_label' in merged_exams.columns:
        merged_exams['study_date_anon'] = merged_exams['study_date_anon_label']
    elif 'study_date_anon_image' in merged_exams.columns:
        merged_exams['study_date_anon'] = merged_exams['study_date_anon_image']
    
    print(f"\n  ✓ After inner merge: {len(merged_exams):,} exams")
    print(f"    Lost {len(exam_labels_df) - len(merged_exams):,} exams without images")
    print(f"    Lost {len(exam_images_df) - len(merged_exams):,} exams without labels")
    
    # ============================================================================
    # Step 7: Check View Completeness
    # ============================================================================
    print("\n[Step 7] Checking view completeness...")
    
    view_stats = {
        'has_left': 0,
        'has_right': 0,
        'has_both': 0,
        'has_4_views': 0,
        'missing_views': 0
    }
    
    for _, exam_row in merged_exams.iterrows():
        l_cc = exam_row.get('L-CC_view')
        l_mlo = exam_row.get('L-MLO_view')
        r_cc = exam_row.get('R-CC_view')
        r_mlo = exam_row.get('R-MLO_view')
        
        has_left = l_cc is not None or l_mlo is not None
        has_right = r_cc is not None or r_mlo is not None
        has_4_views = all(v is not None for v in [l_cc, l_mlo, r_cc, r_mlo])
        
        if has_left:
            view_stats['has_left'] += 1
        if has_right:
            view_stats['has_right'] += 1
        if has_left and has_right:
            view_stats['has_both'] += 1
        if has_4_views:
            view_stats['has_4_views'] += 1
        if not (has_left and has_right):
            view_stats['missing_views'] += 1
    
    total = len(merged_exams)
    print(f"  View completeness:")
    print(f"    Has at least one left view: {view_stats['has_left']:,} ({view_stats['has_left']/total*100:.1f}%)")
    print(f"    Has at least one right view: {view_stats['has_right']:,} ({view_stats['has_right']/total*100:.1f}%)")
    print(f"    Has both left AND right: {view_stats['has_both']:,} ({view_stats['has_both']/total*100:.1f}%)")
    print(f"    Has all 4 views: {view_stats['has_4_views']:,} ({view_stats['has_4_views']/total*100:.1f}%)")
    print(f"    Missing left OR right: {view_stats['missing_views']:,} ({view_stats['missing_views']/total*100:.1f}%)")
    
    print(f"\n  ⚠️  If using 'has_both' filter, will drop {view_stats['missing_views']:,} exams")
    
    # ============================================================================
    # Step 8: Summary
    # ============================================================================
    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)
    
    print(f"\nData flow:")
    print(f"  1. Raw clinical records: {len(clinical_df):,}")
    print(f"  2. → Exam-level labels: {len(exam_labels_df):,}")
    print(f"  3. Raw metadata images: {len(metadata_df):,}")
    print(f"  4. → Exam-level images: {len(exam_images_df):,}")
    print(f"  5. → Merged exams: {len(merged_exams):,}")
    print(f"  6. → With complete views (both sides): {view_stats['has_both']:,}")
    
    loss_at_merge = len(exam_labels_df) - len(merged_exams)
    loss_at_filter = view_stats['missing_views']
    
    print(f"\nLosses:")
    print(f"  At merge (no matching images/labels): {loss_at_merge:,}")
    print(f"  At view filter (missing left or right): {loss_at_filter:,}")
    print(f"  Total loss: {loss_at_merge + loss_at_filter:,}")
    
    final_expected = len(exam_labels_df) - loss_at_merge - loss_at_filter
    print(f"\nExpected final samples: {final_expected:,}")
    
    print("\n" + "="*80)
    
    # Return diagnostic data for further analysis
    return {
        'clinical_df': clinical_df,
        'metadata_df': metadata_df,
        'exam_labels_df': exam_labels_df,
        'exam_images_df': exam_images_df,
        'merged_exams': merged_exams,
        'view_stats': view_stats
    }


def test_actual_dataset(clinical_csv, metadata_csv, sample_fraction=0.01):
    """
    测试实际的数据集加载
    """
    print("\n" + "="*80)
    print("TESTING ACTUAL DATASET LOADING")
    print("="*80)
    
    # Import here to avoid circular dependency
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Sample patients
    df_full = pd.read_csv(clinical_csv, low_memory=False, dtype={'empi_anon': str})
    all_patients = df_full['empi_anon'].unique()
    
    n_sample = int(len(all_patients) * sample_fraction)
    np.random.seed(42)
    sampled_patients = np.random.choice(all_patients, size=n_sample, replace=False)
    
    print(f"\nSampling {sample_fraction*100:.1f}% of patients: {len(sampled_patients):,}")
    
    # Test with your actual dataset class
    print("\n" + "-"*80)
    print("Loading with EMBEDRecallDataset...")
    print("-"*80)
    
    try:
        from dataset import EMBEDRecallDataset
        
        dataset = EMBEDRecallDataset(
            clinical_csv=clinical_csv,
            metadata_csv=metadata_csv,
            mode='train',
            image_size=(2944, 1920),
            use_cache=False,  # Don't use cache for testing
            verbose=True,
            apply_nyu_preprocessing=False,  # Skip preprocessing for speed
            patient_filter=sampled_patients.tolist()
        )
        
        print(f"\n✓ Dataset loaded successfully!")
        print(f"  Final samples: {len(dataset):,}")
        
        # Test getting a sample
        print("\nTesting __getitem__...")
        sample = dataset[0]
        
        print(f"  ✓ Sample structure OK")
        print(f"  Labels: left_malignant={sample['labels']['left_malignant'].item():.0f}, "
              f"right_malignant={sample['labels']['right_malignant'].item():.0f}")
        
        # Test view availability
        print(f"\n  View availability in first sample:")
        for view_key in ['L-CC', 'L-MLO', 'R-CC', 'R-MLO']:
            mask = sample['current_mask'][view_key]
            print(f"    {view_key}: {'✓' if mask > 0 else '✗'}")
        
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()





if __name__ == '__main__':
    import argparse
    import shutil
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--clinical_csv', type=str,
                        default="/projects/standard/lin01231/song0760/embed_recall_reduction/EMBED_OpenData_clinical_relabeled.csv")
    parser.add_argument('--metadata_csv', type=str,
                        default="/projects/standard/lin01231/public/datasets/embed/tables/EMBED_OpenData_metadata_png.csv")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--sample_fraction', type=float, default=0.1,
                        help='Fraction of patients to sample (for testing)')
    parser.add_argument('--image_size', type=int, nargs=2, default=[2944, 1920])
    parser.add_argument('--apply_nyu_preprocessing', action='store_true', default=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train_split', type=float, default=0.7)
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--use_balanced_batch', action='store_true')
    parser.add_argument('--positive_ratio', type=float, default=0.3)
    parser.add_argument('--use_cache', action='store_true', default=True)
    parser.add_argument('--test_dataset', action='store_true', help='Test actual dataset loading')
    
    args = parser.parse_args()
    
    diagnostic_data = diagnose_data_loading(
        args.clinical_csv,
        args.metadata_csv
    )
    
    # Optionally test actual dataset
    if args.test_dataset:
        test_actual_dataset(
            args.clinical_csv,
            args.metadata_csv,
            args.sample_fraction
        )
    
    print("\n✓ Diagnosis complete!")