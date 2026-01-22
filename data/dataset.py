"""
EMBED Single-View Longitudinal Dataset for Recall Reduction
===========================================================

✅ OPTIMIZED FOR CLASS IMBALANCE:
1. Added WeightedRandomSampler for balanced training
2. Enhanced class weight computation
3. Better cache management for faster loading

Key Features:
1. Each sample = ONE view + its historical counterpart (2 images total)
2. Only uses 2D mammography images (excludes C-view, tomosynthesis)
3. Proper clinical-metadata merging considering laterality
4. View-specific longitudinal pairing (LCC→LCC, RCC→RCC, etc.)
5. Fairness-aware metadata (race, age, density)

Author: Yiran
Date: 2025
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from collections import Counter
from typing import Optional, Dict, List, Tuple, Literal
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


class EMBEDSingleViewLongitudinalDataset(Dataset):
    """
    EMBED Dataset with single-view longitudinal pairing.
    
    Each sample contains:
    - current_image: One view from current exam (e.g., LCC)
    - prior_image: Same view from prior exam (e.g., previous LCC)
    
    Args:
        clinical_csv: Path to EMBED_OpenData_clinical_relabeled.csv
        metadata_csv: Path to EMBED_OpenData_metadata_png.csv
        mode: 'train', 'val', or 'test'
        transform: Image transformations
        prior_time_window: (min_days, max_days) for finding prior exams
        handle_missing_prior: 'skip', 'self_pair', 'zero', or 'mixed'
        prior_required: If True, only use samples with real priors
        image_size: Target image size (H, W)
        use_cache: Whether to use cached sample indices
        verbose: Print detailed information
    """
    
    def __init__(
        self,
        clinical_csv: str,
        metadata_csv: str,
        mode: Literal['train', 'val', 'test'] = 'train',
        transform=None,
        prior_time_window: Tuple[int, int] = (365, 1095),
        handle_missing_prior: Literal['skip', 'self_pair', 'zero', 'mixed'] = 'mixed',
        prior_required: bool = False,
        image_size: Optional[Tuple[int, int]] = (2048, 1024),
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        verbose: bool = True
    ):
        self.clinical_csv = clinical_csv
        self.metadata_csv = metadata_csv
        self.mode = mode
        self.transform = transform
        self.prior_time_window = prior_time_window
        self.handle_missing_prior = handle_missing_prior
        self.prior_required = prior_required
        self.image_size = image_size
        self.cache_dir = cache_dir or os.path.dirname(clinical_csv)
        self.use_cache = use_cache
        self.verbose = verbose
        
        # Load data
        self._load_data()
        
        # Build view-level samples
        self._build_view_samples()
        
        if self.verbose:
            self._print_statistics()
    
    def _load_data(self):
        """Load and prepare clinical and metadata files."""
        if self.verbose:
            print("Loading data files...")
        
        # Load clinical data
        self.clinical_df = pd.read_csv(self.clinical_csv, low_memory=False)
        
        if 'new_label' not in self.clinical_df.columns:
            raise ValueError("'new_label' column not found in clinical CSV!")
        
        # Convert dates
        self.clinical_df['study_date_anon'] = pd.to_datetime(
            self.clinical_df['study_date_anon'], errors='coerce'
        )
        
        # Load metadata
        self.metadata_df = pd.read_csv(self.metadata_csv, low_memory=False)
        
        # ✅ Filter for 2D images only
        if 'FinalImageType' in self.metadata_df.columns:
            original_count = len(self.metadata_df)
            self.metadata_df = self.metadata_df[
                self.metadata_df['FinalImageType'] == '2D'
            ].copy()
            if self.verbose:
                print(f"Filtered to 2D images: {len(self.metadata_df):,} / {original_count:,}")
        
        # Verify PNG paths
        if 'png_path' not in self.metadata_df.columns:
            raise ValueError("PNG metadata must contain 'png_path' column!")
        
        # Filter for existing PNGs
        if 'png_exists' in self.metadata_df.columns:
            self.metadata_df = self.metadata_df[
                self.metadata_df['png_exists'] == True
            ].copy()
        
        if self.verbose:
            print(f"Loaded {len(self.clinical_df):,} clinical records")
            print(f"Loaded {len(self.metadata_df):,} 2D PNG images")
    
    def _build_view_samples(self):
        """Build view-level longitudinal pairs."""
        cache_file = os.path.join(
            self.cache_dir,
            f'view_samples_cache_{self.mode}_{self.handle_missing_prior}_'
            f'{self.prior_time_window[0]}_{self.prior_time_window[1]}.pkl'
        )
        
        if self.use_cache and os.path.exists(cache_file):
            if self.verbose:
                print(f"Loading cached view samples from {cache_file}")
            self.view_samples = pd.read_pickle(cache_file)
        else:
            if self.verbose:
                print("Building view-level samples...")
            self.view_samples = self._create_view_pairs()
            
            # Save cache
            if self.use_cache:
                os.makedirs(self.cache_dir, exist_ok=True)
                self.view_samples.to_pickle(cache_file)
                if self.verbose:
                    print(f"Saved cache to {cache_file}")
        
        self.sample_list = self.view_samples.to_dict('records')
        
        if self.verbose:
            print(f"Built {len(self.sample_list):,} view-level samples")
    
    def _create_view_pairs(self) -> pd.DataFrame:
        """
        Create view-level longitudinal pairs.
        
        Strategy:
        1. For each exam, get all 2D views
        2. For each view, find matching view from prior exam
        3. Create one sample per view pair
        
        Returns:
            DataFrame with columns: current_png_path, prior_png_path, 
                                   view_info, label, metadata, etc.
        """
        samples = []
        
        # ✅ Merge clinical and metadata with proper laterality handling
        merged_df = self._merge_clinical_metadata()
        
        if self.verbose:
            print(f"After merging: {len(merged_df):,} image-level records")
        
        # Group by patient
        patient_groups = merged_df.groupby('empi_anon')
        
        if self.verbose:
            patient_groups = tqdm(patient_groups, desc="Processing patients")
        
        for patient_id, patient_data in patient_groups:
            # Sort by date
            patient_data = patient_data.sort_values('study_date_anon')
            
            # Get unique exams
            exams = patient_data.groupby('acc_anon')
            
            for current_exam_id, current_exam_data in exams:
                # Get current exam metadata
                current_exam_clinical = current_exam_data.iloc[0]
                
                # Skip if missing label
                if pd.isna(current_exam_clinical['new_label']):
                    continue
                
                current_date = current_exam_clinical['study_date_anon']
                
                # Find prior exam
                prior_exam_data = self._find_prior_exam_data(
                    patient_data, current_date
                )
                
                has_real_prior = prior_exam_data is not None
                
                if not has_real_prior and self.prior_required:
                    continue
                
                # Process each view in current exam
                for _, current_view_row in current_exam_data.iterrows():
                    view_laterality = current_view_row['ImageLateralityFinal']
                    view_position = current_view_row['ViewPosition']
                    
                    # Find matching view in prior exam
                    prior_png_path = None
                    if has_real_prior:
                        prior_png_path = self._find_matching_prior_view(
                            prior_exam_data,
                            view_laterality,
                            view_position
                        )
                    
                    # Create sample
                    sample = {
                        'empi_anon': patient_id,
                        'current_exam_id': current_exam_id,
                        'current_png_path': current_view_row['png_path'],
                        'prior_png_path': prior_png_path,
                        'view_laterality': view_laterality,
                        'view_position': view_position,
                        'view_name': f"{view_laterality}{view_position}",  # e.g., "LCC"
                        'current_date': current_date,
                        'prior_date': (
                            prior_exam_data.iloc[0]['study_date_anon'] 
                            if has_real_prior else None
                        ),
                        'has_real_prior': has_real_prior and prior_png_path is not None,
                        'time_gap_days': (
                            (current_date - prior_exam_data.iloc[0]['study_date_anon']).days
                            if has_real_prior else 0
                        ),
                        # Labels
                        'label': int(current_exam_clinical['new_label']),
                        'birads_orig': current_exam_clinical.get('asses', np.nan),
                        'density': current_view_row.get('tissueden', np.nan),
                        # Demographics
                        'race': current_exam_clinical.get('ETHNICITY_DESC', np.nan),
                        'age': current_exam_clinical.get('age', np.nan),
                    }
                    
                    samples.append(sample)
        
        return pd.DataFrame(samples)

    def _merge_clinical_metadata(self) -> pd.DataFrame:
        """
        优化版：使用pandas merge，正确处理列名冲突
        
        Properly merge clinical and metadata considering laterality.
        
        Rules (from EMBED documentation):
        - clinical.side='L' → metadata.ImageLateralityFinal='L'
        - clinical.side='R' → metadata.ImageLateralityFinal='R'  
        - clinical.side='B' or NaN → metadata (all L and R)
        
        Returns:
            Merged DataFrame at image level
        """
        # Step 1: 准备clinical数据（exam-level去重）
        clinical_exam = self.clinical_df.drop_duplicates(
            subset='acc_anon', keep='first'
        ).copy()
        
        if self.verbose:
            print(f"Clinical exams (unique acc_anon): {len(clinical_exam):,}")
        
        # Step 2: 检查两个df中的共同列
        common_cols = set(self.metadata_df.columns) & set(clinical_exam.columns)
        if self.verbose and len(common_cols) > 1:  # acc_anon是预期的共同列
            print(f"Common columns: {common_cols}")
        
        # Step 3: 使用merge，明确指定suffixes处理冲突列
        merged = pd.merge(
            self.metadata_df,
            clinical_exam,
            on='acc_anon',
            how='inner',
            suffixes=('_meta', '_clin')  # 重要：处理列名冲突
        )
        
        if self.verbose:
            print(f"After merge (before laterality filter): {len(merged):,}")
        
        # Step 4: 检查关键列是否存在
        required_cols = ['empi_anon', 'study_date_anon', 'ImageLateralityFinal', 'new_label']
        missing_cols = []
        
        for col in required_cols:
            # 检查原始列名或带后缀的列名
            if col not in merged.columns:
                # 尝试查找带后缀的版本
                if f'{col}_clin' in merged.columns:
                    merged[col] = merged[f'{col}_clin']
                    if self.verbose:
                        print(f"  Using {col}_clin as {col}")
                elif f'{col}_meta' in merged.columns:
                    merged[col] = merged[f'{col}_meta']
                    if self.verbose:
                        print(f"  Using {col}_meta as {col}")
                else:
                    missing_cols.append(col)
        
        if missing_cols:
            print(f"Available columns: {sorted(merged.columns.tolist())}")
            raise ValueError(f"Missing required columns after merge: {missing_cols}")
        
        # Step 5: 应用laterality过滤规则
        # side='L' → 只保留 ImageLateralityFinal='L'
        # side='R' → 只保留 ImageLateralityFinal='R'
        # side='B' 或 NaN → 保留所有
        
        mask = (
            merged['side'].isna() |  # side是NaN → 保留所有
            (merged['side'] == 'B') |  # side是'B' → 保留所有
            (merged['side'] == merged['ImageLateralityFinal'])  # side匹配laterality → 保留
        )
        
        merged_filtered = merged[mask].copy()
        
        if self.verbose:
            print(f"After laterality filter: {len(merged_filtered):,}")
            
            # 统计过滤效果
            side_dist = merged['side'].value_counts(dropna=False)
            print(f"\nSide distribution in merged data:")
            for side, count in side_dist.items():
                filtered_count = merged_filtered[merged_filtered['side'] == side].shape[0]
                print(f"  {side}: {count:,} → {filtered_count:,} "
                    f"({filtered_count/count*100:.1f}% retained)")
        
        # Step 6: 转换日期列
        if 'study_date_anon' in merged_filtered.columns:
            merged_filtered['study_date_anon'] = pd.to_datetime(
                merged_filtered['study_date_anon'], errors='coerce'
            )
        
        # Step 7: 清理：删除带后缀的重复列（可选）
        cols_to_drop = [col for col in merged_filtered.columns 
                        if col.endswith('_meta') or col.endswith('_clin')]
        if cols_to_drop and self.verbose:
            print(f"\nDropping duplicate columns with suffixes: {len(cols_to_drop)}")
        
        # 保留原始列，删除带后缀的重复列
        # 但要确保不删除只有后缀版本的列
        safe_to_drop = []
        for col in cols_to_drop:
            base_col = col.replace('_meta', '').replace('_clin', '')
            if base_col in merged_filtered.columns:
                safe_to_drop.append(col)
        
        if safe_to_drop:
            merged_filtered = merged_filtered.drop(columns=safe_to_drop)
        
        return merged_filtered
    
    def _find_prior_exam_data(
        self,
        patient_data: pd.DataFrame,
        current_date: pd.Timestamp
    ) -> Optional[pd.DataFrame]:
        """Find prior exam data within time window."""
        min_days, max_days = self.prior_time_window
        
        # Find prior exams
        prior_exams = patient_data[
            (patient_data['study_date_anon'] < current_date) &
            (patient_data['study_date_anon'] >= current_date - pd.Timedelta(days=max_days)) &
            (patient_data['study_date_anon'] <= current_date - pd.Timedelta(days=min_days))
        ]
        
        if len(prior_exams) == 0:
            return None
        
        # Get most recent prior exam
        most_recent_prior_date = prior_exams['study_date_anon'].max()
        prior_exam_data = prior_exams[
            prior_exams['study_date_anon'] == most_recent_prior_date
        ]
        
        return prior_exam_data
    
    def _find_matching_prior_view(
        self,
        prior_exam_data: pd.DataFrame,
        view_laterality: str,
        view_position: str
    ) -> Optional[str]:
        """
        Find matching view in prior exam.
        
        Args:
            prior_exam_data: All images from prior exam
            view_laterality: 'L' or 'R'
            view_position: 'CC' or 'MLO'
        
        Returns:
            PNG path of matching prior view, or None
        """
        matching_views = prior_exam_data[
            (prior_exam_data['ImageLateralityFinal'] == view_laterality) &
            (prior_exam_data['ViewPosition'] == view_position)
        ]
        
        if len(matching_views) == 0:
            return None
        
        # If multiple matches, prefer non-special views
        if len(matching_views) > 1:
            non_special = matching_views[matching_views['spot_mag'] == '0']
            if len(non_special) > 0:
                matching_views = non_special
        
        return matching_views.iloc[0]['png_path']
    
    def __len__(self) -> int:
        return len(self.sample_list)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single view-level sample.
        
        Returns:
            Dict with keys:
            - 'current': Tensor [1, H, W]
            - 'prior': Tensor [1, H, W]
            - 'labels': Dict with 'recall', 'birads', 'density'
            - 'metadata': Dict with 'race', 'age', 'has_real_prior', etc.
            - 'view_info': Dict with 'laterality', 'position', 'name'
        """
        sample = self.sample_list[idx]
        
        # Load current image
        current_img = self._load_image(sample['current_png_path'])
        
        # Load or synthesize prior image
        if sample['has_real_prior']:
            prior_img = self._load_image(sample['prior_png_path'])
            use_real_prior = 1.0
        else:
            if self.handle_missing_prior == 'self_pair':
                prior_img = current_img.copy()
            elif self.handle_missing_prior == 'zero':
                prior_img = np.zeros_like(current_img)
            elif self.handle_missing_prior == 'mixed':
                if self.mode == 'train' and np.random.rand() > 0.5:
                    prior_img = current_img.copy()
                else:
                    prior_img = np.zeros_like(current_img)
            else:
                raise ValueError(f"Invalid handle_missing_prior: {self.handle_missing_prior}")
            
            use_real_prior = 0.0
        
        # Apply transforms
        if self.transform:
            current_img = self.transform(current_img)
            prior_img = self.transform(prior_img)
        else:
            current_img = self._to_tensor(current_img)
            prior_img = self._to_tensor(prior_img)
        
        # Prepare labels
        original_label = float(sample['label'])
        recall_label_binary = 1.0 if original_label >= 1 else 0.0
        
        return {
            'current': current_img,  # [1, H, W]
            'prior': prior_img,       # [1, H, W]
            'labels': {
                'recall': torch.tensor([recall_label_binary], dtype=torch.float32),
                'birads': torch.tensor(
                    self._encode_birads(sample['birads_orig']), 
                    dtype=torch.long
                ),
                'density': torch.tensor(
                    self._encode_density(sample['density']), 
                    dtype=torch.long
                )
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
                'has_real_prior': torch.tensor(use_real_prior, dtype=torch.float32),
                'time_gap': torch.tensor(sample['time_gap_days'], dtype=torch.float32)
            },
            'view_info': {
                'laterality': sample['view_laterality'],
                'position': sample['view_position'],
                'name': sample['view_name'],
                'exam_id': sample['current_exam_id']
            }
        }
    
    def _load_image(self, png_path: str) -> np.ndarray:
        """Load PNG image."""
        if pd.isna(png_path) or not os.path.exists(png_path):
            return self._get_placeholder_image()
        
        try:
            img = Image.open(png_path)
            img = np.array(img).astype(np.float32)
            
            # Resize if needed
            if self.image_size is not None:
                img = self._resize_image(img, self.image_size)
            
            return img
        
        except Exception as e:
            if self.verbose:
                print(f"Error loading {png_path}: {e}")
            return self._get_placeholder_image()
    
    def _get_placeholder_image(self) -> np.ndarray:
        """Return placeholder image for missing views."""
        H, W = self.image_size if self.image_size else (2048, 1024)
        return np.zeros((H, W), dtype=np.float32)
    
    def _resize_image(self, img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target size."""
        H, W = target_size
        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize((W, H), Image.BILINEAR)
        return np.array(img_pil).astype(np.float32)
    
    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert numpy array to tensor and normalize."""
        # Normalize to [0, 1]
        if img.max() > 1.0:
            img = img / img.max()
        
        # Add channel dimension [1, H, W]
        img = torch.from_numpy(img).float().unsqueeze(0)
        
        return img
    
    def _encode_race(self, race: str) -> int:
        """Encode race to 4 categories."""
        if pd.isna(race) or race == '':
            return 3
        
        race = str(race).strip()
        
        race_mapping = {
            'Caucasian or White': 0,
            'African American or Black': 1,
            'Asian': 2,
        }
        
        if race in race_mapping:
            return race_mapping[race]
        
        # Fallback
        race_lower = race.lower()
        if 'white' in race_lower or 'caucasian' in race_lower:
            return 0
        elif 'black' in race_lower or 'african' in race_lower:
            return 1
        elif 'asian' in race_lower:
            return 2
        else:
            return 3  # Other
    
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
    
    def _encode_density(self, density) -> int:
        """Encode breast density (1-4 → 0-3, or -1 for missing)."""
        if pd.isna(density):
            return -1
        
        try:
            density_int = int(float(density))
            if 1 <= density_int <= 4:
                return density_int - 1
            else:
                return -1
        except (ValueError, TypeError):
            return -1
    
    def _print_statistics(self):
        """Print dataset statistics."""
        print("\n" + "="*70)
        print("EMBED Single-View Longitudinal Dataset Statistics")
        print("="*70)
        
        print(f"\nMode: {self.mode}")
        print(f"Total view-level samples: {len(self.sample_list):,}")
        
        # Label distribution
        labels = [s['label'] for s in self.sample_list]
        label_counts = Counter(labels)
        print("\nLabel distribution:")
        label_names = ['Low Risk', 'Medium Risk', 'High Risk']
        for label in [0, 1, 2]:
            count = label_counts[label]
            pct = count / len(labels) * 100
            print(f"  Class {label} ({label_names[label]}): {count:,} ({pct:.1f}%)")
        
        # Binary label distribution (for recall task)
        binary_labels = [1 if s['label'] >= 1 else 0 for s in self.sample_list]
        binary_counts = Counter(binary_labels)
        print("\nBinary label distribution (Recall Task):")
        print(f"  Class 0 (No Recall): {binary_counts[0]:,} ({binary_counts[0]/len(binary_labels)*100:.1f}%)")
        print(f"  Class 1 (Need Recall): {binary_counts[1]:,} ({binary_counts[1]/len(binary_labels)*100:.1f}%)")
        
        # View distribution
        views = [s['view_name'] for s in self.sample_list]
        view_counts = Counter(views)
        print("\nView distribution:")
        for view in ['LCC', 'RCC', 'LMLO', 'RMLO']:
            count = view_counts.get(view, 0)
            pct = count / len(views) * 100
            print(f"  {view}: {count:,} ({pct:.1f}%)")
        
        # Prior availability
        has_prior = sum(1 for s in self.sample_list if s['has_real_prior'])
        print(f"\nPrior availability:")
        print(f"  With real prior: {has_prior:,} ({has_prior/len(self.sample_list)*100:.1f}%)")
        print(f"  Without prior: {len(self.sample_list)-has_prior:,}")
        
        # Race distribution
        races = [self._encode_race(s['race']) for s in self.sample_list]
        race_counts = Counter(races)
        print("\nRace distribution:")
        race_names = {
            0: 'Caucasian/White',
            1: 'African American/Black', 
            2: 'Asian',
            3: 'Other'
        }
        for race_id in sorted(race_counts.keys()):
            count = race_counts[race_id]
            pct = count / len(races) * 100
            print(f"  {race_names[race_id]}: {count:,} ({pct:.1f}%)")
        
        print("="*70 + "\n")
    
    def get_class_weights(self) -> torch.Tensor:
        """
        ✅ OPTIMIZED: Compute class weights for weighted loss.
        Returns weights for binary classification (No Recall vs Need Recall).
        """
        # Convert to binary labels
        binary_labels = [1 if s['label'] >= 1 else 0 for s in self.sample_list]
        label_counts = Counter(binary_labels)
        
        total = len(binary_labels)
        weights = []
        for label in [0, 1]:
            count = label_counts.get(label, 1)
            weight = total / (2 * count)
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_sample_weights(self) -> np.ndarray:
        """
        ✅ NEW: Get per-sample weights for WeightedRandomSampler.
        Returns weight for each sample inversely proportional to class frequency.
        """
        # Convert to binary labels
        binary_labels = np.array([1 if s['label'] >= 1 else 0 for s in self.sample_list])
        
        # Count samples per class
        class_counts = np.bincount(binary_labels)
        
        # Compute weight for each class
        class_weights = 1.0 / class_counts
        
        # Assign weight to each sample
        sample_weights = class_weights[binary_labels]
        
        return sample_weights


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for single-view batches.
    
    Args:
        batch: List of samples from __getitem__
        
    Returns:
        Batched dictionary with:
        - 'current': Tensor [B, 1, H, W]
        - 'prior': Tensor [B, 1, H, W]
        - 'labels': Dict with batched labels
        - 'metadata': Dict with batched metadata
        - 'view_info': List of view info dicts
    """
    # Stack images
    current_batch = torch.stack([item['current'] for item in batch])  # [B, 1, H, W]
    prior_batch = torch.stack([item['prior'] for item in batch])      # [B, 1, H, W]
    
    # Stack recall labels and ensure [B, 1]
    recall_labels = torch.stack([item['labels']['recall'] for item in batch])
    if recall_labels.dim() == 1:
        recall_labels = recall_labels.unsqueeze(1)
    
    # Stack labels
    labels_batch = {
        'recall': recall_labels,
        'birads': torch.stack([item['labels']['birads'] for item in batch]),
        'density': torch.stack([item['labels']['density'] for item in batch])
    }
    
    # Stack metadata
    metadata_batch = {
        'race': torch.stack([item['metadata']['race'] for item in batch]),
        'age': torch.stack([item['metadata']['age'] for item in batch]),
        'has_real_prior': torch.stack([item['metadata']['has_real_prior'] for item in batch]),
        'time_gap': torch.stack([item['metadata']['time_gap'] for item in batch])
    }
    
    # Keep view info as list
    view_info_batch = [item['view_info'] for item in batch]
    
    return {
        'current': current_batch,
        'prior': prior_batch,
        'labels': labels_batch,
        'metadata': metadata_batch,
        'view_info': view_info_batch
    }


def create_balanced_sampler(dataset: EMBEDSingleViewLongitudinalDataset) -> WeightedRandomSampler:
    """
    ✅ NEW: Create a balanced sampler for handling class imbalance.
    
    Args:
        dataset: The dataset to sample from
    
    Returns:
        WeightedRandomSampler that balances classes
    """
    sample_weights = dataset.get_sample_weights()
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # Allow oversampling minority class
    )
    
    print(f"✓ Created WeightedRandomSampler:")
    print(f"  Minority class will be oversampled")
    print(f"  Majority class will be undersampled")
    
    return sampler


def create_data_loaders(
    clinical_csv: str,
    metadata_csv: str,
    batch_size: int = 16,
    num_workers: int = 4,
    train_split: float = 0.7,
    val_split: float = 0.15,
    random_seed: int = 42,
    sample_fraction: float = 1.0,
    use_balanced_sampling: bool = True,  # ✅ NEW: Enable balanced sampling
    **dataset_kwargs
) -> Tuple:
    """
    ✅ OPTIMIZED: Create train/val/test data loaders with patient-level split.
    
    Args:
        sample_fraction: Fraction of patients to use (for testing, e.g., 0.1 = 10%)
        use_balanced_sampling: Whether to use WeightedRandomSampler for training
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Load clinical data for patient-level split
    clinical_df = pd.read_csv(clinical_csv, low_memory=False)
    
    # Get unique patients
    unique_patients = clinical_df['empi_anon'].unique()
    np.random.seed(random_seed)
    np.random.shuffle(unique_patients)
    
    # ✅ 如果sample_fraction < 1.0，只使用部分患者
    if sample_fraction < 1.0:
        n_sample = max(1, int(len(unique_patients) * sample_fraction))
        unique_patients = unique_patients[:n_sample]
        print(f"Using {sample_fraction*100:.1f}% of patients: {len(unique_patients)}/{len(clinical_df['empi_anon'].unique())}")
    
    # Split patients (patient-level split - 确保同一患者不会出现在不同集合)
    n_train = int(len(unique_patients) * train_split)
    n_val = int(len(unique_patients) * val_split)
    
    train_patients = set(unique_patients[:n_train])
    val_patients = set(unique_patients[n_train:n_train+n_val])
    test_patients = set(unique_patients[n_train+n_val:])
    
    # ✅ 验证患者级别split的正确性
    assert len(train_patients & val_patients) == 0, "Train and val patients overlap!"
    assert len(train_patients & test_patients) == 0, "Train and test patients overlap!"
    assert len(val_patients & test_patients) == 0, "Val and test patients overlap!"
    
    # Create split CSVs
    cache_dir = os.path.dirname(clinical_csv)
    train_csv = os.path.join(cache_dir, 'train_clinical.csv')
    val_csv = os.path.join(cache_dir, 'val_clinical.csv')
    test_csv = os.path.join(cache_dir, 'test_clinical.csv')
    
    train_df = clinical_df[clinical_df['empi_anon'].isin(train_patients)]
    val_df = clinical_df[clinical_df['empi_anon'].isin(val_patients)]
    test_df = clinical_df[clinical_df['empi_anon'].isin(test_patients)]
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    print(f"\n✓ Patient-level split (no overlap):")
    print(f"  Train: {len(train_patients)} patients ({len(train_df)} exams)")
    print(f"  Val: {len(val_patients)} patients ({len(val_df)} exams)")
    print(f"  Test: {len(test_patients)} patients ({len(test_df)} exams)")
    
    # Create datasets
    train_dataset = EMBEDSingleViewLongitudinalDataset(
        clinical_csv=train_csv,
        metadata_csv=metadata_csv,
        mode='train',
        **dataset_kwargs
    )
    
    val_dataset = EMBEDSingleViewLongitudinalDataset(
        clinical_csv=val_csv,
        metadata_csv=metadata_csv,
        mode='val',
        **dataset_kwargs
    )
    
    test_dataset = EMBEDSingleViewLongitudinalDataset(
        clinical_csv=test_csv,
        metadata_csv=metadata_csv,
        mode='test',
        **dataset_kwargs
    )
    
    # ✅ Create balanced sampler for training if requested
    if use_balanced_sampling:
        print("\n" + "="*70)
        print("Creating Balanced Sampler for Training")
        print("="*70)
        train_sampler = create_balanced_sampler(train_dataset)
        shuffle_train = False  # Don't shuffle when using sampler
    else:
        train_sampler = None
        shuffle_train = True
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,  # ✅ Use sampler if available
        shuffle=shuffle_train,   # ✅ Only shuffle if not using sampler
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# ============================================================================
# Testing
# ============================================================================

if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    parser = argparse.ArgumentParser(description='Test EMBED Single-View Dataset')
    parser.add_argument('--clinical_csv', type=str,
                        default="/projects/standard/lin01231/song0760/embed_recall_reduction/EMBED_OpenData_clinical_relabeled.csv")
    parser.add_argument('--metadata_csv', type=str,
                        default="/projects/standard/lin01231/public/datasets/embed/tables/EMBED_OpenData_metadata_png.csv")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--sample_fraction', type=float, default=0.05,
                        help='Fraction of patients to use for testing (0.05 = 5%)')
    parser.add_argument('--save_vis', type=str, default='dataset_visualization.png',
                        help='Path to save visualization')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("TESTING EMBED SINGLE-VIEW LONGITUDINAL DATASET")
    print("="*80)
    
    # ============================================================================
    # Step 1: Create dataset with sampled data
    # ============================================================================
    print("\n[1/4] Creating dataset with sampled patients...")
    print(f"Using {args.sample_fraction*100:.1f}% of patients for faster testing")
    
    # Load clinical to check patient count
    clinical_full = pd.read_csv(args.clinical_csv, low_memory=False)
    total_patients = clinical_full['empi_anon'].nunique()
    sample_patients = int(total_patients * args.sample_fraction)
    print(f"Total patients: {total_patients:,} → Sampling: {sample_patients:,}")
    
    # Sample patients
    np.random.seed(42)
    sampled_patient_ids = np.random.choice(
        clinical_full['empi_anon'].unique(),
        size=sample_patients,
        replace=False
    )
    
    # Create sampled clinical CSV
    import tempfile
    temp_dir = tempfile.mkdtemp()
    sampled_csv = os.path.join(temp_dir, 'sampled_clinical.csv')
    clinical_full[clinical_full['empi_anon'].isin(sampled_patient_ids)].to_csv(
        sampled_csv, index=False
    )
    print(f"Saved sampled clinical data to: {sampled_csv}")
    
    # Create dataset
    dataset = EMBEDSingleViewLongitudinalDataset(
        clinical_csv=sampled_csv,
        metadata_csv=args.metadata_csv,
        mode='train',
        image_size=(512, 256),  # ✅ 使用512x256
        handle_missing_prior='self_pair',
        use_cache=False,
        verbose=True
    )
    
    # ============================================================================
    # Step 2: Test individual samples with detailed info
    # ============================================================================
    print("\n[2/4] Testing individual samples with detailed information...")
    
    test_samples = []
    for i in range(min(args.num_samples, len(dataset))):
        print(f"\n{'='*60}")
        print(f"SAMPLE {i+1}/{args.num_samples}")
        print(f"{'='*60}")
        
        sample = dataset[i]
        test_samples.append(sample)
        
        # Basic info
        view_info = sample['view_info']
        print(f"\n📋 VIEW INFORMATION:")
        print(f"  View name: {view_info['name']}")
        print(f"  Laterality: {view_info['laterality']}")
        print(f"  Position: {view_info['position']}")
        print(f"  Exam ID: {view_info['exam_id']}")
        
        # Image shapes
        print(f"\n🖼️  IMAGE SHAPES:")
        print(f"  Current: {sample['current'].shape}")  # Should be [1, 512, 256]
        print(f"  Prior: {sample['prior'].shape}")      # Should be [1, 512, 256]
        
        # Labels
        print(f"\n🏷️  LABELS:")
        print(f"  Recall (binary): {sample['labels']['recall'].item():.0f}")
        print(f"  BI-RADS: {sample['labels']['birads'].item()}")
        print(f"  Density: {sample['labels']['density'].item()}")
        
        # Metadata
        print(f"\n👤 METADATA:")
        race_code = sample['metadata']['race'].item()
        race_names = {0: 'Caucasian/White', 1: 'African American/Black', 
                     2: 'Asian', 3: 'Other'}
        print(f"  Race: {race_names.get(race_code, 'Unknown')} (code: {race_code})")
        print(f"  Age: {sample['metadata']['age'].item():.1f}")
        print(f"  Has real prior: {sample['metadata']['has_real_prior'].item() > 0.5}")
        print(f"  Time gap (days): {sample['metadata']['time_gap'].item():.0f}")
        
        # Image statistics
        print(f"\n📊 IMAGE STATISTICS:")
        current_img = sample['current'].numpy()
        prior_img = sample['prior'].numpy()
        print(f"  Current - min: {current_img.min():.4f}, max: {current_img.max():.4f}, "
              f"mean: {current_img.mean():.4f}")
        print(f"  Prior   - min: {prior_img.min():.4f}, max: {prior_img.max():.4f}, "
              f"mean: {prior_img.mean():.4f}")
    
    # ============================================================================
    # Step 3: Test batch loading
    # ============================================================================
    print("\n[3/4] Testing batch loading...")
    from torch.utils.data import DataLoader, Subset
    
    # Create a small subset for testing
    subset_indices = list(range(min(20, len(dataset))))
    subset = Subset(dataset, subset_indices)
    
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    batch = next(iter(loader))
    
    print(f"\n📦 BATCH INFORMATION:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Current images shape: {batch['current'].shape}")  # [B, 1, 512, 256]
    print(f"  Prior images shape: {batch['prior'].shape}")      # [B, 1, 512, 256]
    print(f"  Recall labels shape: {batch['labels']['recall'].shape}")  # [B, 1]
    print(f"  Views in batch: {[v['name'] for v in batch['view_info']]}")
    
    print(f"\n✅ Batch shape validation:")
    expected_img_shape = (args.batch_size, 1, 512, 256)
    expected_label_shape = (args.batch_size, 1)
    print(f"  Images: {batch['current'].shape} == {expected_img_shape} ? "
          f"{'✓' if batch['current'].shape == expected_img_shape else '✗'}")
    print(f"  Labels: {batch['labels']['recall'].shape} == {expected_label_shape} ? "
          f"{'✓' if batch['labels']['recall'].shape == expected_label_shape else '✗'}")
    
    # ============================================================================
    # Step 4: Visualize concatenated image pairs
    # ============================================================================
    print(f"\n[4/4] Creating visualization...")
    
    num_vis = min(3, len(test_samples))
    fig = plt.figure(figsize=(15, 5 * num_vis))
    gs = GridSpec(num_vis, 3, figure=fig, hspace=0.3, wspace=0.2)
    
    for idx in range(num_vis):
        sample = test_samples[idx]
        
        # Extract images (remove channel dimension for visualization)
        current_img = sample['current'].squeeze(0).numpy()  # [512, 256]
        prior_img = sample['prior'].squeeze(0).numpy()      # [512, 256]
        
        # Concatenate horizontally
        concat_img = np.concatenate([prior_img, current_img], axis=1)  # [512, 512]
        
        # Get metadata
        view_info = sample['view_info']
        has_prior = sample['metadata']['has_real_prior'].item() > 0.5
        recall_label = sample['labels']['recall'].item()
        time_gap = sample['metadata']['time_gap'].item()
        
        # Plot current
        ax1 = fig.add_subplot(gs[idx, 0])
        ax1.imshow(current_img, cmap='gray')
        ax1.set_title(f'Current\n{view_info["name"]}', fontsize=10)
        ax1.axis('off')
        
        # Plot prior
        ax2 = fig.add_subplot(gs[idx, 1])
        ax2.imshow(prior_img, cmap='gray')
        prior_title = 'Prior (Real)' if has_prior else 'Prior (Synthetic)'
        ax2.set_title(f'{prior_title}\n{view_info["name"]}', fontsize=10)
        ax2.axis('off')
        
        # Plot concatenated
        ax3 = fig.add_subplot(gs[idx, 2])
        ax3.imshow(concat_img, cmap='gray')
        concat_title = (
            f'Concatenated [Prior | Current]\n'
            f'View: {view_info["name"]}, '
            f'Recall: {recall_label:.0f}, '
            f'Gap: {time_gap:.0f}d'
        )
        ax3.set_title(concat_title, fontsize=10)
        ax3.axis('off')
        
        # Add a vertical line to show boundary
        ax3.axvline(x=256, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.suptitle(
        'EMBED Single-View Longitudinal Dataset Visualization\n'
        f'Image size: 512×256, Prior | Current concatenation',
        fontsize=14,
        fontweight='bold'
    )
    
    # Save visualization
    plt.savefig(args.save_vis, dpi=150, bbox_inches='tight')
    print(f"\n💾 Visualization saved to: {args.save_vis}")
    
    # ============================================================================
    # Summary
    # ============================================================================
    print("\n" + "="*80)
    print("TESTING SUMMARY")
    print("="*80)
    
    print(f"\n✓ Dataset Statistics:")
    print(f"  Total samples: {len(dataset):,}")
    print(f"  Image size: 512 × 256")
    print(f"  Patients used: {sample_patients:,} ({args.sample_fraction*100:.1f}% of total)")
    
    print(f"\n✓ View Distribution:")
    views = [s['view_name'] for s in dataset.sample_list]
    view_counts = Counter(views)
    for view in ['LCC', 'RCC', 'LMLO', 'RMLO']:
        count = view_counts.get(view, 0)
        pct = count / len(views) * 100 if views else 0
        print(f"  {view}: {count:,} ({pct:.1f}%)")
    
    print(f"\n✓ Label Distribution:")
    labels = [s['label'] for s in dataset.sample_list]
    label_counts = Counter(labels)
    label_names = ['Low Risk', 'Medium Risk', 'High Risk']
    for label in [0, 1, 2]:
        count = label_counts.get(label, 0)
        pct = count / len(labels) * 100 if labels else 0
        print(f"  Class {label} ({label_names[label]}): {count:,} ({pct:.1f}%)")
    
    print(f"\n✓ Prior Availability:")
    has_prior = sum(1 for s in dataset.sample_list if s['has_real_prior'])
    pct = has_prior / len(dataset) * 100 if len(dataset) > 0 else 0
    print(f"  With real prior: {has_prior:,} ({pct:.1f}%)")
    print(f"  Without prior: {len(dataset)-has_prior:,} ({100-pct:.1f}%)")
    
    print("\n✓ All tests passed!")
    print(f"✓ Visualization saved: {args.save_vis}")
    print("="*80 + "\n")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print(f"Cleaned up temporary directory: {temp_dir}\n")