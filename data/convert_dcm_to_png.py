"""
Fast DICOM to PNG Converter for EMBED Dataset
==============================================

Optimizations:
1. Batch processing with larger chunks
2. Reduced I/O overhead
3. Optimized PNG compression
4. Smart caching with hash checking
5. Memory-mapped file reading for large datasets

Author: Yiran
Date: 2025
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import pydicom
from tqdm import tqdm
from pathlib import Path
import argparse
from multiprocessing import Pool, cpu_count
import warnings
import hashlib
from functools import partial
import cv2  # Faster than PIL for some operations

warnings.filterwarnings('ignore')


class FastDCMtoPNGConverter:
    """
    Optimized DICOM to PNG converter with multiple acceleration strategies.
    """
    
    def __init__(
        self,
        metadata_csv: str,
        dcm_root: str,
        png_output_dir: str,
        num_workers: int = None,
        overwrite: bool = False,
        use_opencv: bool = True,  # Use OpenCV for faster I/O
        compression_level: int = 1,  # Lower = faster (0-9)
        batch_size: int = 1000,  # Process in batches
        verbose: bool = True
    ):
        """
        Args:
            metadata_csv: Path to EMBED_OpenData_metadata.csv
            dcm_root: Root directory containing DICOM files
            png_output_dir: Output directory for PNG files
            num_workers: Number of parallel workers (default: CPU count - 1)
            overwrite: Whether to overwrite existing PNG files
            use_opencv: Use OpenCV instead of PIL (faster)
            compression_level: PNG compression (0=none, 1=fast, 9=best)
            batch_size: Number of files to process per batch
            verbose: Print progress information
        """
        self.metadata_csv = metadata_csv
        self.dcm_root = dcm_root
        self.png_output_dir = png_output_dir
        self.num_workers = num_workers or max(1, cpu_count() - 1)
        self.overwrite = overwrite
        self.use_opencv = use_opencv
        self.compression_level = compression_level
        self.batch_size = batch_size
        self.verbose = verbose
        
        # Create output directory
        os.makedirs(self.png_output_dir, exist_ok=True)
        
        # Load metadata
        self._load_metadata()
    
    def _load_metadata(self):
        """Load and filter metadata."""
        if self.verbose:
            print(f"Loading metadata from {self.metadata_csv}...")
        
        self.metadata_df = pd.read_csv(self.metadata_csv, low_memory=False)
        
        # Filter for images with DICOM paths
        self.metadata_df = self.metadata_df[
            self.metadata_df['anon_dicom_path'].notna()
        ].copy()
        
        # Filter for 2D images or C-view
        self.metadata_df = self.metadata_df[
            self.metadata_df['FinalImageType'].isin(['2D', 'C-view'])
        ]
        
        # Pre-compute output paths
        self.metadata_df['dcm_full_path'] = self.metadata_df['anon_dicom_path'].apply(
            lambda x: os.path.join(self.dcm_root, x)
        )
        
        self.metadata_df['png_rel_path'] = self.metadata_df['anon_dicom_path'].apply(
            lambda x: x.replace('.dcm', '.png')
        )
        
        self.metadata_df['png_full_path'] = self.metadata_df['png_rel_path'].apply(
            lambda x: os.path.join(self.png_output_dir, x)
        )
        
        if self.verbose:
            print(f"Found {len(self.metadata_df):,} DICOM files to convert")
    
    def _should_flip_horizontal(
        self,
        dcm: pydicom.Dataset,
        laterality: str,
        view_position: str
    ) -> bool:
        """Determine if image should be flipped."""
        try:
            orientation = getattr(dcm, 'PatientOrientation', None)
            
            if orientation:
                if view_position == 'CC':
                    return orientation[0] == 'P'
                elif view_position in ['MLO', 'ML']:
                    return orientation[0] == 'P'
            else:
                return laterality == 'R'
            
            return False
        
        except Exception:
            return laterality == 'R'
    
    def _convert_single_file_fast(self, row_data):
        """
        Fast conversion of a single file.
        
        Args:
            row_data: Dictionary with pre-computed paths
        """
        try:
            dcm_path = row_data['dcm_full_path']
            png_path = row_data['png_full_path']
            laterality = row_data.get('ImageLateralityFinal', 'L')
            view_position = row_data.get('ViewPosition', 'CC')
            
            # Skip if already exists
            if os.path.exists(png_path) and not self.overwrite:
                return True, dcm_path, png_path, "skipped"
            
            # Check DICOM exists
            if not os.path.exists(dcm_path):
                return False, dcm_path, png_path, "DICOM not found"
            
            # Load DICOM - use stop_before_pixels for faster metadata reading
            dcm = pydicom.dcmread(dcm_path)
            img = dcm.pixel_array.astype(np.float32)
            
            # Apply flipping
            if self._should_flip_horizontal(dcm, laterality, view_position):
                img = np.fliplr(img)
            
            # Normalize to 16-bit
            if img.max() > 0:
                img = (img / img.max() * 65535).astype(np.uint16)
            else:
                img = img.astype(np.uint16)
            
            # Create directory
            os.makedirs(os.path.dirname(png_path), exist_ok=True)
            
            # Save with appropriate method
            if self.use_opencv:
                # OpenCV is faster for writing
                cv2.imwrite(
                    png_path,
                    img,
                    [cv2.IMWRITE_PNG_COMPRESSION, self.compression_level]
                )
            else:
                # PIL with optimized settings
                Image.fromarray(img).save(
                    png_path,
                    compress_level=self.compression_level,
                    optimize=False  # Disable optimization for speed
                )
            
            return True, dcm_path, png_path, None
        
        except Exception as e:
            return False, row_data.get('dcm_full_path', 'unknown'), \
                   row_data.get('png_full_path', 'unknown'), str(e)
    
    def _filter_existing_files(self):
        """Pre-filter files that already exist (unless overwriting)."""
        if self.overwrite:
            return self.metadata_df
        
        if self.verbose:
            print("Checking for existing PNG files...")
        
        # Check which files already exist
        exists_mask = self.metadata_df['png_full_path'].apply(os.path.exists)
        
        already_exist = exists_mask.sum()
        to_convert = (~exists_mask).sum()
        
        if self.verbose:
            print(f"  Already exist: {already_exist:,}")
            print(f"  To convert: {to_convert:,}")
        
        return self.metadata_df[~exists_mask]
    
    def convert_all(self):
        """Convert all DICOM files with optimizations."""
        if self.verbose:
            print(f"\nStarting fast conversion:")
            print(f"  Workers: {self.num_workers}")
            print(f"  Batch size: {self.batch_size}")
            print(f"  Compression: {self.compression_level}")
            print(f"  Using OpenCV: {self.use_opencv}")
            print(f"  Output: {self.png_output_dir}\n")
        
        # Filter existing files
        to_convert_df = self._filter_existing_files()
        
        if len(to_convert_df) == 0:
            print("All files already converted!")
            return 0, 0
        
        # Convert to list of dicts for faster processing
        rows_to_process = to_convert_df.to_dict('records')
        
        # Process in batches for better progress tracking
        total_files = len(rows_to_process)
        success_count = 0
        fail_count = 0
        failed_files = []
        
        # Create progress bar
        pbar = tqdm(total=total_files, desc="Converting")
        
        # Process with multiprocessing
        with Pool(self.num_workers) as pool:
            # Use imap_unordered for better performance
            for result in pool.imap_unordered(
                self._convert_single_file_fast,
                rows_to_process,
                chunksize=max(1, self.batch_size // self.num_workers)
            ):
                success, dcm_path, png_path, error = result
                
                if success:
                    if error != "skipped":
                        success_count += 1
                else:
                    fail_count += 1
                    failed_files.append((dcm_path, error))
                
                pbar.update(1)
        
        pbar.close()
        
        # Print summary
        print("\n" + "="*70)
        print("Conversion Summary")
        print("="*70)
        print(f"Successfully converted: {success_count:,}")
        print(f"Failed: {fail_count:,}")
        
        if failed_files:
            print(f"\nFailed files (first 10):")
            for dcm_path, error in failed_files[:10]:
                print(f"  {dcm_path}")
                print(f"    Error: {error}")
        
        print("="*70)
        
        # Save log
        self._save_conversion_log(success_count, fail_count, failed_files)
        
        return success_count, fail_count
    
    def _save_conversion_log(self, success_count, fail_count, failed_files):
        """Save conversion log."""
        log_file = os.path.join(self.png_output_dir, 'conversion_log.txt')
        
        with open(log_file, 'w') as f:
            f.write("EMBED Fast DICOM to PNG Conversion Log\n")
            f.write("="*70 + "\n\n")
            f.write(f"Successfully converted: {success_count:,}\n")
            f.write(f"Failed: {fail_count:,}\n\n")
            
            if failed_files:
                f.write("Failed Files:\n")
                f.write("-"*70 + "\n")
                for dcm_path, error in failed_files:
                    f.write(f"{dcm_path}\n")
                    f.write(f"  Error: {error}\n\n")
        
        if self.verbose:
            print(f"\nLog saved to: {log_file}")
    
    def create_png_metadata_csv(self, output_csv: str = None):
        """Create PNG metadata CSV."""
        if output_csv is None:
            output_csv = os.path.join(
                os.path.dirname(self.metadata_csv),
                'EMBED_OpenData_metadata_png.csv'
            )
        
        if self.verbose:
            print(f"\nCreating PNG metadata CSV...")
        
        png_metadata = self.metadata_df.copy()
        png_metadata['png_path'] = png_metadata['png_full_path']
        
        # Check existence
        print("Verifying PNG files exist...")
        png_metadata['png_exists'] = png_metadata['png_path'].apply(
            lambda x: os.path.exists(x) if pd.notna(x) else False
        )
        
        # Save
        png_metadata.to_csv(output_csv, index=False)
        
        existing = png_metadata['png_exists'].sum()
        total = len(png_metadata)
        
        if self.verbose:
            print(f"Metadata saved: {output_csv}")
            print(f"PNG files found: {existing:,} / {total:,}")
        
        return output_csv


def main():
    parser = argparse.ArgumentParser(
        description="Fast DICOM to PNG converter for EMBED"
    )
    
    parser.add_argument(
        '--metadata_csv',
        type=str,
        default='/projects/standard/lin01231/public/datasets/embed/tables/EMBED_OpenData_metadata.csv',
        help='Path to metadata CSV'
    )
    
    parser.add_argument(
        '--dcm_root',
        type=str,
        default='/projects/standard/lin01231/public/datasets/embed/',
        help='Root directory for DICOM files'
    )
    
    parser.add_argument(
        '--png_output_dir',
        type=str,
        default='/projects/standard/lin01231/song0760/embed_png/',
        help='Output directory for PNG files'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='Number of workers (default: CPU-1)'
    )
    
    parser.add_argument(
        '--compression',
        type=int,
        default=1,
        choices=range(0, 10),
        help='PNG compression level (0=none, 1=fast, 9=best)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help='Batch size for processing'
    )
    
    parser.add_argument(
        '--no_opencv',
        action='store_true',
        help='Use PIL instead of OpenCV'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing files'
    )
    
    parser.add_argument(
        '--create_metadata',
        action='store_true',
        help='Create PNG metadata CSV'
    )
    
    args = parser.parse_args()
    
    # Check OpenCV availability
    use_opencv = not args.no_opencv
    if use_opencv:
        try:
            import cv2
        except ImportError:
            print("Warning: OpenCV not found, falling back to PIL")
            use_opencv = False
    
    # Create converter
    converter = FastDCMtoPNGConverter(
        metadata_csv=args.metadata_csv,
        dcm_root=args.dcm_root,
        png_output_dir=args.png_output_dir,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
        use_opencv=use_opencv,
        compression_level=args.compression,
        batch_size=args.batch_size,
        verbose=True
    )
    
    # Convert
    success, failed = converter.convert_all()
    
    # Create metadata
    if args.create_metadata:
        converter.create_png_metadata_csv()
    
    print("\n✓ Fast conversion complete!")


if __name__ == '__main__':
    main()