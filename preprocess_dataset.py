"""
Script untuk preprocessing dataset menggunakan MediaPipe Face Detection
Jalankan script ini SEBELUM training untuk crop dan resize semua gambar
"""

import sys
import os
import warnings

# Suppress all warnings BEFORE importing anything
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GLOG_minloglevel'] = '3'
warnings.filterwarnings('ignore')

# Suppress absl logging
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils.face_crop import FaceCropper


def main():
    """
    Preprocess dataset Train dengan MediaPipe
    """
    print("="*70)
    print("PREPROCESSING DATASET - MediaPipe Face Detection")
    print("="*70)
    
    # Configuration
    INPUT_DIR = "dataset/Train"
    OUTPUT_DIR = "dataset/Train_Cropped"
    PADDING_PERCENT = 20
    TARGET_SIZE = 224
    
    print(f"\nKonfigurasi:")
    print(f"  Input directory: {INPUT_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Padding: {PADDING_PERCENT}%")
    print(f"  Target size: {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"\nMenginisialisasi MediaPipe Face Detection...")
    
    # Suppress stderr during initialization
    import sys
    stderr_backup = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    
    try:
        # Create face cropper (this will suppress TF warnings)
        cropper = FaceCropper(
            padding_percent=PADDING_PERCENT,
            target_size=(TARGET_SIZE, TARGET_SIZE)
        )
    finally:
        # Restore stderr
        sys.stderr.close()
        sys.stderr = stderr_backup
    
    print("✓ MediaPipe berhasil diinisialisasi")
    print(f"\nMemulai preprocessing...")
    
    # Process dataset
    stats = cropper.process_dataset(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        skip_existing=True  # Skip file yang sudah diproses
    )
    
    # Save statistics
    import json
    stats_file = Path(OUTPUT_DIR) / 'processing_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Preprocessing selesai!")
    print(f"✓ Statistik disimpan di: {stats_file}")
    print(f"\nSelanjutnya:")
    print(f"  1. Check hasil di folder: {OUTPUT_DIR}")
    print(f"  2. Jika ada failed files, cek manual gambar tersebut")
    print(f"  3. Jalankan training: python train_swin.py")
    print(f"     (pastikan data_dir='dataset/Train_Cropped')")
    print("="*70)


if __name__ == '__main__':
    main()
