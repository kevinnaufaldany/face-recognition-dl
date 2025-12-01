"""
Face Detection and Cropping using MediaPipe
Deteksi wajah, cropping dengan padding 20%, dan resize ke 224x224
"""

import os
import sys
import warnings

# Suppress all warnings BEFORE importing any libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GLOG_minloglevel'] = '3'
warnings.filterwarnings('ignore')

# Redirect stderr to suppress C++ warnings
import io
import contextlib

# Suppress absl logging
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil


class FaceCropper:
    """
    Face detection dan cropping menggunakan MediaPipe
    Multi-strategy: model_selection 0 & 1, confidence tuning, fallback to center crop
    """
    
    def __init__(self, padding_percent=20, target_size=(224, 224)):
        """
        Args:
            padding_percent: Persentase padding di sekitar wajah (default 20%)
            target_size: Ukuran target setelah resize (default 224x224)
        """
        self.padding_percent = padding_percent
        self.target_size = target_size
        
        # Suppress stderr during MediaPipe initialization
        stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        
        try:
            # Initialize MediaPipe Face Detection - Primary (full range)
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection_full = self.mp_face_detection.FaceDetection(
                model_selection=1,  # Full range detection (0-5 meters)
                min_detection_confidence=0.1  # VERY LOW - aggressive detection
            )
            
            # Secondary detector for close-range (0-2 meters)
            self.face_detection_close = self.mp_face_detection.FaceDetection(
                model_selection=0,  # Short range detection
                min_detection_confidence=0.1  # VERY LOW - aggressive detection
            )
            
            # Dummy call to trigger all initialization warnings
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            self.face_detection_full.process(dummy_img)
            self.face_detection_close.process(dummy_img)
            
        finally:
            # Restore stderr
            sys.stderr.close()
            sys.stderr = stderr
    
    def detect_and_crop_face_from_pil(self, pil_image):
        """
        Deteksi wajah dan crop dari PIL Image
        
        Args:
            pil_image: PIL Image (RGB format)
            
        Returns:
            cropped_face: numpy array BGR (224x224)
            success: True jika wajah terdeteksi, False jika tidak
        """
        try:
            # Convert PIL RGB ke OpenCV BGR
            import cv2
            image_array = np.array(pil_image)  # Convert PIL to numpy (RGB)
            image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        except Exception as e:
            return None, False
        
        if image is None:
            return None, False
        
        # Lanjutkan dengan logic deteksi sama seperti sebelumnya
        return self._detect_and_crop_from_cv2(image)
    
    def detect_and_crop_face(self, image_path):
        """
        Deteksi wajah dan crop dengan padding - Multi-strategy approach
        Support: jpg, jpeg, png, bmp, webp
        
        Args:
            image_path: Path ke gambar input
            
        Returns:
            cropped_face: Gambar wajah yang sudah di-crop dan resize (224x224)
            success: True jika wajah terdeteksi, False jika tidak
        """
        # Baca gambar - OpenCV support webp secara native
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        
        # Fallback untuk webp jika OpenCV gagal
        if image is None:
            try:
                from PIL import Image as PILImage
                pil_img = PILImage.open(image_path).convert('RGB')
                image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception:
                return None, False
        
        if image is None:
            return None, False
        
        return self._detect_and_crop_from_cv2(image)
    
    def _detect_and_crop_from_cv2(self, image):
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        
        bbox = None
        
        # Strategy 1: Full range detector with confidence 0.1
        results = self.face_detection_full.process(image_rgb)
        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box
            print(f"[DEBUG] Strategy 1 (full 0.1): DETECTED")
        else:
            print(f"[DEBUG] Strategy 1 (full 0.1): Not detected")
        
        # Strategy 2: Close range detector dengan confidence 0.1
        if bbox is None:
            results = self.face_detection_close.process(image_rgb)
            if results.detections:
                bbox = results.detections[0].location_data.relative_bounding_box
                print(f"[DEBUG] Strategy 2 (close 0.1): DETECTED")
            else:
                print(f"[DEBUG] Strategy 2 (close 0.1): Not detected")
        
        # Strategy 3: Even lower confidence (0.05) with full range
        if bbox is None:
            try:
                temp_detector = self.mp_face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=0.05
                )
                results = temp_detector.process(image_rgb)
                if results.detections:
                    bbox = results.detections[0].location_data.relative_bounding_box
                    print(f"[DEBUG] Strategy 3 (full 0.05): DETECTED")
                else:
                    print(f"[DEBUG] Strategy 3 (full 0.05): Not detected")
                temp_detector.close()
            except:
                print(f"[DEBUG] Strategy 3: Error")
        
        # Strategy 4: Fallback to intelligent center crop
        if bbox is None:
            print(f"[DEBUG] Strategy 4: Using center crop fallback")
            # Crop center dengan aspect ratio untuk portrait
            # Asumsi: wajah biasanya di tengah gambar portrait
            crop_h = int(h * 0.8)  # 80% tinggi
            crop_w = int(w * 0.8)  # 80% lebar
            
            # Center crop
            y1 = (h - crop_h) // 2
            x1 = (w - crop_w) // 2
            y2 = y1 + crop_h
            x2 = x1 + crop_w
            
            face_crop = image[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                # Last resort: resize entire image
                face_resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
                return face_resized, True
            
            face_resized = cv2.resize(face_crop, self.target_size, interpolation=cv2.INTER_AREA)
            return face_resized, True
        
        # Convert relative coordinates to absolute
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        box_w = int(bbox.width * w)
        box_h = int(bbox.height * h)
        
        print(f"[DEBUG] Face bbox: x={x}, y={y}, w={box_w}, h={box_h}")
        
        # Tambahkan padding 20%
        padding_w = int(box_w * self.padding_percent / 100)
        padding_h = int(box_h * self.padding_percent / 100)
        
        # Calculate padded coordinates
        x1 = max(0, x - padding_w)
        y1 = max(0, y - padding_h)
        x2 = min(w, x + box_w + padding_w)
        y2 = min(h, y + box_h + padding_h)
        
        print(f"[DEBUG] Crop coords: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        
        # Crop wajah
        face_crop = image[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            # Fallback: resize entire image
            face_resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
            return face_resized, True
        
        print(f"[DEBUG] Crop successful: {face_crop.shape} -> resizing to {self.target_size}")
        
        # Resize ke target size (224x224)
        face_resized = cv2.resize(face_crop, self.target_size, interpolation=cv2.INTER_AREA)
        
        return face_resized, True
    
    def process_dataset(self, input_dir, output_dir, skip_existing=True):
        """
        Process seluruh dataset (deteksi dan crop semua gambar)
        
        Args:
            input_dir: Direktori input berisi gambar asli
            output_dir: Direktori output untuk menyimpan hasil crop
            skip_existing: Skip jika file output sudah ada
            
        Returns:
            stats: Dictionary berisi statistik processing
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Hitung total files - SUPPORT WEBP!
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        all_files = []
        
        # Scan semua file dengan extension case-insensitive
        for file_path in input_path.rglob('*'):
            if file_path.suffix.lower() in image_extensions:
                all_files.append(file_path)
        
        total_files = len(all_files)
        successful = 0
        failed = 0
        skipped = 0
        
        print(f"\nMemproses {total_files} gambar dari {input_dir}")
        print(f"Output akan disimpan di {output_dir}")
        print(f"Padding: {self.padding_percent}%, Target size: {self.target_size}\n")
        
        # Process setiap gambar
        failed_files = []
        
        for img_path in tqdm(all_files, desc="Processing images"):
            # Buat path output dengan struktur direktori yang sama
            relative_path = img_path.relative_to(input_path)
            output_file = output_path / relative_path
            
            # Skip jika sudah ada
            if skip_existing and output_file.exists():
                skipped += 1
                continue
            
            # Buat direktori output jika belum ada
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Deteksi dan crop wajah
            face_cropped, success = self.detect_and_crop_face(img_path)
            
            if success:
                # Simpan hasil crop
                cv2.imwrite(str(output_file), face_cropped)
                successful += 1
            else:
                failed += 1
                failed_files.append(str(relative_path))
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Total files: {total_files}")
        print(f"Successfully processed: {successful}")
        print(f"Failed (no face detected): {failed}")
        print(f"Skipped (already exists): {skipped}")
        print(f"{'='*60}")
        
        if failed_files:
            print(f"\nFailed files ({len(failed_files)}):")
            for f in failed_files[:20]:  # Show first 20
                print(f"  - {f}")
            if len(failed_files) > 20:
                print(f"  ... and {len(failed_files) - 20} more")
        
        return {
            'total': total_files,
            'successful': successful,
            'failed': failed,
            'skipped': skipped,
            'failed_files': failed_files
        }
    
    def __del__(self):
        """Cleanup"""
        try:
            self.face_detection_full.close()
            self.face_detection_close.close()
        except:
            pass


def main():
    """
    Contoh penggunaan untuk memproses dataset
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Detection and Cropping using MediaPipe')
    parser.add_argument('--input', type=str, required=True, help='Input directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--padding', type=int, default=20, help='Padding percentage (default: 20)')
    parser.add_argument('--size', type=int, default=224, help='Target size (default: 224)')
    parser.add_argument('--skip-existing', action='store_true', help='Skip if output already exists')
    
    args = parser.parse_args()
    
    # Create face cropper
    cropper = FaceCropper(
        padding_percent=args.padding,
        target_size=(args.size, args.size)
    )
    
    # Process dataset
    stats = cropper.process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        skip_existing=args.skip_existing
    )
    
    # Save stats to JSON
    import json
    stats_file = Path(args.output) / 'processing_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nStatistik disimpan di {stats_file}")


if __name__ == '__main__':
    main()
