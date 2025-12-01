"""
Preprocess satu gambar: deteksi wajah, crop + resize 224x224.
Tanpa MediaPipe (pakai OpenCV Haar Cascade) → aman dari error protobuf.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np


class HaarFaceCropper:
    """
    Deteksi wajah menggunakan Haar Cascade dari OpenCV.
    Cocok untuk preprocessing 1 gambar (single image).
    """

    def __init__(self, padding_percent=20, target_size=(224, 224)):
        self.padding_percent = padding_percent
        self.target_size = target_size

        # Load Haar Cascade bawaan OpenCV
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(haar_path)

        if self.face_cascade.empty():
            raise RuntimeError(f"Gagal load Haar Cascade dari: {haar_path}")

    def detect_and_crop(self, image):
        """
        Deteksi wajah dan crop dari image (numpy BGR).
        Return (cropped_face, success: bool)
        """
        if image is None:
            return None, False

        h, w, _ = image.shape

        # Konversi ke grayscale untuk Haar
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Deteksi wajah
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            # Fallback → center crop 80%
            crop_h = int(h * 0.8)
            crop_w = int(w * 0.8)
            y1 = (h - crop_h) // 2
            x1 = (w - crop_w) // 2
            y2 = y1 + crop_h
            x2 = x1 + crop_w

            face = image[y1:y2, x1:x2]
            if face.size == 0:
                face = image

            face = cv2.resize(face, self.target_size, interpolation=cv2.INTER_AREA)
            return face, True

        # Ambil wajah dengan area terbesar
        areas = [w_ * h_ for (x_, y_, w_, h_) in faces]
        idx_max = int(np.argmax(areas))
        (x, y, bw, bh) = faces[idx_max]

        # Padding
        pad_w = int(bw * self.padding_percent / 100)
        pad_h = int(bh * self.padding_percent / 100)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w, x + bw + pad_w)
        y2 = min(h, y + bh + pad_h)

        face = image[y1:y2, x1:x2]
        if face.size == 0:
            face = image

        face = cv2.resize(face, self.target_size, interpolation=cv2.INTER_AREA)
        return face, True


def process_single_image(input_path, output_path, padding=20):
    print(f"\n🔍 Memproses: {input_path}")

    # Baca gambar
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)

    # Fallback kalau OpenCV gagal (misal WEBP)
    if image is None:
        try:
            from PIL import Image
            pil = Image.open(input_path).convert("RGB")
            image = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"❌ Gagal membuka gambar: {e}")
            return False

    cropper = HaarFaceCropper(
        padding_percent=padding,
        target_size=(224, 224),
    )

    face, ok = cropper.detect_and_crop(image)
    if not ok:
        print("❌ Gagal memproses wajah.")
        return False

    # Pastikan folder output ada
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Simpan gambar
    cv2.imwrite(str(output_path), face)
    print(f"✅ Berhasil disimpan ke: {output_path}")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess satu gambar: deteksi wajah + crop + resize 224x224 (Haar Cascade)."
    )
    parser.add_argument("--image", type=str, required=True, help="Path gambar input")
    parser.add_argument("--output", type=str, required=True, help="Path gambar output")
    parser.add_argument("--padding", type=int, default=20, help="Padding (%) di sekitar wajah")

    args = parser.parse_args()

    ok = process_single_image(args.image, args.output, padding=args.padding)
    if not ok:
        sys.exit(1)
