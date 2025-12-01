import cv2
import numpy as np
from pathlib import Path

class HaarFaceCropper:
    """
    Deteksi wajah menggunakan Haar Cascade (OpenCV).
    Aman untuk Windows & tidak butuh Mediapipe.
    """

    def __init__(self, padding_percent=20, target_size=(224, 224)):
        self.padding_percent = padding_percent
        self.target_size = target_size

        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(haar_path)

        if self.face_cascade.empty():
            raise RuntimeError(f"Gagal load Haar Cascade dari: {haar_path}")

    def crop_from_pil(self, pil_image):
        """Deteksi wajah dari PIL Image."""
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return self.detect_and_crop(image)

    def detect_and_crop(self, image):
        """Deteksi wajah & crop dari numpy array BGR"""

        h, w, _ = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # Jika tidak ada wajah → fallback center crop
        if len(faces) == 0:
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
            return face, False  # False = wajah tidak terdeteksi

        # Ambil wajah terbesar
        areas = [w_ * h_ for (_, _, w_, h_) in faces]
        idx = int(np.argmax(areas))
        (x, y, bw, bh) = faces[idx]

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
        return face, True  # True = wajah terdeteksi
