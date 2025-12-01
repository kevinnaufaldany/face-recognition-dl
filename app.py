import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import streamlit as st
from pathlib import Path
import numpy as np
import sys
import os
import warnings
import logging
import cv2

# ============ SUPPRESS ALL WARNINGS & LOGS ============
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GLOG_minloglevel'] = '3'
warnings.filterwarnings('ignore')

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# Suppress stderr during import
stderr_backup = sys.stderr
sys.stderr = open(os.devnull, 'w')

try:
    from model_convnext import create_model
    from utils.face_crop import FaceCropper
finally:
    sys.stderr.close()
    sys.stderr = stderr_backup

# ===================== #
# 1. CONFIG DASAR
# ===================== #

CHECKPOINT_PATH = r"checkpoints/convnext_tiny_20251201_144518/best_epoch7.pth"
CLASS_NAMES_PATH = "class_names.txt"
IMAGE_SIZE = 224
NUM_CLASSES = 70

# ===================== #
# 2. FUNGSI UTIL
# ===================== #

@st.cache_resource
def load_face_cropper():
    """Load face cropper dengan suppression"""
    try:
        stderr_backup = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        
        try:
            cropper = FaceCropper(padding_percent=20, target_size=(IMAGE_SIZE, IMAGE_SIZE))
            return cropper
        finally:
            sys.stderr.close()
            sys.stderr = stderr_backup
    except:
        return None

@st.cache_data
def load_class_names(path: str):
    with open(path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]
    return class_names

@st.cache_resource
def load_model():
    """Load model dengan struktur yang sama dengan training"""
    model = create_model(num_classes=NUM_CLASSES, pretrained=False, dropout=0.3, device='cpu')
    
    state = torch.load(CHECKPOINT_PATH, map_location="cpu")
    
    if isinstance(state, dict):
        if "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        elif "state_dict" in state:
            state_dict = state["state_dict"]
        else:
            state_dict = state
    else:
        state_dict = state
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

def get_transform():
    """Transform sesuai ImageNet"""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

def pil_to_cv2(pil_image):
    """Convert PIL Image to OpenCV BGR format"""
    rgb_image = np.array(pil_image)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return bgr_image

def cv2_to_pil(cv2_image):
    """Convert OpenCV BGR to PIL Image"""
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    return pil_image

def preprocess_image(image_pil, cropper):
    """
    Pipeline preprocessing seperti preprocess_dataset.py
    1. Convert PIL ke OpenCV
    2. Deteksi & crop wajah dengan MediaPipe
    3. Resize ke 224x224
    
    Returns:
        image_processed (numpy BGR), success (bool), face_detected (bool)
    """
    try:
        # Convert PIL ke OpenCV BGR
        image_cv2 = pil_to_cv2(image_pil)
        
        # Suppress stderr saat cropping
        stderr_backup = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        
        try:
            # Gunakan detect_and_crop_face
            face_cropped, success = cropper.detect_and_crop_face(image_pil)
        finally:
            sys.stderr.close()
            sys.stderr = stderr_backup
        
        if success and face_cropped is not None:
            # Wajah terdeteksi dan di-crop
            return face_cropped, True, True
        else:
            # Fallback: resize image original tanpa cropping
            image_resized = cv2.resize(image_cv2, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            return image_resized, True, False
            
    except Exception as e:
        # Jika ada error, gunakan image original resize
        image_cv2 = pil_to_cv2(image_pil)
        image_resized = cv2.resize(image_cv2, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        return image_resized, True, False

def predict_image(image_pil, model, class_names, cropper):
    """
    Prediksi dengan preprocessing pipeline
    
    Returns:
        pred_name, confidence, probs, top5_names, top5_confs, face_detected
    """
    # Preprocessing
    image_processed, preprocess_ok, face_detected = preprocess_image(image_pil, cropper)
    
    # Convert dari numpy (BGR) ke PIL (RGB) untuk transform
    if isinstance(image_processed, np.ndarray):
        image_for_model = cv2_to_pil(image_processed)
    else:
        image_for_model = image_processed
    
    # Transform ke tensor
    transform = get_transform()
    img_t = transform(image_for_model).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        logits = model(img_t)
        probs = torch.softmax(logits, dim=1)[0]
    
    # Top 1
    conf, idx = torch.max(probs, dim=0)
    pred_name = class_names[idx.item()]
    
    # Top 5
    probs_np = probs.numpy()
    top5_idx = np.argsort(probs_np)[-5:][::-1]
    top5_names = [class_names[i] for i in top5_idx]
    top5_confs = probs_np[top5_idx]
    
    return pred_name, conf.item(), probs_np, top5_names, top5_confs, face_detected

# ===================== #
# 3. SETTING HALAMAN
# ===================== #

st.set_page_config(
    page_title="Face Recognition Mahasiswa",
    page_icon="📸",
    layout="centered"
)

# ===================== #
# 4. LOAD MODEL & LABEL
# ===================== #

try:
    class_names = load_class_names(CLASS_NAMES_PATH)
    assert len(class_names) == NUM_CLASSES
except Exception as e:
    st.error(f"❌ Error load class_names: {e}")
    st.stop()

try:
    face_cropper = load_face_cropper()
except Exception as e:
    face_cropper = None

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Error load model: {e}")
    st.stop()

# ===================== #
# 5. UI UTAMA
# ===================== #

st.title("📸 Face Recognition Mahasiswa")
st.caption("Upload foto wajah untuk prediksi otomatis")

# Info sidebar
with st.sidebar:
    st.subheader("ℹ️ Model Info")
    st.write("- **Model**: ConvNeXt-Tiny")
    st.write("- **Akurasi**: 70.00%")
    st.write(f"- **Kelas**: {NUM_CLASSES} mahasiswa")
    st.write("- **Input**: 224×224 pixels")
    st.markdown("---")
    st.subheader("📝 Preprocessing Pipeline")
    st.write("""
    1. Upload gambar
    2. Deteksi wajah (MediaPipe)
    3. Crop & normalize
    4. Prediksi model
    5. Tampilkan hasil
    """)

# Upload gambar
st.subheader("📤 Upload Gambar")
uploaded_file = st.file_uploader(
    "Pilih foto wajah",
    type=["jpg", "jpeg", "png"],
    help="Format: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    # Baca gambar
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"❌ Error membaca gambar: {e}")
        st.stop()
    
    # Tampilkan preview
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Foto Original", use_container_width=True)
    
    with col2:
        st.write("**Info Gambar:**")
        st.write(f"- Ukuran: {image.size}")
        st.write(f"- Format: {image.format}")
        st.write(f"- Mode: {image.mode}")
    
    st.markdown("---")
    
    # Button prediksi
    if st.button("🚀 Prediksi", key="predict_btn", use_container_width=True):
        with st.spinner("🔄 Sedang memproses..."):
            try:
                # Prediksi
                pred_name, confidence, probs_all, top5_names, top5_confs, face_detected = predict_image(
                    image, model, class_names, face_cropper
                )
                
                # Tampilkan hasil
                st.markdown("---")
                st.subheader("✅ Hasil Prediksi")
                
                # Prediksi utama
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**👤 Nama Mahasiswa:**")
                    st.markdown(f"# {pred_name}")
                    st.write(f"**Confidence:** {confidence*100:.2f}%")
                    
                    # Color based on confidence
                    if confidence >= 0.7:
                        st.success(f"🟢 Sangat Percaya Diri ({confidence*100:.1f}%)")
                    elif confidence >= 0.5:
                        st.warning(f"🟡 Percaya Diri ({confidence*100:.1f}%)")
                    else:
                        st.error(f"🔴 Kurang Percaya Diri ({confidence*100:.1f}%)")
                
                with col2:
                    st.write("**🔍 Info Preprocessing:**")
                    if face_detected:
                        st.info(f"✅ Wajah terdeteksi & di-crop")
                    else:
                        st.warning(f"⚠️ Wajah tidak terdeteksi, gunakan image original")
                
                st.markdown("---")
                
                # Top 5
                st.subheader("🏆 Top 5 Kandidat")
                
                top5_data = {
                    'Rank': ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'],
                    'Nama': top5_names,
                    'Confidence': [f"{c*100:.2f}%" for c in top5_confs]
                }
                
                st.dataframe(
                    top5_data,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Chart
                st.markdown("**Grafik Confidence (Top 5):**")
                chart_df = {"Kandidat": top5_names, "Confidence (%)": top5_confs * 100}
                st.bar_chart(chart_df, x="Kandidat", y="Confidence (%)")
                
            except Exception as e:
                st.error(f"❌ Error prediksi: {e}")

else:
    st.info("👆 Silakan upload foto terlebih dahulu")
