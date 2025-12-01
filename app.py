import io
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import streamlit as st
from pathlib import Path
import numpy as np
import sys
import os

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import model dan preprocessing
from model_convnext import create_model
from utils.face_crop import FaceCropper

# ===================== #
# 1. CONFIG DASAR
# ===================== #

CHECKPOINT_PATH = r"checkpoints/convnext_tiny_20251201_070631/best_epoch7.pth"
CLASS_NAMES_PATH = "class_names.txt"
IMAGE_SIZE = 224  # Model dilatih dengan input 224x224
NUM_CLASSES = 70  # sesuaikan dengan jumlah kelas kamu

# ===================== #
# 2. FUNGSI UTIL
# ===================== #

@st.cache_resource
def load_face_cropper():
    """Load face cropper untuk preprocessing"""
    return FaceCropper(padding_percent=20, target_size=(IMAGE_SIZE, IMAGE_SIZE))

@st.cache_data
def load_class_names(path: str):
    with open(path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]
    return class_names

@st.cache_resource
def load_model():
    """Load model dengan struktur yang sama dengan training"""
    # Create model menggunakan custom ConvNeXtClassifier
    model = create_model(num_classes=NUM_CLASSES, pretrained=False, dropout=0.3, device='cpu')
    
    # Load checkpoint
    state = torch.load(CHECKPOINT_PATH, map_location="cpu")
    
    # Extract model state_dict
    if isinstance(state, dict):
        if "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        elif "state_dict" in state:
            state_dict = state["state_dict"]
        else:
            state_dict = state
    else:
        state_dict = state
    
    # Load state_dict langsung (sudah match dengan model structure)
    model.load_state_dict(state_dict, strict=True)
    
    model.eval()
    return model

# Transform sesuai ImageNet (convnext pretrain style)
def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225],   # ImageNet std
        ),
    ])

def predict_image(image: Image.Image, model, class_names, cropper=None):
    """
    Prediksi gambar dengan preprocessing otomatis
    
    Args:
        image: PIL Image
        model: PyTorch model
        class_names: List nama kelas
        cropper: FaceCropper untuk preprocessing
    
    Returns:
        pred_name, confidence, probs, pred_idx, preprocessing_info
    """
    preprocessing_info = {
        'face_detected': False,
        'method': 'unknown'
    }
    
    # Step 1: Preprocess gambar (deteksi & crop wajah)
    if cropper is not None:
        # Convert PIL to numpy untuk cropper
        img_array = np.array(image)
        
        # Deteksi dan crop wajah
        cropped_face, face_detected = cropper.detect_and_crop_face(image)
        
        if face_detected and cropped_face is not None:
            # Wajah berhasil dideteksi
            image = Image.fromarray(cropped_face)
            preprocessing_info['face_detected'] = True
            preprocessing_info['method'] = 'MediaPipe Detection'
        else:
            # Fallback: gunakan gambar original
            preprocessing_info['face_detected'] = False
            preprocessing_info['method'] = 'No Detection (Original Image)'
    
    # Step 2: Transform ke tensor
    transform = get_transform()
    img_t = transform(image).unsqueeze(0)  # shape: (1, 3, H, W)
    
    # Step 3: Prediksi
    with torch.no_grad():
        logits = model(img_t)
        probs = torch.softmax(logits, dim=1)[0]
    
    conf, idx = torch.max(probs, dim=0)
    pred_name = class_names[idx.item()]
    
    return pred_name, conf.item(), probs.numpy(), idx.item(), preprocessing_info

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
    assert len(class_names) == NUM_CLASSES, (
        f"Jumlah class di file ({len(class_names)}) "
        f"≠ NUM_CLASSES ({NUM_CLASSES})"
    )
except Exception as e:
    st.error(f"Gagal load class_names: {e}")
    st.stop()

try:
    face_cropper = load_face_cropper()
except Exception as e:
    st.warning(f"Preprocessing tidak tersedia: {e}")
    face_cropper = None

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal load model: {e}")
    st.stop()

# ===================== #
# 5. UI UTAMA
# ===================== #

st.title("📸 Face Recognition Mahasiswa")
st.caption("Upload foto wajah, sistem akan otomatis deteksi wajah dan melakukan prediksi.")

with st.sidebar:
    st.subheader("ℹ️ Informasi Model")
    st.write("- **Model**: ConvNeXt-Tiny")
    st.write(f"- **Akurasi**: 64.29%")
    st.write(f"- **Jumlah Kelas**: {NUM_CLASSES} mahasiswa")
    st.write("- **Format Gambar**: JPG, JPEG, PNG")
    st.markdown("---")
    st.subheader("📋 Pipeline")
    st.write("""
    1️⃣ Upload gambar wajah
    2️⃣ Deteksi wajah (MediaPipe)
    3️⃣ Crop & Normalize
    4️⃣ Prediksi dengan model
    5️⃣ Tampilkan hasil
    """)
    st.markdown("---")
    st.write("**Tips**: Pastikan wajah jelas, terang, dan 1 orang per foto.")

st.markdown("### 1️⃣ Upload Foto")

uploaded_file = st.file_uploader(
    "Pilih file gambar wajah",
    type=["jpg", "jpeg", "png"],
    help="Gunakan foto close-up wajah untuk hasil terbaik."
)

if uploaded_file is not None:
    try:
        # Baca gambar
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Tidak bisa membuka gambar: {e}")
        st.stop()

    # Tampilkan preview
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="📸 Foto Original", use_container_width=True)
    
    with col2:
        st.markdown("#### Info Gambar")
        st.write(f"- **Ukuran**: {image.size}")
        st.write(f"- **Format**: {image.format}")
        st.write(f"- **Mode**: {image.mode}")

    st.markdown("### 2️⃣ Processing Pipeline")
    
    # Processing info
    if st.button("🔄 Mulai Preprocessing & Prediksi", key="predict_btn"):
        with st.spinner("🔄 Sedang memproses..."):
            # Step 1: Preprocessing
            progress_placeholder = st.empty()
            
            with progress_placeholder.container():
                st.write("**Processing Steps:**")
                step1 = st.empty()
                step2 = st.empty()
                step3 = st.empty()
                step4 = st.empty()
            
            # Actual prediction
            try:
                step1.info("⏳ Step 1: Deteksi wajah dengan MediaPipe...")
                pred_name, confidence, probs, pred_idx, preprocess_info = predict_image(
                    image, model, class_names, face_cropper
                )
                step1.success(f"✅ Step 1: Wajah {'terdeteksi' if preprocess_info['face_detected'] else 'tidak terdeteksi'} ({preprocess_info['method']})")
                
                step2.info("⏳ Step 2: Cropping & Normalisasi gambar...")
                step2.success("✅ Step 2: Gambar berhasil dinormalisasi ke 224×224")
                
                step3.info("⏳ Step 3: Ekstraksi fitur dengan backbone...")
                step3.success("✅ Step 3: Fitur berhasil diekstrak (768 dimensions)")
                
                step4.info("⏳ Step 4: Prediksi dengan model ConvNeXt-Tiny...")
                step4.success("✅ Step 4: Prediksi selesai!")
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.stop()
        
        # Hasil Prediksi
        st.success("🎉 Prediksi berhasil!")
        
        st.markdown("### 3️⃣ Hasil Prediksi")
        
        col_pred, col_top5 = st.columns([1, 1])
        
        with col_pred:
            st.markdown("#### 🎯 Prediksi Utama")
            # Card dengan confidence
            confidence_pct = confidence * 100
            
            # Color coding based on confidence
            if confidence_pct >= 70:
                color = "🟢"
                level = "Sangat Percaya Diri"
            elif confidence_pct >= 50:
                color = "🟡"
                level = "Percaya Diri"
            else:
                color = "🔴"
                level = "Kurang Percaya Diri"
            
            st.metric(
                label=f"👤 Nama Mahasiswa",
                value=f"{pred_name}",
                delta=f"{confidence_pct:.2f}% {color}"
            )
            st.write(f"**Level Kepercayaan**: {level}")
            
            # Processing details
            st.markdown("#### 📊 Detail Preprocessing")
            st.write(f"- **Wajah Terdeteksi**: {'✅ Ya' if preprocess_info['face_detected'] else '❌ Tidak'}")
            st.write(f"- **Metode**: {preprocess_info['method']}")
            st.write(f"- **Ukuran Output**: {IMAGE_SIZE}×{IMAGE_SIZE}")
        
        with col_top5:
            st.markdown("#### 🏆 Top-5 Kandidat")
            
            probs_np = np.array(probs)
            top5_idx = probs_np.argsort()[-5:][::-1]
            top5_names = [class_names[i] for i in top5_idx]
            top5_probs = probs_np[top5_idx] * 100
            
            # Display as table
            top5_data = {
                'Rank': ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'],
                'Nama': top5_names,
                'Confidence': [f"{p:.2f}%" for p in top5_probs]
            }
            
            st.dataframe(
                top5_data,
                use_container_width=True,
                hide_index=True
            )
            
            # Confidence chart
            st.markdown("#### 📈 Confidence Distribution (Top-5)")
            chart_data = {
                'Kandidat': top5_names,
                'Confidence': top5_probs
            }
            st.bar_chart(chart_data, x='Kandidat', y='Confidence')
        
        # Informasi tambahan
        st.markdown("---")
        st.markdown("### 📌 Informasi Tambahan")
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.write("**Tentang Model:**")
            st.write("""
            - Arsitektur: ConvNeXt-Tiny
            - Parameters: 28 juta
            - Pretrained: ImageNet-1K V1
            - Accuracy: 64.29% (validation)
            """)
        
        with info_col2:
            st.write("**Tentang Dataset:**")
            st.write("""
            - Total kelas: 70 mahasiswa
            - Total gambar training: 210
            - Total gambar validation: 70
            - Augmentasi: ColorJitter, RandomErasing
            """)
        
        # Disclaimer
        st.info(
            "ℹ️ **Catatan**: Jika prediksi terasa tidak akurat, mungkin karena:\n"
            "1. Foto terlalu jauh atau blur\n"
            "2. Pencahayaan kurang optimal\n"
            "3. Wajah terhalangi atau sudut ekstrem\n"
            "4. Dataset training terbatas (hanya 4 gambar/kelas)\n\n"
            "Untuk hasil terbaik, gunakan foto wajah yang jelas, terang, dan frontal."
        )

else:
    st.info("👆 Silakan upload foto terlebih dahulu untuk mulai prediksi.")
    
    # Demo section
    st.markdown("---")
    st.markdown("### 📖 Cara Menggunakan")
    
    st.write("""
    **Langkah 1**: Upload foto wajah (JPG, JPEG, atau PNG)
    - Pastikan wajah terlihat jelas
    - Foto close-up atau medium shot optimal
    - Satu orang per foto
    
    **Langkah 2**: Sistem otomatis akan:
    1. Deteksi wajah menggunakan MediaPipe
    2. Crop dan normalisasi gambar ke 224×224
    3. Ekstraksi fitur dengan ConvNeXt backbone
    4. Prediksi menggunakan model neural network
    
    **Langkah 3**: Lihat hasil:
    - Nama mahasiswa yang diprediksi
    - Confidence score (tingkat kepercayaan)
    - Top-5 kandidat lain
    - Grafik distribusi confidence
    """)
