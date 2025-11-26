# Swin Transformer V2 Small - Image Classification

Sistem klasifikasi gambar menggunakan Swin Transformer V2 Small dengan PyTorch untuk 70 kelas.

## 📋 Requirements

```bash
pip install torch torchvision scikit-learn pillow numpy
```

## 📁 Struktur File

- **datareader.py** - Dataset loader dengan augmentasi dan train/val/test split
- **model.py** - Swin-V2-S model dengan pretrained ImageNet-1K
- **train.py** - Training script lengkap dengan AMP, early stopping, dan metrics

## 🚀 Cara Menggunakan

### 1. Training Model

```bash
python train.py
```

### 2. Test Datareader

```bash
python datareader.py
```

### 3. Test Model

```bash
python model.py
```

## ⚙️ Konfigurasi Training

- **Model**: Swin-V2-S pretrained (ImageNet-1K)
- **Batch Size**: 8
- **Epochs**: 30
- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.01)
- **Scheduler**: CosineAnnealingWarmRestarts (T_0=5, T_mult=1, eta_min=1e-6)
- **Loss**: CrossEntropyLoss (label_smoothing=0.1)
- **AMP**: Enabled (autocast + GradScaler)
- **Gradient Clipping**: max_norm=1.0
- **Early Stopping**: patience=5
- **Data Split**: 70% train, 15% val, 15% test

## 📊 Metrics

- Accuracy
- Precision (macro)
- Recall (macro)
- F1 Score (macro)

## 🎨 Augmentasi Data

**Training:**

- Resize ke 256x256
- Random Crop 224x224
- Random Horizontal Flip (p=0.5)
- Random Rotation (±10°)
- Color Jitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
- Normalization (ImageNet stats)

**Validation/Test:**

- Resize ke 224x224
- Normalization (ImageNet stats)

## 💾 Output

Checkpoints disimpan di folder `checkpoints/swin_v2_s_YYYYMMDD_HHMMSS/`:

- `best_model.pth` - Model dengan validation accuracy terbaik
- `last_checkpoint.pth` - Checkpoint terakhir
- `history.pth` - Training history (loss, accuracy, F1)

## 📖 Struktur Dataset

```
dataset/
└── Train/
    ├── Class1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── Class2/
    │   └── ...
    └── Class70/
        └── ...
```

## 🔧 Customization

Edit konfigurasi di `train.py`:

```python
CONFIG = {
    'data_dir': 'dataset/Train',
    'num_classes': 70,
    'batch_size': 8,
    'epochs': 30,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'save_dir': 'checkpoints'
}
```

## 📝 Catatan

- Menggunakan Mixed Precision Training (AMP) untuk efisiensi memori dan kecepatan
- Early stopping mencegah overfitting
- Gradient clipping untuk stabilitas training
- Semua file dapat dijalankan standalone untuk testing
