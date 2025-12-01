    # Face Recognition - Deep Learning Project

    Sistem klasifikasi wajah menggunakan **ConvNeXt-Tiny** dan **Swin Transformer V2 Tiny** dengan PyTorch untuk mengenali 70 mahasiswa Matakuliah Deep Learning Teknik Informatika ITERA.

## 📋 Daftar Isi

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
- [Training Configuration](#training-configuration)
- [Results & Comparison](#results--comparison)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Requirements](#requirements) ---

  ## 🎯 Overview

  Proyek ini membandingkan performa dua arsitektur modern untuk face recognition:

  - **ConvNeXt-Tiny**: Arsitektur CNN modern dengan design principle dari Vision Transformers
  - **Swin Transformer V2 Tiny**: Pure transformer architecture dengan shifted windows attention

  **Tujuan**: Mengidentifikasi arsitektur terbaik untuk klasifikasi wajah pada small dataset dengan tegangan teknologi deep learning modern.

  ***

  ## 📊 Dataset

  - **Total Images**: 280 gambar
  - **Classes**: 70 mahasiswa
  - **Distribution**: ~4 gambar per kelas
  - **Split Ratio**: 75% training (210 images) / 25% validation (70 images)
  - **Image Format**: JPG, JPEG, PNG, WEBP

  ***

  ## 🏗️ Model Architectures

  ### 1. ConvNeXt-Tiny

  | Property             | Value          |
  | -------------------- | -------------- |
  | **Pretrained**       | ImageNet-1K V1 |
  | **Input Size**       | 512×512×3      |
  | **Parameters**       | 28M total      |
  | **Embedding**        | 768 dimensions |
  | **Stochastic Depth** | 0.1            |
  | **Batch Size**       | 16             |

  **Architecture**:

  ```
  ConvNeXt Backbone (pretrained)
      ↓
  Global Average Pooling
      ↓
  Flatten + LayerNorm
      ↓
  Dropout (0.3)
      ↓
  Linear (768 → 70 classes)
  ```

  ### 2. Swin Transformer V2 Tiny

  | Property        | Value          |
  | --------------- | -------------- |
  | **Pretrained**  | ImageNet-1K V1 |
  | **Input Size**  | 224×224×3      |
  | **Parameters**  | 28M total      |
  | **Embedding**   | 768 dimensions |
  | **Window Size** | 7×7            |
  | **Batch Size**  | 16             |

  **Architecture**:

  ```
  Swin V2 Backbone (pretrained)
      ↓
  Global Features (768-dim)
      ↓
  LayerNorm
      ↓
  Dropout (0.3)
      ↓
  Linear (768 → 70 classes)
  ```

  ***

  ## ⚙️ Training Configuration

  ### Hyperparameters (Both Models)

  | Parameter             | Value                       | Description                 |
  | --------------------- | --------------------------- | --------------------------- |
  | **Optimizer**         | AdamW                       | Weight decay optimizer      |
  | **Learning Rate**     | 5e-4                        | Initial learning rate       |
  | **Weight Decay**      | 0.01                        | L2 regularization           |
  | **Scheduler**         | CosineAnnealingWarmRestarts | T_0=10, T_mult=2            |
  | **Loss Function**     | CrossEntropyLoss            | Label smoothing: 0.1        |
  | **Dropout**           | 0.3                         | Regularization              |
  | **Batch Size**        | 16                          | Both models                 |
  | **Epochs**            | 50                          | Max epochs                  |
  | **Early Stopping**    | Patience: 7                 | Monitor val_acc             |
  | **Gradient Clipping** | max_norm: 1.0               | Prevent exploding gradients |
  | **Mixed Precision**   | Enabled (AMP)               | CUDA only                   |

  ### Data Augmentation

  **Training**:

  - Random Horizontal Flip (p=0.5)
  - ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2)
  - Random Erasing (p=0.2)
  - Normalization (ImageNet stats)

  **Validation**:

  - Resize + Center Crop
  - Normalization only

  ***

  ## 📈 Results & Comparison

  ### Training Summary

  #### ConvNeXt-Tiny

  - **Best Epoch**: 7/14
  - **Training Time**: ~3.2 hours
  - **Best Val Accuracy**: **64.29%**
  - **Training Accuracy**: 100.00%
  - **Val F1 Score**: 0.6214
  - **Overfitting Gap**: 35.71%

  #### Swin Transformer V2 Tiny

  - **Best Epoch**: 27/34
  - **Training Time**: ~7.8 hours
  - **Best Val Accuracy**: **62.86%**
  - **Training Accuracy**: 100.00%
  - **Val F1 Score**: 0.5676
  - **Overfitting Gap**: 37.14%

  ### Performance Metrics Comparison

  | Metric                 | ConvNeXt-Tiny     | Swin V2 Tiny       | Winner      |
  | ---------------------- | ----------------- | ------------------ | ----------- |
  | **Best Val Accuracy**  | 64.29%            | 62.86%             | 🏆 ConvNeXt |
  | **Best Val F1 Score**  | 0.6214            | 0.5676             | 🏆 ConvNeXt |
  | **Best Val Precision** | 0.6119            | 0.5416             | 🏆 ConvNeXt |
  | **Best Val Recall**    | 0.6429            | 0.6286             | 🏆 ConvNeXt |
  | **Training Speed**     | Faster (7 epochs) | Slower (27 epochs) | 🏆 ConvNeXt |
  | **Convergence**        | Epoch 7           | Epoch 27           | 🏆 ConvNeXt |
  | **Overfitting**        | 35.71%            | 37.14%             | 🏆 ConvNeXt |
  | **Parameters**         | 28M               | 28M                | ⚖️ Tie      |

  ### Learning Curves

  #### ConvNeXt-Tiny Training Progress

  ```
  Epoch  Train Acc  Val Acc   Train Loss  Val Loss
  -----  ---------  --------  ----------  --------
  1      1.90%      1.43%     4.548       4.368
  3      16.67%     40.00%    3.872       3.139
  5      91.43%     61.43%    1.290       2.228
  7      100.00%    64.29%    0.832       2.138  ⭐ BEST
  10     100.00%    62.86%    0.804       2.167
  14     100.00%    60.00%    0.838       2.284
  ```

  #### Swin V2 Tiny Training Progress

  ```
  Epoch  Train Acc  Val Acc   Train Loss  Val Loss
  -----  ---------  --------  ----------  --------
  1      0.95%      1.43%     4.629       4.419
  5      4.76%      10.00%    4.160       3.878
  10     66.67%     50.00%    2.313       2.863
  15     90.95%     51.43%    1.147       2.683
  20     98.57%     61.43%    0.911       2.419
  27     100.00%    62.86%    0.818       2.275  ⭐ BEST
  30     100.00%    61.43%    0.806       2.269
  34     98.10%     55.71%    0.973       2.520
  ```

  ### Key Observations

  #### ✅ ConvNeXt-Tiny Advantages:

  1. **Faster Convergence**: Mencapai performa terbaik di epoch 7 vs epoch 27
  2. **Higher Accuracy**: 64.29% vs 62.86% (+1.43% absolute)
  3. **Better Generalization**: F1 score lebih tinggi (0.6214 vs 0.5676)
  4. **Training Efficiency**: ~4× lebih cepat mencapai best model
  5. **More Stable**: Less fluctuation setelah convergence

  #### ⚠️ Swin V2 Tiny Characteristics:

  1. **Slower Convergence**: Butuh 27 epochs untuk mencapai best performance
  2. **Gradual Learning**: Learning curve lebih smooth tapi lambat
  3. **Lower Accuracy**: 62.86% validation accuracy
  4. **Higher Training Time**: Hampir 2× durasi training total

  #### 🔍 Common Issues (Both Models):

  1. **Severe Overfitting**: Gap ~36% antara train dan val accuracy
  2. **Small Dataset**: Hanya 210 training images untuk 70 classes
  3. **Class Imbalance**: ~3 training images per class (sangat sedikit)

  ### Visualizations

  #### Full Model Comparison

  ![Model Comparison](model_comparison/model_comparison_full.png)
  _Comprehensive comparison of training metrics between ConvNeXt-Tiny and Swin V2 Tiny_

  #### Overfitting Analysis

  ![Overfitting Analysis](model_comparison/overfitting_analysis.png)
  _Visual analysis of overfitting gap between training and validation accuracy_

  #### Performance Summary

  ![Performance Summary](model_comparison/performance_summary.png)
  _Bar chart comparison of best performance metrics_

  #### Individual Model Plots

  Training plots tersimpan di direktori checkpoints:

  **ConvNeXt-Tiny** (`checkpoints/convnext_tiny_20251201_070631/`):

  - `loss.png` - Training & Validation Loss
  - `accuracy.png` - Training & Validation Accuracy
  - `f1_score.png` - F1 Score progression
  - `precision_recall.png` - Precision & Recall curves
  - `confusion_matrix.png` - Confusion matrix (normalized)
  - `confusion_matrix_counts.png` - Raw counts

  **Swin V2 Tiny** (`checkpoints/swin_v2_tiny_20251201_084752/`):

  - (Same visualization files)

  ***

  ## 🔬 Generate Comparison Plots

  Untuk regenerate comparison plots:

  ```bash
  python generate_comparison.py
  ```

  Output: `model_comparison/` directory dengan 3 plots

  ***

  ## 💡 Recommendations

  ### For Production Deployment:

  **Gunakan ConvNeXt-Tiny** karena:

  - Accuracy lebih tinggi (64.29%)
  - Convergence lebih cepat
  - Training cost lebih rendah
  - Performance lebih konsisten

  ### For Future Improvements:

  1. **Data Augmentation**: Tambahkan augmentasi lebih agresif
  2. **More Data**: Collect lebih banyak gambar per kelas (minimal 10-20)
  3. **Transfer Learning**: Fine-tune dengan dataset wajah yang lebih besar
  4. **Ensemble**: Combine predictions dari kedua model
  5. **Regularization**: Experiment dengan dropout rates lebih tinggi
  6. **Architecture**: Try ConvNeXt-Small untuk perbandingan lebih detail \*\*\*

  ## 🚀 Installation

  ### 1. Clone Repository

  ```bash
  git clone https://github.com/kevinnaufaldany/face-recognition-dl.git
  cd face-recognition-dl
  ```

  ### 2. Create Virtual Environment

  ```bash
  # Using conda
  conda create -n face-recognition python=3.10
  conda activate face-recognition

  # Or using venv
  python -m venv venv
  venv\Scripts\activate  # Windows
  source venv/bin/activate  # Linux/Mac
  ```

  ### 3. Install Dependencies

  ```bash
  pip install -r requirements.txt
  ```

  ### 4. Verify CUDA (Optional but Recommended)

  ```bash
  python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
  ```

  ***

  ## 📖 Usage

  ### 1. Train Models

  **Train ConvNeXt-Tiny**:

  ```bash
  python train_convnext.py
  ```

  **Train Swin V2 Tiny**:

  ```bash
  python train_swin.py
  ```

  ### 2. Monitor Training

  Checkpoints dan history disimpan di `checkpoints/<model_name_timestamp>/`:

  - `best_epoch{N}.pth` - Best model weights
  - `history_train.json` - Training metrics
  - `*.png` - Visualization plots

  ### 3. Evaluate Model

  ```bash
  python check.py
  ```

  ***

  ## 📁 Project Structure

  ```
  face-recognition-dl/
  │
  ├── dataset/
  │   ├── Train/                    # Original images (280 images, 70 classes)
  │   └── Train_Cropped/            # Preprocessed faces (already ready)
  │
  ├── checkpoints/
  │   ├── convnext_tiny_20251201_070631/
  │   │   ├── best_epoch7.pth
  │   │   ├── history_train.json
  │   │   └── *.png                 # Training plots
  │   └── swin_v2_tiny_20251201_084752/
  │       ├── best_epoch27.pth
  │       ├── history_train.json
  │       └── *.png
  │
  ├── utils/
  │   └── face_crop.py              # MediaPipe face detection (reference)
  │
  ├── model_convnext.py             # ConvNeXt-Tiny model
  ├── model_swin.py                 # Swin V2 Tiny model
  │
  ├── train_convnext.py             # ConvNeXt training script
  ├── train_swin.py                 # Swin V2 training script
  │
  ├── datareader.py                 # Dataset loader & augmentation
  ├── check.py                      # Model evaluation
  ├── generate_comparison.py        # Generate comparison plots
  │
  ├── requirements.txt              # Python dependencies
  └── README.md                     # This file
  ```

  ***

  ## 📦 Requirements

  ### Hardware

  - **GPU**: NVIDIA GPU dengan CUDA support (recommended)
  - **RAM**: Minimum 8GB
  - **Storage**: ~5GB (dataset + models + checkpoints)

  ### Software

  - **Python**: 3.10+
  - **CUDA**: 11.8+ (for GPU training)
  - **OS**: Windows, Linux, or macOS

  ### Key Dependencies

  - PyTorch 2.7.1+
  - torchvision 0.20.1+
  - timm (PyTorch Image Models)
  - MediaPipe
  - OpenCV
  - scikit-learn
  - matplotlib, seaborn
  - tqdm

  See `requirements.txt` for complete list.

```bash
    # Install the dependencies
    pip install -r requirements.txt
```

---

## 🎓 Team

**Teknik Informatika - Institut Teknologi Sumatera**

- Dataset: 70 mahasiswa IF ITERA
- Project: Deep Learning - Face Recognition

---

## 📄 License

This project is for educational purposes.

---

## 🙏 Acknowledgments

- **PyTorch**: Deep learning framework
- **torchvision**: ConvNeXt & Swin Transformer V2
- **OpenCV**: Image processing
- **scikit-learn**: Metrics & evaluation ---

  ## 📞 Contact

  For questions or suggestions, please open an issue on GitHub.

  **Repository**: [kevinnaufaldany/face-recognition-dl](https://github.com/kevinnaufaldany/face-recognition-dl)

  ***

  ## 🏆 Conclusion

  **ConvNeXt-Tiny emerges as the clear winner** untuk face recognition task ini:

  - ✅ 64.29% validation accuracy (highest)
  - ✅ Faster convergence (7 epochs vs 27)
  - ✅ Better F1 score (0.6214 vs 0.5676)
  - ✅ More efficient training

  **Untuk small dataset dengan limited samples per class, ConvNeXt-Tiny menunjukkan performa superior dibanding Swin Transformer V2 Tiny.**
