# Laporan Eksperimen Model Deep Learning
## Klasifikasi Biner ChestMNIST: Cardiomegaly vs Pneumothorax

**Nama:** Nafiz Ahmadin Harily  
**NIM:** 122430051  
**Tanggal:** 8 November 2025

---

## 1. Ringkasan Eksperimen

Eksperimen ini bertujuan untuk mengembangkan model klasifikasi biner pada dataset ChestMNIST untuk membedakan antara dua kondisi medis: Cardiomegaly (pembesaran jantung) dan Pneumothorax (kolaps paru-paru). Model yang dikembangkan merupakan bagian dari strategi ensemble learning dengan identifier Model #42.

### 1.1 Dataset
- **Sumber:** ChestMNIST dari MedMNIST
- **Kelas yang digunakan:**
  - Kelas 0: Cardiomegaly (754 sampel training, 243 sampel validasi)
  - Kelas 1: Pneumothorax (1552 sampel training, 439 sampel validasi)
- **Total data:**
  - Training: 2,306 sampel
  - Validasi: 682 sampel
- **Rasio kelas:** 1:2 (imbalanced dataset)

### 1.2 Arsitektur Model
- **Backbone:** EfficientNet-B0 pre-trained pada ImageNet
- **Parameter:** Sekitar 5 juta parameter
- **Input:** Grayscale image (1 channel) dengan ukuran 96x96 piksel
- **Output:** Binary classification (sigmoid activation)
- **Modifikasi:**
  - Konversi input channel dari 3 (RGB) ke 1 (grayscale)
  - Custom classifier: 1280 → 512 → 256 → 1 dengan BatchNorm dan Dropout
  - Adaptive average pooling untuk fleksibilitas ukuran input

---

## 2. Metodologi Training

### 2.1 Strategi Progressive Training
Training dilakukan dalam dua fase untuk mengoptimalkan transfer learning:

**Fase 1: Frozen Backbone (Epoch 1-15)**
- Backbone EfficientNet-B0 dibekukan (frozen)
- Hanya classifier layer yang dilatih
- Learning rate: 5×10⁻⁴
- Tujuan: Adaptasi classifier terhadap domain medis

**Fase 2: Full Fine-tuning (Epoch 16-85)**
- Seluruh layer (backbone + classifier) dilatih
- Learning rate: 1×10⁻⁴
- Tujuan: Fine-tuning fitur untuk karakteristik spesifik dataset

### 2.2 Hyperparameter
- **Optimizer:** AdamW
- **Learning rate:** 5×10⁻⁴ (Phase 1), 1×10⁻⁴ (Phase 2)
- **Weight decay:** 1×10⁻⁵
- **Batch size:** 32
- **Dropout rate:** 0.25
- **Scheduler:** CosineAnnealingWarmRestarts
- **Loss function:** Focal Loss (alpha=0.3, gamma=2.0)
- **Early stopping patience:** 50 epochs

### 2.3 Augmentasi Data
**Training augmentation:**
- Resize ke 96×96 piksel
- Random horizontal flip (p=0.5)
- Random rotation (±10 derajat)
- Random affine (translate: 0.1, scale: 0.9-1.1)
- Normalization (mean=0.5, std=0.5)

**Validation augmentation:**
- Resize ke 96×96 piksel
- Normalization saja (tanpa augmentasi geometrik)

---

## 3. Hasil Eksperimen

### 3.1 Performa Training
- **Total epochs:** 85 (stopped by early stopping)
- **Best epoch:** Epoch 35
- **Training time:** Sekitar 7 jam (NVIDIA GeForce RTX 3050 Laptop GPU)

### 3.2 Metrik Performa

**Validation Performance (Best Model - Epoch 35)**
| Metrik | Nilai |
|--------|-------|
| Accuracy | 87.54% |
| AUC-ROC | 93.50% |

**Test Performance (Epoch 62)**
| Metrik | Nilai |
|--------|-------|
| Accuracy | 86.95% |
| Precision | 88.55% |
| Recall | 91.57% |
| F1-Score | 90.03% |
| AUC-ROC | 92.98% |

**Per-Class Performance**
| Kelas | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Cardiomegaly (0) | 83.77% | 78.60% | 81.10% | 243 |
| Pneumothorax (1) | 88.55% | 91.57% | 90.03% | 439 |

**Confusion Matrix**
|  | Predicted Cardiomegaly | Predicted Pneumothorax |
|--|------------------------|------------------------|
| **Actual Cardiomegaly** | 191 | 52 |
| **Actual Pneumothorax** | 37 | 402 |

### 3.3 Analisis Performa
1. **High Recall (91.57%):** Model sangat baik dalam mendeteksi kasus Pneumothorax, penting dalam konteks medis untuk menghindari false negative.
2. **Balanced Precision (88.55%):** Tingkat false positive terkendali, mengurangi diagnosis yang tidak perlu.
3. **AUC-ROC tinggi (92.98%):** Menunjukkan kemampuan diskriminasi yang sangat baik antara kedua kelas.
4. **Class Imbalance Handling:** Penggunaan Focal Loss efektif mengatasi rasio kelas 1:2.

---

## 4. Perubahan dari Baseline

### 4.1 Arsitektur
**Baseline (SimpleCNN):**
- Custom CNN sederhana
- Parameter: <1 juta
- Training from scratch

**Model #42 (EfficientNet-B0):**
- Pre-trained architecture
- Parameter: ~5 juta
- Transfer learning strategy

**Dampak:** Peningkatan akurasi signifikan dengan memanfaatkan representasi fitur yang telah dipelajari dari ImageNet.

### 4.2 Input Resolution
**Baseline:** 28×28 piksel (original ChestMNIST)  
**Model #42:** 96×96 piksel

**Dampak:** Resolusi lebih tinggi mempertahankan detail diagnostik penting dalam citra medis, terutama untuk mendeteksi pola kardiomegali dan pneumothorax.

### 4.3 Loss Function
**Baseline:** Binary Cross-Entropy (BCE)  
**Model #42:** Focal Loss (alpha=0.3, gamma=2.0)

**Dampak:** Focal Loss memberikan penekanan lebih pada sampel yang sulit diklasifikasikan dan menangani class imbalance lebih baik.

### 4.4 Training Strategy
**Baseline:** Standard training  
**Model #42:** Progressive training (frozen → fine-tuning)

**Dampak:** Pendekatan bertahap mencegah catastrophic forgetting dan memungkinkan adaptasi yang lebih baik terhadap domain medis.

### 4.5 Data Augmentation
**Baseline:** Minimal augmentation  
**Model #42:** Comprehensive augmentation (flip, rotation, affine)

**Dampak:** Augmentasi yang lebih komprehensif meningkatkan generalisasi model dan mengurangi overfitting.

---

## 5. Observasi dan Temuan

### 5.1 Training Dynamics
- **Epoch 1-15 (Frozen Phase):** Akurasi validasi meningkat dari 51.61% ke 70.23%, menunjukkan adaptasi classifier yang efektif.
- **Epoch 16 (Unfreeze):** Terjadi peningkatan signifikan ke 73.75% setelah backbone di-unfreeze.
- **Epoch 16-35:** Fase peningkatan pesat, mencapai best accuracy 87.54%.
- **Epoch 35-85:** Akurasi berfluktuasi antara 84-87%, menunjukkan model telah mencapai konvergensi.

### 5.2 Overfitting Analysis
- **Train-Val Gap (Epoch 85):** 95.23% - 85.04% = 10.19%
- **Indikator:** Terdapat overfitting moderat pada epoch akhir, namun early stopping berhasil memilih model dengan generalisasi terbaik (Epoch 35).

### 5.3 Class Performance
- Model memiliki performa lebih baik pada kelas Pneumothorax (kelas mayoritas) dengan recall 91.57%.
- Kelas Cardiomegaly (kelas minoritas) memiliki recall lebih rendah (78.60%), menunjukkan kesulitan model pada kelas yang underrepresented.

---

## 6. Limitasi

1. **Dataset Size:** Dataset relatif kecil (2,306 training samples) dapat membatasi kemampuan generalisasi model.
2. **Class Imbalance:** Rasio 1:2 antara kelas dapat menyebabkan bias terhadap kelas mayoritas meskipun telah menggunakan Focal Loss.
3. **Single Modality:** Model hanya dilatih pada single view X-ray, sedangkan diagnosis klinis sering memerlukan multiple views.
4. **Hardware Limitation:** Training terbatas pada GPU 4GB VRAM, membatasi batch size dan ukuran model yang dapat digunakan.

---

## 7. Rekomendasi untuk Penelitian Lanjutan

### 7.1 Peningkatan Performa
1. **Ensemble Learning:** Kombinasi Model #42 dengan model lain (seed berbeda) dapat meningkatkan akurasi hingga 90-93%.
2. **Test-Time Augmentation (TTA):** Menerapkan augmentasi saat inference untuk prediksi yang lebih robust.
3. **Larger Input Size:** Meningkatkan resolusi ke 128×128 atau 224×224 untuk detail yang lebih baik.
4. **Advanced Architectures:** Eksplorasi EfficientNet-B1 atau Vision Transformer (ViT).

### 7.2 Data Strategy
1. **External Data:** Augmentasi dataset dengan sumber eksternal (CheXpert, MIMIC-CXR).
2. **Class Balancing:** Oversampling kelas minoritas atau undersampling kelas mayoritas.
3. **Semi-supervised Learning:** Memanfaatkan data unlabeled untuk pre-training.

### 7.3 Clinical Application
1. **Uncertainty Quantification:** Implementasi Bayesian approach untuk estimasi ketidakpastian prediksi.
2. **Explainability:** Integrasi Grad-CAM atau attention maps untuk interpretasi hasil.
3. **Multi-label Extension:** Ekspansi ke klasifikasi multi-label untuk mendeteksi multiple pathologies.

---

## 8. Kesimpulan

Model #42 berbasis EfficientNet-B0 berhasil mencapai akurasi validasi 87.54% dan akurasi test 86.95% pada task klasifikasi biner ChestMNIST (Cardiomegaly vs Pneumothorax). Implementasi progressive training, Focal Loss, dan augmentasi komprehensif terbukti efektif dalam meningkatkan performa dibandingkan baseline. Model menunjukkan recall tinggi (91.57%) yang penting dalam konteks screening medis untuk meminimalkan missed diagnosis.

Dengan AUC-ROC 92.98%, model memiliki kemampuan diskriminasi yang sangat baik dan berpotensi untuk diintegrasikan dalam sistem computer-aided diagnosis (CAD). Namun, untuk deployment klinis, diperlukan validasi lebih lanjut pada dataset eksternal dan evaluasi oleh tenaga medis profesional.

---

## 9. Referensi

1. Yang, J., et al. (2023). MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification. Scientific Data.
2. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. ICML.
3. Lin, T. Y., et al. (2017). Focal loss for dense object detection. ICCV.
4. Wang, X., et al. (2017). ChestX-ray8: Hospital-scale chest X-ray database and benchmarks. CVPR.

---

## Lampiran

### A. File Output
- `model_ensemble_seed42.pth` - Model checkpoint terbaik (Epoch 62)
- `training_history_model42.png` - Grafik training loss dan accuracy
- `val_predictions_model42.png` - Visualisasi prediksi pada 10 sampel validasi

### B. Spesifikasi Hardware
- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
- **CUDA:** Version 11.8
- **RAM:** 16GB
- **OS:** Windows 11

### C. Environment
- **Python:** 3.11
- **PyTorch:** 2.x
- **Framework:** torchvision, medmnist, numpy, matplotlib, scikit-learn
