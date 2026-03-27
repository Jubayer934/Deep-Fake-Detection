# 🧠 Detecting Deepfake using Hybrid Model

This project focuses on **detecting deepfake videos** using deep learning models — including a **CNN-based model (DFD_CNN)**, a **Hybrid Xception + Vision Transformer (Hybrid_XcepViT)** model, and an additional **ViT-only model** for comparison.

---

## 📂 Project Files
| File | Description |
|------|--------------|
| `DFD_CNN.ipynb` | Deepfake detection using a CNN-based approach. |
| `Hybrid_XcepVit.ipynb` | Hybrid model combining **Xception** and **ViT** architectures for improved feature extraction and classification. |
| `DFD_ViT.ipynb` | **ViT** model for classification. |
| `test.ipynb` | Used for quick tests and experiments. |

---

## 🧩 Dataset
- **Total videos:** 3431  
  - **Fake:** 3068  
  - **Real:** 363  
- **Video length:** 35s – 60s  
- **Frame extraction:** 1 frame every **2 seconds** from each video.

---

## ⚙️ Model Overview
### 🧱 DFD_CNN
A baseline **Convolutional Neural Network** designed to classify real and fake video frames.

### 🔗 Hybrid_XcepVit
A **hybrid model** combining:
- **Xception** network (for spatial feature extraction)
- **Vision Transformer (ViT)** (for contextual and attention-based learning)

### 🔍 ViT Model
A separate **Vision Transformer-only** model was also trained to compare with the hybrid model’s performance.

---

## 🚀 How to Run
1. Open the notebook in **Google Colab** or Jupyter Notebook.  
2. Mount your dataset (folder containing extracted frames).  
3. Run all cells to train and evaluate the model.  

Example:
```python
!pip install tensorflow keras timm

