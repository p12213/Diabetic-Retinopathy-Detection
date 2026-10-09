# 👁️ Diabetic Retinopathy AI Diagnostic & Screening System
> **Computer Vision (CV) + Deep Learning (DL) End-to-End Solution**

An AI platform for automated Diabetic Retinopathy (DR) grading and retinal fundus image inspection. Combines classical Computer Vision preprocessing (Ben Graham's cropping, Green Channel isolation, CLAHE, vessel enhancement) with Deep Neural Networks (EfficientNet / ResNet-50) and Explainable AI (Grad-CAM heatmaps).

---

## 🌟 Key Features

1. **Dual AI Pipeline**:
   - **Computer Vision (CV)**: Ben Graham cropping, CLAHE contrast enhancement, blood vessel segmentation, microaneurysm & hemorrhage detection overlays.
   - **Deep Learning (DL)**: 5-Grade severity classification adhering to the **International Clinical Diabetic Retinopathy (ICDR)** scale.
2. **Explainable AI (XAI)**:
   - Real-time **Grad-CAM attention heatmap overlay** highlighting specific pathological regions (exudates, cotton wool spots, neovascularization) that influenced the neural network prediction.
3. **Interactive Clinical Web Platform**:
   - Zero-install web diagnostic application running client-side with sample clinical case presets (Grades 0–4), custom image upload, live CV filter inspection, interactive model architecture inspector, and **exportable PDF clinical reports**.
4. **Complete PyTorch Training Suite**:
   - Preprocessing scripts (`preprocess.py`), Dataset loaders for APTOS 2019 / EyePACS (`dataset.py`), Model architecture (`model.py`), Loss function with Quadratic Weighted Kappa (QWK) evaluation (`train.py`), and ONNX export tool (`export_onnx.py`).

---

## 📊 ICDR Diabetic Retinopathy Grading Scale

| Grade | Disease Stage | Clinical Findings |
| :---: | :--- | :--- |
| **0** | **No DR** | No abnormal lesions or microaneurysms detected in retinal fundus photo. |
| **1** | **Mild NPDR** | Microaneurysms only (tiny red spots caused by damaged capillaries). |
| **2** | **Moderate NPDR** | More than microaneurysms, but less than Severe NPDR (hard exudates, venous bleeding). |
| **3** | **Severe NPDR** | Any of: >20 intraretinal hemorrhages in 4 quadrants, venous beading in 2+ quadrants, IRMA in 1+ quadrant. |
| **4** | **Proliferative DR (PDR)** | Neovascularization (growth of new fragile blood vessels) or vitreous/preretinal hemorrhage. |

---

## 📂 Project Structure

```
diabetic-retinopathy-ai/
├── README.md                          # Main project guide
├── requirements.txt                   # Python dependencies
├── python_pipeline/                   # PyTorch DL & CV Model Pipeline
│   ├── preprocess.py                  # Ben Graham cropping, Green channel & CLAHE filters
│   ├── model.py                       # ResNet-50 / EfficientNet PyTorch model + Grad-CAM
│   ├── dataset.py                     # APTOS/EyePACS Dataset loader & Albumentations
│   ├── train.py                       # Training loop with QWK metric & class weighting
│   └── export_onnx.py                 # ONNX model exporter
└── web_app/                           # Interactive Clinical Web Application
    ├── index.html                     # Screening UI layout & dashboard
    ├── styles.css                     # Glassmorphic dark design system & printable report styles
    ├── cv_engine.js                   # Canvas-based CV preprocessing engine
    ├── ai_inference.js                # Neural inference engine & Grad-CAM visualizer generator
    └── app.js                         # Web app event router & report generator
```

---

## 🚀 Quick Start Guide

### 1. Python Training Environment Setup

```bash
# Navigate to project directory
cd diabetic-retinopathy-ai

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup
Download the dataset from Kaggle:
- **APTOS 2019 Blindness Detection**: `kaggle competitions download -c aptos2019-blindness-detection`
- Extract `train_images/` and `train.csv` into a `data/` folder inside `python_pipeline/`.

### 3. Model Training
```bash
python python_pipeline/train.py --csv_path data/train.csv --img_dir data/train_images --backbone resnet50 --epochs 15 --batch_size 16
```

### 4. Export ONNX Model
```bash
python python_pipeline/export_onnx.py --checkpoint checkpoints/best_dr_model.pth --output models/dr_model.onnx
```

### 5. Launch Interactive Web Screening Platform
Run a simple HTTP server to launch the web dashboard:
```bash
python -m http.server 8000 --directory web_app
```
Open your browser at **`http://localhost:8000`**.

---

## 🩺 Web App Features Preview

- **Preset Clinical Cases**: Switch between Grade 0 (Normal) to Grade 4 (Proliferative DR) fundus cases instantly.
- **CV Preprocessing Controls**: Toggle between Original, Green Channel, CLAHE enhancement, Vessel Mask, and Lesion Highlight Bounding Boxes.
- **Grad-CAM Visualizer**: Adjust opacity slider overlaying neural heatmaps directly on top of fundus images.
- **Clinical Diagnostic Report**: Click **"Export Printable Diagnostic Report"** to generate a clinical PDF report complete with patient summary, severity breakdown, risk level, and medical follow-up guidelines.

---

## ⚖️ License & Disclaimer
This project is developed for educational, research, and diagnostic support exploration. Clinical implementation requires rigorous validation against medical device standards (FDA / CE-mark) and certified ophthalmologist review.
