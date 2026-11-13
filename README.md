# Deepfake Image Detection & Explainable AI System

A web-based **Deepfake Image Detection and Explainable AI System** that classifies facial images as **REAL or DEEPFAKE** using a PyTorch **EfficientNet-B0** convolutional neural network, visualizes influential facial features using **Grad-CAM heatmaps**, performs **FFT frequency-domain spectral analysis**, and generates human-readable evidence explanations.

---

## Key Features

* **High-Accuracy Classification:** Transfer learning with EfficientNet-B0 fine-tuned for binary classification (`0 = REAL`, `1 = DEEPFAKE`).
* **Explainable AI (Grad-CAM):** Superimposes class activation heatmaps over detected face crops to visually highlight suspicious regions influencing the model's decision.
* **Spatial Evidence Generation:** Maps Grad-CAM activation intensities across anatomical facial zones (Eyes, Mouth, Skin Texture, Facial Boundary, Hairline) to generate measurable, model-backed explanation reasons.
* **Frequency-Domain Analysis (FFT):** 2D Fast Fourier Transform magnitude spectrum analysis to detect high-frequency grid noise and spectral artifacts characteristic of GAN and diffusion deepfakes.
* **Face Detection & Multi-Face Support:** Integrated OpenCV Haar Cascade / DNN face detector that crops and normalizes facial regions automatically.
* **Modern Responsive Dashboard:** Dark-mode dashboard with drag-and-drop uploader, radial confidence gauge, evidence severity badges, Grad-CAM viewer grid, and prediction history.
* **FastAPI Backend & SQLite Log:** Modular RESTful API with SQLite prediction logging (`backend/database/history.db`).
* **Complete Training Pipeline:** Separate training script (`ml/train.py`) supporting Albumentations augmentations, weighted sampling for class imbalance, early stopping, LR scheduling, and evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix).

---

## Machine Learning Pipeline

```text
Image Upload
  │
  ▼
Validate Format & Size
  │
  ▼
Detect Face Bounding Box
  │
  ▼
Crop & Resize to 224x224
  │
  ▼
ImageNet Normalization
  │
  ▼
EfficientNet-B0 Forward Pass
  │
  ▼
Calculate REAL / DEEPFAKE Probability & Confidence %
  ├──────────────────────────┬──────────────────────────┐
  ▼                          ▼                          ▼
Grad-CAM Heatmap     Spatial Zone Activation    FFT Frequency Analysis
  │                          │                          │
  └──────────────────────────┼──────────────────────────┘
                             ▼
              Generate Human-Readable Evidence
                             │
                             ▼
         Display Results on Dashboard & Save History
```

---

## Project Structure

```text
deepfake-detector/
│
├── backend/
│   ├── main.py                 # FastAPI application entrypoint & static mounting
│   ├── api/
│   │   ├── routes.py           # API endpoints (/predict, /health, /history, /train)
│   │   └── schemas.py          # Pydantic request/response schemas
│   ├── services/
│   │   └── detector_service.py # Orchestrates face detection, inference, & DB logging
│   └── database/
│       ├── db.py               # SQLite database setup
│       └── repository.py       # CRUD operations for prediction history
│
├── ml/
│   ├── model.py                # EfficientNet-B0 PyTorch architecture
│   ├── dataset.py              # PyTorch Dataset & Albumentations augmentations
│   ├── train.py                # Model training script with metrics & early stopping
│   ├── inference.py            # Face detection & inference engine
│   ├── gradcam.py              # Grad-CAM heatmap & spatial region evidence generator
│   ├── frequency_analysis.py   # OpenCV FFT frequency domain analysis
│   ├── evaluation.py           # Metrics calculation & loss curve plotting
│   └── create_sample_dataset.py# Utility to generate test dataset for instant verification
│
├── frontend/
│   ├── index.html              # Modern web dashboard layout
│   ├── style.css               # Dark-mode glassmorphic CSS styling
│   └── app.js                  # Frontend logic & API integration
│
├── models/                     # Saved model checkpoint (deepfake_detector.pth) & config
├── data/
│   └── sample_images/          # Sample images for quick UI testing
├── outputs/
│   ├── heatmaps/               # Saved Grad-CAM overlays & FFT spectrums
│   └── reports/                # Training plots & evaluation reports
├── dataset/                    # Standard train/validation/test directory structure
├── requirements.txt            # Python package dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

---

## Installation & Setup

### Prerequisites

* Python 3.11+
* Git

### Step 1: Clone Repository & Create Virtual Environment

```bash
git clone <repository_url>
cd deepfake-detector

# Create virtual environment
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Linux/macOS:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation & Training

### Dataset Layout

Organize your deepfake dataset (e.g. FaceForensics++, Celeb-DF, DFDC, or custom images) into the following directory structure:

```text
dataset/
├── train/
│   ├── real/
│   └── fake/
├── validation/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

### Create Synthetic Sample Dataset (Optional Quick Test)

To test the training pipeline immediately without downloading large datasets:

```bash
python -m ml.create_sample_dataset
```

### Run Model Training

Launch model training with custom parameters:

```bash
python -m ml.train --dataset_dir dataset --epochs 10 --batch_size 16 --lr 0.0001
```

Training Outputs:
* Best model saved to: `models/deepfake_detector.pth` and `models/config.json`.
* Training/Validation Loss Graph: `outputs/reports/loss_curve.png`.
* Confusion Matrix Heatmap: `outputs/reports/confusion_matrix.png`.
* ROC Curve Plot: `outputs/reports/roc_curve.png`.

---

## Running the Web Application

Start the FastAPI backend server:

```bash
python backend/main.py
```

Or using Uvicorn directly:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser and navigate to:
**`http://127.0.0.1:8000`**

---

## API Endpoints Reference

### 1. `POST /predict`
Uploads an image file (`multipart/form-data`) and returns full deepfake detection, Grad-CAM heatmap URLs, frequency score, and evidence reasons.

**Curl Example:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/test_face.jpg"
```

### 2. `GET /health`
Returns backend health status, active execution device (CPU/CUDA), and model checkpoint status.

### 3. `GET /history`
Retrieves past prediction log records stored in the SQLite database.

### 4. `POST /train`
Triggers model training in the background.

---

## Explainable AI (Grad-CAM & FFT)

1. **Grad-CAM Heatmaps:** Backpropagates target class gradients (DEEPFAKE) to the last convolutional layer of EfficientNet-B0 (`features[-1]`) to compute weighted feature activations.
2. **Spatial Zone Mapping:** Evaluates activation intensities across anatomical face regions (Eye Region, Mouth/Lips, Skin Texture, Facial Contour, Hairline) to pinpoint suspicious localized anomalies.
3. **FFT Frequency Spectrum:** Computes 2D Discrete Fourier Transform magnitude spectrum to flag artificial high-frequency grid noise typical of synthetic image generators.

---

## Limitations & Future Improvements

* **Single/Multi Face:** Optimized for frontal faces; severe side profiles or heavy occlusions may fallback to full image analysis.
* **Video Deepfakes:** Currently processes still images. Future updates can extract video keyframes for temporal consistency analysis.
* **Ensemble Models:** Integrating Vision Transformers (ViT) or Swin Transformers alongside EfficientNet to further boost robustness against novel deepfake generators.
