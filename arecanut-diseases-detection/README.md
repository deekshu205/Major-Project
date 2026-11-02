# Arecanut Disease Detection & Multimodal Agro-Climatic Decision Support System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/framework-PyTorch%20%7C%20FastAPI-red.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/model-XGBoost%20Multimodal%20Arbiter-green.svg)](https://xgboost.readthedocs.io/)
[![XAI](https://img.shields.io/badge/explainability-Grad--CAM-yellow.svg)](https://github.com/jacobgil/pytorch-grad-cam)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An intelligent, multimodal deep learning and decision support system for automated detection, Explainable AI (XAI) localization, and micro-climate risk arbitration of diseases affecting arecanut (*Areca catechu* L.) crops.

---

## Screenshots & Visualizations

![System Interface and Workflow](./Screenshot.png)

*Diagnostic Interface: Deep learning vision backbone with Grad-CAM heatmap visualization and micro-climatic arbitration.*

---

## Project Overview

Arecanut (betel nut) is an economically vital plantation crop in tropical regions. However, arecanut palms are vulnerable to destructive pathogens affecting roots, trunks, foliage, and fruits throughout their growth cycle. 

This project delivers a multi-stage **Smart Areca Nut Crop Advisor** combining:
1. **Computer Vision Backbone:** Pre-trained **EfficientNet (B0 / B4)** transfer learning backbones fine-tuned on a 9-class dataset of arecanut plant organs and diseases.
2. **Explainable AI (Grad-CAM):** Gradient-weighted Class Activation Mapping hooked to the final convolutional layer to visualize pathological lesions and ensure the network focuses on plant tissue rather than background clutter.
3. **Multimodal Environmental Fusion Arbiter:** Tabular machine learning model (**XGBoost**) that fuses vision softmax probabilities and bottleneck feature embeddings with real-time **agro-climatic telemetry** (Relative Humidity, 7-day Rainfall, Temperature, Soil Moisture) and **Sentinel-2 NDVI canopy trends** (Current, 15-day delta, 30-day delta).
4. **Agronomic Heuristic Arbitration:** Domain-specific decision rules preventing visual disease detections from being overridden into healthy classifications during high-risk micro-climatic events.
5. **Full-Stack Deployment:** Production-grade **FastAPI** asynchronous REST backend coupled with a Progressive Web App (**PWA**) featuring GPS-based auto-weather fetching, camera capture, offline caching, and actionable agronomic remedies.

---

## Disease Taxonomy & Management

The system evaluates both healthy anatomical organs and major plant diseases across 9 distinct categories:

| Category | Type | Pathogen / Cause | Prescribed Agronomic Remedy |
| :--- | :--- | :--- | :--- |
| **`male_koleroga`** | Disease | *Phytophthora meadii* (Fruit rot / Mahali) | Pre-monsoon prophylactic spraying of 1% Bordeaux mixture with rosin-soda adhesive. In case of active rot, apply Metalaxyl + Mancozeb (2 g/L) or Fosetyl-Al (2.5 g/L). |
| **`stem_bleeding`** | Disease | *Thielaviopsis paradoxa* | Chisel out infected dark-bleeding bark tissue; swab wound with coal tar or Bordeaux paste. Drench root zone with Tridemorph (0.1%) or Hexaconazole (2 ml/L). |
| **`stem_cracking`** | Condition | Physiological stress / water surges | Apply Copper Oxychloride (3 g/L) paste to fissures to prevent secondary fungal infestation. Regulate irrigation and apply 10% lime wash to trunk. |
| **`yellow_leaf`** | Disease | Phytoplasma / nutrient deficiency | Foliar spray of Fosetyl-Al (0.2%) or Potassium Phosphonate (3 ml/L). Apply balanced micronutrient drenches (Mg, Zn, B) and ensure soil drainage. |
| **`bud_borer`** | Pest | *Conogethes punctiferalis* | Spray Chlorpyrifos 20 EC (2 ml/L) or Neem oil (5 ml/L) into the spindle leaf axils. Install light and pheromone traps to control adult moths. |
| **`healthy_leaf`** | Healthy | Normal foliage | Vigorous foliage. Maintain balanced NPK nutrition (100:40:140 g/palm/year), mulch base, and monitor regularly. |
| **`healthy_nut`** | Healthy | Normal nut bunch | Normal fruit bunch development. Maintain drip irrigation during dry intervals and clear crowns of debris. |
| **`healthy_trunk`** | Healthy | Normal stem | Normal trunk structure. In high sunlight areas, apply slaked lime wash (10%) on southwest side to prevent scorch. |
| **`healthy_foot`** | Healthy | Normal basal bole | Basal root zone healthy. Maintain field drainage, avoid mechanical root injury, and apply organic compost annually. |

---

## System Architecture

```
[ Plantation Camera / Field Upload ]
                 │
                 ▼
     [ Auto-Crop Borders (OpenCV) ]
                 │
                 ▼
[ Background-Invariant Augmentation Pipeline ]
 (RandomResizedCrop, RandomErasing, Jitter)
                 │
                 ▼
   [ EfficientNet Vision Backbone ]
          │                   │
          ▼                   ▼
 [ Feature Embeddings & ] [ Grad-CAM XAI Hook ]
 [ Softmax Probabilities]             │
          │                           ▼
          │               [ Visual Lesion Heatmap Overlay ]
          │                           │
          ▼                           │
[ Concatenated Multimodal Vector ]   │
(Vision + Weather + Sentinel-2 NDVI)  │
          │                           │
          ▼                           │
[ Multimodal XGBoost Decision Arbiter ]│
          │                           │
          ▼                           │
[ Agronomic Heuristic Rule Layer ]     │
          │                           │
          ▼                           ▼
[ Final Calibrated Diagnosis + Top-3 Confidence + Actionable Treatment Plan ]
```

---

## Key Technical Innovations

### 1. Border-Aware & Background-Invariant Preprocessing
- **`autocrop_borders`**: Analyzes grayscale bounding contours to automatically detect and remove letterbox banners, scanner edges, and digital photo borders.
- **Lesion-Focused Augmentation**: Utilizes `RandomResizedCrop(scale=(0.5, 1.0))` and `RandomErasing(p=0.3)` to enforce learning on plant tissue and eliminate shortcut learning from background skies, fences, or wires.

### 2. Explainable AI with Grad-CAM
Grad-CAM calculates the gradient of the predicted class score with respect to the activation maps of the final convolutional head (`conv_head`):

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)$$

This allows agricultural experts and farmers to immediately see where the network is focusing, verifying that predictions are driven by authentic lesions.

### 3. Multimodal Decision Arbitration
Fuses visual evidence with seven agro-environmental indicators:
- **Relative Humidity (%)**: Fungal sporulation driver.
- **7-Day Cumulative Rainfall (mm)**: Moisture saturation metric.
- **Ambient Temperature (°C)**: Thermal profile.
- **Sentinel-2 NDVI Current**: Canopy vigor index ($0.0 - 1.0$).
- **Sentinel-2 NDVI 15-Day Delta**: Rate of canopy decline.
- **Sentinel-2 NDVI 30-Day Delta**: Seasonal canopy trend.
- **Volumetric Soil Moisture (%)**: Root zone water retention.

Domain heuristic arbitration guarantees that visual disease signs are **never overridden into healthy status** when environmental telemetry reflects high disease risk (e.g. monsoon humidity $\ge 85\%$ or rain $\ge 90\text{ mm}$).

---

## Experimental Results

The upgraded vision pipeline evaluated on held-out test data achieves robust classification metrics across all classes:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `bud_borer` | 1.00 | 1.00 | 1.00 | 12 |
| `healthy_foot` | 1.00 | 1.00 | 1.00 | 12 |
| `healthy_leaf` | 1.00 | 1.00 | 1.00 | 12 |
| `healthy_nut` | 1.00 | 1.00 | 1.00 | 13 |
| `healthy_trunk` | 1.00 | 1.00 | 1.00 | 12 |
| `male_koleroga` | 1.00 | 1.00 | 1.00 | 11 |
| `stem_bleeding` | 1.00 | 1.00 | 1.00 | 12 |
| `stem_cracking` | 1.00 | 1.00 | 1.00 | 12 |
| `yellow_leaf` | 1.00 | 1.00 | 1.00 | 12 |
| **Overall Accuracy** | — | — | **1.00** | **108** |
| **Macro Average** | **1.00** | **1.00** | **1.00** | **108** |
| **Weighted Average** | **1.00** | **1.00** | **1.00** | **108** |

*Evaluation metrics recorded from `models/evaluation_summary.json`.*

---

## Repository Structure

```
arecanut-diseases-detection/
├── api/
│   ├── main.py              # Production FastAPI server (Endpoints, CORS, Grad-CAM generation)
│   └── install.txt          # API service dependencies
├── data/                    # Dataset directory partitioned by class
│   ├── train/               # Training image sets across 9 classes
│   └── val/                 # Validation image sets across 9 classes
├── frontend/                # Web Interface & Progressive Web App
│   ├── index.html           # Responsive PWA markup
│   ├── app.js               # Client controller (GPS weather fetch, camera, prediction rendering)
│   ├── styles.css           # Modern mobile-optimized stylesheets
│   ├── sw.js                # Service Worker for offline asset caching
│   ├── manifest.json        # PWA application manifest
│   └── src/                 # React component library
├── models/                  # Trained checkpoints & serialized models
│   ├── best_arecanut_efficientnet_b4.pth    # PyTorch EfficientNet-B4 weights
│   ├── best_arecanut_efficientnet_b0.pth    # PyTorch EfficientNet-B0 weights
│   ├── multimodal_xgboost_arbiter.joblib    # Serialized multimodal XGBoost arbiter
│   ├── evaluation_summary.json             # Test set quantitative evaluation metrics
│   └── test_confusion_matrix.png           # Evaluated confusion matrix plot
├── src/                     # Core deep learning & fusion source modules
│   ├── dataset.py           # Dataloaders, autocrop_borders, tissue augmentations
│   ├── model.py             # EfficientNet model wrappers & head definitions
│   ├── gradcam.py           # Grad-CAM PyTorch hooks and colormap blending
│   ├── fusion.py            # Multimodal XGBoost arbiter & heuristic arbitration
│   └── metrics.py           # Performance calculation and confusion matrix generation
├── train.py                 # Multi-stage transfer learning training script
├── predict.py               # Command-line single & multimodal inference runner
├── demo_fusion.py           # End-to-end multimodal simulation benchmark
├── requirements.txt         # Python project dependencies
└── README.md                # Project documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+ (CUDA-enabled GPU optional for accelerated inference)
- Node.js (optional, for React frontend development)

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/example-org/arecanut-diseases-detection.git
cd arecanut-diseases-detection

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Running Model Training
To train the vision backbone using two-stage transfer learning:
```bash
python train.py --data_dir data --epochs 30 --batch_size 16 --img_size 380
```

### 3. Launching the Backend API
Start the FastAPI server on port 8000:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be accessible at: `http://localhost:8000/docs`

### 4. Running the Progressive Web App (PWA)
Serve the `frontend/` directory with any static web server:
```bash
# Using Python built-in HTTP server:
cd frontend
python -m http.server 3000
```
Open your browser and navigate to `http://localhost:3000`.

---

## API Endpoints Reference

### `GET /health`
Returns service status, vision model architecture, and GPU initialization state.

### `POST /predict`
Uploads a plantation image and returns:
- `predicted_class`: Top predicted category
- `confidence`: Confidence score ($0.0 - 1.0$)
- `vision_top3`: Top 3 candidate classes with percentages
- `remedy`: Agronomic recommendations and chemical prescriptions
- `gradcam_base64`: Rendered Grad-CAM heatmap overlay for verification

### `POST /predict-multimodal`
Uploads a plantation image along with environmental telemetry:
- Multipart Form Fields:
  - `file`: Image file
  - `relative_humidity_pct`: float (e.g. 85.0)
  - `rainfall_mm_7d`: float (e.g. 120.0)
  - `temperature_c`: float (e.g. 26.5)
  - `ndvi_current`: float (e.g. 0.55)
  - `ndvi_delta_15d`: float (e.g. -0.06)
  - `soil_moisture_pct`: float (e.g. 70.0)

Returns the arbitrated diagnosis, fused confidence scores, environmental risk profile, and Grad-CAM visual artifact.

---

## Academic & Technical Documentation

This codebase and system architecture are designed for research, education, and field deployment in smart agricultural monitoring:
- **Research Paper Preparation:** Refer to the modular design in `src/` for methodology, equations, and benchmark evaluation tables.
- **Explainability Validation:** Use the Grad-CAM artifacts in `artifacts/gradcam/` to demonstrate model transparency and absence of background shortcut learning.

---

