# How to Run the Smart Areca Nut Crop Advisor Application

This guide provides simple, step-by-step instructions to run the **Smart Areca Nut Crop Advisor** (FastAPI backend + Progressive Web App frontend) on Windows, macOS, or Linux.

---

## 📋 Prerequisites

1. **Python 3.10+** installed on your system.
2. (Optional) **NVIDIA GPU with CUDA** enabled (CPU is also supported automatically).
3. Modern Web Browser (Google Chrome, Microsoft Edge, Mozilla Firefox, or Brave).

---

## ⚡ Quick Start (2 Steps)

### Step 1: Start the FastAPI Backend Server

Open your terminal or PowerShell in the root project directory (`MajorProject`):

```powershell
# 1. Activate your virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (CMD):
.venv\Scripts\activate.bat
# On Linux/macOS:
source .venv/bin/activate

# 2. Start the FastAPI backend server
python -m uvicorn arecanut-diseases-detection.api.main:app --host 127.0.0.1 --port 8000 --reload
```

> **Backend is ready when you see:**  
> `INFO: Uvicorn running on http://127.0.0.1:8000`

---

### Step 2: Start the Web Frontend (PWA)

Open a **second** terminal window or tab in the root project directory (`MajorProject`):

```powershell
# 1. Activate virtual environment
.venv\Scripts\Activate.ps1

# 2. Serve the frontend directory on port 3000
python -m http.server 3000 --directory arecanut-diseases-detection\frontend
```

> **Frontend is ready when you see:**  
> `Serving HTTP on :: port 3000 (http://[::]:3000/) ...`

---

## 🌐 Accessing the Application

Once both services are running:

| Service | URL | Description |
| :--- | :--- | :--- |
| **User Interface (PWA)** | **[http://localhost:3000](http://localhost:3000)** | Upload crop images, adjust environmental sliders, run diagnosis, and inspect Grad-CAM heatmaps. |
| **API Health Status** | **[http://localhost:8000/health](http://localhost:8000/health)** | Verify active model status and GPU acceleration. |
| **Interactive API Docs** | **[http://localhost:8000/docs](http://localhost:8000/docs)** | Swagger UI to test endpoints (`/predict`, `/predict-multimodal`). |

---

## 🌴 How to Use the Advisor Interface

1. Open **[http://localhost:3000](http://localhost:3000)** in your browser.
2. **Provide an Image**:
   - Drag and drop a picture of an arecanut palm, leaf, trunk, or nut bunch into the dropzone.
   - Or click **Browse File** / **Use Camera** to capture a picture.
3. **Choose Diagnostic Mode**:
   - **Quick Vision Diagnosis**: Click `Run Vision Diagnosis` for fast visual assessment + Grad-CAM heatmap.
   - **Multimodal Agro-Climatic Diagnosis**: Expand the **Environmental & Satellite Telemetry** panel:
     - Click **Auto-Fetch Field Weather** to retrieve live micro-climate data using GPS, or manually adjust sliders (Relative Humidity, Rainfall, Temperature, NDVI).
     - Click `Run Multimodal Diagnostic Scan`.
4. **Inspect Results**:
   - View top diagnosis and confidence percentage.
   - Inspect Top-3 likelihood distribution.
   - Review the tailored agronomic medicine/cultural remedy.
   - Examine the **Grad-CAM visual heatmap** to verify the model is pinpointing genuine pathological lesions.

---

## 🏷️ Supported Crop Health Classes (7 Categories)

1. **`male_koleroga`**: Koleroga / Mahali (Severe Fruit Rot caused by *Phytophthora meadii*)
2. **`bud_borer`**: Bud Borer Infestation (*Conogethes punctiferalis*)
3. **`stem_bleeding_and_cracking`**: Stem Bleeding & Structural Cracking Disorder
4. **`healthy_nut`**: Healthy Developing Nut Bunch
5. **`healthy_leaf`**: Optimal Foliar Canopy
6. **`healthy_trunk`**: Healthy Trunk Structure
7. **`healthy_foot`**: Healthy Basal Bole / Root Zone

---

## 🛠️ Useful Helper Commands

### Retraining the Vision Model
If you ever want to retrain the vision backbone on the dataset:
```powershell
python arecanut-diseases-detection\train.py --data_dir arecanut-diseases-detection\data --output_dir arecanut-diseases-detection\models --epochs_stage1 5 --epochs_stage2 10 --batch_size 16 --img_size 380
```

### Retraining the XGBoost Multimodal Arbiter
```powershell
python arecanut-diseases-detection\demo_fusion.py
```

### Stopping the Servers
Press `Ctrl + C` in each terminal window to gracefully stop the backend and frontend servers.

