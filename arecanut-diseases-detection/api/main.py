"""Production FastAPI Backend for Smart Areca Nut Crop Advisor.
Serves EfficientNet-B4 predictions, Grad-CAM XAI diagnostic heatmaps,
and Multimodal Agro-Climatic (Weather & Sentinel-2 NDVI) Fusion.
"""

import base64
from io import BytesIO
import os
import sys
from typing import Optional
import numpy as np
from PIL import Image
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# Add parent project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import autocrop_borders
from src.fusion import (
    MultimodalFusionArbiter,
    TabularFeatureSchema,
    arbitrate_multimodal_decision,
    construct_multimodal_feature_vector,
    extract_vision_features,
)
from src.gradcam import GradCAM, apply_colormap_on_image, generate_gradcam_visualization
from src.model import DEFAULT_CLASSES, MEDICINE_RECOMMENDATIONS, build_model

app = FastAPI(
    title="Smart Areca Nut Crop Advisor API",
    description="Upgraded Deep Learning (EfficientNet-B4) & Multimodal XGBoost Decision Support System",
    version="2.0.0",
)

# Allow CORS for React/Vite/Vue frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "models", "best_arecanut_efficientnet_b4.pth")
ARBITER_PATH = os.path.join(PROJECT_ROOT, "models", "multimodal_xgboost_arbiter.joblib")
STATIC_GRADCAM_DIR = os.path.join(PROJECT_ROOT, "artifacts", "gradcam")
os.makedirs(STATIC_GRADCAM_DIR, exist_ok=True)

# Load Vision Model
MODEL = None
CLASS_NAMES = DEFAULT_CLASSES
ARBITER = None


def load_artifacts():
    global MODEL, CLASS_NAMES, ARBITER
    if MODEL is not None:
        return
    print(f"[API Startup] Initializing on device: {DEVICE}")

    # Vision Model
    if os.path.exists(CHECKPOINT_PATH):
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
            CLASS_NAMES = ckpt.get("class_names", DEFAULT_CLASSES)
            MODEL = build_model(
                num_classes=len(CLASS_NAMES),
                pretrained=False,
                class_names=CLASS_NAMES,
                checkpoint_path=CHECKPOINT_PATH,
                device=DEVICE,
            )
            print(f"[API Startup] Loaded trained weights from {CHECKPOINT_PATH}")
        except Exception as e:
            print(f"[API Startup] Failed to load checkpoint ({e}). Using initialized backbone.")
            MODEL = build_model(num_classes=len(CLASS_NAMES), pretrained=True, class_names=CLASS_NAMES, device=DEVICE)
    else:
        print("[API Startup] Checkpoint not found. Initializing EfficientNet-B4 backbone.")
        MODEL = build_model(num_classes=len(CLASS_NAMES), pretrained=True, class_names=CLASS_NAMES, device=DEVICE)

    MODEL.eval()

    # Multimodal Arbiter
    if os.path.exists(ARBITER_PATH):
        try:
            ARBITER = MultimodalFusionArbiter.load(ARBITER_PATH)
            print(f"[API Startup] Loaded Multimodal Arbiter from {ARBITER_PATH}")
        except Exception as e:
            print(f"[API Startup] Could not load arbiter: {e}")


@app.on_event("startup")
def startup_event():
    load_artifacts()


def get_model():
    if MODEL is None:
        load_artifacts()
    return MODEL


def get_arbiter():
    if ARBITER is None and os.path.exists(ARBITER_PATH):
        load_artifacts()
    return ARBITER


@app.get("/ping")
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Smart Areca Nut Crop Advisor",
        "vision_backbone": "EfficientNet-B4",
        "classes": CLASS_NAMES,
        "device": str(DEVICE),
        "multimodal_arbiter_ready": ARBITER is not None,
    }


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """Single-image vision diagnosis endpoint with letterbox autocropping and Grad-CAM explainability."""
    try:
        file_bytes = await file.read()
        pil_img = Image.open(BytesIO(file_bytes)).convert("RGB")
        pil_img = autocrop_borders(pil_img)

        # Save temporary image for Grad-CAM processing
        temp_img_name = f"upload_{os.urandom(4).hex()}.jpg"
        temp_img_path = os.path.join(STATIC_GRADCAM_DIR, temp_img_name)
        pil_img.save(temp_img_path)

        gradcam_artifact_name = f"gradcam_{temp_img_name}"
        gradcam_artifact_path = os.path.join(STATIC_GRADCAM_DIR, gradcam_artifact_name)

        model = get_model()
        # Generate prediction and Grad-CAM visualization
        gradcam_res = generate_gradcam_visualization(
            model=model,
            image_path=temp_img_path,
            output_path=gradcam_artifact_path,
            img_size=380,
            class_names=CLASS_NAMES,
            device=DEVICE,
        )

        pred_class = gradcam_res["predicted_class"]
        confidence = gradcam_res["confidence"]

        # Extract vision probabilities for Top-3 breakdown
        vis_data = extract_vision_features(model, pil_img, device=DEVICE, img_size=380)
        vis_probs = vis_data["probabilities"]
        vis_sorted_indices = np.argsort(vis_probs)[::-1]
        vision_top3 = [
            {
                "class": CLASS_NAMES[idx],
                "confidence": round(float(vis_probs[idx]), 4),
                "confidence_pct": f"{float(vis_probs[idx]) * 100:.2f}%",
            }
            for idx in vis_sorted_indices[:3]
        ]

        with open(gradcam_artifact_path, "rb") as f:
            b64_artifact = base64.b64encode(f.read()).decode("utf-8")

        remedy = MEDICINE_RECOMMENDATIONS.get(
            pred_class,
            "Inspect plantation canopy, prune infected limbs, and spray recommended copper-based fungicides."
        )

        return {
            "predicted_class": pred_class,
            "confidence": round(float(confidence), 4),
            "confidence_pct": f"{confidence * 100:.2f}%",
            "vision_prediction": pred_class,
            "vision_confidence": round(float(confidence), 4),
            "vision_confidence_pct": f"{confidence * 100:.2f}%",
            "vision_top3": vision_top3,
            "probabilities": {CLASS_NAMES[i]: round(float(vis_probs[i]), 4) for i in range(len(CLASS_NAMES))},
            "remedy": remedy,
            "gradcam_artifact_url": f"/gradcam/{gradcam_artifact_name}",
            "gradcam_base64": f"data:image/png;base64,{b64_artifact}",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict-multimodal")
async def predict_multimodal_endpoint(
    file: UploadFile = File(...),
    relative_humidity_pct: float = Form(85.0),
    rainfall_mm_7d: float = Form(120.0),
    temperature_c: float = Form(26.5),
    ndvi_current: float = Form(0.55),
    ndvi_delta_15d: float = Form(-0.06),
    ndvi_delta_30d: float = Form(-0.12),
    soil_moisture_pct: float = Form(70.0),
):
    """Multimodal decision endpoint combining visual evidence with weather & Sentinel-2 NDVI.
    Ensures that visual disease evidence is never overridden by healthy priors under high-risk weather.
    """
    try:
        file_bytes = await file.read()
        pil_img = Image.open(BytesIO(file_bytes)).convert("RGB")
        pil_img = autocrop_borders(pil_img)
        model = get_model()
        arbiter = get_arbiter()

        # 1. Vision Feature Extraction
        vis_data = extract_vision_features(model, pil_img, device=DEVICE, img_size=380)

        # 2. Tabular Schema
        tab_schema = TabularFeatureSchema(
            relative_humidity_pct=relative_humidity_pct,
            rainfall_mm_7d=rainfall_mm_7d,
            temperature_c=temperature_c,
            ndvi_current=ndvi_current,
            ndvi_delta_15d=ndvi_delta_15d,
            ndvi_delta_30d=ndvi_delta_30d,
            soil_moisture_pct=soil_moisture_pct,
        )

        # 3. Multimodal Decision Arbitration
        arb_res = arbitrate_multimodal_decision(
            vision_features=vis_data,
            tabular_features=tab_schema,
            arbiter_model=arbiter,
            class_names=CLASS_NAMES,
        )

        # 4. Generate Grad-CAM visualization artifact for explainability
        temp_img_name = f"upload_{os.urandom(4).hex()}.jpg"
        temp_img_path = os.path.join(STATIC_GRADCAM_DIR, temp_img_name)
        pil_img.save(temp_img_path)

        gradcam_artifact_name = f"gradcam_{temp_img_name}"
        gradcam_artifact_path = os.path.join(STATIC_GRADCAM_DIR, gradcam_artifact_name)

        gradcam_res = generate_gradcam_visualization(
            model=model,
            image_path=temp_img_path,
            output_path=gradcam_artifact_path,
            img_size=380,
            class_names=CLASS_NAMES,
            device=DEVICE,
        )

        with open(gradcam_artifact_path, "rb") as f:
            b64_artifact = base64.b64encode(f.read()).decode("utf-8")

        final_class = arb_res["final_predicted_class"]
        remedy = MEDICINE_RECOMMENDATIONS.get(final_class, "Standard agronomic orchard management.")

        return {
            "final_predicted_class": final_class,
            "final_confidence": arb_res["final_confidence"],
            "final_confidence_pct": arb_res["final_confidence_pct"],
            "vision_prediction": arb_res["vision_prediction"],
            "vision_confidence": arb_res["vision_confidence"],
            "vision_confidence_pct": arb_res["vision_confidence_pct"],
            "vision_only_prediction": arb_res["vision_prediction"],  # Backwards compatibility
            "vision_only_confidence": arb_res["vision_confidence"],  # Backwards compatibility
            "vision_top3": arb_res["vision_top3"],
            "multimodal_prediction": arb_res["final_predicted_class"],
            "multimodal_confidence": arb_res["final_confidence"],
            "multimodal_confidence_pct": arb_res["final_confidence_pct"],
            "fused_top3": arb_res["fused_top3"],
            "arbiter_reasoning": arb_res["arbiter_reasoning"],
            "is_override": arb_res["is_override"],
            "remedy": remedy,
            "environmental_context": tab_schema.to_dict(),
            "multimodal_probabilities": arb_res["fused_probabilities"],
            "gradcam_artifact_url": f"/gradcam/{gradcam_artifact_name}",
            "gradcam_base64": f"data:image/png;base64,{b64_artifact}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multimodal inference error: {str(e)}")


@app.get("/gradcam/{filename}")
async def serve_gradcam_image(filename: str):
    path = os.path.join(STATIC_GRADCAM_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path, media_type="image/png")


# Mount frontend directory for Progressive Web App (PWA)
from fastapi.staticfiles import StaticFiles

FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
