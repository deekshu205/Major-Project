"""Modular Inference Script for Smart Areca Nut Crop Advisor.
Predicts disease class, generates Grad-CAM XAI diagnostics, and exports multimodal fusion vectors.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional
import numpy as np
import torch
from PIL import Image

from src.fusion import (
    TabularFeatureSchema,
    construct_multimodal_feature_vector,
    extract_vision_features,
)
from src.gradcam import generate_gradcam_visualization
from src.model import DEFAULT_CLASSES, MEDICINE_RECOMMENDATIONS, build_model


def predict_image(
    image_path: str,
    checkpoint_path: Optional[str] = None,
    output_gradcam_path: Optional[str] = None,
    tabular_features: Optional[Dict[str, float]] = None,
    img_size: int = 380,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Runs end-to-end inference on a single image.

    Outputs:
      - Predicted disease class & confidence score
      - Agronomic remedy recommendation
      - Grad-CAM heatmap visualization artifact
      - Downstream multimodal fusion vector (1792 bottleneck + class probs + tabular metadata)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load class names from checkpoint if present
    class_names = DEFAULT_CLASSES
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "class_names" in ckpt:
            class_names = ckpt["class_names"]

    model = build_model(
        num_classes=len(class_names),
        pretrained=(checkpoint_path is None or not os.path.exists(checkpoint_path)),
        class_names=class_names,
        checkpoint_path=checkpoint_path if (checkpoint_path and os.path.exists(checkpoint_path)) else None,
        device=device,
    )
    model.eval()

    # 1. Vision Feature & Probability Extraction
    vis_data = extract_vision_features(model, image_path, device=device, img_size=img_size)
    pred_idx = vis_data["top_class_idx"]
    predicted_class = class_names[pred_idx] if pred_idx < len(class_names) else f"Class_{pred_idx}"
    confidence = vis_data["confidence"]

    # 2. Grad-CAM Explainability Artifact
    if output_gradcam_path is None:
        base, _ = os.path.splitext(image_path)
        output_gradcam_path = f"{base}_gradcam_overlay.png"

    gradcam_res = generate_gradcam_visualization(
        model=model,
        image_path=image_path,
        output_path=output_gradcam_path,
        img_size=img_size,
        class_names=class_names,
        device=device,
    )

    # 3. Multimodal Decision Arbitration
    tab_schema = TabularFeatureSchema(**(tabular_features or {}))
    arbiter_model = None
    arb_path = os.path.join(os.path.dirname(checkpoint_path or ""), "multimodal_xgboost_arbiter.joblib")
    if os.path.exists(arb_path):
        try:
            arbiter_model = MultimodalFusionArbiter.load(arb_path)
        except Exception:
            arbiter_model = None

    from src.fusion import arbitrate_multimodal_decision
    arb_res = arbitrate_multimodal_decision(
        vision_features=vis_data,
        tabular_features=tab_schema,
        arbiter_model=arbiter_model,
        class_names=class_names,
    )

    final_class = arb_res["final_predicted_class"]
    final_conf = arb_res["final_confidence"]

    # 4. Agronomic Medicine / Action Recommendation
    remedy = MEDICINE_RECOMMENDATIONS.get(
        final_class,
        "Consult local Krishi Vigyan Kendra (KVK) or horticulture officer for specialized fungicide treatment."
    )

    # 5. Multimodal Fusion Vector for Downstream Export
    fusion_vector, feature_names = construct_multimodal_feature_vector(
        vision_features=vis_data,
        tabular_features=tab_schema,
        include_full_embeddings=True,
        class_names=class_names,
    )

    result = {
        "image_path": os.path.abspath(image_path),
        "predicted_class": final_class,
        "confidence": round(float(final_conf), 4),
        "confidence_percentage": f"{final_conf * 100:.2f}%",
        "vision_prediction": predicted_class,
        "vision_confidence": round(float(confidence), 4),
        "vision_confidence_percentage": f"{confidence * 100:.2f}%",
        "vision_top3": arb_res["vision_top3"],
        "fused_top3": arb_res["fused_top3"],
        "arbiter_reasoning": arb_res["arbiter_reasoning"],
        "is_override": arb_res["is_override"],
        "probabilities": arb_res["fused_probabilities"],
        "remedy": remedy,
        "gradcam_artifact": os.path.abspath(output_gradcam_path),
        "multimodal_fusion_vector_shape": list(fusion_vector.shape),
        "multimodal_fusion_vector": fusion_vector,
        "multimodal_feature_names": feature_names,
        "tabular_context_used": tab_schema.to_dict(),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Smart Areca Nut Crop Advisor - Inference & Diagnostic CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to leaf image file")
    parser.add_argument("--model", type=str, default="models/best_arecanut_efficientnet_b4.pth", help="Checkpoint path")
    parser.add_argument("--gradcam_out", type=str, default=None, help="Output path for Grad-CAM diagnostic image")
    parser.add_argument("--save_vector", type=str, default=None, help="Output path (.npy or .json) for fusion vector")
    parser.add_argument("--rh", type=float, default=85.0, help="Relative Humidity (%)")
    parser.add_argument("--rainfall", type=float, default=120.0, help="7-day Cumulative Rainfall (mm)")
    parser.add_argument("--ndvi", type=float, default=0.55, help="Sentinel-2 NDVI value (0-1)")
    parser.add_argument("--temp", type=float, default=26.5, help="Temperature (C)")
    parser.add_argument("--soil_moisture", type=float, default=None, help="Soil Moisture (%%)")
    parser.add_argument("--ndvi_delta", type=float, default=None, help="15-day NDVI delta")
    parser.add_argument("--img_size", type=int, default=380, help="Model input resolution")

    args = parser.parse_args()

    # Determine agronomic defaults for NDVI delta and soil moisture if unspecified
    soil_moist = args.soil_moisture if args.soil_moisture is not None else (75.0 if args.rainfall > 80.0 else 45.0)
    ndvi_delta = args.ndvi_delta if args.ndvi_delta is not None else (0.01 if args.ndvi >= 0.70 else -0.06)

    tab_input = {
        "relative_humidity_pct": args.rh,
        "rainfall_mm_7d": args.rainfall,
        "temperature_c": args.temp,
        "ndvi_current": args.ndvi,
        "ndvi_delta_15d": ndvi_delta,
        "ndvi_delta_30d": ndvi_delta * 2,
        "soil_moisture_pct": soil_moist,
    }

    checkpoint = args.model if os.path.exists(args.model) else None
    if checkpoint is None:
        print(f"[Notice] Model checkpoint '{args.model}' not found. Initializing with ImageNet backbone for zero-shot testing.")

    result = predict_image(
        image_path=args.image,
        checkpoint_path=checkpoint,
        output_gradcam_path=args.gradcam_out,
        tabular_features=tab_input,
        img_size=args.img_size,
    )

    # Save vector if requested
    if args.save_vector:
        vec_dir = os.path.dirname(args.save_vector)
        if vec_dir:
            os.makedirs(vec_dir, exist_ok=True)
        if args.save_vector.endswith(".npy"):
            np.save(args.save_vector, result["multimodal_fusion_vector"])
        else:
            with open(args.save_vector, "w") as f:
                json.dump({
                    "vector": result["multimodal_fusion_vector"].tolist(),
                    "feature_names": result["multimodal_feature_names"],
                }, f)
        print(f"[Inference] Saved fusion vector to {args.save_vector}")

    # Display clean CLI output summary
    print("\n" + "=" * 70)
    print("[SMART ARECA NUT CROP ADVISOR - MULTIMODAL DIAGNOSTIC REPORT]")
    print("=" * 70)
    print(f"Specimen Image:        {result['image_path']}")
    print(f"Vision Prediction:     {result['vision_prediction']} ({result['vision_confidence_percentage']})")
    print(f"Fused Final Decision:  {result['predicted_class']} ({result['confidence_percentage']})")
    print(f"Override Status:       {'Yes (Calibrated by Telemetry)' if result['is_override'] else 'No (Direct Agreement)'}")
    print(f"\n--- Multimodal Arbiter Reasoning ---")
    print(f"  {result['arbiter_reasoning']}")
    print(f"\n--- Vision Top-3 Candidates ---")
    for item in result["vision_top3"]:
        print(f"  {item['class']:<20}: {item['confidence_pct']:>8}")
    print(f"\n--- Fused Top-3 Candidates ---")
    for item in result["fused_top3"]:
        print(f"  {item['class']:<20}: {item['confidence_pct']:>8}")
    print(f"\n--- Grad-CAM Heatmap Artifact ---")
    print(f"  {result['gradcam_artifact']}")
    print(f"\n--- Actionable Agronomic Remedy ---")
    print(f"  {result['remedy']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
