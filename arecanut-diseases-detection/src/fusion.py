"""Multi-Modal Fusion Architecture for Smart Areca Nut Crop Advisor.
Integrates Vision Backbone features with Tabular Agro-Climatic Metadata
and Sentinel-2 NDVI trends. Implements expert agronomic risk arbitration
to prevent false overrides of disease detections into healthy classes.
"""

from dataclasses import asdict, dataclass
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from sklearn.ensemble import GradientBoostingClassifier
from src.dataset import autocrop_borders


HEALTHY_CLASSES = {"healthy_leaf", "healthy_nut", "healthy_trunk", "healthy_foot"}
DISEASE_CLASSES = {
    "bud_borer",
    "male_koleroga",
    "stem_bleeding",
    "stem_cracking",
    "yellow_leaf",
}


@dataclass
class TabularFeatureSchema:
    """Standardized schema for multimodal tabular agro-climatic metadata.

    Fields:
      relative_humidity_pct: Air relative humidity (0 - 100%). Crucial for Koleroga (Mahali) fungal sporulation.
      rainfall_mm_7d: Cumulative 7-day precipitation in mm. Drives waterlogging and fungal dispersal.
      temperature_c: Mean ambient temperature in Celsius.
      ndvi_current: Sentinel-2 normalized difference vegetation index (0.0 to 1.0).
      ndvi_delta_15d: 15-day rate of change in NDVI canopy density (negative indicates defoliation/rot).
      ndvi_delta_30d: 30-day seasonal trend in NDVI canopy vigor.
      soil_moisture_pct: Topsoil volumetric water content (0 - 100%).
    """

    relative_humidity_pct: float = 85.0
    rainfall_mm_7d: float = 120.0
    temperature_c: float = 26.5
    ndvi_current: float = 0.58
    ndvi_delta_15d: float = -0.08
    ndvi_delta_30d: float = -0.14
    soil_moisture_pct: float = 72.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.relative_humidity_pct,
            self.rainfall_mm_7d,
            self.temperature_c,
            self.ndvi_current,
            self.ndvi_delta_15d,
            self.ndvi_delta_30d,
            self.soil_moisture_pct,
        ], dtype=np.float32)

    @classmethod
    def feature_names(cls) -> List[str]:
        return [
            "relative_humidity_pct",
            "rainfall_mm_7d",
            "temperature_c",
            "ndvi_current",
            "ndvi_delta_15d",
            "ndvi_delta_30d",
            "soil_moisture_pct",
        ]


def extract_vision_features(
    model: nn.Module,
    image_input: Union[str, Image.Image, torch.Tensor],
    device: Optional[torch.device] = None,
    img_size: int = 380,
) -> Dict[str, Any]:
    """Extracts bottleneck embedding vector and softmax probabilities from vision backbone.
    Applies autocrop_borders to remove letterbox/white/black borders before inference.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    elif isinstance(image_input, torch.Tensor):
        pil_img = None
        tensor = image_input.to(device)
    else:
        raise ValueError("Unsupported image_input type")

    if pil_img is not None:
        # Autocrop letterboxes, banners, or scanning margins
        pil_img = autocrop_borders(pil_img)
        resized_pil = pil_img.resize((img_size, img_size))
        rgb_np = np.array(resized_pil, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_np = (rgb_np - mean) / std
        tensor = torch.from_numpy(norm_np.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        if hasattr(model, "extract_features"):
            emb, probs, logits = model.extract_features(tensor)
        else:
            logits = model(tensor)
            probs = torch.softmax(logits, dim=-1)
            emb = logits  # fallback

    emb_np = emb.squeeze(0).cpu().numpy()
    probs_np = probs.squeeze(0).cpu().numpy()
    logits_np = logits.squeeze(0).cpu().numpy()
    top_idx = int(np.argmax(probs_np))

    return {
        "embeddings": emb_np,
        "probabilities": probs_np,
        "logits": logits_np,
        "top_class_idx": top_idx,
        "confidence": float(probs_np[top_idx]),
    }


def construct_multimodal_feature_vector(
    vision_features: Dict[str, Any],
    tabular_features: Union[TabularFeatureSchema, Dict[str, float], np.ndarray],
    include_full_embeddings: bool = False,
    class_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Concatenates vision predictions/embeddings with weather and Sentinel-2 NDVI metadata."""
    probs = vision_features["probabilities"]
    num_classes = len(probs)
    if class_names and len(class_names) == num_classes:
        prob_names = [f"vis_prob_{name}" for name in class_names]
    else:
        prob_names = [f"vis_prob_class{i}" for i in range(num_classes)]

    if isinstance(tabular_features, TabularFeatureSchema):
        tab_vec = tabular_features.to_vector()
        tab_names = TabularFeatureSchema.feature_names()
    elif isinstance(tabular_features, dict):
        schema = TabularFeatureSchema(**tabular_features)
        tab_vec = schema.to_vector()
        tab_names = TabularFeatureSchema.feature_names()
    else:
        tab_vec = np.asarray(tabular_features, dtype=np.float32)
        tab_names = [f"tabular_feat_{i}" for i in range(len(tab_vec))]

    if include_full_embeddings and "embeddings" in vision_features:
        embeddings = vision_features["embeddings"]
        emb_names = [f"backbone_emb_{i}" for i in range(len(embeddings))]
        feature_vector = np.concatenate([embeddings, probs, tab_vec], axis=0)
        all_names = emb_names + prob_names + tab_names
    else:
        feature_vector = np.concatenate([probs, tab_vec], axis=0)
        all_names = prob_names + tab_names

    return feature_vector.astype(np.float32), all_names


class MultimodalFusionArbiter:
    """XGBoost Arbiter that reconciles visual evidence with agro-environmental context."""

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        use_xgboost: bool = HAS_XGBOOST,
        n_estimators: int = 150,
        max_depth: int = 4,
        learning_rate: float = 0.05,
    ):
        from src.model import DEFAULT_CLASSES
        self.class_names = class_names or DEFAULT_CLASSES
        self.use_xgboost = use_xgboost and HAS_XGBOOST
        self.feature_names: Optional[List[str]] = None

        if self.use_xgboost:
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                eval_metric="mlogloss",
                random_state=42,
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
            )

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> None:
        self.feature_names = feature_names
        self.model.fit(X, y)
        print(f"[Fusion Arbiter] Fitted successfully on {len(X)} samples with {X.shape[1]} features.")

    def predict(self, feature_vector: np.ndarray) -> Tuple[str, float, np.ndarray]:
        if feature_vector.ndim == 1:
            X = feature_vector.reshape(1, -1)
        else:
            X = feature_vector

        probs = self.model.predict_proba(X)[0]
        top_idx = int(np.argmax(probs))
        pred_name = self.class_names[top_idx] if top_idx < len(self.class_names) else f"Class_{top_idx}"
        confidence = float(probs[top_idx])
        return pred_name, confidence, probs

    def save(self, filepath: str) -> None:
        import joblib
        payload = {
            "model": self.model,
            "class_names": self.class_names,
            "feature_names": self.feature_names,
            "use_xgboost": self.use_xgboost,
        }
        joblib.dump(payload, filepath)
        print(f"[Fusion Arbiter] Saved model to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "MultimodalFusionArbiter":
        import joblib
        payload = joblib.load(filepath)
        instance = cls(class_names=payload["class_names"], use_xgboost=payload["use_xgboost"])
        instance.model = payload["model"]
        instance.feature_names = payload.get("feature_names")
        return instance


def arbitrate_multimodal_decision(
    vision_features: Dict[str, Any],
    tabular_features: TabularFeatureSchema,
    arbiter_model: Optional[MultimodalFusionArbiter] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Reconciles visual disease evidence with micro-climate & Sentinel-2 telemetry.

    Guarantees:
      1. NEVER overrides a visual disease detection into a healthy class when micro-climate
         indicates elevated risk (e.g. RH > 80% or heavy rain).
      2. If visual confidence is spread across disease categories, reflects aggregate
         disease risk instead of defaulting to healthy trunk.
      3. Disambiguates Koleroga / Mahali under heavy monsoon conditions (RH > 85%, rain > 100mm).
      4. Corrects low-confidence false positives on uncurated healthy trunks when telemetry
         is benign and canopy NDVI is stable.
      5. Generates Vision Top-3 vs Fused Top-3 with detailed agronomic reasoning.
    """
    from src.model import DEFAULT_CLASSES
    classes = class_names or DEFAULT_CLASSES
    num_classes = len(classes)

    # 1. Vision Metrics
    vis_probs = np.array(vision_features["probabilities"], dtype=np.float32)
    if vis_probs.sum() > 0:
        vis_probs = vis_probs / vis_probs.sum()
    vis_top_idx = int(np.argmax(vis_probs))
    vis_pred = classes[vis_top_idx]
    vis_conf = float(vis_probs[vis_top_idx])

    # Vision Top-3
    vis_sorted_indices = np.argsort(vis_probs)[::-1]
    vision_top3 = [
        {
            "class": classes[idx],
            "confidence": round(float(vis_probs[idx]), 4),
            "confidence_pct": f"{float(vis_probs[idx]) * 100:.2f}%",
        }
        for idx in vis_sorted_indices[:3]
    ]

    disease_indices = [i for i, c in enumerate(classes) if c in DISEASE_CLASSES]
    healthy_indices = [i for i, c in enumerate(classes) if c in HEALTHY_CLASSES]

    vis_disease_mass = float(sum(vis_probs[i] for i in disease_indices))
    vis_healthy_mass = float(sum(vis_probs[i] for i in healthy_indices))

    # 2. Environmental Telemetry
    rh = float(tabular_features.relative_humidity_pct)
    rain = float(tabular_features.rainfall_mm_7d)
    temp = float(tabular_features.temperature_c)
    ndvi = float(tabular_features.ndvi_current)
    ndvi_delta = float(tabular_features.ndvi_delta_15d)
    soil_moist = float(tabular_features.soil_moisture_pct)

    is_high_disease_risk = (rh >= 80.0) or (rain >= 80.0) or (soil_moist >= 75.0 and rh >= 70.0)
    is_monsoon_koleroga = (rh >= 85.0 and rain >= 90.0)
    is_yld_stress = (temp >= 28.0 and ndvi_delta <= -0.06 and rain <= 60.0)
    is_healthy_env = (rh < 75.0 and rain < 60.0 and ndvi >= 0.70 and ndvi_delta >= -0.02)

    # 3. Model Prior / Raw Arbiter Output (if available)
    raw_arb_probs = None
    if arbiter_model is not None:
        try:
            multimodal_vec, _ = construct_multimodal_feature_vector(
                vision_features=vision_features,
                tabular_features=tabular_features,
                include_full_embeddings=False,
                class_names=classes,
            )
            _, _, raw_arb_probs = arbiter_model.predict(multimodal_vec)
            raw_arb_probs = np.array(raw_arb_probs, dtype=np.float32)
        except Exception:
            raw_arb_probs = None

    # 4. Calibrated Multipliers
    multipliers = np.ones(num_classes, dtype=np.float32)
    for i, c in enumerate(classes):
        if c == "male_koleroga":
            if is_monsoon_koleroga:
                multipliers[i] = 2.4
            elif is_high_disease_risk:
                multipliers[i] = 1.6
        elif c in ["bud_borer", "stem_bleeding"]:
            if is_high_disease_risk:
                multipliers[i] = 1.5
        elif c == "yellow_leaf":
            if is_yld_stress:
                multipliers[i] = 1.8
            elif is_high_disease_risk:
                multipliers[i] = 1.2
        elif c in HEALTHY_CLASSES:
            if is_healthy_env:
                multipliers[i] = 1.6
            elif is_high_disease_risk:
                multipliers[i] = 0.25  # Suppress healthy classes when weather is high disease risk

    if raw_arb_probs is not None:
        base_probs = 0.70 * vis_probs + 0.30 * raw_arb_probs
    else:
        base_probs = vis_probs.copy()

    fused_probs = base_probs * multipliers
    fused_probs = fused_probs / fused_probs.sum()

    cand_idx = int(np.argmax(fused_probs))
    cand_class = classes[cand_idx]
    cand_conf = float(fused_probs[cand_idx])

    # 5. Core Decision Logic & Non-Override Safeguards
    # Rule 1: NEVER override disease detection into healthy trunk when micro-climate risk is high
    if (vis_pred in DISEASE_CLASSES or vis_disease_mass >= 0.45) and is_high_disease_risk:
        if cand_class in HEALTHY_CLASSES:
            # Overriding to healthy is strictly blocked! Pick leading disease
            best_disease_idx = max(disease_indices, key=lambda idx: fused_probs[idx])
            fused_pred = classes[best_disease_idx]
            
            # Consolidate disease confidence: normalize among disease classes
            disease_sum = max(1e-6, sum(fused_probs[i] for i in disease_indices))
            normalized_disease_conf = fused_probs[best_disease_idx] / disease_sum
            fused_conf = float(min(0.95, max(vis_conf, normalized_disease_conf)))
            
            arbiter_reasoning = (
                f"Maintained visual disease diagnosis ({fused_pred.replace('_', ' ').title()}) due to elevated "
                f"micro-climate disease risk factors (RH: {rh:.1f}%, 7-day Rainfall: {rain:.1f}mm, "
                f"Aggregate Disease Likelihood: {vis_disease_mass * 100:.1f}%). Blocked statistical bias toward healthy palm."
            )
            # Rebalance array to ensure leading disease class is at top
            fused_probs[best_disease_idx] = max(fused_probs[best_disease_idx], fused_conf)
            fused_probs /= fused_probs.sum()
        else:
            fused_pred = cand_class
            fused_conf = cand_conf
            if fused_pred == "male_koleroga" and is_monsoon_koleroga:
                arbiter_reasoning = (
                    f"Heavy monsoon rainfall ({rain:.1f}mm) and high humidity ({rh:.1f}%) confirmed "
                    f"Koleroga / Mahali Fruit Rot diagnosis."
                )
            else:
                arbiter_reasoning = (
                    f"Visual detection of {fused_pred.replace('_', ' ').title()} ({vis_conf * 100:.1f}%) "
                    f"corroborated by humid agro-climatic conditions (RH: {rh:.1f}%, Rain: {rain:.1f}mm)."
                )

    # Rule 2: Ambiguous prediction on genuine healthy trunk (low confidence < 48% and favorable environment)
    elif vis_pred in DISEASE_CLASSES and vis_conf < 0.48 and is_healthy_env:
        fused_pred = "healthy_trunk" if "healthy_trunk" in classes else cand_class
        fused_conf = max(0.68, cand_conf)
        arbiter_reasoning = (
            f"Visual confidence was low and ambiguous ({vis_conf * 100:.1f}%). Favorable environment "
            f"(RH: {rh:.1f}%, stable NDVI: {ndvi:.2f}) calibrated diagnosis toward Healthy Structural Trunk."
        )
        fused_probs[classes.index(fused_pred)] = fused_conf
        fused_probs /= fused_probs.sum()

    # Rule 3: Visual model identified healthy class
    elif vis_pred in HEALTHY_CLASSES:
        fused_pred = vis_pred
        fused_conf = max(vis_conf, cand_conf)
        if is_high_disease_risk and (rh >= 90.0 or rain >= 150.0):
            arbiter_reasoning = (
                f"Foliar specimen appears visually healthy ({vis_pred.replace('_', ' ').title()}), "
                f"but extreme weather (RH: {rh:.1f}%, Rain: {rain:.1f}mm) indicates high imminent risk for fungal infection."
            )
        else:
            arbiter_reasoning = (
                f"Visual crop health ({vis_pred.replace('_', ' ').title()}: {vis_conf * 100:.1f}%) "
                f"confirmed by stable agro-environmental telemetry."
            )

    # Rule 4: Default multimodal reconciliation
    else:
        fused_pred = cand_class
        fused_conf = cand_conf
        arbiter_reasoning = (
            f"Multimodal synthesis combined {vis_pred.replace('_', ' ').title()} visual evidence "
            f"with ambient temperature ({temp:.1f}°C) and soil moisture ({soil_moist:.1f}%)."
        )

    # 6. Fused Top-3
    fused_sorted_indices = list(np.argsort(fused_probs)[::-1])
    # Ensure fused_pred is ranked #1 in top-3
    pred_idx = classes.index(fused_pred)
    if fused_sorted_indices[0] != pred_idx:
        fused_sorted_indices = [pred_idx] + [i for i in fused_sorted_indices if i != pred_idx]

    fused_top3 = [
        {
            "class": classes[idx],
            "confidence": round(float(fused_probs[idx]), 4),
            "confidence_pct": f"{float(fused_probs[idx]) * 100:.2f}%",
        }
        for idx in fused_sorted_indices[:3]
    ]

    is_override = (fused_pred != vis_pred)

    return {
        "final_predicted_class": fused_pred,
        "final_confidence": round(float(fused_conf), 4),
        "final_confidence_pct": f"{fused_conf * 100:.2f}%",
        "vision_prediction": vis_pred,
        "vision_confidence": round(float(vis_conf), 4),
        "vision_confidence_pct": f"{vis_conf * 100:.2f}%",
        "vision_top3": vision_top3,
        "fused_top3": fused_top3,
        "arbiter_reasoning": arbiter_reasoning,
        "is_override": is_override,
        "fused_probabilities": {
            classes[i]: round(float(fused_probs[i]), 4) for i in range(num_classes)
        },
        "environmental_context": tabular_features.to_dict(),
    }
