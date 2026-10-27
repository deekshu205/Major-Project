"""EfficientNet-B0 Model Architecture for Arecanut Disease Classification.
Provides transfer learning, multi-stage fine-tuning, dropout regularization,
and feature extraction hooks optimized to prevent background memorization.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

from torchvision import models as tv_models


DEFAULT_CLASSES = [
    "bud_borer",
    "healthy_foot",
    "healthy_leaf",
    "healthy_nut",
    "healthy_trunk",
    "male_koleroga",
    "stem_bleeding",
    "stem_cracking",
    "yellow_leaf",
]

MEDICINE_RECOMMENDATIONS = {
    "bud_borer": (
        "Spray Chlorpyrifos 20 EC (2 ml/L) or Neem oil (5 ml/L) into the spindle leaf axils. "
        "Install light traps in the plantation to monitor and control adult borer moths."
    ),
    "healthy_foot": (
        "Palm foot and basal bole are healthy. Maintain field drainage, avoid direct physical injuries "
        "during tilling, and apply organic compost annually around the base."
    ),
    "healthy_leaf": (
        "Foliage is vigorous and healthy. Continue balanced NPK nutrition (100:40:140 g/palm/year), "
        "maintain mulch, and monitor for early pest attacks."
    ),
    "healthy_nut": (
        "Nut bunches are developing normally. Ensure regular drip irrigation during dry periods "
        "and maintain clean crowns free from debris."
    ),
    "healthy_trunk": (
        "Trunk is healthy. In high sunlight regions, apply slaked lime wash (10%) on southwest "
        "side of trunk to prevent sun scorching."
    ),
    "male_koleroga": (
        "Pre-monsoon prophylactic spraying of 1% Bordeaux mixture with rosin-soda adhesive. "
        "If fruit/bunch rot symptoms appear, spray Metalaxyl + Mancozeb (2 g/L) or Fosetyl-Al (2.5 g/L). "
        "Ensure prompt field drainage."
    ),
    "stem_bleeding": (
        "Chisel out infected bleeding bark tissue and swab the wound with coal tar or Bordeaux paste. "
        "Drench root feeding zone with Tridemorph (0.1% / 2 ml/L) or Hexaconazole (2 ml/L)."
    ),
    "stem_cracking": (
        "Apply Copper Oxychloride (3 g/L) paste on cracks to prevent secondary fungal infection. "
        "Regulate irrigation frequency to avoid sudden water surge and trunk expansion."
    ),
    "yellow_leaf": (
        "Spray Fosetyl-Al (0.2%) or Potassium Phosphonate (3 ml/L) on leaf undersides. "
        "Correct potassium and magnesium deficiencies via soil micronutrient drenches, and improve soil aeration."
    ),
}


class ArecaEfficientNetB0(nn.Module):
    """Downscaled Vision Backbone using EfficientNet-B0 with regularized classification head.
    EfficientNet-B0 (~5.3M params vs 19M in B4) prevents background shortcut memorization
    on limited-sample agricultural datasets.
    """

    def __init__(
        self,
        num_classes: int = len(DEFAULT_CLASSES),
        pretrained: bool = True,
        dropout_rate: float = 0.4,
        hidden_dim: int = 512,
        class_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names or DEFAULT_CLASSES
        self.in_features = 1280  # EfficientNet-B0 bottleneck feature dimension

        # Prefer timm backbone; fallback to torchvision if unavailable
        if HAS_TIMM:
            self.backend = "timm"
            try:
                self.backbone = timm.create_model(
                    "efficientnet_b0",
                    pretrained=pretrained,
                    num_classes=0,  # pooled 1280-dim output
                )
            except Exception as e:
                print(f"[Warning] Failed to fetch pretrained timm weights ({e}). Initializing random weights.")
                self.backbone = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
            self.target_layer = self.backbone.conv_head
        else:
            self.backend = "torchvision"
            weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            tv_eff = tv_models.efficientnet_b0(weights=weights)
            self.backbone = tv_eff.features
            self.target_layer = self.backbone[-1]
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Modern Regularized Classification Head with nn.Dropout(p=0.4) before final layer
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.in_features),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.in_features, hidden_dim),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=0.4),  # Enforced dropout before final classification head
            nn.Linear(hidden_dim, num_classes),
        )

    def get_target_layer_for_gradcam(self) -> nn.Module:
        """Returns the final convolutional layer for Grad-CAM explanation."""
        return self.target_layer

    def extract_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts pooled 1280-dim bottleneck embedding from input image tensor."""
        if self.backend == "timm":
            embedding = self.backbone(x)
        else:
            feat_map = self.backbone(x)
            pooled = self.global_pool(feat_map)
            embedding = torch.flatten(pooled, 1)
        return embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw class logits."""
        embedding = self.extract_backbone_features(x)
        logits = self.classifier(embedding)
        return logits

    def extract_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Multi-modal feature extraction hook.

        Returns:
            bottleneck_embedding: (B, 1280) feature vector
            probabilities: (B, num_classes) softmax confidence distribution
            logits: (B, num_classes) raw model logits
        """
        embedding = self.extract_backbone_features(x)
        logits = self.classifier(embedding)
        probabilities = F.softmax(logits, dim=-1)
        return embedding, probabilities, logits

    def freeze_backbone(self) -> None:
        """Freeze all backbone feature extractor layers (Phase 1 Training)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
        print("[Model] Backbone layers frozen. Classifier head is trainable.")

    def unfreeze_top_stages(self, num_blocks: int = 3) -> None:
        """Unfreeze top convolutional blocks for fine-tuning (Phase 2 Training)."""
        if self.backend == "timm" and hasattr(self.backbone, "blocks"):
            total_blocks = len(self.backbone.blocks)
            unfreeze_from = max(0, total_blocks - num_blocks)
            for i in range(unfreeze_from, total_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True
            if hasattr(self.backbone, "conv_head"):
                for param in self.backbone.conv_head.parameters():
                    param.requires_grad = True
                if hasattr(self.backbone, "bn2"):
                    for param in self.backbone.bn2.parameters():
                        param.requires_grad = True
            print(f"[Model] Unfroze top {num_blocks} blocks + conv_head for stage-2 fine-tuning.")
        else:
            self.unfreeze_all()

    def unfreeze_all(self) -> None:
        """Unfreeze all backbone and head parameters."""
        for param in self.parameters():
            param.requires_grad = True
        print("[Model] All layers unfrozen for end-to-end fine-tuning.")


# Backward compatibility alias
ArecaEfficientNetB4 = ArecaEfficientNetB0


def build_model(
    num_classes: int = len(DEFAULT_CLASSES),
    pretrained: bool = True,
    dropout_rate: float = 0.4,
    class_names: Optional[List[str]] = None,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> ArecaEfficientNetB0:
    """Factory helper to build and optionally restore model weights."""
    model = ArecaEfficientNetB0(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        class_names=class_names,
    )
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"[Model] Loaded weights from {checkpoint_path}")

    if device:
        model.to(device)
    return model
