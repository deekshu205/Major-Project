"""Smart Areca Nut Crop Advisor - Core Module
Upgraded Computer Vision & Multimodal Architecture for Arecanut Leaf Disease Detection
"""

from .model import ArecaEfficientNetB4, build_model
from .dataset import create_dataloaders, get_transforms
from .gradcam import GradCAM, generate_gradcam_visualization
from .fusion import MultimodalFusionArbiter, TabularFeatureSchema
from .metrics import evaluate_model, plot_confusion_matrix

__all__ = [
    "ArecaEfficientNetB4",
    "build_model",
    "create_dataloaders",
    "get_transforms",
    "GradCAM",
    "generate_gradcam_visualization",
    "MultimodalFusionArbiter",
    "TabularFeatureSchema",
    "evaluate_model",
    "plot_confusion_matrix",
]
