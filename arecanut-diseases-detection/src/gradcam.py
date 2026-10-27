"""Explainable AI (XAI) Grad-CAM Module for EfficientNet-B4.
Generates class-discriminative localization maps highlighting disease lesions.
"""

import os
from typing import Optional, Tuple, Union
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset import autocrop_borders


class GradCAM:
    """Computes Grad-CAM visualizations on the final convolutional layer of a model."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        # If target layer is not explicitly provided, try helper method
        if target_layer is None:
            if hasattr(model, "get_target_layer_for_gradcam"):
                self.target_layer = model.get_target_layer_for_gradcam()
            elif hasattr(model, "backbone") and hasattr(model.backbone, "conv_head"):
                self.target_layer = model.backbone.conv_head
            elif hasattr(model, "conv_head"):
                self.target_layer = model.conv_head
            else:
                raise ValueError("Could not automatically locate final conv layer. Please pass target_layer.")
        else:
            self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """Generates normalized [0, 1] Grad-CAM heatmap for the given image tensor.

        Args:
            input_tensor: (1, 3, H, W) normalized torch tensor with requires_grad=True.
            target_class: Target class index. If None, uses top predicted class.

        Returns:
            cam_map: (H, W) float32 numpy array normalized to [0, 1]
            predicted_class_idx: int
            confidence: float (softmax probability 0.0 - 1.0)
        """
        self.model.zero_grad()
        if not input_tensor.requires_grad:
            input_tensor = input_tensor.clone().detach().requires_grad_(True)

        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=-1)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=-1).item())

        confidence = float(probs[0, target_class].item())

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not captured. Check target_layer hooks.")

        # Global average pooling of gradients: weights alpha_k
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H_feat, W_feat)
        cam = F.relu(cam)  # Only features with positive influence on class

        # Upscale CAM to original input resolution
        _, _, H, W = input_tensor.shape
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32), target_class, confidence


def apply_colormap_on_image(
    org_img: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.55,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Applies a colormap to CAM and overlays it on the original RGB image.

    Args:
        org_img: (H, W, 3) uint8 RGB image
        cam: (H, W) float32 array in [0, 1]
        alpha: Weight for CAM overlay
        colormap: OpenCV colormap constant
    """
    heatmap_uint8 = np.uint8(255 * cam)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = alpha * heatmap_colored.astype(np.float32) + (1.0 - alpha) * org_img.astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def generate_gradcam_visualization(
    model: nn.Module,
    image_path: str,
    output_path: Optional[str] = None,
    img_size: int = 380,
    class_names: Optional[list] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Takes an image file path, computes Grad-CAM, and saves a side-by-side diagnostic figure.

    Returns:
        Dictionary containing prediction results, confidence, and output file path.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load and preprocess image with letterbox border removal
    pil_img = Image.open(image_path).convert("RGB")
    pil_img = autocrop_borders(pil_img)
    resized_pil = pil_img.resize((img_size, img_size))
    rgb_np = np.array(resized_pil, dtype=np.uint8)

    # Normalized tensor for model input
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm_np = ((rgb_np / 255.0) - mean) / std
    norm_tensor = torch.from_numpy(norm_np.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    # Compute Grad-CAM
    gradcam = GradCAM(model)
    cam, pred_idx, confidence = gradcam.generate_cam(norm_tensor)

    pred_name = class_names[pred_idx] if class_names and pred_idx < len(class_names) else f"Class_{pred_idx}"
    overlay_img = apply_colormap_on_image(rgb_np, cam, alpha=0.55)

    # Create diagnostic side-by-side visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), dpi=150)
    fig.patch.set_facecolor("#0F172A")

    titles = [
        "Original Specimen",
        "Grad-CAM Heatmap (EfficientNet-B0)",
        f"Diagnostic Overlay ({pred_name}: {confidence * 100:.1f}%)",
    ]
    images = [rgb_np, cam, overlay_img]

    for ax, img, title in zip(axes, images, titles):
        ax.set_facecolor("#0F172A")
        if img.ndim == 2:
            im = ax.imshow(img, cmap="jet")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.imshow(img)
        ax.set_title(title, color="#E2E8F0", fontsize=12, pad=10, fontweight="bold")
        ax.axis("off")

    plt.suptitle(
        f"Smart Areca Nut Advisor - Explainability Audit | Prediction: {pred_name} (Confidence: {confidence * 100:.2f}%)",
        color="#38BDF8",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    if output_path is None:
        base_name = image_path.rsplit(".", 1)[0]
        output_path = f"{base_name}_gradcam.png"

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    return {
        "predicted_class": pred_name,
        "class_index": pred_idx,
        "confidence": confidence,
        "gradcam_artifact": output_path,
        "cam_array": cam,
    }
