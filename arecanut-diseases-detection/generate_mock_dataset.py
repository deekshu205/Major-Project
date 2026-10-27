"""Utility to generate a representative mock dataset and test image for Arecanut leaf diseases.
Creates realistic synthetic leaf patterns for:
  - Bud_Rot
  - Healthy
  - Koleroga_Mahali
  - Yellow_Leaf_Spot
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def create_leaf_image(disease_type: str, size: int = 380) -> Image.Image:
    """Generates a synthetic leaf image with characteristic disease visual patterns."""
    img = Image.new("RGB", (size, size), color=(20, 25, 20))
    draw = ImageDraw.Draw(img)

    # Base leaf shape (green elliptical arecanut frond)
    leaf_color = (45, 125, 45) if disease_type != "Yellow_Leaf_Spot" else (160, 150, 40)
    margin = 40
    draw.polygon(
        [
            (size // 2, margin),
            (size - margin, size // 2),
            (size // 2, size - margin),
            (margin, size // 2),
        ],
        fill=leaf_color,
        outline=(25, 75, 25),
    )

    # Central rachis/vein
    draw.line([(size // 2, margin), (size // 2, size - margin)], fill=(30, 90, 30), width=4)

    # Lateral pinnae veins
    for y in range(margin + 20, size - margin - 20, 30):
        draw.line([(size // 2, y), (size - margin - 20, y + 15)], fill=(35, 100, 35), width=2)
        draw.line([(size // 2, y), (margin + 20, y + 15)], fill=(35, 100, 35), width=2)

    # Disease specific lesion patterns
    if disease_type == "Koleroga_Mahali":
        # Water-soaked dark necrotic rotting patches with irregular borders
        for center_x, center_y, radius in [(size // 2, size // 2, 45), (size // 2 + 30, size // 2 - 40, 30)]:
            draw.ellipse(
                [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
                fill=(70, 45, 25),
                outline=(40, 25, 15),
            )
            # Fungal felt/whitish mycelial growth at center
            draw.ellipse(
                [center_x - 12, center_y - 12, center_x + 12, center_y + 12],
                fill=(210, 210, 200),
            )

    elif disease_type == "Yellow_Leaf_Spot":
        # Yellow chlorotic halos with central dark brown necrotic specks
        np.random.seed(42)
        for _ in range(8):
            sx = np.random.randint(margin + 30, size - margin - 30)
            sy = np.random.randint(margin + 30, size - margin - 30)
            draw.ellipse([sx - 18, sy - 18, sx + 18, sy + 18], fill=(225, 205, 30))
            draw.ellipse([sx - 6, sy - 6, sx + 6, sy + 6], fill=(90, 40, 20))

    elif disease_type == "Bud_Rot":
        # Dark brown to black rot spreading down spindle leaves
        draw.polygon(
            [
                (size // 2, margin),
                (size // 2 + 35, margin + 120),
                (size // 2 - 35, margin + 120),
            ],
            fill=(30, 20, 15),
            outline=(15, 10, 5),
        )

    # Add realistic texture noise and subtle blur
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    return img


def generate_mock_dataset(base_dir: str = "sample_dataset", samples_per_class: int = 5):
    """Generates a small directory structure with sample images for pipeline testing."""
    classes = ["Bud_Rot", "Healthy", "Koleroga_Mahali", "Yellow_Leaf_Spot"]
    for split in ["train", "val", "test"]:
        n_samples = samples_per_class if split == "train" else max(2, samples_per_class // 2)
        for cls in classes:
            dir_path = os.path.join(base_dir, split, cls)
            os.makedirs(dir_path, exist_ok=True)
            for i in range(n_samples):
                img = create_leaf_image(cls)
                img.save(os.path.join(dir_path, f"{cls}_{i:03d}.jpg"))

    print(f"[Dataset Generator] Created sample dataset in '{base_dir}' with classes: {classes}")


if __name__ == "__main__":
    generate_mock_dataset()
    # Also save one standalone test image
    test_img = create_leaf_image("Koleroga_Mahali")
    test_img.save("sample_test_leaf.jpg")
    print("Saved 'sample_test_leaf.jpg' for single-image diagnosis testing.")
