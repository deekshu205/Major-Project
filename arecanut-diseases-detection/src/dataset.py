"""Modernized Dataset Loading and Background-Invariant Augmentation Pipeline.
Enforces true plant tissue and anatomical feature learning through RandomResizedCrop
and RandomErasing to suppress background shortcut learning (e.g. skies, wires, fences).
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms


import cv2

DEFAULT_IMG_SIZE = 380
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def autocrop_borders(image: Union[Image.Image, np.ndarray, str], tolerance: int = 20) -> Image.Image:
    """Removes solid black, white, or uniform letterbox borders from an image.
    Ensures convolutional filters focus on plant anatomy rather than artificial canvas margins.
    """
    if isinstance(image, str):
        pil_img = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            pil_img = Image.fromarray(image).convert("RGB")
        elif image.shape[2] == 3:
            pil_img = Image.fromarray(image).convert("RGB")
        else:
            pil_img = Image.fromarray(image[..., :3]).convert("RGB")
    elif isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    np_img = np.array(pil_img)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

    # Content mask excluding solid black (<= tolerance) and solid white (>= 255 - tolerance)
    content_mask = (gray > tolerance) & (gray < (255 - tolerance))
    coords = cv2.findNonZero(content_mask.astype(np.uint8))
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        if w >= 0.15 * np_img.shape[1] and h >= 0.15 * np_img.shape[0]:
            return pil_img.crop((x, y, x + w, y + h))

    return pil_img


def get_transforms(
    img_size: int = DEFAULT_IMG_SIZE,
    is_train: bool = True,
) -> transforms.Compose:
    """Returns background-invariant data augmentation and normalization transforms.
    
    Training transforms include:
      - RandomResizedCrop(scale=(0.5, 1.0)): Forces the model to look at zoomed-in plant tissue
      - RandomHorizontalFlip(p=0.5)
      - RandomRotation(degrees=15)
      - ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
      - ToTensor()
      - Normalize(IMAGENET_MEAN, IMAGENET_STD)
      - RandomErasing(p=0.3, scale=(0.02, 0.25), value='random'): Prevents corner shortcut memorization
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(size=img_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.25), value="random"),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class TransformedSubset(Dataset):
    """Wraps a Subset with a specific torchvision transform."""

    def __init__(self, subset: Subset, transform: Optional[transforms.Compose] = None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.subset.dataset[self.subset.indices[idx]]
        if isinstance(img, torch.Tensor):
            return img, label
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self) -> int:
        return len(self.subset)


def create_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    img_size: int = DEFAULT_IMG_SIZE,
    num_workers: int = 0,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Creates train, validation, and test DataLoaders with background-invariant augmentations."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory '{data_dir}' does not exist.")

    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    has_explicit_splits = ("train" in subdirs) and ("val" in subdirs or "test" in subdirs)

    train_transform = get_transforms(img_size=img_size, is_train=True)
    val_transform = get_transforms(img_size=img_size, is_train=False)

    if has_explicit_splits:
        train_path = os.path.join(data_dir, "train")
        val_name = "val" if "val" in subdirs else "validation"
        val_path = os.path.join(data_dir, val_name) if val_name in subdirs else None
        test_path = os.path.join(data_dir, "test") if "test" in subdirs else None

        train_ds = datasets.ImageFolder(train_path, transform=train_transform)
        class_names = train_ds.classes

        if val_path and os.path.exists(val_path):
            val_ds = datasets.ImageFolder(val_path, transform=val_transform)
        else:
            val_ds = train_ds

        if test_path and os.path.exists(test_path):
            test_ds = datasets.ImageFolder(test_path, transform=val_transform)
        else:
            test_ds = val_ds
    else:
        raw_dataset = datasets.ImageFolder(data_dir, transform=None)
        class_names = raw_dataset.classes
        total_len = len(raw_dataset)

        generator = torch.Generator().manual_seed(seed)
        n_train = int(train_split * total_len)
        n_val = int(val_split * total_len)
        n_test = total_len - n_train - n_val

        train_sub, val_sub, test_sub = random_split(
            raw_dataset, [n_train, n_val, n_test], generator=generator
        )

        train_ds = TransformedSubset(train_sub, transform=train_transform)
        val_ds = TransformedSubset(val_sub, transform=val_transform)
        test_ds = TransformedSubset(test_sub, transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f"[Dataset] Background-invariant loaders initialized. Classes: {class_names} | "
        f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}"
    )

    return train_loader, val_loader, test_loader, class_names
