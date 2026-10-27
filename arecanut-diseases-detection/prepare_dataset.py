"""Stratified 80/10/10 Dataset Splitter for Arecanut Plant Disease Classification Dataset.
Consolidates source images from dataset/train and dataset/test into clean human-readable
class folders under data/train, data/val, and data/test.
"""

import os
import shutil
from typing import Dict, List
import numpy as np
from sklearn.model_selection import train_test_split

CLASS_MAPPING: Dict[int, str] = {
    0: "bud_borer",
    1: "healthy_foot",
    2: "healthy_leaf",
    3: "healthy_nut",
    4: "healthy_trunk",
    5: "male_koleroga",
    6: "stem_cracking",
    7: "stem_bleeding",
    8: "yellow_leaf",
}


def prepare_stratified_dataset(
    src_dir: str = "dataset",
    dest_dir: str = "data",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_seed: int = 42,
):
    np.random.seed(random_seed)
    print(f"[Dataset Prep] Reading raw dataset from '{src_dir}'...")

    # Collect all image paths and labels across train & test subfolders
    all_filepaths: List[str] = []
    all_labels: List[int] = []

    for sub in ["train", "test"]:
        sub_path = os.path.join(src_dir, sub)
        if not os.path.exists(sub_path):
            continue
        for class_id_str in sorted(os.listdir(sub_path)):
            if not class_id_str.isdigit():
                continue
            class_id = int(class_id_str)
            if class_id not in CLASS_MAPPING:
                continue

            folder = os.path.join(sub_path, class_id_str)
            files = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]
            all_filepaths.extend(files)
            all_labels.extend([class_id] * len(files))

    print(f"[Dataset Prep] Collected {len(all_filepaths)} total images across {len(CLASS_MAPPING)} classes.")

    # Stratified split: First 80% train, 20% temp (which will be split 50/50 into val and test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_filepaths,
        all_labels,
        test_size=(val_ratio + test_ratio),
        stratify=all_labels,
        random_state=random_seed,
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.5,
        stratify=temp_labels,
        random_state=random_seed,
    )

    splits = {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }

    # Clean existing destination if needed
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)

    for split_name, (paths, labels) in splits.items():
        print(f"\n--- Populating {split_name.upper()} split ({len(paths)} images) ---")
        for p, lbl in zip(paths, labels):
            c_name = CLASS_MAPPING[lbl]
            target_folder = os.path.join(dest_dir, split_name, c_name)
            os.makedirs(target_folder, exist_ok=True)
            dest_file = os.path.join(target_folder, os.path.basename(p))
            shutil.copy2(p, dest_file)

        # Print per-class summary
        for lbl, c_name in CLASS_MAPPING.items():
            c_folder = os.path.join(dest_dir, split_name, c_name)
            count = len(os.listdir(c_folder)) if os.path.exists(c_folder) else 0
            print(f"  [{c_name:<16}]: {count} images")

    print(f"\n[Dataset Prep] Successfully prepared stratified 80/10/10 dataset in '{dest_dir}'!")


if __name__ == "__main__":
    prepare_stratified_dataset()
