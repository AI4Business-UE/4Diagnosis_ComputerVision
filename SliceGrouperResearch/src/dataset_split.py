"""
src/dataset_split.py — Reproducible Stratified Train/Test Dataset Partitioning
=============================================================================

Splits the 108 labeled slides into:
- Train Set: ~80% (86 slides) — used for training models, tuning thresholds, and synthetic augmentation.
- Test Set:  ~20% (22 slides) — FROZEN test set, strictly unseen during training and tuning.

Stratification is performed across the number of physical slices (1, 2, 3, 4 slices)
to ensure representative distribution in both partitions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.labeler import SlideLabelerWidget

_logger = logging.getLogger("SliceGrouper.split")

SPLITS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def create_stratified_split(
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates and saves a stratified Train/Test split across the labeled dataset.
    """
    labeler = SlideLabelerWidget()
    labeled_slides = [s for s in labeler.slides if s["labels_path"].exists()]

    records = []
    for s in labeled_slides:
        with open(s["labels_path"], "r", encoding="utf-8") as f:
            data = json.load(f)
        comps = data.get("components", [])
        slices = set(c.get("slice_id", 0) for c in comps)
        n_slices = max(1, len(slices))
        
        records.append({
            "slide_name": s["slide_name"],
            "sample_id": s["sample_id"],
            "n_components": len(comps),
            "n_slices": n_slices,
            "tiff_path": str(s["tiff_path"]),
            "mask_path": str(s["mask_path"]),
            "labels_path": str(s["labels_path"]),
        })

    df = pd.DataFrame(records).sort_values("slide_name").reset_index(drop=True)

    # For stratification: bin 3 and 4 together so every class has >= 2 samples
    strat_labels = df["n_slices"].clip(upper=3)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_labels,
    )

    # Ensure the single 4-slice slide is placed in Train so models can learn 4-slice patterns
    four_slice_test = test_df[test_df["n_slices"] == 4]
    if len(four_slice_test) > 0:
        # Swap with a 3-slice slide from train
        three_slice_train = train_df[train_df["n_slices"] == 3].iloc[:1]
        train_df = pd.concat([train_df.drop(three_slice_train.index), four_slice_test])
        test_df = pd.concat([test_df.drop(four_slice_test.index), three_slice_train])

    train_df = train_df.sort_values("slide_name").reset_index(drop=True)
    test_df = test_df.sort_values("slide_name").reset_index(drop=True)

    # Save to disk
    train_csv = SPLITS_DIR / "train_slides.csv"
    test_csv = SPLITS_DIR / "test_slides.csv"
    json_path = SPLITS_DIR / "train_test_split.json"

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    split_manifest = {
        "metadata": {
            "total_slides": len(df),
            "train_count": len(train_df),
            "test_count": len(test_df),
            "test_ratio": test_size,
            "random_state": random_state,
        },
        "train_slides": train_df["slide_name"].tolist(),
        "test_slides": test_df["slide_name"].tolist(),
        "train_distribution": train_df["n_slices"].value_counts().to_dict(),
        "test_distribution": test_df["n_slices"].value_counts().to_dict(),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)

    print(f"Stratified dataset split created successfully!")
    print(f"Train set: {len(train_df)} slides -> {train_csv}")
    print(f"Test set:  {len(test_df)} slides (FROZEN) -> {test_csv}")
    print("\nTrain slice distribution:")
    print(train_df["n_slices"].value_counts().sort_index())
    print("\nTest slice distribution:")
    print(test_df["n_slices"].value_counts().sort_index())

    return train_df, test_df


def load_dataset_split() -> Tuple[List[str], List[str]]:
    """Loads existing train/test split slide names."""
    json_path = SPLITS_DIR / "train_test_split.json"
    if not json_path.exists():
        create_stratified_split()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["train_slides"], data["test_slides"]


if __name__ == "__main__":
    create_stratified_split()
