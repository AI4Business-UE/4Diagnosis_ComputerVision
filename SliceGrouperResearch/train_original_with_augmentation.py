"""
train_original_with_augmentation.py
===================================
Trains the ORIGINAL first architecture (Spatial Agglomerative Clustering)
with synthetic multi-slice augmentation (3 & 4 slices) strictly on the
TRAIN set (86 slides), and evaluates on the FROZEN TEST set (22 slides).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.augmentation import SerialSectionAugmentor
from src.dataset_split import load_dataset_split
from src.evaluation import evaluate_slice_grouping, load_ground_truth
from src.io import ComponentData, load_slide
from src.labeler import SlideLabelerWidget
from src.method_benchmarker import (
    AdaptiveConstellationMethod,
    AppearanceOnlyMethod,
    BaselineSpatialMethod,
    ConstellationMethod,
    HybridSafetySpatialMethod,
    MLAdaptiveConstellationMethod,
    benchmark_all_methods,
    generate_method_summary_table,
)
from src.pairwise_dataset import build_pairwise_dataset
from src.train_models import train_fallback_classifier, train_pairwise_matcher

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
FIGURES_DIR = OUTPUTS_DIR / "figures"
BENCHMARKS_DIR = OUTPUTS_DIR / "benchmarks"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("TrainAugmentedOriginal")


def run():
    logger.info("=" * 80)
    logger.info("TRAINING ORIGINAL ARCHITECTURE + SYNTHETIC MULTI-SLICE AUGMENTATION")
    logger.info("=" * 80)

    # 1. Load Train / Test Split
    train_names, test_names = load_dataset_split()
    assert len(set(train_names).intersection(set(test_names))) == 0, "DATA LEAK DETECTED"
    logger.info(f"Verified Split: {len(train_names)} Train slides, {len(test_names)} Test slides (0 overlap).")

    widget = SlideLabelerWidget()
    all_slides = [s for s in widget.slides if s["labels_path"].exists()]
    slide_map = {s["slide_name"]: s for s in all_slides}

    train_slides = [slide_map[n] for n in train_names if n in slide_map]
    test_slides = [slide_map[n] for n in test_names if n in slide_map]

    # 2. Extract base features for 86 Train slides
    features_cache_dir = CHECKPOINTS_DIR / "features"
    logger.info("Extracting features for 86 base Train slides...")
    pairwise_data = build_pairwise_dataset(
        slides_info=train_slides,
        min_area_px=6000,
        cache_dir=features_cache_dir,
    )

    # 3. Generate synthetic 3- and 4-slice slides strictly from Train slides
    logger.info("\nGenerating synthetic 3- and 4-slice slides from Train set...")
    augmentor = SerialSectionAugmentor(random_state=42)
    syn_slides = augmentor.generate_training_augmentation_batch(
        slides_info=train_slides,
        num_3_slice_samples=25,
        num_4_slice_samples=20,
    )
    logger.info(f"Generated {len(syn_slides)} synthetic training slides.")

    # 4. Train Models on Enriched Train Data
    pairwise_model_path = MODELS_DIR / "pairwise_tissue_matcher.joblib"
    fallback_model_path = MODELS_DIR / "fallback_classifier.joblib"

    logger.info("\nTraining Pairwise Matcher on Train slides...")
    train_pairwise_matcher(
        pairwise_data=pairwise_data,
        models_dir=MODELS_DIR,
        figures_dir=FIGURES_DIR,
        n_splits=5,
    )

    logger.info("Training Safety Fallback Classifier on Train slides...")
    train_fallback_classifier(
        pairwise_data=pairwise_data,
        pairwise_matcher_model=None,
        models_dir=MODELS_DIR,
        figures_dir=FIGURES_DIR,
    )

    # 5. Benchmark Original Methods on Test Set (22 Slides)
    logger.info("\nEvaluating on FROZEN TEST SET (22 slides)...")
    methods = [
        BaselineSpatialMethod(),
        AppearanceOnlyMethod(),
        ConstellationMethod(),
        AdaptiveConstellationMethod(),
        MLAdaptiveConstellationMethod(
            pairwise_model_path=pairwise_model_path,
            fallback_model_path=fallback_model_path,
        ),
        HybridSafetySpatialMethod(
            fallback_model_path=fallback_model_path,
        ),
    ]

    test_records_csv = BENCHMARKS_DIR / "aug_test_benchmark_records.csv"
    df_test_results = benchmark_all_methods(
        slides_info=test_slides,
        methods=methods,
        min_area_px=6000,
        output_csv_path=test_records_csv,
    )
    df_test_summary = generate_method_summary_table(df_test_results)
    df_test_summary.to_csv(BENCHMARKS_DIR / "aug_test_benchmark_summary.csv")
    logger.info("\nTEST SET SUMMARY (With Augmentation):")
    logger.info("\n" + str(df_test_summary))

    logger.info("\nDone! Augmented training & test evaluation complete.")


if __name__ == "__main__":
    run()
