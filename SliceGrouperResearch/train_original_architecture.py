"""
train_original_architecture.py
==============================
Trains the original (first) architecture strictly on the Train set (86 slides),
with NO synthetic augmentation, and evaluates on the frozen Test set (22 slides).
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
logger = logging.getLogger("TrainOriginal")

def run():
    logger.info("=" * 80)
    logger.info("TRAINING ORIGINAL (FIRST) ARCHITECTURE WITH STRICT TRAIN/TEST SPLIT")
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

    # 2. Extract Features only for Train slides
    features_cache_dir = CHECKPOINTS_DIR / "features"
    logger.info("Extracting features for 86 Train slides...")
    pairwise_data = build_pairwise_dataset(
        slides_info=train_slides,
        min_area_px=6000,
        cache_dir=features_cache_dir,
    )

    # 3. Train Original Models on Train slides only
    pairwise_model_path = MODELS_DIR / "pairwise_tissue_matcher.joblib"
    fallback_model_path = MODELS_DIR / "fallback_classifier.joblib"

    logger.info("Training Pairwise Matcher on Train slides...")
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

    # 4. Benchmark Original Methods on Test Set (22 Slides)
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

    test_records_csv = BENCHMARKS_DIR / "test_benchmark_records.csv"
    df_test_results = benchmark_all_methods(
        slides_info=test_slides,
        methods=methods,
        min_area_px=6000,
        output_csv_path=test_records_csv,
    )
    df_test_summary = generate_method_summary_table(df_test_results)
    df_test_summary.to_csv(BENCHMARKS_DIR / "test_benchmark_summary.csv")
    logger.info("\nTEST SET SUMMARY:")
    logger.info("\n" + str(df_test_summary))

    # Also evaluate on Train Set (86 Slides)
    logger.info("\nEvaluating on TRAIN SET (86 slides)...")
    train_records_csv = BENCHMARKS_DIR / "train_benchmark_records.csv"
    df_train_results = benchmark_all_methods(
        slides_info=train_slides,
        methods=methods,
        min_area_px=6000,
        output_csv_path=train_records_csv,
    )
    df_train_summary = generate_method_summary_table(df_train_results)
    df_train_summary.to_csv(BENCHMARKS_DIR / "train_benchmark_summary.csv")
    logger.info("\nTRAIN SET SUMMARY:")
    logger.info("\n" + str(df_train_summary))

    logger.info("\nDone! Training and evaluation complete.")

if __name__ == "__main__":
    run()
