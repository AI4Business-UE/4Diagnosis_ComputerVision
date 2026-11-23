"""
src/train_models.py — Train & Calibrate SliceGrouper Machine Learning Models
============================================================================

Trains and evaluates:
1. Pairwise Tissue Matcher: P(same_tissue_core | component_i, component_j)
2. Safety Fallback Classifier: P(fallback_to_single_slice | slide_features)

Includes strict GroupKFold cross-validation (zero slide data leakage),
probability calibration, feature importance extraction, and immediate
serialization of .joblib and .json artifacts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold

_logger = logging.getLogger("SliceGrouper.train")


def train_pairwise_matcher(
    pairwise_data: Dict[str, Any],
    models_dir: Path,
    figures_dir: Path,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """
    Trains and cross-validates a calibrated Pairwise Tissue Matcher using GroupKFold.
    Saves model and diagnostic plots immediately to disk.
    """
    X = pairwise_data["X"]
    y = pairwise_data["y_same_tissue"]
    groups = pairwise_data["groups"]
    feat_names = pairwise_data["feature_names"]

    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if len(X) == 0:
        _logger.warning("No pairwise data available to train pairwise matcher.")
        return {}

    n_unique_groups = len(np.unique(groups))
    effective_splits = min(n_splits, max(2, n_unique_groups))

    gkf = GroupKFold(n_splits=effective_splits)

    cv_rocs = []
    cv_prs = []
    cv_f1s = []
    cv_accs = []
    all_y_true = []
    all_y_prob = []

    _logger.info(f"Cross-validating Pairwise Matcher ({len(X)} pairs across {n_unique_groups} slides)...")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Check class presence
        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            continue

        base_clf = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            class_weight="balanced",
            random_state=42 + fold,
        )
        base_clf.fit(X_train, y_train)
        probs = base_clf.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.5).astype(int)

        roc = roc_auc_score(y_val, probs)
        pr = average_precision_score(y_val, probs)
        f1 = f1_score(y_val, preds, zero_division=0)
        acc = accuracy_score(y_val, preds)

        cv_rocs.append(roc)
        cv_prs.append(pr)
        cv_f1s.append(f1)
        cv_accs.append(acc)

        all_y_true.extend(y_val)
        all_y_prob.extend(probs)

    # Train final full model with probability calibration
    final_base = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        max_leaf_nodes=31,
        class_weight="balanced",
        random_state=42,
    )
    final_base.fit(X, y)
    calibrated_matcher = CalibratedClassifierCV(estimator=final_base, cv="prefit", method="isotonic")
    calibrated_matcher.fit(X, y)

    # Compute Feature Importances using a fast Random Forest surrogate
    rf_surrogate = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=42)
    rf_surrogate.fit(X, y)
    importances = rf_surrogate.feature_importances_

    # Save model and artifacts immediately
    model_path = models_dir / "pairwise_tissue_matcher.joblib"
    joblib.dump(calibrated_matcher, model_path)

    metrics = {
        "n_samples": int(len(X)),
        "n_slides": int(n_unique_groups),
        "mean_roc_auc": float(np.mean(cv_rocs)) if cv_rocs else 0.0,
        "std_roc_auc": float(np.std(cv_rocs)) if cv_rocs else 0.0,
        "mean_pr_auc": float(np.mean(cv_prs)) if cv_prs else 0.0,
        "mean_f1": float(np.mean(cv_f1s)) if cv_f1s else 0.0,
        "mean_accuracy": float(np.mean(cv_accs)) if cv_accs else 0.0,
        "feature_names": feat_names,
        "feature_importances": {feat_names[i]: float(importances[i]) for i in np.argsort(-importances)},
    }

    with open(models_dir / "pairwise_matcher_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Plot ROC & PR curves
    if all_y_true:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
        axes[0].plot(fpr, tpr, color="#2ecc71", lw=2.5, label=f"ROC-AUC = {metrics['mean_roc_auc']:.3f}")
        axes[0].plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6)
        axes[0].set_title("Pairwise Tissue Matcher: ROC Curve", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        precision, recall, _ = precision_recall_curve(all_y_true, all_y_prob)
        axes[1].plot(recall, precision, color="#3498db", lw=2.5, label=f"PR-AUC = {metrics['mean_pr_auc']:.3f}")
        axes[1].set_title("Precision-Recall Curve", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        fig.savefig(figures_dir / "pairwise_roc_pr.png", dpi=160)
        plt.close(fig)

    # Plot Feature Importances
    sorted_idx = np.argsort(importances)[-12:]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(sorted_idx)), importances[sorted_idx], color="#9b59b6", edgecolor="black", alpha=0.85)
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feat_names[i] for i in sorted_idx], fontsize=9)
    ax.set_title("Top Pairwise Matcher Feature Importances (Gini)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Importance Weight")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(figures_dir / "pairwise_feature_importance.png", dpi=160)
    plt.close(fig)

    _logger.info(f"Pairwise Matcher trained & saved: ROC-AUC={metrics['mean_roc_auc']:.3f}, PR-AUC={metrics['mean_pr_auc']:.3f}")
    return metrics


def train_fallback_classifier(
    pairwise_data: Dict[str, Any],
    pairwise_matcher_model: Any,
    models_dir: Path,
    figures_dir: Path,
) -> Dict[str, Any]:
    """
    Trains a slide-level Safety Fallback Classifier:
    Predicts whether a slide should fallback to full-slide (no slicing)
    or be split into serial section slices.
    """
    slide_records = pairwise_data.get("slide_records", [])
    if not slide_records:
        return {}

    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    X_slide = []
    y_slide = []
    slide_names = []

    for rec in slide_records:
        n_comps = rec["n_components"]
        areas = np.array(rec["areas"]) if rec["areas"] else np.array([1.0])
        cents = np.array(rec["centroids"]) if rec["centroids"] else np.zeros((1, 2))
        bboxes = np.array(rec.get("bboxes", [])) if rec.get("bboxes") else np.zeros((len(areas), 4))
        c_feats = rec.get("color_feats")

        # Morphology distributions
        area_cv = float(np.std(areas) / (np.mean(areas) + 1e-5))
        area_min_max = float(np.min(areas) / (np.max(areas) + 1e-5))

        # Spatial regularity & Inter-cluster gap
        if len(cents) >= 2:
            from scipy.spatial.distance import pdist, squareform
            dists = pdist(cents)
            dist_mean = float(np.mean(dists))
            dist_std = float(np.std(dists))
            dist_cv = float(dist_std / (dist_mean + 1e-5))

            dx_span = np.ptp(cents[:, 0])
            dy_span = np.ptp(cents[:, 1])
            axis = 0 if dx_span >= dy_span else 1
            order = np.argsort(cents[:, axis])
            sorted_cents = cents[order, axis]
            gaps = np.diff(sorted_cents)
            max_gap = float(np.max(gaps))
            med_dim = float(np.median([bboxes[i, 2] if axis == 0 else bboxes[i, 3] for i in range(len(bboxes))])) if len(bboxes) == len(areas) else 100.0
            gap_ratio = max_gap / (med_dim + 1e-5)

            if c_feats is not None and len(c_feats) >= 2:
                sim_m = 1.0 - squareform(pdist(c_feats, metric="cosine"))
                np.fill_diagonal(sim_m, 0.0)
                max_twin_sim = float(np.max(sim_m))
            else:
                max_twin_sim = 0.0
        else:
            dist_cv = 0.0
            gap_ratio = 0.0
            max_twin_sim = 0.0

        # Ground truth: 1 = fallback to single full slide, 0 = multi-slice serial
        is_fallback_gt = 1 if rec["is_single_slice"] else 0

        feat_row = [
            float(n_comps),
            area_cv,
            area_min_max,
            gap_ratio,
            dist_cv,
            max_twin_sim,
            1.0 if n_comps <= 1 else 0.0,
            1.0 if n_comps in (2, 3) else 0.0,
        ]

        X_slide.append(feat_row)
        y_slide.append(is_fallback_gt)
        slide_names.append(rec["slide_name"])

    X_arr = np.array(X_slide, dtype=float)
    y_arr = np.array(y_slide, dtype=int)

    feature_names = [
        "n_components",
        "area_cv",
        "area_min_max_ratio",
        "gap_ratio",
        "distance_cv",
        "max_twin_sim",
        "is_single_comp",
        "is_small_set",
    ]

    # Cross-validation with high safety penalty for false splitting
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_rocs = []
    cv_accs = []
    cv_sens = []
    cv_spec = []
    all_y_true = []
    all_y_prob = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
        X_train, y_train = X_arr[train_idx], y_arr[train_idx]
        X_val, y_val = X_arr[val_idx], y_arr[val_idx]

        clf = LogisticRegression(class_weight={0: 1.0, 1: 3.5}, C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.5).astype(int)

        roc = roc_auc_score(y_val, probs) if len(np.unique(y_val)) > 1 else 0.5
        acc = accuracy_score(y_val, preds)
        cm = confusion_matrix(y_val, preds, labels=[0, 1])

        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        cv_rocs.append(roc)
        cv_accs.append(acc)
        cv_sens.append(sens)
        cv_spec.append(spec)

        all_y_true.extend(y_val)
        all_y_prob.extend(probs)

    # Train final model
    final_fallback_clf = LogisticRegression(class_weight={0: 1.0, 1: 3.5}, C=1.0, max_iter=1000, random_state=42)
    final_fallback_clf.fit(X_arr, y_arr)

    model_path = models_dir / "fallback_classifier.joblib"
    joblib.dump(final_fallback_clf, model_path)

    metrics = {
        "n_slides": int(len(X_arr)),
        "n_single_slice_fallback": int(np.sum(y_arr == 1)),
        "n_multi_slice": int(np.sum(y_arr == 0)),
        "mean_roc_auc": float(np.mean(cv_rocs)),
        "mean_accuracy": float(np.mean(cv_accs)),
        "mean_sensitivity": float(np.mean(cv_sens)),
        "mean_specificity": float(np.mean(cv_spec)),
        "feature_names": feature_names,
        "feature_coefficients": {feature_names[i]: float(final_fallback_clf.coef_[0, i]) for i in range(len(feature_names))},
    }

    with open(models_dir / "fallback_classifier_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Diagnostic plots
    if all_y_true:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
        axes[0].plot(fpr, tpr, color="#e67e22", lw=2.5, label=f"ROC-AUC = {metrics['mean_roc_auc']:.3f}")
        axes[0].plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6)
        axes[0].set_title("Safety Fallback Classifier: ROC", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("False Fallback Rate")
        axes[0].set_ylabel("True Fallback Rate")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        cm = confusion_matrix(all_y_true, (np.array(all_y_prob) >= 0.5).astype(int))
        import itertools
        axes[1].imshow(cm, interpolation="nearest", cmap=plt.cm.Oranges)
        axes[1].set_title("Fallback Confusion Matrix (CV)", fontsize=11, fontweight="bold")
        tick_marks = np.arange(2)
        axes[1].set_xticks(tick_marks)
        axes[1].set_yticks(tick_marks)
        axes[1].set_xticklabels(["Multi-slice", "Fallback (Full)"])
        axes[1].set_yticklabels(["Multi-slice", "Fallback (Full)"])

        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            axes[1].text(j, i, f"{cm[i, j]:,}", horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontweight="bold")

        axes[1].set_ylabel("Ground Truth")
        axes[1].set_xlabel("Predicted")

        plt.tight_layout()
        fig.savefig(figures_dir / "fallback_roc_and_confusion.png", dpi=160)
        plt.close(fig)

    _logger.info(f"Fallback Classifier trained & saved: Acc={metrics['mean_accuracy']:.3f}, ROC={metrics['mean_roc_auc']:.3f}")
    return metrics
