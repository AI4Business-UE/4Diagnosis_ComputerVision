"""
src/evaluation.py — Ground Truth Evaluation and Supervised Metrics
================================================================

Provides rigorous, objective evaluation of slice grouping and tissue matching
against ground-truth annotations across a dataset.

Supported Metrics:
- Adjusted Rand Index (ARI): Standard clustering metric (1.0 = perfect, 0.0 = random).
- Normalized Mutual Information (NMI).
- Slice Set Purity: Fraction of ground truth slices perfectly reconstructed.
- Core Alignment Accuracy: Accuracy of matching corresponding tissue cores across slices.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score

_logger = logging.getLogger("SliceGrouper.evaluation")


@dataclass
class SlideEvaluationResult:
    """Evaluation result for a single slide."""
    slide_name: str
    sample_id: str
    n_components: int
    n_true_slices: int
    n_pred_slices: int
    n_true_tissues: int
    n_pred_tissues: int
    slice_ari: float
    slice_nmi: float
    slice_v_measure: float
    tissue_ari: float
    tissue_nmi: float
    perfect_slice_ratio: float
    details: Dict[str, Any] = field(default_factory=dict)


def load_ground_truth(slide_dir_or_file: Path) -> Optional[Dict[str, Any]]:
    """
    Load ground truth labels for a slide.

    Looks for 'labels.json' or 'ground_truth.json' in the slide directory,
    or accepts direct path to JSON file.
    """
    path = Path(slide_dir_or_file)
    if path.is_dir():
        for candidate in ["labels.json", "ground_truth.json"]:
            cand_path = path / candidate
            if cand_path.exists():
                path = cand_path
                break
        else:
            return None

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as exc:
        _logger.warning(f"Failed to read ground truth from {path}: {exc}")
        return None


def evaluate_slice_grouping(
    pred_slice_labels: np.ndarray,
    pred_tissue_labels: np.ndarray,
    ground_truth: Dict[str, Any],
    slide_name: str = "",
    sample_id: str = "",
) -> SlideEvaluationResult:
    """
    Evaluate predicted slice and tissue labels against ground truth.

    Parameters
    ----------
    pred_slice_labels : np.ndarray, shape (N,)
        Predicted slice index per component.
    pred_tissue_labels : np.ndarray, shape (N,)
        Predicted tissue core type per component.
    ground_truth : dict
        Ground truth dict containing 'components': [{'id': ..., 'slice_id': ..., 'tissue_id': ...}, ...]
    """
    gt_comps = ground_truth.get("components", [])
    if not gt_comps:
        raise ValueError("Ground truth contains no component annotations.")

    gt_by_id = {c["id"]: c for c in gt_comps}
    N = len(pred_slice_labels)

    true_slices = []
    true_tissues = []
    valid_mask = []

    for comp_id in range(N):
        if comp_id in gt_by_id:
            c = gt_by_id[comp_id]
            true_slices.append(c["slice_id"])
            true_tissues.append(c.get("tissue_id", 0))
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    valid_mask = np.array(valid_mask, dtype=bool)
    if not np.any(valid_mask):
        raise ValueError("No matching component IDs between predictions and ground truth.")

    y_true_slices = np.array(true_slices)
    y_pred_slices = pred_slice_labels[valid_mask]

    y_true_tissues = np.array(true_tissues)
    y_pred_tissues = pred_tissue_labels[valid_mask]

    # Compute clustering similarity metrics
    slice_ari = float(adjusted_rand_score(y_true_slices, y_pred_slices))
    slice_nmi = float(normalized_mutual_info_score(y_true_slices, y_pred_slices))
    slice_v = float(v_measure_score(y_true_slices, y_pred_slices))

    tissue_ari = float(adjusted_rand_score(y_true_tissues, y_pred_tissues))
    tissue_nmi = float(normalized_mutual_info_score(y_true_tissues, y_pred_tissues))

    # Compute fraction of perfectly reconstructed slices
    unique_gt_slices = np.unique(y_true_slices)
    perfect_count = 0
    for gt_s in unique_gt_slices:
        members_mask = (y_true_slices == gt_s)
        pred_assignments = y_pred_slices[members_mask]
        # Perfect if all members got the same predicted slice, and no extra items got that prediction
        if len(set(pred_assignments)) == 1:
            pred_s = pred_assignments[0]
            total_with_pred = np.sum(y_pred_slices == pred_s)
            if total_with_pred == np.sum(members_mask):
                perfect_count += 1

    perfect_ratio = perfect_count / max(1, len(unique_gt_slices))

    return SlideEvaluationResult(
        slide_name=slide_name or ground_truth.get("slide_name", "unknown"),
        sample_id=sample_id or ground_truth.get("sample_id", "unknown"),
        n_components=int(valid_mask.sum()),
        n_true_slices=len(unique_gt_slices),
        n_pred_slices=len(set(y_pred_slices[y_pred_slices >= 0])),
        n_true_tissues=len(set(y_true_tissues)),
        n_pred_tissues=len(set(y_pred_tissues[y_pred_tissues >= 0])),
        slice_ari=slice_ari,
        slice_nmi=slice_nmi,
        slice_v_measure=slice_v,
        tissue_ari=tissue_ari,
        tissue_nmi=tissue_nmi,
        perfect_slice_ratio=perfect_ratio,
        details={
            "perfect_slices": perfect_count,
            "total_gt_slices": len(unique_gt_slices),
        }
    )


def evaluate_dataset(
    results_list: List[SlideEvaluationResult],
) -> pd.DataFrame:
    """Format a list of slide evaluation results into a clean benchmark table."""
    records = []
    for r in results_list:
        records.append({
            "Slide": r.slide_name,
            "Components": r.n_components,
            "True Slices": r.n_true_slices,
            "Pred Slices": r.n_pred_slices,
            "Slice ARI": round(r.slice_ari, 3),
            "Slice NMI": round(r.slice_nmi, 3),
            "Tissue ARI": round(r.tissue_ari, 3),
            "Perfect Slices (%)": round(r.perfect_slice_ratio * 100, 1),
        })

    df = pd.DataFrame(records)
    if not df.empty:
        summary_row = {
            "Slide": "MEAN (Dataset)",
            "Components": int(df["Components"].mean()),
            "True Slices": round(df["True Slices"].mean(), 1),
            "Pred Slices": round(df["Pred Slices"].mean(), 1),
            "Slice ARI": round(df["Slice ARI"].mean(), 3),
            "Slice NMI": round(df["Slice NMI"].mean(), 3),
            "Tissue ARI": round(df["Tissue ARI"].mean(), 3),
            "Perfect Slices (%)": round(df["Perfect Slices (%)"].mean(), 1),
        }
        df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

    return df
