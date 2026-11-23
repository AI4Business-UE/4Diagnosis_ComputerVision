"""
src/pairwise_dataset.py — Feature Extraction and Pairwise Dataset Generation
=============================================================================

Extracts multi-modal features (Color, Texture, Geometry) for all components
across labeled slides and constructs pairwise feature vectors with Ground Truth
labels (same_tissue_core, same_slice).

Includes per-slide disk caching so feature extraction can resume seamlessly
if interrupted.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.color_features import extract_color_features_batch
from src.geometry import extract_geometry_features
from src.io import ComponentData, load_slide
from src.labeler import SlideLabelerWidget
from src.texture_features import extract_texture_features_batch

_logger = logging.getLogger("SliceGrouper.pairwise")


def get_cache_dir(base_dir: Optional[Path] = None) -> Path:
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent / "outputs" / "checkpoints" / "features"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def extract_or_load_slide_features(
    slide_info: Dict[str, Any],
    min_area_px: int = 6000,
    cache_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """
    Extract or load cached features for all components in a slide.
    Saves compressed .npz cache to avoid recomputing.
    """
    cache_dir = get_cache_dir(cache_dir)
    slide_name = slide_info["slide_name"]
    cache_file = cache_dir / f"{slide_name.replace(' ', '_')}_feats.npz"

    if cache_file.exists():
        try:
            cached = np.load(cache_file, allow_pickle=True)
            return {
                "slide_name": slide_name,
                "sample_id": slide_info["sample_id"],
                "color_feats": cached["color_feats"],
                "texture_feats": cached["texture_feats"],
                "geom_feats": cached["geom_feats"],
                "comp_ids": cached["comp_ids"].tolist(),
                "centroids": cached["centroids"],
                "bboxes": cached["bboxes"],
                "areas": cached["areas"],
                "img_shape": tuple(cached["img_shape"]),
            }
        except Exception as e:
            _logger.warning(f"Failed to read cache {cache_file}: {e}, recomputing.")

    # Load slide and detect components
    try:
        t_path = slide_info.get("tiff_path") or slide_info.get("slide_path")
        m_path = slide_info.get("mask_path")
        image, mask, comps = load_slide(t_path, m_path, min_area_px=min_area_px)
        img_shape = np.array(image.shape[:2])  # (H, W)
    except Exception as e:
        _logger.error(f"Error loading slide {slide_name}: {e}")
        return None

    if not comps:
        return None

    color_feats = extract_color_features_batch(comps)
    texture_feats = extract_texture_features_batch(comps)
    geom_feats = np.array([extract_geometry_features(c).to_vector() for c in comps], dtype=float)

    comp_ids = np.array([c.component_id for c in comps], dtype=int)
    centroids = np.array([c.centroid for c in comps], dtype=float)
    bboxes = np.array([c.bbox for c in comps], dtype=float)
    areas = np.array([c.area for c in comps], dtype=float)

    # Save to disk cache
    try:
        np.savez_compressed(
            cache_file,
            color_feats=color_feats,
            texture_feats=texture_feats,
            geom_feats=geom_feats,
            comp_ids=comp_ids,
            centroids=centroids,
            bboxes=bboxes,
            areas=areas,
            img_shape=img_shape,
        )
    except Exception as e:
        _logger.warning(f"Could not save cache to {cache_file}: {e}")

    return {
        "slide_name": slide_name,
        "sample_id": slide_info["sample_id"],
        "color_feats": color_feats,
        "texture_feats": texture_feats,
        "geom_feats": geom_feats,
        "comp_ids": comp_ids.tolist(),
        "centroids": centroids,
        "bboxes": bboxes,
        "areas": areas,
        "img_shape": tuple(img_shape),
    }


def compute_pairwise_vector(
    c1_color: np.ndarray,
    c2_color: np.ndarray,
    c1_texture: np.ndarray,
    c2_texture: np.ndarray,
    c1_geom: np.ndarray,
    c2_geom: np.ndarray,
    c1_cent: np.ndarray,
    c2_cent: np.ndarray,
    c1_bbox: np.ndarray,
    c2_bbox: np.ndarray,
    c1_area: float,
    c2_area: float,
    img_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Computes a symmetric, normalized pairwise feature vector between two components.
    """
    H, W = img_shape
    diag = np.sqrt(H**2 + W**2) + 1e-6

    # 1. Color features
    eps = 1e-8
    norm1 = np.linalg.norm(c1_color) + eps
    norm2 = np.linalg.norm(c2_color) + eps
    color_cosine = float(np.dot(c1_color, c2_color) / (norm1 * norm2))
    color_l1 = float(np.sum(np.abs(c1_color - c2_color)))

    # 2. Texture features
    norm_t1 = np.linalg.norm(c1_texture) + eps
    norm_t2 = np.linalg.norm(c2_texture) + eps
    texture_cosine = float(np.dot(c1_texture, c2_texture) / (norm_t1 * norm_t2))
    texture_l1 = float(np.sum(np.abs(c1_texture - c2_texture)))

    # 3. Geometry / Shape
    area_ratio = float(min(c1_area, c2_area) / max(c1_area, c2_area, 1.0))
    geom_diff = np.abs(c1_geom - c2_geom)

    # 4. Spatial relative features
    dx = abs(c1_cent[0] - c2_cent[0])
    dy = abs(c1_cent[1] - c2_cent[1])
    euc_dist = np.sqrt(dx**2 + dy**2)

    norm_dx = dx / (W + 1e-6)
    norm_dy = dy / (H + 1e-6)
    norm_dist = euc_dist / diag

    # Alignment features (are they aligned horizontally or vertically?)
    min_w = min(c1_bbox[2], c2_bbox[2]) + 1e-6
    min_h = min(c1_bbox[3], c2_bbox[3]) + 1e-6
    h_align_ratio = dx / min_w
    v_align_ratio = dy / min_h

    # Pack into 1D feature array
    feats = [
        color_cosine,
        color_l1,
        texture_cosine,
        texture_l1,
        area_ratio,
        norm_dx,
        norm_dy,
        norm_dist,
        h_align_ratio,
        v_align_ratio,
    ]
    feats.extend(geom_diff.tolist())

    return np.array(feats, dtype=float)


def build_pairwise_dataset(
    slides_info: List[Dict[str, Any]],
    min_area_px: int = 6000,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Iterates over all labeled slides, builds pairwise component features,
    and attaches Ground Truth labels:
      - y_same_tissue: 1 if components have the same tissue_id, 0 otherwise
      - y_same_slice: 1 if components have the same slice_id, 0 otherwise
      - groups: slide_id for strict GroupKFold evaluation
    """
    X_pairs = []
    y_same_tissue = []
    y_same_slice = []
    slide_groups = []
    pair_metadata = []

    slide_level_records = []

    n_slides = len(slides_info)
    _logger.info(f"Extracting features and pairs from {n_slides} slides...")

    for s_idx, s in enumerate(slides_info):
        labels_path = s.get("labels_path")
        if not labels_path or not Path(labels_path).exists():
            continue

        with open(labels_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        gt_comps = {c["id"]: c for c in gt_data.get("components", [])}
        if not gt_comps:
            continue

        slide_feats = extract_or_load_slide_features(s, min_area_px=min_area_px, cache_dir=cache_dir)
        if not slide_feats:
            continue

        comp_ids = slide_feats["comp_ids"]
        centroids = slide_feats["centroids"]
        bboxes = slide_feats["bboxes"]
        areas = slide_feats["areas"]
        c_feats = slide_feats["color_feats"]
        t_feats = slide_feats["texture_feats"]
        g_feats = slide_feats["geom_feats"]
        img_shape = slide_feats["img_shape"]

        # Track slide-level statistics
        slice_ids_in_gt = [gt_comps[cid]["slice_id"] for cid in comp_ids if cid in gt_comps]
        is_single_slice = len(set(slice_ids_in_gt)) <= 1 if slice_ids_in_gt else True

        slide_level_records.append({
            "slide_idx": s_idx,
            "slide_name": s["slide_name"],
            "sample_id": s["sample_id"],
            "n_components": len(comp_ids),
            "is_single_slice": is_single_slice,
            "n_gt_slices": len(set(slice_ids_in_gt)) if slice_ids_in_gt else 1,
            "areas": areas.tolist(),
            "centroids": centroids.tolist(),
            "bboxes": bboxes.tolist(),
            "color_feats": c_feats,
        })

        # Generate pairwise combinations
        n = len(comp_ids)
        for i in range(n):
            cid_i = comp_ids[i]
            if cid_i not in gt_comps:
                continue
            gt_i = gt_comps[cid_i]

            for j in range(i + 1, n):
                cid_j = comp_ids[j]
                if cid_j not in gt_comps:
                    continue
                gt_j = gt_comps[cid_j]

                vec = compute_pairwise_vector(
                    c1_color=c_feats[i],
                    c2_color=c_feats[j],
                    c1_texture=t_feats[i],
                    c2_texture=t_feats[j],
                    c1_geom=g_feats[i],
                    c2_geom=g_feats[j],
                    c1_cent=centroids[i],
                    c2_cent=centroids[j],
                    c1_bbox=bboxes[i],
                    c2_bbox=bboxes[j],
                    c1_area=areas[i],
                    c2_area=areas[j],
                    img_shape=img_shape,
                )

                same_t = 1 if (gt_i.get("tissue_id") == gt_j.get("tissue_id")) else 0
                same_s = 1 if (gt_i.get("slice_id") == gt_j.get("slice_id")) else 0

                X_pairs.append(vec)
                y_same_tissue.append(same_t)
                y_same_slice.append(same_s)
                slide_groups.append(s_idx)
                pair_metadata.append({
                    "slide_name": s["slide_name"],
                    "comp_i": cid_i,
                    "comp_j": cid_j,
                    "tissue_i": gt_i.get("tissue_id"),
                    "tissue_j": gt_j.get("tissue_id"),
                    "slice_i": gt_i.get("slice_id"),
                    "slice_j": gt_j.get("slice_id"),
                })

    X_arr = np.array(X_pairs, dtype=float) if X_pairs else np.empty((0, 22))
    yt_arr = np.array(y_same_tissue, dtype=int) if y_same_tissue else np.empty((0,), dtype=int)
    ys_arr = np.array(y_same_slice, dtype=int) if y_same_slice else np.empty((0,), dtype=int)
    grps_arr = np.array(slide_groups, dtype=int) if slide_groups else np.empty((0,), dtype=int)

    feature_names = [
        "color_cosine",
        "color_l1",
        "texture_cosine",
        "texture_l1",
        "area_ratio",
        "norm_dx",
        "norm_dy",
        "norm_dist",
        "h_align_ratio",
        "v_align_ratio",
        "diff_area",
        "diff_perimeter",
        "diff_circularity",
        "diff_compactness",
        "diff_solidity",
        "diff_convex_area",
        "diff_aspect_ratio",
        "diff_eccentricity",
        "diff_orientation",
        "diff_major_axis",
        "diff_minor_axis",
        "diff_skeleton_length",
    ]

    return {
        "X": X_arr,
        "y_same_tissue": yt_arr,
        "y_same_slice": ys_arr,
        "groups": grps_arr,
        "feature_names": feature_names,
        "pair_metadata": pair_metadata,
        "slide_records": slide_level_records,
    }
