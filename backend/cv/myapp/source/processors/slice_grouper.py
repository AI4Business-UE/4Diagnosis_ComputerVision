from __future__ import annotations

from functools import lru_cache
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import joblib
import numpy as np
from django.conf import settings
from scipy import stats as scipy_stats
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _load_fallback_model(model_path_str: str):
    model = joblib.load(model_path_str)
    logger.info(f"SliceGrouper: Loaded Safety Fallback Model from {model_path_str}")
    return model


def _channel_stats(pixels: np.ndarray) -> np.ndarray:
    if len(pixels) == 0:
        return np.zeros(3, dtype=np.float64)
    return np.array([
        float(np.mean(pixels)),
        float(np.std(pixels)),
        float(scipy_stats.skew(pixels)),
    ], dtype=np.float64)


def _extract_space_stats(img_space: np.ndarray, tissue_mask: np.ndarray) -> np.ndarray:
    img_f32 = img_space.astype(np.float32)
    stats = [_channel_stats(img_f32[:, :, ch][tissue_mask]) for ch in range(3)]
    means = np.array([s[0] for s in stats], dtype=np.float64)
    stds = np.array([s[1] for s in stats], dtype=np.float64)
    skews = np.array([s[2] for s in stats], dtype=np.float64)
    return np.concatenate([means, stds, skews])


def _build_space_histogram(
    img_space: np.ndarray,
    mask_u8: np.ndarray,
    ranges: list[tuple[int, int]],
    bins: int = 32,
) -> np.ndarray:
    hists = []
    for ch in range(3):
        lo, hi = ranges[ch]
        h = cv2.calcHist([img_space], [ch], mask_u8, [bins], [lo, hi]).flatten().astype(np.float64)
        total = h.sum()
        if total > 0:
            h /= total
        hists.append(h)
    return np.concatenate(hists)


def extract_component_color_vector(crop_rgb: np.ndarray, crop_mask_bool: np.ndarray, bins: int = 32) -> np.ndarray:
    crop_hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    crop_lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2Lab)
    mask_u8 = crop_mask_bool.astype(np.uint8) * 255

    rgb_stats = _extract_space_stats(crop_rgb, crop_mask_bool)
    hsv_stats = _extract_space_stats(crop_hsv, crop_mask_bool)
    lab_stats = _extract_space_stats(crop_lab, crop_mask_bool)

    rgb_hist = _build_space_histogram(crop_rgb, mask_u8, [(0, 256), (0, 256), (0, 256)], bins=bins)
    hsv_hist = _build_space_histogram(crop_hsv, mask_u8, [(0, 181), (0, 256), (0, 256)], bins=bins)
    lab_hist = _build_space_histogram(crop_lab, mask_u8, [(0, 256), (0, 256), (0, 256)], bins=bins)

    return np.concatenate([rgb_stats, hsv_stats, lab_stats, rgb_hist, hsv_hist, lab_hist]).astype(np.float64)


class SliceGrouper:
    def __init__(
        self,
        fallback_model_path: Optional[Path] = None,
        eps_scale: Optional[float] = None,
        fallback_threshold: Optional[float] = None,
        use_fallback_ml: Optional[bool] = None,
    ):
        self.eps_scale = (
            float(eps_scale)
            if eps_scale is not None
            else float(getattr(settings, "SLICE_AGGLOMERATIVE_EPS_SCALE", 0.35))
        )
        self.fallback_threshold = (
            float(fallback_threshold)
            if fallback_threshold is not None
            else float(getattr(settings, "SLICE_FALLBACK_THRESHOLD", 0.50))
        )
        self.use_fallback_ml = (
            bool(use_fallback_ml)
            if use_fallback_ml is not None
            else bool(getattr(settings, "SLICE_GROUPER_USE_FALLBACK_ML", True))
        )
        self.fallback_model = None

        if self.use_fallback_ml:
            model_p = fallback_model_path
            if model_p is None:
                model_p = getattr(
                    settings,
                    "FALLBACK_MODEL_PATH",
                    settings.BASE_DIR / "myapp" / "source" / "model" / "fallback_classifier.joblib",
                )
            model_path = Path(model_p)
            if model_path.exists():
                try:
                    self.fallback_model = _load_fallback_model(str(model_path.resolve()))
                except Exception as e:
                    logger.warning(f"SliceGrouper: Could not load fallback model: {e}")
            else:
                logger.warning(f"SliceGrouper: Model file not found at {model_path}, using heuristic fallback.")

    def extract_slide_features(
        self,
        img_bgr: np.ndarray,
        comp_infos: List[Dict[str, Any]],
    ) -> tuple[np.ndarray, float, int]:
        N = len(comp_infos)
        areas = np.array([c["area"] for c in comp_infos], dtype=float)
        cents = np.array([c["centroid"] for c in comp_infos], dtype=float)
        bboxes = np.array([c["bbox"] for c in comp_infos], dtype=float)

        area_cv = float(np.std(areas) / (np.mean(areas) + 1e-5))
        area_min_max = float(np.min(areas) / (np.max(areas) + 1e-5))

        dx_span = np.ptp(cents[:, 0])
        dy_span = np.ptp(cents[:, 1])
        axis = 0 if dx_span >= dy_span else 1

        if N >= 2:
            dists = pdist(cents)
            dist_cv = float(np.std(dists) / (np.mean(dists) + 1e-5))

            order = np.argsort(cents[:, axis])
            sorted_cents = cents[order, axis]
            gaps = np.diff(sorted_cents)
            max_gap = float(np.max(gaps)) if len(gaps) > 0 else 0.0
            med_dim = float(np.median([bboxes[i, 2] if axis == 0 else bboxes[i, 3] for i in range(N)]))
            gap_ratio = max_gap / (med_dim + 1e-5)

            c_feats = np.array([c["color_vec"] for c in comp_infos], dtype=np.float64)
            sim_m = 1.0 - squareform(pdist(c_feats, metric="cosine"))
            np.fill_diagonal(sim_m, 0.0)
            sim_m = np.nan_to_num(sim_m, nan=0.0)
            max_twin_sim = float(np.max(sim_m))
        else:
            dist_cv = 0.0
            gap_ratio = 0.0
            max_twin_sim = 0.0

        fallback_feats = np.array([[
            float(N),
            area_cv,
            area_min_max,
            gap_ratio,
            dist_cv,
            max_twin_sim,
            1.0 if N <= 1 else 0.0,
            1.0 if N in (2, 3) else 0.0,
        ]], dtype=float)

        return fallback_feats, gap_ratio, axis

    def group(self, img_bgr: np.ndarray, components: list) -> dict:
        if not components:
            logger.warning("SliceGrouper.group(): no components received")
            return {
                "representative_slice_id": 0,
                "items": [],
                "fallback_used": False,
                "fallback_prob": 0.0,
            }

        N = len(components)
        need_color = N > 1

        comp_infos: List[Dict[str, Any]] = []
        for idx, comp in enumerate(components):
            if isinstance(comp, dict) and "crop_mask" in comp:
                x_min, y_min, w, h = comp["bbox"]
                area = int(comp["area"])
                if area == 0 or w <= 0 or h <= 0:
                    continue
                bbox = [int(x_min), int(y_min), int(w), int(h)]
                cx, cy = comp["centroid"]
                crop_mask = np.asarray(comp["crop_mask"], dtype=bool)
            else:
                mask_bool = np.asarray(comp, dtype=bool)
                ys, xs = np.where(mask_bool)
                if len(ys) == 0:
                    continue
                x_min, x_max = int(xs.min()), int(xs.max())
                y_min, y_max = int(ys.min()), int(ys.max())
                w = x_max - x_min + 1
                h = y_max - y_min + 1
                bbox = [x_min, y_min, w, h]
                area = int(len(ys))
                cx = float(xs.mean())
                cy = float(ys.mean())
                crop_mask = mask_bool[y_min:y_min + h, x_min:x_min + w]

            color_vec = None
            if need_color:
                crop_rgb = cv2.cvtColor(img_bgr[y_min:y_min + h, x_min:x_min + w], cv2.COLOR_BGR2RGB)
                color_vec = extract_component_color_vector(crop_rgb, crop_mask)

            comp_infos.append({
                "idx": idx,
                "bbox": bbox,
                "area": area,
                "centroid": (cx, cy),
                "color_vec": color_vec,
            })

        if not comp_infos:
            return {
                "representative_slice_id": 0,
                "items": [],
                "fallback_used": False,
                "fallback_prob": 0.0,
            }

        if len(comp_infos) <= 1:
            item = comp_infos[0]
            return {
                "representative_slice_id": 0,
                "items": [{
                    "slice_id": 0,
                    "is_representative": True,
                    "bbox_tiff": item["bbox"],
                    "area_tiff_px": item["area"],
                }],
                "fallback_used": False,
                "fallback_prob": 1.0,
            }

        fallback_feats, gap_ratio, axis = self.extract_slide_features(img_bgr, comp_infos)
        cents = np.array([c["centroid"] for c in comp_infos], dtype=float)

        should_fallback = False
        fallback_prob = 0.0

        if self.fallback_model is not None:
            try:
                fallback_prob = float(self.fallback_model.predict_proba(fallback_feats)[0, 1])
                should_fallback = fallback_prob >= self.fallback_threshold
            except Exception as e:
                logger.warning(f"SliceGrouper: Model evaluation error: {e}")
                should_fallback = (gap_ratio < 2.5)
        else:
            should_fallback = (gap_ratio < 2.5)

        if should_fallback:
            logger.info(
                f"[SAFETY FALLBACK] SliceGrouper: Fallback triggered (P={fallback_prob:.2f}, gap_ratio={gap_ratio:.2f}) -> "
                f"Preserving all {len(comp_infos)} components as 1 single slice."
            )
            return {
                "representative_slice_id": 0,
                "items": [self._merge_comp_infos_to_slice(comp_infos, slice_id=0, is_repr=True)],
                "fallback_used": True,
                "fallback_prob": fallback_prob,
            }

        h_img, w_img = img_bgr.shape[:2]
        threshold = max(h_img, w_img) * self.eps_scale
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            linkage="average",
        )
        raw_slice_labels = clustering.fit_predict(cents)

        unique_slices = np.unique(raw_slice_labels)
        means = [cents[raw_slice_labels == s, axis].mean() for s in unique_slices]
        slice_sort_order = np.argsort(means)
        remap = {orig: new for new, orig in enumerate(unique_slices[slice_sort_order])}
        slice_labels = np.array([remap[s] for s in raw_slice_labels])

        n_slices = len(unique_slices)
        repr_idx = n_slices // 2

        items = []
        for s_idx in range(n_slices):
            comp_indices = np.where(slice_labels == s_idx)[0]
            cluster_comps = [comp_infos[i] for i in comp_indices]
            items.append(self._merge_comp_infos_to_slice(cluster_comps, slice_id=s_idx, is_repr=(s_idx == repr_idx)))

        logger.info(
            f"[SLICES DETECTED] SliceGrouper: {len(items)} serial slice(s) grouped (P(fallback)={fallback_prob:.2f}), "
            f"representative = slice_id {repr_idx}"
        )

        return {
            "representative_slice_id": repr_idx,
            "items": items,
            "fallback_used": False,
            "fallback_prob": fallback_prob,
        }

    @staticmethod
    def _merge_comp_infos_to_slice(cluster_comps: List[Dict[str, Any]], slice_id: int, is_repr: bool) -> dict:
        s_xmin = int(min(c["bbox"][0] for c in cluster_comps))
        s_ymin = int(min(c["bbox"][1] for c in cluster_comps))
        s_xmax = int(max(c["bbox"][0] + c["bbox"][2] for c in cluster_comps))
        s_ymax = int(max(c["bbox"][1] + c["bbox"][3] for c in cluster_comps))
        s_area = int(sum(c["area"] for c in cluster_comps))
        return {
            "slice_id": slice_id,
            "is_representative": is_repr,
            "bbox_tiff": [s_xmin, s_ymin, s_xmax - s_xmin, s_ymax - s_ymin],
            "area_tiff_px": s_area,
        }
