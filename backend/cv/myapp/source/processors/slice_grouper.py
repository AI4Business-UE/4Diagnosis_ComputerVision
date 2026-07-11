import logging
import cv2
import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)


class SliceGrouper:
    """
    Groups connected tissue components (from mask.py) into slices of the same tissue.

    Algorithm:
    1. Compute bbox, area, and HSV histogram for each component
    2. Compare each pair: area ratio + histogram correlation
    3. Union-Find -> connected groups (same tissue)
    4. Select middle slice as representative
    """

    def __init__(
        self,
        hist_thresh: float = settings.SLICE_SIMILARITY_HIST_THRESH,
        area_ratio_thresh: float = settings.SLICE_SIMILARITY_AREA_RATIO,
    ):
        self.hist_thresh = hist_thresh
        self.area_ratio_thresh = area_ratio_thresh

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def group(self, img_bgr: np.ndarray, components: list) -> dict:
        """
        Group tissue components into slices based on spatial layout.
        
        This handles scans where a single slice consists of multiple 
        disconnected tissue fragments by clustering components that align
        along the main layout axis.
        """
        if not components:
            logger.warning("SliceGrouper.group(): no components received")
            return {"representative_slice_id": 0, "items": []}

        # 1. Get bounding boxes and areas for all components
        comp_infos = []
        for idx, comp in enumerate(components):
            bbox = self._bbox_from_mask(comp)
            area = int(comp.sum())
            comp_infos.append({
                "idx": idx,
                "bbox": bbox, # [x, y, w, h]
                "area": area
            })

        # Find total span on X and Y to determine layout direction
        all_x_min = min(info["bbox"][0] for info in comp_infos)
        all_x_max = max(info["bbox"][0] + info["bbox"][2] for info in comp_infos)
        all_y_min = min(info["bbox"][1] for info in comp_infos)
        all_y_max = max(info["bbox"][1] + info["bbox"][3] for info in comp_infos)

        span_x = all_x_max - all_x_min
        span_y = all_y_max - all_y_min

        # Layout direction: True = horizontal, False = vertical
        is_horizontal = span_x >= span_y
        total_span = span_x if is_horizontal else span_y

        # Clustering threshold: 8% of the total span, min 100 pixels
        threshold = max(100.0, total_span * 0.08)

        # 2. Cluster components along the layout axis
        # Each component has an interval [start, end] along the axis
        intervals = []
        for info in comp_infos:
            x, y, w, h = info["bbox"]
            start = x if is_horizontal else y
            end = start + (w if is_horizontal else h)
            intervals.append({
                "info": info,
                "start": start,
                "end": end
            })

        # Sort intervals by start position
        intervals.sort(key=lambda item: item["start"])

        # Merge overlapping/close intervals into slice clusters
        clusters = [] # list of lists of interval dicts
        for item in intervals:
            if not clusters:
                clusters.append([item])
            else:
                # Compare with the last cluster's maximum end position
                last_cluster = clusters[-1]
                max_end = max(c_item["end"] for c_item in last_cluster)
                if item["start"] - max_end < threshold:
                    last_cluster.append(item)
                else:
                    clusters.append([item])

        # 3. Build slices from clusters
        slices = []
        for cluster_idx, cluster in enumerate(clusters):
            # Union of bounding boxes
            x_min = min(c_item["info"]["bbox"][0] for c_item in cluster)
            y_min = min(c_item["info"]["bbox"][1] for c_item in cluster)
            x_max = max(c_item["info"]["bbox"][0] + c_item["info"]["bbox"][2] for c_item in cluster)
            y_max = max(c_item["info"]["bbox"][1] + c_item["info"]["bbox"][3] for c_item in cluster)
            
            bbox_tiff = [x_min, y_min, x_max - x_min, y_max - y_min]
            area_tiff_px = sum(c_item["info"]["area"] for c_item in cluster)
            
            slices.append({
                "slice_id": cluster_idx, # temporary, sorted below
                "bbox_tiff": bbox_tiff,
                "area_tiff_px": area_tiff_px,
                "sort_val": x_min if is_horizontal else y_min
            })

        # Sort slices along the axis (left-to-right or top-to-bottom)
        slices.sort(key=lambda s: s["sort_val"])

        # Pick middle slice as representative (middle index)
        n_slices = len(slices)
        repr_idx = n_slices // 2

        items = []
        representative_slice_id = None
        for local_id, s in enumerate(slices):
            is_repr = (local_id == repr_idx)
            if is_repr:
                representative_slice_id = local_id
            items.append({
                "slice_id": local_id,
                "is_representative": is_repr,
                "bbox_tiff": s["bbox_tiff"],
                "area_tiff_px": s["area_tiff_px"],
            })

        logger.info(
            f"SliceGrouper (spatial): {len(items)} slice(s) detected, "
            f"representative = slice_id {representative_slice_id}"
        )

        return {
            "representative_slice_id": representative_slice_id,
            "items": items,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _bbox_from_mask(self, mask_bool: np.ndarray):
        """Return (x, y, w, h) bounding box of a boolean mask."""
        rows = np.any(mask_bool, axis=1)
        cols = np.any(mask_bool, axis=0)
        if not np.any(rows):
            return (0, 0, 1, 1)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        x, y = int(cmin), int(rmin)
        w, h = int(cmax - cmin + 1), int(rmax - rmin + 1)
        return (x, y, w, h)

    def _compute_hist(self, img_hsv: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
        """Compute normalised HSV histogram for the masked region."""
        mask_u8 = (mask_bool.astype(np.uint8)) * 255
        hist = cv2.calcHist(
            [img_hsv], [0, 1], mask_u8, [50, 60], [0, 180, 0, 256]
        )
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def _similar(self, a: dict, b: dict) -> bool:
        """Return True if two components look like slices of the same tissue."""
        # Area ratio check
        min_area = min(a["area"], b["area"])
        max_area = max(a["area"], b["area"])
        if max_area == 0:
            return False
        area_ratio = min_area / max_area
        if area_ratio < (1.0 - self.area_ratio_thresh):
            return False

        # Histogram correlation check
        corr = cv2.compareHist(a["hist"], b["hist"], cv2.HISTCMP_CORREL)
        return corr >= self.hist_thresh
