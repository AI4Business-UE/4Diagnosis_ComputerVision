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
        Group tissue components into slices of the same tissue.

        Parameters
        ----------
        img_bgr : np.ndarray
            Full TIFF image in BGR format (used for histogram computation).
        components : list[np.ndarray]
            List of boolean masks (one per connected component), as returned
            by mask.generate_mask()["components"].

        Returns
        -------
        dict with structure:
            {
              "representative_slice_id": int,
              "items": [
                { "slice_id": int, "is_representative": bool,
                  "bbox_tiff": [x, y, w, h], "area_tiff_px": int }
              ]
            }
        """
        if not components:
            logger.warning("SliceGrouper.group(): no components received")
            return {"representative_slice_id": 0, "items": []}

        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Build descriptor for each component
        descriptors = []
        for comp in components:
            bbox = self._bbox_from_mask(comp)
            area = int(comp.sum())
            hist = self._compute_hist(img_hsv, comp)
            descriptors.append({"bbox": bbox, "area": area, "hist": hist})

        n = len(descriptors)

        # Union-Find
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            parent[find(i)] = find(j)

        for i in range(n):
            for j in range(i + 1, n):
                if self._similar(descriptors[i], descriptors[j]):
                    union(i, j)

        # Collect groups (all components -> one tissue -> one group)
        groups = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        # We assume one tissue per scan -> take the largest group
        largest_group = max(groups.values(), key=lambda g: sum(descriptors[i]["area"] for i in g))

        if len(groups) > 1:
            logger.info(
                f"SliceGrouper: {len(groups)} groups found, using largest "
                f"({len(largest_group)} slices). Others discarded."
            )

        # Sort slices by horizontal position (left -> right) then vertical (top -> bottom)
        def sort_key(idx):
            x, y, w, h = descriptors[idx]["bbox"]
            return (x, y)

        sorted_indices = sorted(largest_group, key=sort_key)

        # Pick middle slice as representative (index N//2)
        repr_pos = len(sorted_indices) // 2
        repr_global_idx = sorted_indices[repr_pos]

        # Build output items (local slice_id = position in sorted list)
        items = []
        representative_slice_id = None
        for local_id, global_idx in enumerate(sorted_indices):
            is_repr = (global_idx == repr_global_idx)
            if is_repr:
                representative_slice_id = local_id
            x, y, w, h = descriptors[global_idx]["bbox"]
            items.append({
                "slice_id": local_id,
                "is_representative": is_repr,
                "bbox_tiff": [x, y, w, h],
                "area_tiff_px": descriptors[global_idx]["area"],
            })

        logger.info(
            f"SliceGrouper: {len(items)} slice(s) detected, "
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
