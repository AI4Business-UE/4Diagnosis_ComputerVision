import math
import logging
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)


def _apply_tissue_mask(tile_rgb: np.ndarray) -> np.ndarray:
    """Replace irrelevant background (glass, blue dye) with white.
    Same HSV params as generate_mask: sat_min=5, val_max=250, kernel 3x3."""
    tile_bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([0, 5, 0], dtype=np.uint8),
        np.array([179, 255, 250], dtype=np.uint8),
    )
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    result = tile_rgb.copy()
    result[mask == 0] = 255  # background -> white
    return result


class GlomeruliProcessor:
    def __init__(
        self,
        path_mrxs: str,
        model_path: str,
        conf: float = settings.YOLO_CONF,
        iou: float = settings.YOLO_IOU,
        imgsz: int = settings.YOLO_IMG_SIZE,
        tile_size: int = settings.GLOMERULI_TILE_SIZE,
        overlap: int = settings.GLOMERULI_OVERLAP,
        wsi_level: int = settings.GLOMERULI_WSI_LEVEL,
        batch_size: int = settings.GLOMERULI_BATCH_SIZE,
        mask_path: str | None = None,
        scan_bbox: tuple | None = None,  # (x, y, w, h) in level-0 coords; None = full slide
    ):
        self.path = Path(path_mrxs)
        self.model_path = Path(model_path)
        self.mask_path = Path(mask_path) if mask_path else None
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.tile_size = tile_size
        self.overlap = overlap
        self.wsi_level = wsi_level
        self.batch_size = batch_size
        self.scan_bbox = scan_bbox  # optional (x, y, w, h) in level-0 px

        self.model = None
        self.glomeruli: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self):
        if self.model is None:
            import torch
            from ultralytics import YOLO
            _orig = torch.load
            torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
            try:
                self.model = YOLO(str(self.model_path))
            finally:
                torch.load = _orig
        return self.model

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_glomeruli(self, on_batch=None, on_tile=None) -> List[Dict[str, Any]]:
        import openslide

        self.glomeruli = []
        model = self.load_model()

        slide = openslide.OpenSlide(str(self.path))
        W, H = slide.level_dimensions[self.wsi_level]

        scale = self.tile_size / self.imgsz
        step = self.tile_size - self.overlap

        # Determine scan region (full slide or specific slice bbox)
        if self.scan_bbox is not None:
            bx, by, bw, bh = self.scan_bbox
            x_start = max(0, bx)
            y_start = max(0, by)
            x_end = min(W, bx + bw)
            y_end = min(H, by + bh)
        else:
            x_start, y_start = 0, 0
            x_end, y_end = W, H

        # TIFF mask filter — tiles all-white in the mask preview are skipped
        tiff_mask = None
        if self.mask_path and self.mask_path.exists():
            _m = cv2.imread(str(self.mask_path), cv2.IMREAD_GRAYSCALE)
            if _m is not None:
                if _m.ndim == 3:
                    _m = _m.squeeze(axis=2)
                tiff_mask = (_m > 0)  # bool — True = tissue

        def is_tissue_in_tiff_mask(x, y, w, h) -> bool:
            if tiff_mask is None:
                return True  # fallback: no mask -> scan everything
            mh, mw = tiff_mask.shape
            mx1 = int(x * mw / W); my1 = int(y * mh / H)
            mx2 = max(mx1 + 1, int((x + w) * mw / W))
            my2 = max(my1 + 1, int((y + h) * mh / H))
            region = tiff_mask[my1:my2, mx1:mx2]
            if region.size == 0:
                return False
            return region.sum() / region.size >= 0.05

        # Thumbnail pre-filter: fast check without level-0 I/O
        THUMB_SCALE = 32
        thumb_w = max(1, W // THUMB_SCALE)
        thumb_h = max(1, H // THUMB_SCALE)
        thumb = slide.get_thumbnail((thumb_w, thumb_h))
        thumb_gray = cv2.cvtColor(np.array(thumb.convert("RGB")), cv2.COLOR_RGB2GRAY)
        actual_tw, actual_th = thumb.size
        sx = actual_tw / W
        sy = actual_th / H

        def is_tissue_thumb(x, y, w, h) -> bool:
            tx1 = int(x * sx); ty1 = int(y * sy)
            tx2 = max(tx1 + 1, int((x + w) * sx))
            ty2 = max(ty1 + 1, int((y + h) * sy))
            region = thumb_gray[ty1:ty2, tx1:tx2]
            if region.size == 0:
                return False
            tissue_px = int(np.sum((region > 15) & (region < 230)))
            return tissue_px >= max(1, int(region.size * 0.03))

        batch_tiles = []
        batch_offsets = []

        def process_batch():
            if not batch_tiles:
                return
            results = model.predict(
                batch_tiles,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                verbose=False,
            )
            for i, result in enumerate(results):
                x_off, y_off = batch_offsets[i]
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                    cls_id = int(box.cls[0].cpu().item()) if box.cls is not None else 0
                    score = float(box.conf[0].cpu().item()) if box.conf is not None else 0.0
                    self.glomeruli.append({
                        "x1": int(x1 * scale + x_off),
                        "y1": int(y1 * scale + y_off),
                        "x2": int(x2 * scale + x_off),
                        "y2": int(y2 * scale + y_off),
                        "cls": cls_id,
                        "cls_name": settings.GLOMERULI_CLASSES.get(cls_id, str(cls_id)),
                        "conf": score,
                    })
            if on_batch:
                on_batch(self.glomeruli)
            batch_tiles.clear()
            batch_offsets.clear()

        for y in range(y_start, y_end, step):
            for x in range(x_start, x_end, step):
                actual_w = min(self.tile_size, x_end - x)
                actual_h = min(self.tile_size, y_end - y)
                if actual_w < 50 or actual_h < 50:
                    continue

                if not is_tissue_thumb(x, y, actual_w, actual_h):
                    if on_tile:
                        on_tile(x, y, actual_w, actual_h, False)
                    continue

                if not is_tissue_in_tiff_mask(x, y, actual_w, actual_h):
                    if on_tile:
                        on_tile(x, y, actual_w, actual_h, False)
                    continue

                region = slide.read_region((x, y), self.wsi_level, (actual_w, actual_h))
                tile = np.array(region.convert("RGB"))
                tile = _apply_tissue_mask(tile)

                if on_tile:
                    on_tile(x, y, actual_w, actual_h, True)

                tile_resized = cv2.resize(tile, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
                batch_tiles.append(tile_resized)
                batch_offsets.append((x, y))

                if len(batch_tiles) >= self.batch_size:
                    process_batch()

        process_batch()
        slide.close()

        self.glomeruli = self.simple_global_merge(self.glomeruli, overlap_thresh=0.3)
        return self.glomeruli

    def count_glomeruli(self) -> int:
        return len(self.glomeruli or [])

    # ------------------------------------------------------------------
    # Geometry helpers — cross-slice bbox comparison
    # ------------------------------------------------------------------

    @staticmethod
    def boxes_correspond(a: dict, b: dict, tol_px: int = 200) -> bool:
        """
        True if two glomeruli bboxes from *different slices* correspond to
        the same anatomical structure.

        Comparison is based on centre-point Euclidean distance (level-0 px).
        Coordinates in both dicts must be in the same coordinate system
        (either both TIFF-relative or both level-0-relative).
        """
        ca_x = (a["x1"] + a["x2"]) / 2
        ca_y = (a["y1"] + a["y2"]) / 2
        cb_x = (b["x1"] + b["x2"]) / 2
        cb_y = (b["y1"] + b["y2"]) / 2
        return math.hypot(ca_x - cb_x, ca_y - cb_y) <= tol_px

    # ------------------------------------------------------------------
    # NMS / deduplication within one detection run
    # ------------------------------------------------------------------

    def simple_global_merge(self, detections, overlap_thresh=0.3):
        """Merge overlapping detections using union-find (IoMin).
        If >= overlap_thresh of the smaller box is covered -> merge."""
        if not detections:
            return []

        n = len(detections)
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            parent[find(i)] = find(j)

        for i in range(n):
            a = detections[i]
            area_a = max(0, (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]))
            for j in range(i + 1, n):
                b = detections[j]
                area_b = max(0, (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))
                min_area = min(area_a, area_b)
                if min_area <= 0:
                    continue
                ix1 = max(a["x1"], b["x1"]); iy1 = max(a["y1"], b["y1"])
                ix2 = min(a["x2"], b["x2"]); iy2 = min(a["y2"], b["y2"])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if inter / min_area >= overlap_thresh:
                    union(i, j)

        groups: dict[int, list] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(detections[i])

        result = []
        for group in groups.values():
            best = max(group, key=lambda d: d["conf"])
            result.append({
                "x1": min(d["x1"] for d in group),
                "y1": min(d["y1"] for d in group),
                "x2": max(d["x2"] for d in group),
                "y2": max(d["y2"] for d in group),
                "cls": best["cls"],
                "cls_name": best["cls_name"],
                "conf": best["conf"],
            })
        return result
