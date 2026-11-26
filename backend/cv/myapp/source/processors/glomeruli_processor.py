from functools import lru_cache
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from django.conf import settings

from .mask import compute_initial_mask

logger = logging.getLogger(__name__)
_MODEL_LOAD_LOCK = threading.Lock()
_MODEL_PREDICT_LOCK = threading.Lock()


@lru_cache(maxsize=2)
def _load_yolo_model(model_path_str: str):
    """Load and cache YOLO model once per process to avoid reloading weights on every request."""
    import torch
    from ultralytics import YOLO

    with _MODEL_LOAD_LOCK:
        _orig = torch.load
        torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
        try:
            model = YOLO(model_path_str)
        finally:
            torch.load = _orig
    logger.info(f"GlomeruliProcessor: Loaded and cached YOLO model from {model_path_str}")
    return model


def _apply_tissue_mask(tile_rgb: np.ndarray) -> np.ndarray:
    """Replace irrelevant background (glass, faint dye) with white using shared HSV mask params."""
    tile_bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
    mask = compute_initial_mask(tile_bgr, sat_min=5, val_max=250, open_ksize=3, close_ksize=3)
    result = tile_rgb.copy()
    result[mask == 0] = 255
    return result


def _select_read_level(slide, base_level: int, tile_size: int, imgsz: int) -> tuple[int, float, float]:
    """
    Pick the coarsest OpenSlide pyramid level >= base_level whose downsampled
    tile resolution is still >= imgsz (no upscaling).

    Returns (read_level, rel_downsample_from_base, base_downsample_from_level0).
    """
    downsamples = getattr(slide, "level_downsamples", None)
    if not downsamples or base_level >= len(downsamples):
        return base_level, 1.0, 1.0

    base_ds = float(downsamples[base_level])
    max_rel_ds = tile_size / float(imgsz)
    best_level = base_level
    best_rel_ds = 1.0

    for lvl in range(base_level, len(downsamples)):
        rel_ds = float(downsamples[lvl]) / base_ds
        if rel_ds <= max_rel_ds + 1e-6:
            best_level = lvl
            best_rel_ds = rel_ds
        else:
            break

    return best_level, best_rel_ds, base_ds


class GlomeruliProcessor:
    def __init__(
        self,
        path_mrxs: str,
        model_path: str,
        crop_offset_x: int,
        crop_offset_y: int,
        ds: float,
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
        if ds <= 0:
            raise ValueError("downsample factor (ds) must be a positive float")
        if crop_offset_x < 0 or crop_offset_y < 0:
            raise ValueError("crop offsets must be non-negative integers")
        if overlap < 0 or overlap >= tile_size:
            raise ValueError("overlap must be >= 0 and strictly less than tile_size")

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
        self.scan_bbox = scan_bbox
        self.crop_offset_x = crop_offset_x
        self.crop_offset_y = crop_offset_y
        self.ds = ds

        self.model = None
        self.glomeruli: List[Dict[str, Any]] = []

    def load_model(self):
        if self.model is None:
            self.model = _load_yolo_model(str(self.model_path.resolve()))
        return self.model

    def detect_glomeruli(self, on_batch=None, on_tile=None, cancel_event=None) -> List[Dict[str, Any]]:
        import openslide
        import torch

        self.glomeruli = []
        edge_flags: List[bool] = []
        model = self.load_model()

        step = self.tile_size - self.overlap
        edge_margin_px = float(getattr(settings, "GLOMERULI_EDGE_MARGIN_PX", 12))

        # TIFF mask filter — tiles all-white in the mask preview are skipped
        tiff_mask = None
        if self.mask_path and self.mask_path.exists():
            _m = cv2.imread(str(self.mask_path), cv2.IMREAD_GRAYSCALE)
            if _m is not None:
                # Ultralytics monkey-patches cv2.imread globally and can return (H, W, 3) or (H, W, 1)
                if _m.ndim == 3:
                    _m = _m[:, :, 0]
                tiff_mask = (_m > 0)

        def is_tissue_in_tiff_mask(x, y, w, h) -> bool:
            if tiff_mask is None:
                return True
            mx1 = int(x / self.ds) - self.crop_offset_x
            my1 = int(y / self.ds) - self.crop_offset_y
            mx2 = int((x + w) / self.ds) - self.crop_offset_x
            my2 = int((y + h) / self.ds) - self.crop_offset_y

            mh, mw = tiff_mask.shape
            mx1 = max(0, min(mx1, mw - 1))
            my1 = max(0, min(my1, mh - 1))
            mx2 = max(mx1 + 1, min(mx2, mw))
            my2 = max(my1 + 1, min(my2, mh))

            region = tiff_mask[my1:my2, mx1:mx2]
            if region.size == 0:
                return False
            return int(np.count_nonzero(region)) / region.size >= 0.05

        batch_tiles: List[np.ndarray] = []
        batch_offsets: List[tuple[int, int, int, int, float]] = []

        def process_batch():
            if not batch_tiles:
                return
            with _MODEL_PREDICT_LOCK, torch.inference_mode():
                results = model.predict(
                    batch_tiles,
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    verbose=False,
                )
            for i, result in enumerate(results):
                x_off, y_off, tile_w, tile_h, coord_scale = batch_offsets[i]
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                valid_w_imgsz = tile_w / coord_scale
                valid_h_imgsz = tile_h / coord_scale
                for box in result.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().tolist()
                    cls_id = int(box.cls[0].cpu().item()) if box.cls is not None else 0
                    score = float(box.conf[0].cpu().item()) if box.conf is not None else 0.0
                    # Clamp to the unpadded region of edge tiles
                    gx1 = int(max(x_off, min(x_off + tile_w, bx1 * coord_scale + x_off)))
                    gy1 = int(max(y_off, min(y_off + tile_h, by1 * coord_scale + y_off)))
                    gx2 = int(max(x_off, min(x_off + tile_w, bx2 * coord_scale + x_off)))
                    gy2 = int(max(y_off, min(y_off + tile_h, by2 * coord_scale + y_off)))
                    if gx2 <= gx1 or gy2 <= gy1:
                        continue
                    on_edge = (
                        bx1 <= edge_margin_px
                        or by1 <= edge_margin_px
                        or bx2 >= valid_w_imgsz - edge_margin_px
                        or by2 >= valid_h_imgsz - edge_margin_px
                    )
                    self.glomeruli.append({
                        "x1": gx1,
                        "y1": gy1,
                        "x2": gx2,
                        "y2": gy2,
                        "cls": cls_id,
                        "cls_name": settings.GLOMERULI_CLASSES.get(cls_id, str(cls_id)),
                        "conf": score,
                    })
                    edge_flags.append(on_edge)
            if on_batch:
                on_batch(self.glomeruli)
            batch_tiles.clear()
            batch_offsets.clear()

        cancelled = False
        mpp_base = 0.0
        try:
            with openslide.OpenSlide(str(self.path)) as slide:
                W, H = slide.level_dimensions[self.wsi_level]
                read_level, rel_ds, base_ds = _select_read_level(
                    slide, self.wsi_level, self.tile_size, self.imgsz
                )
                full_read_size = max(1, int(round(self.tile_size / rel_ds)))
                coord_scale = (full_read_size * rel_ds) / float(self.imgsz)

                props = getattr(slide, "properties", None)
                if props:
                    try:
                        mpp_x = float(props.get("openslide.mpp-x", 0.0))
                        mpp_y = float(props.get("openslide.mpp-y", 0.0))
                        if mpp_x > 0 and mpp_y > 0:
                            mpp_base = ((mpp_x + mpp_y) / 2.0) * base_ds
                    except (TypeError, ValueError):
                        mpp_base = 0.0

                logger.info(
                    f"GlomeruliProcessor: reading OpenSlide level {read_level} "
                    f"(rel_ds={rel_ds:.2f}, tile={full_read_size}px -> {self.imgsz}px, "
                    f"overlap={self.overlap}px, step={step}px)"
                )

                if self.scan_bbox is not None:
                    bx, by, bw, bh = self.scan_bbox
                    x_start = max(0, bx)
                    y_start = max(0, by)
                    x_end = min(W, bx + bw)
                    y_end = min(H, by + bh)
                else:
                    x_start, y_start = 0, 0
                    x_end, y_end = W, H

                # Thumbnail pre-filter: fast check without high-res I/O
                THUMB_SCALE = 32
                thumb_w = max(1, W // THUMB_SCALE)
                thumb_h = max(1, H // THUMB_SCALE)
                thumb = slide.get_thumbnail((thumb_w, thumb_h))
                try:
                    thumb_gray = cv2.cvtColor(np.array(thumb.convert("RGB")), cv2.COLOR_RGB2GRAY)
                    actual_tw, actual_th = thumb.size
                finally:
                    thumb.close()
                sx = actual_tw / W
                sy = actual_th / H

                def is_tissue_thumb(x, y, w, h) -> bool:
                    tx1 = int(x * sx)
                    ty1 = int(y * sy)
                    tx2 = max(tx1 + 1, int((x + w) * sx))
                    ty2 = max(ty1 + 1, int((y + h) * sy))
                    region = thumb_gray[ty1:ty2, tx1:tx2]
                    if region.size == 0:
                        return False
                    tissue_px = int(np.count_nonzero((region > 15) & (region < 230)))
                    return tissue_px >= max(1, int(region.size * 0.03))

                for y in range(y_start, y_end, step):
                    if cancel_event is not None and cancel_event.is_set():
                        cancelled = True
                        break
                    for x in range(x_start, x_end, step):
                        if cancel_event is not None and cancel_event.is_set():
                            cancelled = True
                            break
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

                        read_w = max(1, int(round(actual_w / rel_ds)))
                        read_h = max(1, int(round(actual_h / rel_ds)))
                        loc_l0 = (int(round(x * base_ds)), int(round(y * base_ds)))
                        region = slide.read_region(loc_l0, read_level, (read_w, read_h))
                        try:
                            tile = np.array(region.convert("RGB"))
                        finally:
                            region.close()

                        # Apply HSV tissue mask BEFORE resize
                        tile = _apply_tissue_mask(tile)

                        if on_tile:
                            on_tile(x, y, actual_w, actual_h, True)

                        # Pad non-square edge tiles with white to preserve aspect ratio and coordinate scale
                        th_cur, tw_cur = tile.shape[:2]
                        if tw_cur < full_read_size or th_cur < full_read_size:
                            padded = np.full((full_read_size, full_read_size, 3), 255, dtype=np.uint8)
                            copy_h = min(th_cur, full_read_size)
                            copy_w = min(tw_cur, full_read_size)
                            padded[:copy_h, :copy_w] = tile[:copy_h, :copy_w]
                            tile = padded

                        tile_resized = cv2.resize(tile, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
                        del tile
                        # Ultralytics BasePredictor.preprocess() assumes numpy inputs are BGR and reverses
                        # channels (im[..., ::-1]) to RGB before feeding the network.
                        tile_bgr = cv2.cvtColor(tile_resized, cv2.COLOR_RGB2BGR)

                        batch_tiles.append(tile_bgr)
                        batch_offsets.append((x, y, actual_w, actual_h, coord_scale))

                        if len(batch_tiles) >= self.batch_size:
                            process_batch()

                if not cancelled:
                    process_batch()
        finally:
            batch_tiles.clear()
            batch_offsets.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if cancelled:
            logger.info("Glomeruli detection cancelled by client")
            return self.glomeruli

        merged, merged_edge_only = self._merge_with_edge_flags(
            self.glomeruli, edge_flags, overlap_thresh=0.3
        )
        self.glomeruli = self._filter_small_edge_boxes(merged, merged_edge_only, mpp_base=mpp_base)
        return self.glomeruli

    def _merge_with_edge_flags(
        self,
        detections: List[Dict[str, Any]],
        edge_flags: List[bool] | None = None,
        overlap_thresh: float = 0.3,
    ) -> tuple[List[Dict[str, Any]], List[bool]]:
        """Merge overlapping detections using union-find (IoMin).
        A merged box is flagged edge-only iff every contributing detection touched a tile edge."""
        if not detections:
            return [], []

        n = len(detections)
        flags = edge_flags if (edge_flags is not None and len(edge_flags) == n) else [False] * n
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
                ix1 = max(a["x1"], b["x1"])
                iy1 = max(a["y1"], b["y1"])
                ix2 = min(a["x2"], b["x2"])
                iy2 = min(a["y2"], b["y2"])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if inter / min_area >= overlap_thresh:
                    union(i, j)

        groups: dict[int, list[tuple[Dict[str, Any], bool]]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append((detections[i], flags[i]))

        result: List[Dict[str, Any]] = []
        result_edge_only: List[bool] = []
        for group in groups.values():
            dets = [d for d, _ in group]
            best = max(dets, key=lambda d: d["conf"])
            result.append({
                "x1": min(d["x1"] for d in dets),
                "y1": min(d["y1"] for d in dets),
                "x2": max(d["x2"] for d in dets),
                "y2": max(d["y2"] for d in dets),
                "cls": best["cls"],
                "cls_name": best["cls_name"],
                "conf": best["conf"],
            })
            result_edge_only.append(all(is_edge for _, is_edge in group))
        return result, result_edge_only

    def _filter_small_edge_boxes(
        self,
        detections: List[Dict[str, Any]],
        edge_only_flags: List[bool],
        mpp_base: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Drop tiny or severely elongated boxes that only ever appeared truncated on a tile edge."""
        if not detections:
            return []

        min_dim_um = float(getattr(settings, "GLOMERULI_EDGE_MIN_DIM_UM", 70.0))
        max_ar = float(getattr(settings, "GLOMERULI_EDGE_MAX_ASPECT_RATIO", 2.0))
        if mpp_base > 0:
            min_dim_px = min_dim_um / mpp_base
        else:
            min_dim_px = float(getattr(settings, "GLOMERULI_EDGE_MIN_DIM_PX", 580.0))

        kept: List[Dict[str, Any]] = []
        dropped = 0
        for det, is_edge_only in zip(detections, edge_only_flags):
            if is_edge_only:
                w = max(0, det["x2"] - det["x1"])
                h = max(0, det["y2"] - det["y1"])
                short_side = min(w, h)
                long_side = max(w, h)
                ar = long_side / max(1, short_side)
                if short_side < min_dim_px or ar > max_ar:
                    dropped += 1
                    continue
            kept.append(det)

        if dropped:
            logger.info(
                f"GlomeruliProcessor: filtered {dropped} small/truncated edge box(es) "
                f"(min_dim={min_dim_px:.0f}px, max_ar={max_ar:.1f})"
            )
        return kept

    def simple_global_merge(self, detections, overlap_thresh=0.3):
        """Merge overlapping detections using union-find (IoMin).
        If >= overlap_thresh of the smaller box is covered -> merge."""
        merged, _ = self._merge_with_edge_flags(detections, None, overlap_thresh=overlap_thresh)
        return merged
