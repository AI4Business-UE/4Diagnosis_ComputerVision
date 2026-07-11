"""
SSE streaming for glomeruli detection.

Supports two modes (settings.SLICE_MODE):
  "one-slice"  — stream on representative slice only
  "all-slices" — stream live on representative; analyse other slices in
                 background threads; merge + grid after completion
"""
import json
import logging
import queue
import threading
from pathlib import Path

import cv2
import numpy as np
import openslide
from django.conf import settings

from .processors.glomeruli_processor import GlomeruliProcessor
from .processors.metadata import SlideMetadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers (used by views.py)
# ---------------------------------------------------------------------------

def get_tiff_path(job_id: str) -> Path:
    """Resolve the converted TIFF path for a given job_id."""
    job_dir = Path(settings.SLIDES_DIR) / job_id
    if not job_dir.exists():
        raise FileNotFoundError(f"Job not found: {job_id}")
    mrxs_files = list(job_dir.glob("*.mrxs"))
    if not mrxs_files:
        raise FileNotFoundError("Source .mrxs not found")
    tiff_path = mrxs_files[0].with_suffix(".tiff")
    if not tiff_path.exists():
        raise FileNotFoundError(f"TIFF not found: {tiff_path.name}")
    return tiff_path


def get_tiff_path_detect_glomeruli(job_id: str) -> Path:
    """Resolve the origin_detect TIFF (representative slice preview)."""
    job_dir = Path(settings.SLIDES_DIR) / job_id
    if not job_dir.exists():
        raise FileNotFoundError(f"Job not found: {job_id}")
    detect_files = list(job_dir.glob("*_origin_detect.tiff"))
    if not detect_files:
        raise FileNotFoundError("Origin detect TIFF not found")
    return detect_files[0]


# ---------------------------------------------------------------------------
# Cross-slice aggregation helpers
# ---------------------------------------------------------------------------

def _merge_cross_slice(
    per_slice_results: list[list[dict]],
    representative_idx: int,
    conf_warn_diff: float = settings.SLICE_GLOM_CONF_WARN_DIFF,
) -> list[dict]:
    """
    For each glomerulus on the representative slice, find matching ones on
    other slices (using GlomeruliProcessor.boxes_correspond), compute
    average confidence, and assign a consistency status.
    """
    n_slices = len(per_slice_results)
    if n_slices == 0:
        return []

    repr_glom = per_slice_results[representative_idx] if representative_idx < n_slices else []
    other_slices = [
        g for i, g in enumerate(per_slice_results) if i != representative_idx
    ]

    merged = []
    for g in repr_glom:
        confs = [g["conf"]]
        for other in other_slices:
            match = next(
                (o for o in other if GlomeruliProcessor.boxes_correspond(g, o)),
                None,
            )
            confs.append(match["conf"] if match else 0.0)

        conf_avg = sum(confs) / n_slices
        max_diff = max(confs) - min(confs)
        status = "inconsistent" if max_diff >= conf_warn_diff else "consistent"

        merged.append({
            **g,
            "conf": round(conf_avg, 4),
            "conf_per_slice": [round(c, 4) for c in confs],
            "status": status,
        })

    return merged


def _build_glomeruli_grid(
    job_dir: Path,
    tiff_path: Path,
    slices_meta,            # SlicesInfo dataclass
    per_slice_results: list[list[dict]],
) -> str | None:
    """
    Build a horizontal grid image of all slices with annotated glomeruli.
    Green bbox = consistent, Orange bbox = inconsistent.
    Returns path to saved grid image or None on failure.
    """
    try:
        img_bgr = cv2.imread(str(tiff_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None

        crops = []
        for item in slices_meta.items:
            sid = item.slice_id
            x, y, w, h = item.bbox_tiff
            crop = img_bgr[y:y+h, x:x+w].copy()

            slice_glom = per_slice_results[sid] if sid < len(per_slice_results) else []
            for g in slice_glom:
                gx1 = g["x1"] - x; gy1 = g["y1"] - y
                gx2 = g["x2"] - x; gy2 = g["y2"] - y
                color = (0, 200, 0) if g.get("status", "consistent") == "consistent" else (0, 140, 255)
                cv2.rectangle(crop, (gx1, gy1), (gx2, gy2), color, 3)
                label = f"{g.get('cls_name', '')} {g['conf']:.2f}"
                cv2.putText(crop, label, (gx1, max(gy1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            crops.append(crop)

        if not crops:
            return None

        target_h = min(c.shape[0] for c in crops)
        resized = [
            cv2.resize(c, (max(1, int(c.shape[1] * target_h / c.shape[0])), target_h))
            for c in crops
        ]

        total_w = sum(c.shape[1] for c in resized)
        bar_h = 40
        legend = np.ones((bar_h, total_w, 3), dtype=np.uint8) * 240
        cv2.putText(legend, "Consistent",   (10,  28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150,   0), 2)
        cv2.rectangle(legend, (150, 8), (180, 32), (0, 200,   0), -1)
        cv2.putText(legend, "Inconsistent", (200, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 200), 2)
        cv2.rectangle(legend, (360, 8), (390, 32), (0, 140, 255), -1)

        full = np.vstack([legend, np.hstack(resized)])
        grid_path = job_dir / "glom_grid.jpg"
        cv2.imwrite(str(grid_path), full, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return str(grid_path)

    except Exception as e:
        logger.warning(f"Glomeruli grid generation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main streaming generator
# ---------------------------------------------------------------------------

def generate_glomeruli_stream(job_id: str):
    """
    SSE generator for glomeruli detection.
    Yields JSON-encoded data: events for tiles, glomeruli batches, and done.
    """
    try:
        tiff_path = get_tiff_path(job_id)
        job_dir = tiff_path.parent
        mrxs_path = next(job_dir.glob("*.mrxs"), None)
        if mrxs_path is None:
            raise FileNotFoundError("No .mrxs in job dir")

        metadata = SlideMetadata.load(job_dir)
        slices_meta = metadata.slices

        with openslide.OpenSlide(str(mrxs_path)) as slide:
            slide_w, slide_h = slide.level_dimensions[0]

        mask_path = job_dir / f"{tiff_path.stem}_mask.tiff"
        slice_mode = settings.SLICE_MODE
        slice_items = slices_meta.items
        repr_id = slices_meta.representative_slice_id

        def make_processor(item):
            bbox_l0 = item.bbox_level0 if item else None
            return GlomeruliProcessor(
                path_mrxs=str(mrxs_path),
                model_path=str(settings.MODEL_PATH),
                mask_path=str(mask_path) if mask_path.exists() else None,
                scan_bbox=tuple(bbox_l0) if bbox_l0 else None,
            )

        # -------------------------------------------------------------------
        # ONE-SLICE mode
        # -------------------------------------------------------------------
        if slice_mode == "one-slice" or not slice_items:
            repr_item = next((s for s in slice_items if s.slice_id == repr_id), None)
            processor = make_processor(repr_item)

            yield f"data: {json.dumps({'slide_info': {'w': slide_w, 'h': slide_h, 'conf': processor.conf}})}\n\n"

            q: queue.Queue = queue.Queue()
            sent_count = [0]
            tile_buffer: list = []
            TILE_BATCH = 20

            def on_batch(current_all):
                new = current_all[sent_count[0]:]
                sent_count[0] = len(current_all)
                if new:
                    q.put({"__glomeruli__": new})
                if tile_buffer:
                    q.put({"__tiles__": list(tile_buffer)})
                    tile_buffer.clear()

            def on_tile(x, y, w, h, is_tissue):
                tile_buffer.append({"x": x, "y": y, "w": w, "h": h, "tissue": is_tissue})
                if len(tile_buffer) >= TILE_BATCH:
                    q.put({"__tiles__": list(tile_buffer)})
                    tile_buffer.clear()

            def run():
                try:
                    processor.detect_glomeruli(on_batch=on_batch, on_tile=on_tile)
                except Exception as e:
                    q.put({"__error__": str(e)})
                finally:
                    if tile_buffer:
                        q.put({"__tiles__": list(tile_buffer)})
                        tile_buffer.clear()
                    q.put(None)

            threading.Thread(target=run, daemon=True).start()

            while True:
                item = q.get()
                if item is None:
                    break
                if "__error__" in item:
                    yield f"data: {json.dumps({'error': item['__error__']})}\n\n"
                    return
                if "__glomeruli__" in item:
                    yield f"data: {json.dumps({'glomeruli': item['__glomeruli__']})}\n\n"
                elif "__tiles__" in item:
                    yield f"data: {json.dumps({'tiles': item['__tiles__']})}\n\n"

            final = processor.glomeruli
            try:
                (job_dir / "glomeruli.json").write_text(json.dumps(final), encoding="utf-8")
            except Exception:
                pass
            yield f"data: {json.dumps({'done': True, 'count': len(final), 'final_glomeruli': final})}\n\n"
            return

        # -------------------------------------------------------------------
        # ALL-SLICES mode
        # -------------------------------------------------------------------
        sorted_items = sorted(slice_items, key=lambda s: s.slice_id)
        n_slices = len(sorted_items)

        repr_item = next((s for s in sorted_items if s.slice_id == repr_id), sorted_items[0])
        other_items = [s for s in sorted_items if s.slice_id != repr_id]

        repr_processor = make_processor(repr_item)
        other_processors = [make_processor(it) for it in other_items]

        yield f"data: {json.dumps({'slide_info': {'w': slide_w, 'h': slide_h, 'conf': repr_processor.conf, 'slice_mode': 'all-slices', 'n_slices': n_slices}})}\n\n"

        q: queue.Queue = queue.Queue()
        sent_count = [0]
        tile_buffer: list = []
        TILE_BATCH = 20

        per_slice_results: list[list[dict]] = [[] for _ in range(n_slices)]

        def on_batch_repr(current_all):
            new = current_all[sent_count[0]:]
            sent_count[0] = len(current_all)
            if new:
                q.put({"__glomeruli__": new})
            if tile_buffer:
                q.put({"__tiles__": list(tile_buffer)})
                tile_buffer.clear()

        def on_tile_repr(x, y, w, h, is_tissue):
            tile_buffer.append({"x": x, "y": y, "w": w, "h": h, "tissue": is_tissue})
            if len(tile_buffer) >= TILE_BATCH:
                q.put({"__tiles__": list(tile_buffer)})
                tile_buffer.clear()

        def run_repr():
            try:
                repr_processor.detect_glomeruli(on_batch=on_batch_repr, on_tile=on_tile_repr)
                per_slice_results[repr_item.slice_id] = repr_processor.glomeruli
            except Exception as e:
                q.put({"__error__": str(e)})
            finally:
                if tile_buffer:
                    q.put({"__tiles__": list(tile_buffer)})
                    tile_buffer.clear()
                q.put(None)

        def run_other(proc, it):
            try:
                proc.detect_glomeruli()
                per_slice_results[it.slice_id] = proc.glomeruli
            except Exception as e:
                logger.warning(f"Slice {it.slice_id} detection failed: {e}")

        other_threads = [
            threading.Thread(target=run_other, args=(p, it), daemon=True)
            for p, it in zip(other_processors, other_items)
        ]
        repr_thread = threading.Thread(target=run_repr, daemon=True)

        for t in other_threads:
            t.start()
        repr_thread.start()

        while True:
            item = q.get()
            if item is None:
                break
            if "__error__" in item:
                yield f"data: {json.dumps({'error': item['__error__']})}\n\n"
                return
            if "__glomeruli__" in item:
                yield f"data: {json.dumps({'glomeruli': item['__glomeruli__']})}\n\n"
            elif "__tiles__" in item:
                yield f"data: {json.dumps({'tiles': item['__tiles__']})}\n\n"

        for t in other_threads:
            t.join(timeout=600)

        merged = _merge_cross_slice(per_slice_results, representative_idx=repr_id)

        grid_path = _build_glomeruli_grid(job_dir, tiff_path, slices_meta, per_slice_results)
        grid_url = f"/api/result-image/{job_id}/glom_grid.jpg/" if grid_path else None

        try:
            (job_dir / "glomeruli.json").write_text(json.dumps(merged), encoding="utf-8")
        except Exception:
            pass

        yield f"data: {json.dumps({'done': True, 'count': len(merged), 'final_glomeruli': merged, 'glom_grid_url': grid_url})}\n\n"

    except Exception as e:
        logger.error(f"Stream error for job {job_id}: {e}", exc_info=True)
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
