import json
import logging
import queue
import threading
from pathlib import Path

import openslide
from django.conf import settings

from .processors.glomeruli_processor import GlomeruliProcessor
from .processors.metadata import SlideMetadata

logger = logging.getLogger(__name__)


def resolve_job_dir(job_id: str) -> Path:
    """
    Safely resolve slides/<job_id> with path traversal protection.
    Raises FileNotFoundError if job_id escapes SLIDES_DIR or does not exist.
    """
    if not job_id or not isinstance(job_id, str):
        raise FileNotFoundError("Invalid job_id")
    slides_root = Path(settings.SLIDES_DIR).resolve()
    job_dir = (slides_root / job_id).resolve()
    if slides_root not in job_dir.parents or not job_dir.is_dir():
        raise FileNotFoundError(f"Job not found: {job_id}")
    return job_dir


def get_tiff_path(job_id: str) -> Path:
    """Resolve the converted TIFF path for a given job_id."""
    job_dir = resolve_job_dir(job_id)
    mrxs_files = list(job_dir.glob("*.mrxs"))
    if not mrxs_files:
        raise FileNotFoundError("Source .mrxs not found")
    tiff_path = mrxs_files[0].with_suffix(".tiff")
    if not tiff_path.exists():
        raise FileNotFoundError(f"TIFF not found: {tiff_path.name}")
    return tiff_path


def get_glomeruli_view_frame(
    metadata: SlideMetadata,
    slide_w: int,
    slide_h: int,
) -> tuple[int, int, int, int]:
    """
    Return (offset_x, offset_y, view_w, view_h) in level-0 coordinates matching
    the active base preview image (representative crop when grouped, or full TIFF).
    Shared by SSE streaming, synchronous /glomeruli/count/, and glomeruli.json.
    """
    repr_item = metadata.get_representative()
    is_grouped = metadata.slices.to_summary()["is_grouped"]
    crop_x = metadata.scan.crop_offset_x
    crop_y = metadata.scan.crop_offset_y
    ds = metadata.calibration.downsample

    if is_grouped and repr_item is not None:
        rx, ry, rw, rh = repr_item.bbox_level0
        return rx, ry, rw, rh

    if (
        metadata.scan.tiff_shape
        and len(metadata.scan.tiff_shape) == 2
        and all(d > 0 for d in metadata.scan.tiff_shape)
    ):
        tiff_h, tiff_w = metadata.scan.tiff_shape
        return int(crop_x * ds), int(crop_y * ds), int(tiff_w * ds), int(tiff_h * ds)

    return 0, 0, slide_w, slide_h


def shift_glomeruli_coords(gloms: list[dict], offset_x: int, offset_y: int) -> list[dict]:
    """Shift level-0 glomeruli bounding boxes into the active view's local coordinate space."""
    return [
        {
            **g,
            "x1": g["x1"] - offset_x,
            "y1": g["y1"] - offset_y,
            "x2": g["x2"] - offset_x,
            "y2": g["y2"] - offset_y,
        }
        for g in gloms
    ]


def shift_tile_coords(tiles: list[dict], offset_x: int, offset_y: int) -> list[dict]:
    """Shift level-0 tile rectangles into the active view's local coordinate space."""
    return [
        {
            **t,
            "x": t["x"] - offset_x,
            "y": t["y"] - offset_y,
        }
        for t in tiles
    ]


def generate_glomeruli_stream(job_id: str):
    """
    SSE generator for glomeruli detection on the representative slice.
    Yields JSON-encoded data: events for slide_info, tiles, glomeruli batches, and done.
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
        repr_item = metadata.get_representative()

        crop_x = metadata.scan.crop_offset_x
        crop_y = metadata.scan.crop_offset_y
        ds = metadata.calibration.downsample
        offset_x, offset_y, repr_w, repr_h = get_glomeruli_view_frame(metadata, slide_w, slide_h)

        bbox_l0 = repr_item.bbox_level0 if repr_item is not None else None
        processor = GlomeruliProcessor(
            path_mrxs=str(mrxs_path),
            model_path=str(settings.MODEL_PATH),
            mask_path=str(mask_path) if mask_path.exists() else None,
            scan_bbox=tuple(bbox_l0) if bbox_l0 else None,
            crop_offset_x=crop_x,
            crop_offset_y=crop_y,
            ds=ds,
        )

        yield f"data: {json.dumps({'slide_info': {'w': repr_w, 'h': repr_h, 'conf': processor.conf, 'n_slices': len(slices_meta.items)}})}\n\n"

        q: queue.Queue = queue.Queue()
        cancel_event = threading.Event()
        sent_count = [0]
        tile_buffer: list = []
        TILE_BATCH = 20

        def on_batch(current_all):
            if cancel_event.is_set():
                return
            new = current_all[sent_count[0]:]
            sent_count[0] = len(current_all)
            if new:
                q.put({"__glomeruli__": new})
            if tile_buffer:
                q.put({"__tiles__": list(tile_buffer)})
                tile_buffer.clear()

        def on_tile(x, y, w, h, is_tissue):
            if cancel_event.is_set():
                return
            tile_buffer.append({"x": x, "y": y, "w": w, "h": h, "tissue": is_tissue})
            if len(tile_buffer) >= TILE_BATCH:
                q.put({"__tiles__": list(tile_buffer)})
                tile_buffer.clear()

        def run():
            try:
                processor.detect_glomeruli(
                    on_batch=on_batch,
                    on_tile=on_tile,
                    cancel_event=cancel_event,
                )
            except Exception as e:
                logger.error(f"Glomeruli detection thread error for {job_id}: {e}", exc_info=True)
                if not cancel_event.is_set():
                    q.put({"__error__": str(e)})
            finally:
                if tile_buffer and not cancel_event.is_set():
                    q.put({"__tiles__": list(tile_buffer)})
                    tile_buffer.clear()
                q.put(None)

        threading.Thread(target=run, daemon=True).start()

        try:
            while True:
                item = q.get()
                if item is None:
                    break
                if "__error__" in item:
                    yield f"data: {json.dumps({'error': item['__error__']})}\n\n"
                    return
                if "__glomeruli__" in item:
                    yield f"data: {json.dumps({'glomeruli': shift_glomeruli_coords(item['__glomeruli__'], offset_x, offset_y)})}\n\n"
                elif "__tiles__" in item:
                    yield f"data: {json.dumps({'tiles': shift_tile_coords(item['__tiles__'], offset_x, offset_y)})}\n\n"

            if not cancel_event.is_set():
                shifted_final = shift_glomeruli_coords(processor.glomeruli, offset_x, offset_y)
                try:
                    (job_dir / "glomeruli.json").write_text(json.dumps(shifted_final), encoding="utf-8")
                except Exception as e:
                    logger.warning(f"Could not save glomeruli.json for {job_id}: {e}")
                yield f"data: {json.dumps({'done': True, 'count': len(shifted_final), 'final_glomeruli': shifted_final})}\n\n"
        finally:
            cancel_event.set()

    except Exception as e:
        logger.error(f"Stream error for job {job_id}: {e}", exc_info=True)
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
