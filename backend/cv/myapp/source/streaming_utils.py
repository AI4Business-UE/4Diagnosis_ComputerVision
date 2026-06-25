import json
import queue
import threading
from pathlib import Path
import openslide
from django.conf import settings
from .glomeruli_processor import GlomeruliProcessor

def get_tiff_path(job_id):
    """Resolve the converted TIFF path for a given job_id."""
    slides_root = Path(settings.SLIDES_DIR)
    job_dir = slides_root / job_id

    if not job_dir.exists():
        raise FileNotFoundError("Job not found")

    mrxs_files = list(job_dir.glob("*.mrxs"))
    if not mrxs_files:
        raise FileNotFoundError("Source .mrxs not found")

    tiff_path = mrxs_files[0].with_suffix(".tiff")
    if not tiff_path.exists():
        raise FileNotFoundError(f"TIFF not found: {tiff_path.name}")

    return tiff_path

def get_tiff_path_detect_glomerule(job_id):
    """Resolve the origin_detect TIFF path for glomeruli detection."""
    slides_root = Path(settings.SLIDES_DIR)
    job_dir = slides_root / job_id

    if not job_dir.exists():
        raise FileNotFoundError("Job not found")

    detect_files = list(job_dir.glob("*_origin_detect.tiff"))

    if not detect_files:
        raise FileNotFoundError("Origin detect TIFF not found")

    return detect_files[0]

def generate_glomeruli_stream(job_id):
    """
    Generator function that yields Server-Sent Events for glomeruli detection.
    """
    try:
        tiff_path = get_tiff_path(job_id)
        job_dir = tiff_path.parent
        mrxs_files = list(job_dir.glob("*.mrxs"))
        if not mrxs_files:
            yield f"data: {json.dumps({'error': 'Brak pliku .mrxs'})}\n\n"
            return

        _slide = openslide.OpenSlide(str(mrxs_files[0]))
        slide_w, slide_h = _slide.level_dimensions[0]
        _slide.close()

        mask_path = job_dir / f"{tiff_path.stem}_mask.tiff"
        processor = GlomeruliProcessor(
            path_mrxs=str(mrxs_files[0]),
            model_path=str(settings.MODEL_PATH),
            mask_path=str(mask_path) if mask_path.exists() else None,
        )

        yield f"data: {json.dumps({'slide_info': {'w': slide_w, 'h': slide_h, 'conf': processor.conf}})}\n\n"

        q = queue.Queue()
        sent_count = [0]
        tile_buffer = []
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
        glomeruli_json = job_dir / "glomeruli.json"
        try:
            glomeruli_json.write_text(json.dumps(final), encoding="utf-8")
        except Exception as _e:
            pass
        yield f"data: {json.dumps({'done': True, 'count': len(final), 'final_glomeruli': final})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
