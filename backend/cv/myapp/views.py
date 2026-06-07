from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt
import json
from django.http import JsonResponse, FileResponse, StreamingHttpResponse
import logging
from django.conf import settings
from pathlib import Path
from urllib.parse import quote

from .source.slide_converter import SlideConverter
from .source.processed_image import ProcessedImage

logger = logging.getLogger(__name__)

@csrf_exempt
def select_folder(request):
    pass


### Receives multiple slide files via POST, 
### converts them to single TIFF using SlideConverter, returns job_id and TIFF download URL.
@csrf_exempt
def convert(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    try:
        files = request.FILES.getlist("files")
        logger.info(f"FILES RECEIVED: {len(files)}")

        if not files:
            return JsonResponse({"error": "No files uploaded"}, status=400)

        job_id, tiff_path, mask_preview_path, origin_detect_path = SlideConverter.convert_to_tiff(
            files,
            settings.BASE_DIR,
        )

        mask_preview_url = None
        if mask_preview_path:
            mask_preview_filename = Path(mask_preview_path).name
            mask_preview_url = f"/api/result-image/{job_id}/{mask_preview_filename}/"

        origin_detect_url = None
        if origin_detect_path:
            origin_detect_filename = Path(origin_detect_path).name
            origin_detect_url = f"/api/result-image/{job_id}/{origin_detect_filename}/"

        return JsonResponse({
            "status": "ok",
            "job_id": job_id,
            "tiff": str(tiff_path),
            "tiff_url": f"/api/tiff/{job_id}/",
            "mask_preview_url": mask_preview_url,
            "origin_detect_url": origin_detect_url,
        })

    except Exception as e:
        logger.error(f"Convert error: {str(e)}", exc_info=True)
        return JsonResponse({
            "status": "error",
            "error": str(e)
        }, status=500)

    

### GET endpoint serving TIFF file by job_id as binary stream with image/tiff content type.
@csrf_exempt
def get_tiff(request, job_id):
    if request.method != "GET":
        return JsonResponse({"error": "GET only"}, status=405)

    try:
        tiff_path = get_tiff_path(job_id)
        return FileResponse(open(tiff_path, "rb"), content_type="image/tiff")
    except FileNotFoundError as e:
        return JsonResponse({"error": str(e)}, status=404)
    except Exception as e:
        logger.error(f"TIFF fetch error: {str(e)}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)

### GET endpoint serving analysis result images from /cv/result_analyze/ with path traversal protection.
@csrf_exempt
def get_result_image(request, job_id,image_name):
    if request.method != "GET":
        return JsonResponse({"error": "GET only"}, status=405)

    if not image_name.lower().endswith((".tiff", ".tif", ".jpg", "jpeg")):
        return JsonResponse({"error": "Unsupported image format"}, status=400)

    slides_root = Path(settings.BASE_DIR) / "slides"
    job_dir = (slides_root / job_id).resolve()
    image_path = (job_dir / image_name).resolve()

    if job_dir not in image_path.parents and image_path != job_dir:
        return JsonResponse({"error": "Invalid image path"}, status=400)

    if not image_path.exists():
        return JsonResponse({"error": "Result image not found"}, status=404)

    return FileResponse(open(image_path, "rb"), content_type="image/tiff")


### POST endpoint: loads TIFF by job_id, runs ProcessedImage.calculate_fibrosis_degree(), 
### returns ratio, pixel counts, result image
@csrf_exempt
def analyze_fibrosis_degree(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body)
        job_id = data.get("job_id")

        if not job_id:
            return JsonResponse({"error": "job_id missing"}, status=400)

        tiff_path = get_tiff_path(job_id)

        logger.info(f"Fibrosis analysis started: {job_id}")

        processor = ProcessedImage(str(tiff_path))

        result = processor.calculate_fibrosis_degree()

        return JsonResponse({
            "job_id": job_id,
            "fibrosis_ratio": result.get("fibrosis_ratio"),
            "fibrotic_pixels": result.get("fibrotic_pixels"),
            "tissue_pixels": result.get("tissue_pixels"),
            "image_path": result.get("image_path"),
            "error": result.get("error"),
        })

    except Exception as e:
        logger.error(f"Fibrosis error: {str(e)}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)


### POST endpoint: loads TIFF by job_id, runs ProcessedImage.calculate_tissue_length(), 
### returns length measurement and result image.
@csrf_exempt
def measure_tissue_length(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body)
        job_id = data.get("job_id")

        if not job_id:
            return JsonResponse({"error": "job_id missing"}, status=400)

        tiff_path = get_tiff_path(job_id)

        processor = ProcessedImage(str(tiff_path))

        result = processor.calculate_tissue_length()

        return JsonResponse({
            "job_id": job_id,
            "length": result.get("length"),
            "image_path": result.get("image_path"),
            "error": result.get("error"),
        })

    except Exception as e:
        logger.error(f"Length error: {str(e)}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def detect_glomeruli_stream(request):
    """SSE endpoint — wysyła kłębuszki do frontendu na bieżąco, batch po batchu."""
    if request.method != "GET":
        return JsonResponse({"error": "GET only"}, status=405)

    job_id = request.GET.get("job_id")
    if not job_id:
        return JsonResponse({"error": "job_id missing"}, status=400)

    def generate():
        import queue
        import threading
        import openslide
        from .source.glomeruli_processor import GlomeruliProcessor

        try:
            tiff_path = get_tiff_path(job_id)
            job_dir = tiff_path.parent
            mrxs_files = list(job_dir.glob("*.mrxs"))
            if not mrxs_files:
                yield f"data: {json.dumps({'error': 'Brak pliku .mrxs'})}\n\n"
                return

            # Wyślij wymiary slajdu — frontend potrzebuje ich do skalowania bboxów.
            # WAŻNE: slide_info zawiera też 'conf' (próg ufności modelu) — frontend
            # używa go jako minimum suwaka progu ufności. Jeśli zmienisz conf w
            # GlomeruliProcessor.__init__, zmień też domyślny stan confThresholds
            # w App.tsx (inicjalizowany na podstawie onSlideInfo).
            _slide = openslide.OpenSlide(str(mrxs_files[0]))
            slide_w, slide_h = _slide.level_dimensions[0]
            _slide.close()

            mask_path = tiff_path.parent / f"{tiff_path.stem}_mask.tiff"
            processor = GlomeruliProcessor(
                path_mrxs=str(mrxs_files[0]),
                model_path=str(ProcessedImage.MODEL_PATH),
                mask_path=str(mask_path) if mask_path.exists() else None,
            )

            yield f"data: {json.dumps({'slide_info': {'w': slide_w, 'h': slide_h, 'conf': processor.conf}})}\n\n"

            q = queue.Queue()
            sent_count = [0]
            tile_buffer = []
            TILE_BATCH = 20  # flush tile events co 20 kafelków

            def on_batch(current_all):
                new = current_all[sent_count[0]:]
                sent_count[0] = len(current_all)
                if new:
                    q.put({"__glomeruli__": new})
                # Flush remaining tile buffer with this batch
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
                    logger.error(f"Detection thread error: {e}", exc_info=True)
                    q.put({"__error__": str(e)})
                finally:
                    # Flush remaining tiles
                    if tile_buffer:
                        q.put({"__tiles__": list(tile_buffer)})
                        tile_buffer.clear()
                    q.put(None)

            threading.Thread(target=run, daemon=True).start()

            while True:
                item = q.get()
                if item is None:
                    break
                if isinstance(item, dict):
                    if "__error__" in item:
                        yield f"data: {json.dumps({'error': item['__error__']})}\n\n"
                        return
                    if "__glomeruli__" in item:
                        yield f"data: {json.dumps({'glomeruli': item['__glomeruli__']})}\n\n"
                    elif "__tiles__" in item:
                        yield f"data: {json.dumps({'tiles': item['__tiles__']})}\n\n"

            # processor.glomeruli jest już po simple_global_merge — wyślij jako finalna lista
            final = processor.glomeruli
            # Zapisz wyniki do pliku JSON, żeby endpoint eksportu mógł je odczytać
            glomeruli_json = tiff_path.parent / "glomeruli.json"
            try:
                glomeruli_json.write_text(json.dumps(final), encoding="utf-8")
            except Exception as _e:
                logger.warning(f"Cannot save glomeruli.json: {_e}")
            yield f"data: {json.dumps({'done': True, 'count': len(final), 'final_glomeruli': final})}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    response = StreamingHttpResponse(generate(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


@csrf_exempt
def count_glomeruli(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body)
        job_id = data.get("job_id")

        if not job_id:
            return JsonResponse({"error": "job_id missing"}, status=400)

        tiff_path = get_tiff_path(job_id)
        processor = ProcessedImage(str(tiff_path))
        glomeruli = processor.detect_glomeruli()

        logger.info(f"Glomeruli detected for job_id={job_id}: {len(glomeruli)}")

        return JsonResponse({
            "job_id": job_id,
            "count": len(glomeruli),
            "glomeruli": glomeruli,
        })

    except Exception as e:
        logger.error(f"Glomeruli count error: {str(e)}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)


### additional funciton - to get path for tiff
def get_tiff_path(job_id):
    slides_root = Path(settings.BASE_DIR) / "slides"
    job_dir = slides_root / job_id

    if not job_dir.exists():
        raise FileNotFoundError("Job not found")

    tiff_files = list(job_dir.glob("*.tiff"))

    if not tiff_files:
        raise FileNotFoundError("TIFF not found")
    return tiff_files[0]

