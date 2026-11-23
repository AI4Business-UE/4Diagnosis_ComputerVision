import io
import json
import logging
import shutil
from pathlib import Path
from urllib.parse import quote

import cv2
from django.conf import settings
from django.http import FileResponse, HttpResponse, JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt

from .source.slide_converter import SlideConverter
from .source.processed_image import ProcessedImage
from .source.streaming_utils import generate_glomeruli_stream, get_tiff_path, get_tiff_path_detect_glomeruli

logger = logging.getLogger(__name__)

@csrf_exempt
def select_folder(request):
    if request.method != "DELETE":
        return JsonResponse({"error": "Only DELETE method allowed"}, status=405)

    try:
        slides_dir = Path(settings.SLIDES_DIR)
        cleared = False

        if slides_dir.exists() and any(slides_dir.iterdir()):
            for item in slides_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            cleared = True
            logger.info(f"Cleared directory: {slides_dir}")

        if cleared:
            return JsonResponse({"status": "ok", "message": f"Cleared folder: {slides_dir.name}"})
        
        return JsonResponse({"status": "ok", "message": "Folders already empty"})

    except Exception as e:
        logger.error(f"Error clearing folders: {str(e)}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)



@csrf_exempt
def convert(request):
    """Receive slide files via POST, convert to TIFF, return job_id and preview URLs."""
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    try:
        files = request.FILES.getlist("files")
        logger.info(f"FILES RECEIVED: {len(files)}")

        if not files:
            return JsonResponse({"error": "No files uploaded"}, status=400)

        presets = getattr(settings, "CONVERSION_LEVEL_PRESETS", {"S": 7, "M": 6, "L": 5, "XL": 4})
        preset_key = (request.POST.get("conversion_preset") or "").strip().upper()
        user_lvl = presets.get(preset_key)
        if user_lvl is None and request.POST.get("user_lvl") is not None:
            try:
                user_lvl = int(request.POST.get("user_lvl"))
            except (TypeError, ValueError):
                return JsonResponse({"error": "Invalid user_lvl"}, status=400)

        enable_grouping_raw = request.POST.get("enable_slice_grouping")
        enable_slice_grouping = None
        if enable_grouping_raw is not None:
            enable_slice_grouping = enable_grouping_raw.strip().lower() in ("true", "1", "yes", "on")

        cleaning_level = None
        cleaning_raw = request.POST.get("cleaning_level")
        if cleaning_raw is not None and cleaning_raw.strip() != "":
            try:
                cleaning_level = max(0, min(10, int(cleaning_raw)))
            except (TypeError, ValueError):
                return JsonResponse({"error": "Invalid cleaning_level"}, status=400)

        job_id, tiff_path, cleaned_path, origin_detect_path, metadata = SlideConverter.convert_to_tiff(
            files,
            settings.BASE_DIR,
            user_lvl=user_lvl,
            enable_slice_grouping=enable_slice_grouping,
            cleaning_level=cleaning_level,
        )

        cleaned_url = (
            f"/api/result-image/{job_id}/{quote(Path(cleaned_path).name)}/"
            if cleaned_path else None
        )
        origin_detect_url = (
            f"/api/result-image/{job_id}/{quote(Path(origin_detect_path).name)}/"
            if origin_detect_path else None
        )

        return JsonResponse({
            "status": "ok",
            "job_id": job_id,
            "tiff": str(tiff_path),
            "tiff_url": f"/api/tiff/{job_id}/",
            "cleaned_url": cleaned_url,
            "origin_detect_url": origin_detect_url,
            "conversion_level": metadata.calibration.level,
            "cleaning_level": metadata.slices.cleaning_level,
            "slices_summary": metadata.slices.to_summary(),
        })

    except Exception as e:
        logger.error(f"Convert error: {str(e)}", exc_info=True)
        return JsonResponse({
            "status": "error",
            "error": str(e)
        }, status=500)


@csrf_exempt
def update_slide_grouping(request, job_id):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body or "{}")
        if "grouped" not in data or not isinstance(data["grouped"], bool):
            return JsonResponse({"error": "Missing boolean 'grouped' field"}, status=400)

        slides_root = Path(settings.SLIDES_DIR).resolve()
        job_dir = (slides_root / job_id).resolve()
        if slides_root not in job_dir.parents or not job_dir.exists():
            return JsonResponse({"error": "Job not found"}, status=404)

        result = SlideConverter.set_grouping(job_dir, grouped=data["grouped"])

        return JsonResponse({
            "status": "ok",
            "job_id": job_id,
            "cleaned_url": f"/api/result-image/{job_id}/{quote(Path(result['cleaned_path']).name)}/",
            "origin_detect_url": f"/api/result-image/{job_id}/{quote(Path(result['origin_detect_path']).name)}/",
            "slices_summary": result["slices_summary"],
        })
    except FileNotFoundError as e:
        return JsonResponse({"error": str(e)}, status=404)
    except Exception as e:
        logger.error(f"Grouping toggle error for {job_id}: {str(e)}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)



PREVIEW_MAX_DIM = 8192
PREVIEW_LARGE_TIFF_BYTES = 20 * 1024 * 1024


def _serve_browser_safe_image(image_path: Path, prefer_preview: bool = False):
    lower_name = image_path.name.lower()
    is_tiff = lower_name.endswith((".tiff", ".tif"))

    if prefer_preview and is_tiff:
        try:
            file_size = image_path.stat().st_size
            if file_size >= PREVIEW_LARGE_TIFF_BYTES:
                img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if img is not None:
                    h, w = img.shape[:2]
                    max_side = max(h, w)
                    if max_side > PREVIEW_MAX_DIM:
                        scale = PREVIEW_MAX_DIM / max_side
                        img = cv2.resize(
                            img,
                            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                            interpolation=cv2.INTER_AREA,
                        )
                    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    if ok:
                        return HttpResponse(buf.tobytes(), content_type="image/jpeg")
        except Exception as e:
            logger.warning(f"JPEG preview fallback failed for {image_path.name}: {e}")

    if lower_name.endswith((".jpg", ".jpeg")):
        content_type = "image/jpeg"
    elif lower_name.endswith(".png"):
        content_type = "image/png"
    else:
        content_type = "image/tiff"

    return FileResponse(open(image_path, "rb"), content_type=content_type)


@csrf_exempt
def get_tiff(request, job_id):
    """Serve a TIFF file by job_id as a binary stream (or browser-safe JPEG preview when ?preview=1)."""
    if request.method != "GET":
        return JsonResponse({"error": "GET only"}, status=405)

    try:
        tiff_path = get_tiff_path(job_id)
        prefer_preview = request.GET.get("preview") in ("1", "true")
        return _serve_browser_safe_image(tiff_path, prefer_preview=prefer_preview)
    except FileNotFoundError as e:
        return JsonResponse({"error": str(e)}, status=404)
    except Exception as e:
        logger.error(f"TIFF fetch error: {str(e)}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def get_result_image(request, job_id, image_name):
    """Serve analysis result images with path traversal protection."""
    if request.method != "GET":
        return JsonResponse({"error": "GET only"}, status=405)

    lower_name = image_name.lower()
    if not lower_name.endswith((".tiff", ".tif", ".jpg", ".jpeg", ".png")):
        return JsonResponse({"error": "Unsupported image format"}, status=400)

    slides_root = Path(settings.SLIDES_DIR)
    job_dir = (slides_root / job_id).resolve()
    image_path = (job_dir / image_name).resolve()

    if job_dir not in image_path.parents and image_path != job_dir:
        return JsonResponse({"error": "Invalid image path"}, status=400)

    if not image_path.exists():
        return JsonResponse({"error": "Result image not found"}, status=404)

    prefer_preview = request.GET.get("preview") in ("1", "true")
    return _serve_browser_safe_image(image_path, prefer_preview=prefer_preview)


@csrf_exempt
def analyze_fibrosis_degree(request):
    """Run fibrosis analysis and return ratio, pixel counts, and result image."""
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body)
        job_id = data.get("job_id")

        if not job_id:
            return JsonResponse({"error": "job_id missing"}, status=400)

        threshold = data.get("threshold")
        if threshold is not None:
            try:
                threshold = max(0.0, min(1.0, float(threshold)))
            except (TypeError, ValueError):
                return JsonResponse({"error": "threshold must be a number between 0 and 1"}, status=400)

        tiff_path = get_tiff_path(job_id)

        logger.info(f"Fibrosis analysis started: {job_id} (threshold={threshold})")

        processor = ProcessedImage(str(tiff_path))

        result = processor.calculate_fibrosis_degree(threshold=threshold)

        return JsonResponse({
            "job_id": job_id,
            "fibrosis_ratio": result.get("fibrosis_ratio"),
            "fibrosis_ratio_avg": result.get("fibrosis_ratio_avg"),
            "fibrosis_ratio_per_slice": result.get("fibrosis_ratio_per_slice"),
            "fibrosis_warning": result.get("fibrosis_warning", False),
            "fibrotic_pixels": result.get("fibrotic_pixels"),
            "tissue_pixels": result.get("tissue_pixels"),
            "image_path": result.get("image_path"),
            "threshold": result.get("threshold"),
            "error": result.get("error"),
        })

    except Exception as e:
        logger.error(f"Fibrosis error: {str(e)}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)



@csrf_exempt
def measure_tissue_length(request):
    """Run tissue length measurement and return length and result image."""
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
def count_glomeruli(request):
    """Run glomeruli detection and return the count."""
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body)
        job_id = data.get("job_id")

        if not job_id:
            return JsonResponse({"error": "job_id missing"}, status=400)

        tiff_path = get_tiff_path_detect_glomeruli(job_id)

        processor = ProcessedImage(str(tiff_path))
        count = processor.count_glomeruli()

        slides_root = Path(settings.SLIDES_DIR)
        job_dir = slides_root / job_id
        image_path = next(job_dir.glob("*_origin_detect_glomeruli.jpg"), None)

        if image_path is None:
            return JsonResponse({"error": "Glomeruli image not found"}, status=404)

        logger.info(f"Glomeruli count for job_id={job_id}: {count}")

        return JsonResponse({
            "job_id": job_id,
            "count": count,
            "image_url": f"/api/result-image/{job_id}/{quote(image_path.name)}/",
        })

    except Exception as e:
        logger.error(f"Glomeruli count error: {str(e)}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)



@csrf_exempt
def detect_glomeruli_stream(request):
    """SSE endpoint — streams glomeruli detections batch by batch."""
    if request.method != "GET":
        return JsonResponse({"error": "GET only"}, status=405)

    job_id = request.GET.get("job_id")
    if not job_id:
        return JsonResponse({"error": "job_id missing"}, status=400)

    response = StreamingHttpResponse(generate_glomeruli_stream(job_id), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response
