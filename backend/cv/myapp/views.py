import json
import logging
import shutil
from pathlib import Path
from urllib.parse import quote

from django.conf import settings
from django.http import FileResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .source.slide_converter import SlideConverter
from .source.processed_image import ProcessedImage

logger = logging.getLogger(__name__)

@csrf_exempt
def select_folder(request):
    """Clear 'slides' and 'result_analyze' directories if they are not empty."""
    if request.method != "DELETE":
        return JsonResponse({"error": "Only DELETE method allowed"}, status=405)

    try:
        base_dir = Path(settings.BASE_DIR)
        slides_dir = base_dir / "slides"
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
    """Receive slide files via POST, convert to TIFF, return job_id and download URL."""
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



@csrf_exempt
def get_tiff(request, job_id):
    """Serve a TIFF file by job_id as a binary stream."""
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


@csrf_exempt
def get_result_image(request, job_id, image_name):
    """Serve analysis result images with path traversal protection."""
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

        tiff_path = get_tiff_path_detect_glomerule(job_id)

        processor = ProcessedImage(str(tiff_path))
        count = processor.count_glomeruli()

        slides_root = Path(settings.BASE_DIR) / "slides"
        job_dir = slides_root / job_id
        image_path = next(job_dir.glob("*_origin_detect_glomeruli.jpg"), None)

        if image_path is None:
            return JsonResponse({"error": "Glomeruli image not found"}, status=404)

        logger.info(f"Glomeruli count for job_id={job_id}: {count}")

        return JsonResponse({
            "job_id": job_id,
            "count": count,
            "image_url": f"/api/result-image/{job_id}/{quote(image_path.name)}/"
        })

    except Exception as e:
        logger.error(f"Glomeruli count error: {str(e)}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)



def get_tiff_path(job_id):
    """Resolve the converted TIFF path for a given job_id."""
    slides_root = Path(settings.BASE_DIR) / "slides"
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
    slides_root = Path(settings.BASE_DIR) / "slides"
    job_dir = slides_root / job_id

    if not job_dir.exists():
        raise FileNotFoundError("Job not found")

    detect_files = list(job_dir.glob("*_origin_detect.tiff"))

    if not detect_files:
        raise FileNotFoundError("Origin detect TIFF not found")

    return detect_files[0]
