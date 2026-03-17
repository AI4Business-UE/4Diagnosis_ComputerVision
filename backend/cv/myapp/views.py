from django.shortcuts import render
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt
import json
from django.http import JsonResponse
from django.http import FileResponse
import logging
import tempfile
from django.conf import settings
from pathlib import Path


from .source.SlideConverter import SlideConverter
from .source.ProcessedImage import ProcessedImage

logger = logging.getLogger(__name__)

@csrf_exempt
def select_folder(request):
    pass



@csrf_exempt
def convert(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    try:
        files = request.FILES.getlist("files")
        logger.info("FILES RECEIVED:", len(files))

        if not files:
            return JsonResponse({"error": "No files uploaded"}, status=400)

        job_id, tiff_path = SlideConverter.convert_to_tiff(
            files,
            settings.BASE_DIR,
        )

        return JsonResponse({
            "status": "ok",
            "job_id": job_id,
            "tiff": str(tiff_path),
            "tiff_url": f"/api/tiff/{job_id}/"
        })

    except Exception as e:

        logger.error(f"Convert error: {str(e)}", exc_info=True)

        return JsonResponse({
            "status": "error",
            "error": str(e)
        }, status=500)
    

def get_tiff_path(job_id):
    slides_root = Path(settings.BASE_DIR) / "slides"
    job_dir = slides_root / job_id

    if not job_dir.exists():
        raise FileNotFoundError("Job not found")

    tiff_files = list(job_dir.glob("*.tiff"))

    if not tiff_files:
        raise FileNotFoundError("TIFF not found")

    return tiff_files[0]


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


@csrf_exempt
def get_result_image(request, image_name):
    if request.method != "GET":
        return JsonResponse({"error": "GET only"}, status=405)

    if not image_name.lower().endswith((".tiff", ".tif")):
        return JsonResponse({"error": "Unsupported image format"}, status=400)

    result_dir = (Path(settings.BASE_DIR) / "cv" / "result_analyze").resolve()
    image_path = (result_dir / image_name).resolve()

    # Prevent path traversal outside the result directory.
    if result_dir not in image_path.parents and image_path != result_dir:
        return JsonResponse({"error": "Invalid image path"}, status=400)

    if not image_path.exists():
        return JsonResponse({"error": "Result image not found"}, status=404)

    return FileResponse(open(image_path, "rb"), content_type="image/tiff")


@csrf_exempt
def fibrosis(request):
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
def length(request):
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







# @csrf_exempt

# def analyze(request):
#     if request.method != "POST":
#         return JsonResponse({"error": "Only POST allowed"}, status=405)

#     try:
#         data = json.loads(request.body)
#         job_id = data.get("job_id")

#         if not job_id:
#             logger.warning("Analyze request missing job_id")
#             return JsonResponse({"error": "job_id missing"}, status=400)

#         slides_root = Path(settings.BASE_DIR) / "slides"
#         job_dir = slides_root / job_id

#         if not job_dir.exists():
#             logger.warning(f"Analyze job not found: {job_id}")
#             return JsonResponse({"error": "Job not found"}, status=404)

#         tiff_files = list(job_dir.glob("*.tiff"))
#         if not tiff_files:
#             logger.error(f"TIFF not found in job dir: {job_id}")
#             return JsonResponse({"error": "TIFF not found"}, status=404)

#         tiff_path = tiff_files[0]

#         logger.info(f"Starting analysis for job: {job_id}, tiff: {tiff_path.name}")
#         processor = TissueLengthProcessor(str(tiff_path))
#         result = processor.process_image()
#         logger.info(f"Analysis finished for job: {job_id}")

#         return JsonResponse({
#             "job_id": job_id,
#             "tiff": str(tiff_path),
#             "length": result.get("length"),
#             "fibrosis_percent": None,
#             "image_path": result.get("image_path"),
#             "error": result.get("error"),
#         })

#     except Exception as e:
#         job_info = job_id if 'job_id' in locals() else 'unknown'
#         logger.error(f"Critical error in analyze for job {job_info}: {str(e)}", exc_info=True)
#         return JsonResponse({"error": str(e)}, status=500)

