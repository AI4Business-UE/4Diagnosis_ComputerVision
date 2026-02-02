from django.shortcuts import render
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt
import json
from django.http import JsonResponse
import logging
import tempfile
from django.conf import settings
from pathlib import Path
import os
import uuid
import shutil

import openslide

from .source.tissue_length_processor import TissueLengthProcessor
from .source.fibrosis_processor import FibrosisProcessor
from .source.converter_tiff import SlideProcessor, save_result

logger = logging.getLogger(__name__)

@csrf_exempt
def convert(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    try:
        files = request.FILES.getlist("files")
        if not files:
            logger.warning("Convert attempt without files.")
            return JsonResponse({"error": "No files uploaded"}, status=400)

        job_id = str(uuid.uuid4())
        logger.info(f"Starting convert job: {job_id}")
        slides_root = Path(settings.BASE_DIR) / "slides"
        slides_root.mkdir(exist_ok=True)

        job_dir = slides_root / job_id
        job_dir.mkdir()

        mrxs_path = None
        data_dir = None

        # 1️⃣ zapisz MRXS
        for f in files:
            name = Path(f.name).name
            if name.lower().endswith(".mrxs"):
                mrxs_path = job_dir / name
                with open(mrxs_path, "wb") as out:
                    for chunk in f.chunks():
                        out.write(chunk)

        if not mrxs_path:
            logger.error(f"MRXS missing for job: {job_id}")
            return JsonResponse({"error": "MRXS missing"}, status=400)

        data_dir = job_dir / mrxs_path.stem
        data_dir.mkdir()

        for f in files:
            name = Path(f.name).name
            if not name.lower().endswith(".mrxs"):
                target = data_dir / name
                with open(target, "wb") as out:
                    for chunk in f.chunks():
                        out.write(chunk)

        for f in data_dir.iterdir():
            if f.name.lower() == "index.dat" and f.name != "Index.dat":
                f.rename(data_dir / "Index.dat")

        if not (data_dir / "Index.dat").exists():
            return JsonResponse({"error": "Index.dat missing"}, status=400)

        if not list(data_dir.glob("Data*.dat")):
            return JsonResponse({"error": "Data*.dat missing"}, status=400)

        if not list(data_dir.glob("*.ini")):
            return JsonResponse({"error": "Slidedat.ini missing"}, status=400)
        
        logger.info(f"Before processor for job: {job_id}")

        processor = SlideProcessor(
            slide_path=str(mrxs_path),
            level=0,              # pełna rozdzielczość
            tile_size=1024,       # bezpieczne dla RAM
            threshold=10,         # próg tła
            use_associated="auto" # fallback
        )
        logger.info(f"After processor initialization for job: {job_id}")

        result_img = processor.process()

        if result_img is None:
            logger.error(f"TIFF conversion returned None for job: {job_id}")
            return JsonResponse({"error": "TIFF conversion failed"}, status=500)
        

        tiff_path = job_dir / f"{mrxs_path.stem}.tiff"

        if not save_result(result_img, str(tiff_path)):
            logger.error(f"TIFF save failed for job: {job_id}")
            return JsonResponse({"error": "TIFF save failed"}, status=500)
        logger.info(f"Convert job success: {job_id}")
        return JsonResponse({
            "status": "ok",
            "job_id": job_id,
            "tiff": str(tiff_path)
    })

    except Exception as e:
        job_info = job_id if 'job_id' in locals() else 'unknown'
        logger.error(f"Critical error in convert for job {job_info}: {str(e)}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def analyze(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body)
        job_id = data.get("job_id")

        if not job_id:
            logger.warning("Analyze request missing job_id")
            return JsonResponse({"error": "job_id missing"}, status=400)

        slides_root = Path(settings.BASE_DIR) / "slides"
        job_dir = slides_root / job_id

        if not job_dir.exists():
            logger.warning(f"Analyze job not found: {job_id}")
            return JsonResponse({"error": "Job not found"}, status=404)

        # 🔎 SZUKAMY TIFF
        tiff_files = list(job_dir.glob("*.tiff"))
        if not tiff_files:
            logger.error(f"TIFF not found in job dir: {job_id}")
            return JsonResponse({"error": "TIFF not found"}, status=404)

        tiff_path = tiff_files[0]

        logger.info(f"Starting analysis for job: {job_id}, tiff: {tiff_path.name}")
        
        # Tissue length analysis
        processor = TissueLengthProcessor(str(tiff_path))
        result = processor.process_image()
        logger.info(f"Tissue length analysis finished for job: {job_id}")
        
        # Fibrosis analysis
        fibrosis_processor = FibrosisProcessor(str(tiff_path))
        fibrosis_result = fibrosis_processor.process_image(visualize=True)
        logger.info(f"Fibrosis analysis finished for job: {job_id}")

        return JsonResponse({
            "job_id": job_id,
            "tiff": str(tiff_path),
            "length": result.get("length"),
            "fibrosis_percent": fibrosis_result.get("fibrosis_ratio"),
            "image_path": result.get("image_path"),
            "fibrosis_image_path": fibrosis_result.get("image_path"),
            "error": result.get("error") or fibrosis_result.get("error"),
        })

    except Exception as e:
        job_info = job_id if 'job_id' in locals() else 'unknown'
        logger.error(f"Critical error in analyze for job {job_info}: {str(e)}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)

