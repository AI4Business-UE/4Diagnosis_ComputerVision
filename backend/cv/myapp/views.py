from django.shortcuts import render

# Create your views here.

from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt
import json
from django.http import JsonResponse
import tempfile
import os
from django.conf import settings
from pathlib import Path
import os
import uuid
import traceback
import shutil

import openslide

from .source.tissue_length_processor import TissueLengthProcessor
from .source.converter_tiff import SlideProcessor, save_result



@csrf_exempt
def import_slide(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    data = json.loads(request.body)
    source = data.get("source_path")

    if not source:
        return JsonResponse({"error": "Missing source_path"}, status=400)

    source_mrxs = Path(source)
    source_folder = source_mrxs.with_suffix("")

    if not source_mrxs.exists() or not source_folder.exists():
        return JsonResponse({"error": "Source MRXS or folder missing"}, status=404)

    slides_dir = Path(settings.BASE_DIR) / "slides"
    slides_dir.mkdir(exist_ok=True)

    dest_mrxs = slides_dir / source_mrxs.name
    dest_folder = slides_dir / source_folder.name

    if dest_mrxs.exists():
        return JsonResponse({"error": "Slide already imported"}, status=409)

    # 🔥 PRAWDZIWA KOPIA — filesystem → filesystem
    shutil.copy2(source_mrxs, dest_mrxs)
    shutil.copytree(source_folder, dest_folder)

    return JsonResponse({
        "status": "imported",
        "slide": dest_mrxs.name
    })



@csrf_exempt
def convert(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    try:
        files = request.FILES.getlist("files")
        if not files:
            return JsonResponse({"error": "No files uploaded"}, status=400)

        job_id = str(uuid.uuid4())
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
        print("Before proccessor)")
        processor = SlideProcessor(
            slide_path=str(mrxs_path),
            level=0,              # pełna rozdzielczość
            tile_size=1024,       # bezpieczne dla RAM
            threshold=10,         # próg tła
            use_associated="auto" # fallback
        )
        print("After proccesor")

        result_img = processor.process()

        if result_img is None:
            return JsonResponse({"error": "TIFF conversion failed"}, status=500)

        tiff_path = job_dir / f"{mrxs_path.stem}.tiff"

        if not save_result(result_img, str(tiff_path)):
            return JsonResponse({"error": "TIFF save failed"}, status=500)
        return JsonResponse({
            "status": "ok",
            "job_id": job_id,
            "tiff": str(tiff_path)
    })

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)




@csrf_exempt
def analyze(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body)
        job_id = data.get("job_id")

        if not job_id:
            return JsonResponse({"error": "job_id missing"}, status=400)

        slides_root = Path(settings.BASE_DIR) / "slides"
        job_dir = slides_root / job_id

        if not job_dir.exists():
            return JsonResponse({"error": "Job not found"}, status=404)

        # 🔎 SZUKAMY TIFF
        tiff_files = list(job_dir.glob("*.tiff"))
        if not tiff_files:
            return JsonResponse({"error": "TIFF not found"}, status=404)

        tiff_path = tiff_files[0]


        processor = TissueLengthProcessor(str(tiff_path))
        result = processor.process_image()

        return JsonResponse({
            "job_id": job_id,
            "tiff": str(tiff_path),
            "length": result.get("length"),
            "fibrosis_percent": None,
            "image_path": result.get("image_path"),
            "error": result.get("error"),
        })

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)

