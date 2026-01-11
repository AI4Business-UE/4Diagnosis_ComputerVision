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
from .source.converter_tiff import SlideProcessor



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

        # 2️⃣ folder danych = STEM MRXS
        data_dir = job_dir / mrxs_path.stem
        data_dir.mkdir()

        # 3️⃣ zapisz pliki pomocnicze DO FOLDERU
        for f in files:
            name = Path(f.name).name
            if not name.lower().endswith(".mrxs"):
                target = data_dir / name
                with open(target, "wb") as out:
                    for chunk in f.chunks():
                        out.write(chunk)

        # 4️⃣ wymuś poprawne nazwy
        for f in data_dir.iterdir():
            if f.name.lower() == "index.dat" and f.name != "Index.dat":
                f.rename(data_dir / "Index.dat")

        # 5️⃣ walidacja
        if not (data_dir / "Index.dat").exists():
            return JsonResponse({"error": "Index.dat missing"}, status=400)

        if not list(data_dir.glob("Data*.dat")):
            return JsonResponse({"error": "Data*.dat missing"}, status=400)

        if not list(data_dir.glob("*.ini")):
            return JsonResponse({"error": "Slidedat.ini missing"}, status=400)

        # 6️⃣ TEST OPENSLIDE
        slide = openslide.OpenSlide(str(mrxs_path))
        slide.close()

        return JsonResponse({
            "status": "ok",
            "job_id": job_id,
            "mrxs": str(mrxs_path),
            "data_dir": str(data_dir)
        })

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)




@csrf_exempt
def analyze(request):
    print("FILES:", request.FILES)
    print("POST:", request.POST)
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    files = request.FILES.getlist("files")
    if not files:
        return JsonResponse({"error": "No files uploaded"}, status=400)

    results = []

    for f in files:
        # sprawdzamy rozszerzenie
        if not f.name.endswith(".mrxs"):
            continue  # pomijamy inne pliki

        # zapisujemy tymczasowo
        with tempfile.NamedTemporaryFile(delete=False, suffix=f.name) as tmp:
            for chunk in f.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        # przetwarzamy plik
        processor = TissueLengthProcessor(tmp_path)
        result = processor.process_image()
        results.append({
            "file_name": f.name,
            "length": result.get("length"),
            "image_path": result.get("image_path"),
            "error": result.get("error")
        })

        # usuwamy tymczasowy plik
        os.remove(tmp_path)

    if not results:
        return JsonResponse({"error": "No valid .mrxs files found"}, status=400)

    return JsonResponse(results, safe=False)

