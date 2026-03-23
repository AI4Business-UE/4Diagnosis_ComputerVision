import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "best_100.pt"

yolo_model = YOLO(str(MODEL_PATH))


def load_heavy_tiff(path):
    print(f"Próba wczytania: {path}")
    try:
        print("1. Próbuję OpenCV...")
        img = cv2.imread(path)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"OpenCV nie dało rady: {e}")

    try:
        print("2. Próbuję PIL (Pillow)...")
        img = Image.open(path)
        return np.array(img.convert("RGB"))
    except Exception as e:
        print(f"PIL nie dało rady: {e}")

    try:
        print("3. Próbuję tifffile...")
        import tifffile
        return tifffile.imread(path)
    except Exception as e:
        print(f"Tifffile też nie dało rady: {e}")

    return None


def process_tiff(tiff_path, model=yolo_model, conf=0.15, patch_size=1024):
    tiff_path = Path(tiff_path)
    job_dir = tiff_path.parent                    # slides/<job_id>
    out_dir = job_dir / "glomeruli"               # slides/<job_id>/glomeruli
    out_dir.mkdir(exist_ok=True)

    print(f"[YOLO] Start analizy TIFF: {tiff_path}")
    print(f"[YOLO] conf={conf}, patch_size={patch_size}")
    print(f"[YOLO] Katalog wyników: {out_dir}")

    large_image = load_heavy_tiff(str(tiff_path))

    if large_image is None:
        print("[YOLO] ❌ NIE UDAŁO SIĘ WCZYTAĆ PLIKU")
        return {"error": "cannot load tiff", "found_count": 0, "images": []}

    h, w = large_image.shape[:2]
    print(f"[YOLO] Obraz wczytany, rozmiar: {w} x {h}")

    found_count = 0
    saved_paths: list[str] = []
    total_patches = 0

    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)
            patch = large_image[y:y_end, x:x_end]
            ph, pw = patch.shape[:2]
            if ph < 50 or pw < 50:
                continue

            total_patches += 1
            print(f"[YOLO] Patch #{total_patches} @ ({x},{y}) size={pw}x{ph}")

            results = model.predict(patch, conf=conf, verbose=False)
            n_boxes = len(results[0].boxes)
            print(f"[YOLO]   -> wykryto boxów: {n_boxes}")

            if n_boxes > 0:
                found_count += 1
                res_plotted = results[0].plot()
                filename = f"glomeruli_x{x}_y{y}.jpg"
                save_path = out_dir / filename
                cv2.imwrite(str(save_path), cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR))
                saved_paths.append(str(save_path))
                print(f"[YOLO]   -> zapisano wynik: {save_path}")

    print("=" * 40)
    print(f"[YOLO] ZAKOŃCZONO. Przetworzono patchy: {total_patches}")
    print(f"[YOLO] Patchy z kłębuszkami: {found_count}")
    print(f"[YOLO] Pliki wynikowe w folderze: {out_dir}")

    return {"error": None, "found_count": found_count, "images": saved_paths}


