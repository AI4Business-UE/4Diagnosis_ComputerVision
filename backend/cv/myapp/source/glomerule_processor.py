import os
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "best_100.pt"

yolo_model = YOLO(str(MODEL_PATH))


def load_heavy_tiff(path: str):
    logger.info(f"[YOLO] Attempting to load TIFF: {path}")

    try:
        logger.info("[YOLO] 1. Trying OpenCV...")
        img = cv2.imread(path)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.warning(f"[YOLO] OpenCV failed: {e}")

    try:
        logger.info("[YOLO] 2. Trying Pillow...")
        img = Image.open(path)
        return np.array(img.convert("RGB"))
    except Exception as e:
        logger.warning(f"[YOLO] Pillow failed: {e}")

    try:
        logger.info("[YOLO] 3. Trying tifffile...")
        import tifffile
        return tifffile.imread(path)
    except Exception as e:
        logger.warning(f"[YOLO] tifffile failed: {e}")

    logger.error("[YOLO] All TIFF loading methods failed")
    return None


def process_tiff(tiff_path: str, model=yolo_model, conf: float = 0.15, patch_size: int = 1024):
    tiff_path = Path(tiff_path)
    job_dir = tiff_path.parent                # slides/<job_id>
    out_dir = job_dir / "glomeruli"           # slides/<job_id>/glomeruli
    out_dir.mkdir(exist_ok=True)

    logger.info(f"[YOLO] Starting TIFF analysis: {tiff_path}")
    logger.info(f"[YOLO] conf={conf}, patch_size={patch_size}")
    logger.info(f"[YOLO] Output directory: {out_dir}")

    large_image = load_heavy_tiff(str(tiff_path))

    if large_image is None:
        logger.error("[YOLO] Failed to load TIFF image")
        return {"error": "cannot load tiff", "found_count": 0, "images": []}

    h, w = large_image.shape[:2]
    logger.info(f"[YOLO] Image loaded, size: {w} x {h}")

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
            logger.debug(f"[YOLO] Patch #{total_patches} @ ({x},{y}) size={pw}x{ph}")

            results = model.predict(patch, conf=conf, verbose=False)
            n_boxes = len(results[0].boxes)
            logger.debug(f"[YOLO]   -> detected boxes: {n_boxes}")

            if n_boxes > 0:
                found_count += 1
                res_plotted = results[0].plot()
                filename = f"glomeruli_x{x}_y{y}.jpg"
                save_path = out_dir / filename
                cv2.imwrite(str(save_path), cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR))
                saved_paths.append(str(save_path))
                logger.info(f"[YOLO]   -> saved result: {save_path}")

    logger.info("=" * 40)
    logger.info(f"[YOLO] Finished. Processed patches: {total_patches}")
    logger.info(f"[YOLO] Patches with glomeruli: {found_count}")
    logger.info(f"[YOLO] Output directory: {out_dir}")

    return {"error": None, "found_count": found_count, "images": saved_paths}
