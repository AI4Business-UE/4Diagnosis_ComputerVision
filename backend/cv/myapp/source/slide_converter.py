import uuid
from pathlib import Path
from .converter_tiff import SlideProcessor, save_result
from .mask import generate_mask

from PIL import Image
import cv2
import numpy as np
import logging

# configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def crop_all_images(tiff_path, mask_path, mask_preview_path, margin=100):
    logger.info("Starting global cropping process")
    
    # Load the mask file
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logger.error("Failed to load the mask file!")
        return

    _, binary_mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary_mask)
    
    if coords is None:
        logger.error("Mask file is completely empty!")
        return
        
    # Calculate common coordinates for all files
    x, y, w, h = cv2.boundingRect(coords)
    height, width = mask.shape
    
    x_start = max(0, x - margin)
    y_start = max(0, y - margin)
    x_end = min(width, x + w + margin)
    y_end = min(height, y + h + margin)
    
    logger.info(f"Tissue detected! Cropping to coordinates: Y({y_start}:{y_end}), X({x_start}:{x_end})")

    # Helper function for cropping and overwriting
    def crop_and_save(img_path):
        if not img_path: 
            return
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is not None:
            cropped = img[y_start:y_end, x_start:x_end]
            cv2.imwrite(str(img_path), cropped)
            logger.debug(f"Cropped file saved: {img_path}")

    crop_and_save(tiff_path)
    crop_and_save(mask_path)
    crop_and_save(mask_preview_path)
    logger.info("Global cropping process finished")


class SlideConverter:

    @staticmethod
    def convert_to_tiff(files, base_dir, user_lvl=5):
        job_id = str(uuid.uuid4())

        slides_root = Path(base_dir) / "slides"
        slides_root.mkdir(exist_ok=True)

        job_dir = slides_root / job_id
        job_dir.mkdir()

        mrxs_path = None

        # Save MRXS
        logger.info("Saving MRXS file")
        for f in files:
            name = Path(f.name).name
            if name.lower().endswith(".mrxs"):

                mrxs_path = job_dir / name

                with open(mrxs_path, "wb") as out:
                    for chunk in f.chunks():
                        out.write(chunk)

        if not mrxs_path:
            raise Exception("MRXS missing")

        data_dir = job_dir / mrxs_path.stem
        data_dir.mkdir()

        for f in files:

            name = Path(f.name).name

            if not name.lower().endswith(".mrxs"):

                target = data_dir / name

                with open(target, "wb") as out:
                    for chunk in f.chunks():
                        out.write(chunk)

        # Fix Index.dat
        logger.info("Fixing Index.dat file casing")
        for f in data_dir.iterdir():
            if f.name.lower() == "index.dat" and f.name != "Index.dat":
                f.rename(data_dir / "Index.dat")

        # Validate MRXS
        logger.info("Validating MRXS structure")
        if not (data_dir / "Index.dat").exists():
            raise Exception("Index.dat missing")

        if not list(data_dir.glob("Data*.dat")):
            raise Exception("Data*.dat missing")

        if not list(data_dir.glob("*.ini")):
            raise Exception("Slidedat.ini missing")

        # Convert to TIFF
        logger.info("Converting MRXS to TIFF format")
        processor = SlideProcessor(
            slide_path=str(mrxs_path),
            level=user_lvl,
            tile_size=1024,
            threshold=10,
            use_associated="auto"
        )

        result_img = processor.process()

        if result_img is None:
            raise Exception("TIFF conversion failed")

        tiff_path = job_dir / f"{mrxs_path.stem}.tiff"

        if not save_result(result_img, str(tiff_path)):
            raise Exception("TIFF save failed")
        
        # Generate tissue mask automatically
        try:
            mask_result = generate_mask(str(tiff_path), mode="all", visualize=False, save_mask=True, save_preview=True)
            mask_preview_path = mask_result.get("preview_path")
            mask_path = mask_result.get("mask_path")
            logger.info(f"Preview generated at: {mask_preview_path}")
            
            if mask_path:
                crop_all_images(tiff_path, mask_path, mask_preview_path, margin=100)
            
        except Exception as e:
            logger.warning(f"Mask generation failed: {str(e)}")
            mask_preview_path = None

        origin_detect_path = None
        if mask_preview_path:
            origin_detect_path = job_dir / f"{mrxs_path.stem}_origin_detect.tiff"
            try:
                with Image.open(str(mask_preview_path)) as im:
                    im.convert("RGB").save(str(origin_detect_path), format="TIFF")
            except Exception as e:
                raise Exception(f"Origin detect TIFF save failed: {e}")

        return job_id, tiff_path, mask_preview_path, origin_detect_path