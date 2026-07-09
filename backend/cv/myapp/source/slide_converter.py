import logging
import uuid
from pathlib import Path
from .converter_tiff import SlideProcessor, save_result
from .mask import generate_mask
from django.conf import settings
import json
import openslide

from PIL import Image

logger = logging.getLogger(__name__)

class SlideConverter:

    @staticmethod
    def convert_to_tiff(files, base_dir, user_lvl=settings.TIFF_USER_LVL):

        job_id = str(uuid.uuid4())

        slides_root = Path(settings.SLIDES_DIR)
        slides_root.mkdir(exist_ok=True)

        job_dir = slides_root / job_id
        job_dir.mkdir()

        mrxs_path = None

        # Save MRXS file
        for f in files:
            name = Path(f.name).name
            if name.lower().endswith(".mrxs"):

                mrxs_path = job_dir / name

                with open(mrxs_path, "wb") as out:
                    for chunk in f.chunks():
                        out.write(chunk)

        if not mrxs_path:
            raise Exception("MRXS missing")

        # Slide data folder
        data_dir = job_dir / mrxs_path.stem
        data_dir.mkdir()

        # Save DATA / INI files
        for f in files:

            name = Path(f.name).name

            if not name.lower().endswith(".mrxs"):

                target = data_dir / name

                with open(target, "wb") as out:
                    for chunk in f.chunks():
                        out.write(chunk)

        # Fix Index.dat casing (common case-sensitivity issue on Linux)
        for f in data_dir.iterdir():
            if f.name.lower() == "index.dat" and f.name != "Index.dat":
                f.rename(data_dir / "Index.dat")

        # Validate MRXS structure
        if not (data_dir / "Index.dat").exists():
            raise Exception("Index.dat missing")

        if not list(data_dir.glob("Data*.dat")):
            raise Exception("Data*.dat missing")

        if not list(data_dir.glob("*.ini")):
            raise Exception("Slidedat.ini missing")

        # Convert to TIFF
        processor = SlideProcessor(
            slide_path=str(mrxs_path),
            level=user_lvl,
            tile_size=settings.TIFF_TILE_SIZE,
            threshold=settings.TIFF_THRESHOLD,
            use_associated="auto"
        )

        result_img = processor.process()

        if result_img is None:
            raise Exception("TIFF conversion failed")

        tiff_path = job_dir / f"{mrxs_path.stem}.tiff"

        if not save_result(result_img, str(tiff_path)):
            raise Exception("TIFF save failed")

        try:
            with openslide.OpenSlide(str(mrxs_path)) as slide:
                source_mpp_x = float(slide.properties.get("openslide.mpp-x"))
                source_mpp_y = float(slide.properties.get("openslide.mpp-y"))
                downsample = float(slide.level_downsamples[user_lvl])

            calibration = {
                "source": "mrxs",
                "source_path": str(mrxs_path.name),
                "source_mpp_x": source_mpp_x,
                "source_mpp_y": source_mpp_y,
                "level": user_lvl,
                "downsample": downsample,
                "mpp_x": source_mpp_x * downsample,
                "mpp_y": source_mpp_y * downsample,
                "unit": "um_per_pixel",
            }

            calibration_path = tiff_path.with_name(tiff_path.stem + "_calibration.json")
            calibration_path.write_text(
                json.dumps(calibration, indent=2),
                encoding="utf-8"
            )

        except Exception as e:
            logger.warning(f"Calibration metadata save failed: {e}")

        # Generate tissue mask automatically
        try:
            mask_result = generate_mask(str(tiff_path), mode="all", visualize=False, save_mask=True, save_preview=True)
            mask_preview_path = mask_result.get("preview_path")
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
