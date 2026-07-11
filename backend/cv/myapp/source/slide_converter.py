import json
import logging
import uuid
from pathlib import Path

import cv2
import openslide
from django.conf import settings
from PIL import Image

from .processors.converter_tiff import SlideProcessor, save_result
from .processors.mask import generate_mask
from .processors.metadata import (
    CalibrationData, ScanInfo, SliceItem, SlicesInfo, SlideMetadata,
)
from .processors.slice_grouper import SliceGrouper

logger = logging.getLogger(__name__)


class SlideConverter:

    @staticmethod
    def convert_to_tiff(files, base_dir, user_lvl=settings.TIFF_USER_LVL):
        job_id = str(uuid.uuid4())

        slides_root = Path(settings.SLIDES_DIR)
        slides_root.mkdir(exist_ok=True)

        job_dir = slides_root / job_id
        job_dir.mkdir()

        mrxs_path = SlideConverter._save_uploaded_files(files, job_dir)
        tiff_path = SlideConverter._convert_to_tiff(mrxs_path, job_dir, user_lvl)

        build_result = SlideConverter._build_metadata(mrxs_path, tiff_path, job_dir, user_lvl)
        metadata = build_result[0]

        metadata_path = job_dir / f"{mrxs_path.stem}.json"
        metadata.save(metadata_path)

        origin_detect_path = SlideConverter._generate_preview(
            tiff_path, job_dir, mrxs_path, build_result
        )

        # origin_detect_path — representative slice crop; shown to the user as main preview
        return job_id, tiff_path, origin_detect_path

    @staticmethod
    def _save_uploaded_files(files, job_dir: Path) -> Path:
        """Save .mrxs + companion data files; return mrxs_path."""
        mrxs_path = None

        for f in files:
            name = Path(f.name).name
            if name.lower().endswith(".mrxs"):
                mrxs_path = job_dir / name
                with open(mrxs_path, "wb") as out:
                    for chunk in f.chunks():
                        out.write(chunk)

        if not mrxs_path:
            raise ValueError("No .mrxs file in upload")

        data_dir = job_dir / mrxs_path.stem
        data_dir.mkdir()

        for f in files:
            name = Path(f.name).name
            if not name.lower().endswith(".mrxs"):
                target = data_dir / name
                with open(target, "wb") as out:
                    for chunk in f.chunks():
                        out.write(chunk)

        # Fix Index.dat casing (case-sensitivity on Linux)
        for f in data_dir.iterdir():
            if f.name.lower() == "index.dat" and f.name != "Index.dat":
                f.rename(data_dir / "Index.dat")

        if not (data_dir / "Index.dat").exists():
            raise FileNotFoundError("Index.dat missing")
        if not list(data_dir.glob("Data*.dat")):
            raise FileNotFoundError("Data*.dat missing")
        if not list(data_dir.glob("*.ini")):
            raise FileNotFoundError("Slidedat.ini missing")

        return mrxs_path

    @staticmethod
    def _convert_to_tiff(mrxs_path: Path, job_dir: Path, user_lvl: int) -> Path:
        """Run MRXS -> TIFF conversion; return tiff_path."""
        processor = SlideProcessor(
            slide_path=str(mrxs_path),
            level=user_lvl,
            tile_size=settings.TIFF_TILE_SIZE,
            threshold=settings.TIFF_THRESHOLD,
            use_associated="auto",
        )
        result_img = processor.process()
        if result_img is None:
            raise RuntimeError("TIFF conversion failed")

        tiff_path = job_dir / f"{mrxs_path.stem}.tiff"
        if not save_result(result_img, str(tiff_path)):
            raise RuntimeError("TIFF save failed")

        return tiff_path

    @staticmethod
    def _build_metadata(
        mrxs_path: Path,
        tiff_path: Path,
        job_dir: Path,
        user_lvl: int,
    ) -> SlideMetadata:
        """
        Read calibration from openslide, detect tissue components,
        group into slices, and return a fully populated SlideMetadata.
        """
        # -- Calibration from openslide --
        calibration = CalibrationData(
            source_mpp_x=0.0, source_mpp_y=0.0,
            level=user_lvl, downsample=1.0,
            mpp_x=0.0, mpp_y=0.0,
        )
        mrxs_level0_shape = [0, 0]
        try:
            with openslide.OpenSlide(str(mrxs_path)) as slide:
                mpp_x = float(slide.properties.get("openslide.mpp-x", 0))
                mpp_y = float(slide.properties.get("openslide.mpp-y", 0))
                downsample = float(slide.level_downsamples[user_lvl])
                slide_w, slide_h = slide.level_dimensions[0]
                mrxs_level0_shape = [slide_h, slide_w]
            calibration = CalibrationData(
                source_mpp_x=mpp_x,
                source_mpp_y=mpp_y,
                level=user_lvl,
                downsample=downsample,
                mpp_x=mpp_x * downsample,
                mpp_y=mpp_y * downsample,
            )
        except Exception as e:
            logger.warning(f"Calibration read failed: {e}")

        # -- Tissue mask --
        mask_result = None
        mask_preview_path = None
        img_bgr = cv2.imread(str(tiff_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Cannot read converted TIFF: {tiff_path}")
        tiff_h, tiff_w = img_bgr.shape[:2]

        try:
            mask_result = generate_mask(
                str(tiff_path), mode="all",
                visualize=False, save_mask=True, save_preview=False,
            )
        except Exception as e:
            logger.warning(f"Mask generation failed — using full-image fallback: {e}")

        # -- Slice grouping --
        slices_info = SlicesInfo(
            representative_slice_id=0,
            items=[
                SliceItem(
                    slice_id=0, is_representative=True,
                    bbox_tiff=[0, 0, tiff_w, tiff_h],
                    bbox_level0=[0, 0, mrxs_level0_shape[1], mrxs_level0_shape[0]],
                    area_tiff_px=tiff_w * tiff_h,
                )
            ],
        )
        if mask_result is not None:
            try:
                grouper = SliceGrouper()
                slices_dict = grouper.group(img_bgr, mask_result.get("components", []))
                ds = calibration.downsample
                slice_items = []
                for item in slices_dict["items"]:
                    x, y, w, h = item["bbox_tiff"]
                    slice_items.append(SliceItem(
                        slice_id=item["slice_id"],
                        is_representative=item["is_representative"],
                        bbox_tiff=[x, y, w, h],
                        bbox_level0=[
                            int(x * ds), int(y * ds),
                            int(w * ds), int(h * ds),
                        ],
                        area_tiff_px=item["area_tiff_px"],
                    ))
                slices_info = SlicesInfo(
                    representative_slice_id=slices_dict["representative_slice_id"],
                    items=slice_items,
                )
            except Exception as e:
                logger.warning(f"Slice grouping failed — using full-image fallback: {e}")

        return (
            SlideMetadata(
                source="mrxs",
                source_path=mrxs_path.name,
                calibration=calibration,
                scan=ScanInfo(tiff_shape=[tiff_h, tiff_w], mrxs_level0_shape=mrxs_level0_shape),
                slices=slices_info,
            ),
            img_bgr,
            mask_result, 
            mask_preview_path, 
        )

    @staticmethod
    def _generate_preview(
        tiff_path: Path,
        job_dir: Path,
        mrxs_path: Path,
        build_result,  
    ) -> Path | None:
        """
        Crop the representative slice, apply tissue mask (white BG),
        save as <stem>_origin_detect.tiff.

        Fallback chain:
          - Grouping OK  → crop of representative slice with mask applied
          - Mask failed  → crop of representative slice (raw TIFF region)
          - No slices    → full TIFF saved as origin_detect
        """
        metadata, img_bgr, mask_result, _ = build_result

        repr_item = metadata.get_representative()
        origin_detect_path = job_dir / f"{mrxs_path.stem}_origin_detect.tiff"

        if repr_item is None:
            logger.warning("No representative slice — saving full TIFF as preview")
            cv2.imwrite(str(origin_detect_path), img_bgr)
            return origin_detect_path

        x, y, w, h = repr_item.bbox_tiff
        crop = img_bgr[y:y+h, x:x+w].copy()

        if mask_result is not None:
            mask_bool = mask_result["mask"][y:y+h, x:x+w]
            crop[~mask_bool] = (255, 255, 255)

        cv2.imwrite(str(origin_detect_path), crop)
        logger.info(f"Preview saved -> {origin_detect_path.name}")
        return origin_detect_path
