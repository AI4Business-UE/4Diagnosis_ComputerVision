import logging
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
import openslide
from django.conf import settings

from .processors.converter_tiff import SlideProcessor, save_result
from .processors.mask import compute_min_component_area_px, extract_components, generate_mask, load_mask_for_image
from .processors.metadata import (
    CalibrationData, ScanInfo, SliceItem, SlicesInfo, SlideMetadata,
)
from .processors.slice_grouper import SliceGrouper

logger = logging.getLogger(__name__)


class SlideConverter:

    @staticmethod
    def convert_to_tiff(
        files,
        base_dir,
        user_lvl: int | None = None,
        enable_slice_grouping: bool | None = None,
    ):
        job_id = str(uuid.uuid4())

        slides_root = Path(settings.SLIDES_DIR)
        slides_root.mkdir(exist_ok=True)

        job_dir = slides_root / job_id
        job_dir.mkdir()

        requested_lvl = settings.TIFF_USER_LVL if user_lvl is None else int(user_lvl)
        enable_slicing = (
            getattr(settings, "ENABLE_SLICE_GROUPING", True)
            if enable_slice_grouping is None
            else bool(enable_slice_grouping)
        )

        mrxs_path = SlideConverter._save_uploaded_files(files, job_dir)
        effective_lvl = SlideConverter._resolve_effective_level(mrxs_path, requested_lvl)
        tiff_path, min_x, min_y = SlideConverter._convert_to_tiff(mrxs_path, job_dir, effective_lvl)

        build_result = SlideConverter._build_metadata(
            mrxs_path,
            tiff_path,
            job_dir,
            effective_lvl,
            min_x=min_x,
            min_y=min_y,
            enable_slice_grouping=enable_slicing,
        )
        metadata = build_result[0]

        metadata_path = job_dir / f"{mrxs_path.stem}.json"
        metadata.save(metadata_path)

        cleaned_path, origin_detect_path = SlideConverter._generate_previews(
            tiff_path, job_dir, mrxs_path, build_result
        )

        return job_id, tiff_path, cleaned_path, origin_detect_path, metadata

    @staticmethod
    def _resolve_effective_level(mrxs_path: Path, requested_lvl: int) -> int:
        try:
            with openslide.OpenSlide(str(mrxs_path)) as slide:
                max_lvl = max(0, slide.level_count - 1)
                effective = max(0, min(int(requested_lvl), max_lvl))
                if effective != requested_lvl:
                    logger.warning(
                        f"Requested level {requested_lvl} out of range (max={max_lvl}); using level {effective}"
                    )
                return effective
        except Exception as e:
            logger.warning(f"Could not inspect OpenSlide level_count: {e}")
            return max(0, int(requested_lvl))

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
    def _convert_to_tiff(mrxs_path: Path, job_dir: Path, user_lvl: int) -> tuple[Path, int, int]:
        """Run MRXS -> TIFF conversion; return (tiff_path, min_x, min_y)."""
        processor = SlideProcessor(
            slide_path=str(mrxs_path),
            level=user_lvl,
            tile_size=settings.TIFF_TILE_SIZE,
            threshold=settings.TIFF_THRESHOLD,
            use_associated="auto",
        )
        result_img = processor.process()
        if result_img is None:
            raise RuntimeError("Nie wykryto tkanki na skanie (konwersja TIFF nie zwróciła obrazu)")

        tiff_path = job_dir / f"{mrxs_path.stem}.tiff"
        if not save_result(result_img, str(tiff_path)):
            raise RuntimeError("TIFF save failed")

        return tiff_path, processor.min_x, processor.min_y

    @staticmethod
    def _build_full_slide_item(
        tiff_w: int,
        tiff_h: int,
        min_x: int,
        min_y: int,
        ds: float,
        mask_bool: np.ndarray | None = None,
    ) -> SliceItem:
        area_px = int(mask_bool.sum()) if mask_bool is not None else (tiff_w * tiff_h)
        return SliceItem(
            slice_id=0,
            is_representative=True,
            bbox_tiff=[0, 0, tiff_w, tiff_h],
            bbox_level0=[
                int(min_x * ds),
                int(min_y * ds),
                int(tiff_w * ds),
                int(tiff_h * ds),
            ],
            area_tiff_px=area_px,
        )

    @staticmethod
    def _to_slice_items(items_raw: list[dict], min_x: int, min_y: int, ds: float) -> list[SliceItem]:
        slice_items = []
        for item in items_raw:
            x, y, w, h = item["bbox_tiff"]
            slice_items.append(SliceItem(
                slice_id=item["slice_id"],
                is_representative=item["is_representative"],
                bbox_tiff=[x, y, w, h],
                bbox_level0=[
                    int((x + min_x) * ds),
                    int((y + min_y) * ds),
                    int(w * ds),
                    int(h * ds),
                ],
                area_tiff_px=item["area_tiff_px"],
            ))
        return slice_items

    @staticmethod
    def _build_metadata(
        mrxs_path: Path,
        tiff_path: Path,
        job_dir: Path,
        user_lvl: int,
        min_x: int = 0,
        min_y: int = 0,
        enable_slice_grouping: bool = True,
    ):
        """
        Read calibration from openslide, compute scale-aware tissue mask,
        group into slices (when enabled), and return populated SlideMetadata.
        """
        # -- Calibration from openslide --
        calibration = CalibrationData(
            source_mpp_x=0.0, source_mpp_y=0.0,
            level=user_lvl, downsample=float(2 ** user_lvl),
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

        mask_result = None
        img_bgr = cv2.imread(str(tiff_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Cannot read converted TIFF: {tiff_path}")
        tiff_h, tiff_w = img_bgr.shape[:2]

        min_component_area = compute_min_component_area_px(
            mpp_x=calibration.mpp_x,
            mpp_y=calibration.mpp_y,
            downsample=calibration.downsample,
        )
        logger.info(
            f"Scale-aware mask cleaning: level={calibration.level}, ds={calibration.downsample:.1f}, "
            f"mpp=({calibration.mpp_x:.3f}, {calibration.mpp_y:.3f}) -> min_component_area={min_component_area} px"
        )

        try:
            mask_result = generate_mask(
                str(tiff_path),
                mode="all",
                visualize=False,
                save_mask=True,
                save_preview=False,
                min_component_area=min_component_area,
            )
        except Exception as e:
            logger.warning(f"Mask generation failed — using full-image fallback: {e}")

        # -- Slice grouping --
        ds = calibration.downsample
        mask_bool = mask_result["mask"] if mask_result is not None else None
        full_item = SlideConverter._build_full_slide_item(tiff_w, tiff_h, min_x, min_y, ds, mask_bool)

        slices_info = SlicesInfo(
            representative_slice_id=0,
            items=[full_item],
            grouping_enabled=enable_slice_grouping,
            fallback_used=False,
            fallback_prob=0.0,
            user_ungrouped=False,
            auto_representative_slice_id=0,
            auto_items=[full_item],
        )

        if enable_slice_grouping and mask_result is not None:
            try:
                grouper = SliceGrouper()
                slices_dict = grouper.group(img_bgr, mask_result.get("components", []))
                slice_items = SlideConverter._to_slice_items(slices_dict["items"], min_x, min_y, ds)
                if slice_items:
                    repr_id = slices_dict["representative_slice_id"]
                    slices_info = SlicesInfo(
                        representative_slice_id=repr_id,
                        items=slice_items,
                        grouping_enabled=True,
                        fallback_used=bool(slices_dict.get("fallback_used", False)),
                        fallback_prob=float(slices_dict.get("fallback_prob", 0.0)),
                        user_ungrouped=False,
                        auto_representative_slice_id=repr_id,
                        auto_items=list(slice_items),
                    )
            except Exception as e:
                logger.warning(f"Slice grouping failed — using full-image fallback: {e}")

        return (
            SlideMetadata(
                source="mrxs",
                source_path=mrxs_path.name,
                calibration=calibration,
                scan=ScanInfo(
                    tiff_shape=[tiff_h, tiff_w],
                    mrxs_level0_shape=mrxs_level0_shape,
                    crop_offset_x=min_x,
                    crop_offset_y=min_y,
                ),
                slices=slices_info,
            ),
            img_bgr,
            mask_result,
        )

    @staticmethod
    def _generate_previews(
        tiff_path: Path,
        job_dir: Path,
        mrxs_path: Path,
        build_result,
    ) -> tuple[Path, Path]:
        metadata, img_bgr, mask_result = build_result
        stem = mrxs_path.stem
        cleaned_path = job_dir / f"{stem}_cleaned.tiff"
        origin_detect_path = job_dir / f"{stem}_origin_detect.tiff"

        cleaned = img_bgr.copy()
        if mask_result is not None:
            cleaned[~mask_result["mask"]] = (255, 255, 255)
        cv2.imwrite(str(cleaned_path), cleaned)
        logger.info(f"Cleaned preview saved -> {cleaned_path.name}")

        summary = metadata.slices.to_summary()
        repr_item = metadata.get_representative()
        if summary["is_grouped"] and repr_item is not None:
            x, y, w, h = repr_item.bbox_tiff
            crop = cleaned[y:y + h, x:x + w].copy()
            cv2.imwrite(str(origin_detect_path), crop)
            logger.info(f"Grouped representative preview saved -> {origin_detect_path.name}")
        else:
            shutil.copy2(cleaned_path, origin_detect_path)
            logger.info(f"Full cleaned preview copied -> {origin_detect_path.name}")

        return cleaned_path, origin_detect_path

    @staticmethod
    def set_grouping(job_dir: Path, grouped: bool) -> dict:
        meta_path = SlideMetadata.get_metadata_path(job_dir)
        metadata = SlideMetadata.load(job_dir)

        stem = meta_path.stem
        tiff_path = job_dir / f"{stem}.tiff"
        cleaned_path = job_dir / f"{stem}_cleaned.tiff"
        origin_detect_path = job_dir / f"{stem}_origin_detect.tiff"

        if not tiff_path.exists():
            raise FileNotFoundError(f"Converted TIFF not found: {tiff_path.name}")

        img_bgr = cv2.imread(str(tiff_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Cannot read TIFF: {tiff_path}")
        tiff_h, tiff_w = img_bgr.shape[:2]

        mask_bool = load_mask_for_image(str(tiff_path))
        if not cleaned_path.exists():
            cleaned = img_bgr.copy()
            if mask_bool is not None:
                cleaned[~mask_bool] = (255, 255, 255)
            cv2.imwrite(str(cleaned_path), cleaned)

        ds = metadata.calibration.downsample
        min_x = metadata.scan.crop_offset_x
        min_y = metadata.scan.crop_offset_y

        if not grouped:
            if not metadata.slices.auto_items and len(metadata.slices.items) > 1:
                metadata.slices.auto_items = list(metadata.slices.items)
                metadata.slices.auto_representative_slice_id = metadata.slices.representative_slice_id

            full_item = SlideConverter._build_full_slide_item(tiff_w, tiff_h, min_x, min_y, ds, mask_bool)
            metadata.slices.items = [full_item]
            metadata.slices.representative_slice_id = 0
            metadata.slices.user_ungrouped = True

            shutil.copy2(cleaned_path, origin_detect_path)
            logger.info(f"Slide {job_dir.name}: user ungrouped -> full cleaned slide")
        else:
            if len(metadata.slices.auto_items) > 1:
                metadata.slices.items = list(metadata.slices.auto_items)
                metadata.slices.representative_slice_id = (
                    metadata.slices.auto_representative_slice_id
                    if metadata.slices.auto_representative_slice_id is not None
                    else len(metadata.slices.items) // 2
                )
                metadata.slices.grouping_enabled = True
                metadata.slices.user_ungrouped = False
            else:
                min_area = compute_min_component_area_px(
                    mpp_x=metadata.calibration.mpp_x,
                    mpp_y=metadata.calibration.mpp_y,
                    downsample=ds,
                )
                components = (
                    extract_components(mask_bool.astype(np.uint8), min_area)
                    if mask_bool is not None
                    else []
                )
                grouper = SliceGrouper()
                slices_dict = grouper.group(img_bgr, components)
                slice_items = SlideConverter._to_slice_items(slices_dict["items"], min_x, min_y, ds)
                if slice_items:
                    repr_id = slices_dict["representative_slice_id"]
                    metadata.slices.items = slice_items
                    metadata.slices.representative_slice_id = repr_id
                    metadata.slices.auto_items = list(slice_items)
                    metadata.slices.auto_representative_slice_id = repr_id
                    metadata.slices.fallback_used = bool(slices_dict.get("fallback_used", False))
                    metadata.slices.fallback_prob = float(slices_dict.get("fallback_prob", 0.0))
                metadata.slices.grouping_enabled = True
                metadata.slices.user_ungrouped = False

            summary = metadata.slices.to_summary()
            repr_item = metadata.get_representative()
            if summary["is_grouped"] and repr_item is not None:
                cleaned = cv2.imread(str(cleaned_path), cv2.IMREAD_COLOR)
                x, y, w, h = repr_item.bbox_tiff
                cv2.imwrite(str(origin_detect_path), cleaned[y:y + h, x:x + w].copy())
            else:
                shutil.copy2(cleaned_path, origin_detect_path)
            logger.info(f"Slide {job_dir.name}: grouping restored ({len(metadata.slices.items)} slice(s))")

        metadata.save(meta_path)

        for pattern in (
            "*_fibrosis.tiff",
            "*_length.tiff",
            "*_repr_crop*.tiff",
            "*_fibrosis_crop*.tiff",
            "*_length_crop*.tiff",
            "*_slice*_crop*.tiff",
            "glomeruli.json",
        ):
            for stale in job_dir.glob(pattern):
                try:
                    stale.unlink()
                except OSError:
                    pass

        return {
            "cleaned_path": cleaned_path,
            "origin_detect_path": origin_detect_path,
            "slices_summary": metadata.slices.to_summary(),
        }
