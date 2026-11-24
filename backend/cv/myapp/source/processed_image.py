"""
ProcessedImage — thin orchestration facade.

All heavy computation lives in the processors/ package.
This class resolves paths, reads metadata, and delegates to processors.
"""
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import openslide
from django.conf import settings

from .processors.fibrosis_processor import FibrosisProcessor
from .processors.glomeruli_processor import GlomeruliProcessor
from .processors.mask import generate_mask, load_mask_for_image
from .processors.metadata import SlideMetadata
from .processors.tissue_length_processor import TissueLengthProcessor
from .streaming_utils import get_glomeruli_view_frame, shift_glomeruli_coords

logger = logging.getLogger(__name__)


class ProcessedImage:
    def __init__(self, path_tiff: str):
        self.path = Path(path_tiff)
        self.job_dir = self.path.parent          # slides/<job_id>
        self.mask_path = self.job_dir / f"{self.path.stem}_mask.tiff"

        self.glomeruli = None
        self.glomeruli_slide_info = None
        self.tissue_length = None

        self._metadata: SlideMetadata | None = None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> SlideMetadata | None:
        if self._metadata is None:
            try:
                self._metadata = SlideMetadata.load(self.job_dir)
            except FileNotFoundError:
                logger.warning(f"No metadata JSON in {self.job_dir} — using full-image fallback")
        return self._metadata

    def _repr_bbox(self) -> list[int] | None:
        """Return [x, y, w, h] (TIFF px) for the representative slice when grouped, else None (full slide)."""
        if self.metadata and self.metadata.slices.to_summary()["is_grouped"]:
            item = self.metadata.get_representative()
            return item.bbox_tiff if item else None
        return None

    def _repr_bbox_level0(self) -> list[int] | None:
        """Return [x, y, w, h] (level-0 px) for the representative slice, or None."""
        if self.metadata:
            item = self.metadata.get_representative()
            return item.bbox_level0 if item else None
        return None

    def _load_tiff_and_mask(self) -> tuple[np.ndarray, np.ndarray | None]:
        img_bgr = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Cannot read TIFF: {self.path}")
        mask_bool = load_mask_for_image(str(self.path))
        return img_bgr, mask_bool

    def _crop_tiff(self, bbox: list[int]):
        """Return (crop_bgr, mask_crop_bool) for a given [x, y, w, h] bbox."""
        img_bgr, mask_bool = self._load_tiff_and_mask()
        x, y, w, h = bbox
        crop = img_bgr[y:y + h, x:x + w].copy()
        mask_crop = mask_bool[y:y + h, x:x + w] if mask_bool is not None else None
        return crop, mask_crop

    # ------------------------------------------------------------------
    # Tissue length
    # ------------------------------------------------------------------

    def calculate_tissue_length(self):
        """Measure tissue length on the representative slice (or full TIFF)."""
        bbox = self._repr_bbox()
        if bbox is None:
            processor = TissueLengthProcessor(str(self.path), output_dir=self.job_dir)
            result = processor.process_image()
        else:
            crop, mask_crop = self._crop_tiff(bbox)
            crop_path = self.job_dir / f"{self.path.stem}_length_crop.tiff"
            cv2.imwrite(str(crop_path), crop)

            if mask_crop is not None:
                crop_mask_path = self.job_dir / f"{self.path.stem}_length_crop_mask.tiff"
                cv2.imwrite(str(crop_mask_path), mask_crop.astype(np.uint8) * 255)

            processor = TissueLengthProcessor(str(crop_path), output_dir=self.job_dir)
            result = processor.process_image()

        self.tissue_length = result.get("length")
        return result

    # ------------------------------------------------------------------
    # Fibrosis
    # ------------------------------------------------------------------

    def calculate_fibrosis_degree(self, threshold: float | None = None):
        """
        Analyse fibrosis per settings.FIBROSIS_SLICE_MODE.

        Returns dict:
          fibrosis_ratio, fibrotic_pixels, tissue_pixels, image_path, threshold, error,
          + (all-slices) fibrosis_ratio_avg, fibrosis_ratio_per_slice, fibrosis_warning
        """
        items = self.metadata.slices.items if self.metadata else []
        repr_id = self.metadata.slices.representative_slice_id if self.metadata else 0
        threshold = threshold if threshold is not None else settings.FIBROSIS_THRESHOLD

        fibrosis_mode = getattr(settings, "FIBROSIS_SLICE_MODE", "one-slice")
        if fibrosis_mode == "one-slice" or not items:
            return self._fibrosis_one_slice(repr_id, items, threshold)
        return self._fibrosis_all_slices(repr_id, items, threshold)

    def _fibrosis_one_slice(self, repr_id, items, threshold):
        bbox = self._repr_bbox()

        if bbox is None:
            processor = FibrosisProcessor(str(self.path), threshold=threshold, output_dir=self.job_dir)
            return processor.process_image()

        crop, mask_crop = self._crop_tiff(bbox)
        crop_path = self.job_dir / f"{self.path.stem}_fibrosis_crop.tiff"
        processor = FibrosisProcessor(str(crop_path), threshold=threshold, output_dir=self.job_dir)
        result = processor.compute_fibrosis_ratio(
            str(crop_path),
            mask_bool=mask_crop,
            threshold=threshold,
            save_overlay=True,
            overlay_suffix="_fibrosis.tiff",
            img_bgr=crop,
        )
        return {
            "fibrosis_ratio": result.get("fibrosis_ratio"),
            "fibrotic_pixels": result.get("fibrotic_pixels"),
            "tissue_pixels": result.get("tissue_pixels"),
            "image_path": result.get("overlay_path"),
            "threshold": result.get("threshold"),
            "error": result.get("error"),
        }

    def _fibrosis_all_slices(self, repr_id, items, threshold):
        # Load full TIFF and mask ONCE instead of once per slice
        img_bgr, mask_bool = self._load_tiff_and_mask()
        ratios = []
        repr_ratio = None
        repr_result = None

        for item in items:
            try:
                x, y, w, h = item.bbox_tiff
                crop = img_bgr[y:y + h, x:x + w]
                mask_crop = mask_bool[y:y + h, x:x + w] if mask_bool is not None else None
                crop_path = self.job_dir / f"{self.path.stem}_slice{item.slice_id}_crop.tiff"
                is_repr = (item.slice_id == repr_id)
                processor = FibrosisProcessor(str(crop_path), threshold=threshold, output_dir=self.job_dir)
                result = processor.compute_fibrosis_ratio(
                    str(crop_path),
                    mask_bool=mask_crop,
                    threshold=threshold,
                    save_overlay=is_repr,
                    overlay_suffix="_fibrosis.tiff",
                    img_bgr=crop,
                )
                ratio = result.get("fibrosis_ratio", 0.0)
                ratios.append(ratio)
                if is_repr:
                    repr_ratio = ratio
                    repr_result = result
            except Exception as e:
                logger.warning(f"Fibrosis failed for slice {item.slice_id}: {e}")

        if not ratios:
            raise RuntimeError("Fibrosis analysis failed for all slices")

        avg_ratio = sum(ratios) / len(ratios)
        fibrosis_warning = (
            repr_ratio is not None
            and abs(avg_ratio - repr_ratio) > settings.SLICE_FIBROSIS_WARN_DIFF
        )

        if fibrosis_warning:
            logger.warning(
                f"Fibrosis discrepancy: representative={repr_ratio:.3f}, "
                f"avg={avg_ratio:.3f} (diff={abs(avg_ratio - repr_ratio):.3f})"
            )

        return {
            "fibrosis_ratio": repr_ratio if repr_ratio is not None else avg_ratio,
            "fibrosis_ratio_avg": round(avg_ratio, 4),
            "fibrosis_ratio_per_slice": [round(r, 4) for r in ratios],
            "fibrosis_warning": fibrosis_warning,
            "fibrotic_pixels": repr_result.get("fibrotic_pixels") if repr_result else None,
            "tissue_pixels": repr_result.get("tissue_pixels") if repr_result else None,
            "image_path": repr_result.get("overlay_path") if repr_result else None,
            "threshold": threshold,
            "error": None,
        }

    # ------------------------------------------------------------------
    # Glomeruli
    # ------------------------------------------------------------------

    def detect_glomeruli(self):
        """
        Run glomeruli detection on the representative slice and return boxes in the
        exact same view-relative coordinate space as GET /api/glomeruli/stream/.
        """
        if not self.mask_path.exists():
            self.generate_tissue_mask(visualize=False, save_preview=False)

        mrxs_files = list(self.job_dir.glob("*.mrxs"))
        if not mrxs_files:
            raise FileNotFoundError("No .mrxs found in job dir")
        mrxs_path = mrxs_files[0]

        metadata = self.metadata
        if metadata is None:
            raise FileNotFoundError("Metadata JSON not found, cannot run glomeruli detection")

        scan_bbox = self._repr_bbox_level0()
        crop_x = metadata.scan.crop_offset_x
        crop_y = metadata.scan.crop_offset_y
        ds = metadata.calibration.downsample

        if (
            metadata.scan.mrxs_level0_shape
            and len(metadata.scan.mrxs_level0_shape) == 2
            and all(d > 0 for d in metadata.scan.mrxs_level0_shape)
        ):
            slide_h, slide_w = metadata.scan.mrxs_level0_shape
        else:
            with openslide.OpenSlide(str(mrxs_path)) as slide:
                slide_w, slide_h = slide.level_dimensions[0]

        offset_x, offset_y, view_w, view_h = get_glomeruli_view_frame(metadata, slide_w, slide_h)

        processor = GlomeruliProcessor(
            path_mrxs=str(mrxs_path),
            model_path=str(settings.MODEL_PATH),
            crop_offset_x=crop_x,
            crop_offset_y=crop_y,
            ds=ds,
            mask_path=str(self.mask_path) if self.mask_path.exists() else None,
            scan_bbox=tuple(scan_bbox) if scan_bbox else None,
        )
        raw_glomeruli = processor.detect_glomeruli()
        shifted = shift_glomeruli_coords(raw_glomeruli, offset_x, offset_y)

        self.glomeruli = shifted
        self.glomeruli_slide_info = {
            "w": view_w,
            "h": view_h,
            "conf": processor.conf,
            "n_slices": len(metadata.slices.items),
        }
        try:
            (self.job_dir / "glomeruli.json").write_text(json.dumps(shifted), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not save glomeruli.json in {self.job_dir}: {e}")

        return self.glomeruli

    def count_glomeruli(self) -> int:
        if self.glomeruli is None:
            self.detect_glomeruli()
        return len(self.glomeruli or [])

    # ------------------------------------------------------------------
    # Mask generation
    # ------------------------------------------------------------------

    def generate_tissue_mask(self, mode="all", **kwargs):
        return generate_mask(str(self.path), mode=mode, **kwargs)
