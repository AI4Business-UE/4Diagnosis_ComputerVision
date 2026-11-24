import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

try:
    from django.conf import settings as django_settings
except Exception:
    django_settings = None

from .mask import load_mask_for_image

logger = logging.getLogger(__name__)

_DEFAULT_FIBROSIS_THRESHOLD = 0.4


def _default_threshold() -> float:
    if django_settings is not None:
        return float(getattr(django_settings, "FIBROSIS_THRESHOLD", _DEFAULT_FIBROSIS_THRESHOLD))
    return _DEFAULT_FIBROSIS_THRESHOLD


class FibrosisProcessor:
    """
    Processor class for calculating fibrosis ratio in tissue samples.

    Analyzes tissue images (typically stained with trichrome staining)
    to detect and quantify fibrotic areas based on color analysis
    in the LAB color space. Collagen fibers appear blue and are
    detected by low B-channel values in the LAB space.
    """

    def __init__(self, file_path: str, threshold: Optional[float] = None, output_dir: Optional[str | Path] = None):
        self.file_path = file_path
        self.threshold = float(threshold) if threshold is not None else _default_threshold()
        self.output_dir = str(output_dir) if output_dir else str(Path(self.file_path).parent)
        logger.debug(f"FibrosisProcessor initialized with file: {file_path}, threshold: {self.threshold}")

    def process_image(self) -> Dict[str, Any]:
        """Main method to execute the fibrosis analysis pipeline."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        logger.info(f"Starting fibrosis analysis for: {self.file_path}")
        result = self.compute_fibrosis_ratio(
            self.file_path,
            threshold=self.threshold,
            save_overlay=True,
            overlay_suffix="_fibrosis.tiff",
        )

        return {
            "fibrosis_ratio": result.get("fibrosis_ratio"),
            "fibrotic_pixels": result.get("fibrotic_pixels"),
            "tissue_pixels": result.get("tissue_pixels"),
            "threshold": result.get("threshold"),
            "image_path": result.get("overlay_path"),
            "error": None,
        }

    def compute_fibrosis_ratio(
        self,
        image_path: str,
        mask_bool: np.ndarray | None = None,
        threshold: float | None = None,
        save_overlay: bool = True,
        overlay_suffix: str = "_fibrosis.tiff",
        img_bgr: np.ndarray | None = None,
    ) -> dict:
        """
        Compute the fibrosis ratio based on LAB color space B-channel analysis.
        Accepts optional pre-loaded img_bgr and mask_bool to avoid redundant disk I/O.
        """
        effective_threshold = float(threshold) if threshold is not None else self.threshold

        if img_bgr is None:
            img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError(f"Failed to load image: {image_path}")

        if mask_bool is None:
            mask_bool = load_mask_for_image(image_path)
        if mask_bool is None:
            raise ValueError(f"Tissue mask not found for image: {image_path}")

        mask_bool = np.asarray(mask_bool, dtype=bool)
        if mask_bool.ndim == 3 and mask_bool.shape[-1] == 1:
            mask_bool = mask_bool.squeeze(-1)

        if mask_bool.shape != img_bgr.shape[:2]:
            raise ValueError(f"Mask shape {mask_bool.shape} != image shape {img_bgr.shape[:2]}")

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        b_channel = lab[:, :, 2]
        fibrotic_mask_bool = (b_channel < (effective_threshold * 255.0)) & mask_bool
        del lab

        fibrotic_pixels = int(np.count_nonzero(fibrotic_mask_bool))
        tissue_pixels = int(np.count_nonzero(mask_bool))
        fibrosis_ratio = fibrotic_pixels / tissue_pixels if tissue_pixels > 0 else 0.0

        overlay_path = None
        if save_overlay:
            overlay_bgr = img_bgr.copy()
            overlay_bgr[~mask_bool] = (255, 255, 255)
            overlay_bgr[fibrotic_mask_bool] = (0, 255, 0)

            out_dir = Path(self.output_dir) if self.output_dir else Path(image_path).parent
            overlay_path = str(out_dir / (Path(image_path).stem + overlay_suffix))
            cv2.imwrite(overlay_path, overlay_bgr)

        return {
            "fibrosis_ratio": float(fibrosis_ratio),
            "fibrotic_pixels": fibrotic_pixels,
            "tissue_pixels": tissue_pixels,
            "threshold": float(effective_threshold),
            "overlay_path": overlay_path,
        }
