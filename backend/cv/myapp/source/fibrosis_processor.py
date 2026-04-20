import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
import matplotlib
# Set backend to Agg to prevent GUI errors in server environment
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Django settings import is optional for standalone usage
try:
    from django.conf import settings
    DJANGO_AVAILABLE = True
except ImportError:
    DJANGO_AVAILABLE = False

from .mask import load_mask_for_image

# Setup Logging
logger = logging.getLogger(__name__)


class FibrosisProcessor:
    """
    Processor class for calculating fibrosis ratio in tissue samples.
    
    Analyzes tissue images (typically stained with trichrome staining)
    to detect and quantify fibrotic areas based on color analysis
    in the LAB color space. Collagen fibers appear blue and are
    detected by low B-channel values in the LAB space.
    """

    # Default threshold for normalized B channel (0-1 scale)
    # Lower B values indicate more blue (fibrotic collagen)
    DEFAULT_THRESHOLD = 0.4

    def __init__(self, file_path: str, threshold: Optional[float] = None, output_dir: Optional[str] = None):
        """
        Initializes the FibrosisProcessor.
        
        Parameters
        ----------
        file_path : str
            Path to the tissue image file (e.g., .tiff).
        threshold : float, optional
            Custom threshold for B channel classification.
            If None, uses DEFAULT_THRESHOLD (0.4).
        output_dir : str, optional
            Directory to save output files. If None, uses Django settings
            (if available) or a temporary directory.
        """
        self.file_path = file_path
        self.threshold = threshold if threshold is not None else self.DEFAULT_THRESHOLD
        self.output_dir = output_dir if output_dir else str(Path(self.file_path).parent)
        logger.debug(f"FibrosisProcessor initialized with file: {file_path}, threshold: {self.threshold}")

    def process_image(self, visualize: bool = True) -> Dict[str, Any]:
        """
        Main method to execute the fibrosis analysis pipeline.
        
        Parameters
        ----------
        visualize : bool, default True
            If True, generates and saves visualization images.
        
        Returns
        -------
        dict
            Dictionary containing:
            - fibrosis_ratio: float (0-1 range)
            - fibrotic_pixels: int
            - tissue_pixels: int
            - threshold: float
            - image_path: str (path to visualization, empty if not generated)
            - error: str or None
        """
        if not os.path.exists(self.file_path):
            logger.error(f"File not found: {self.file_path}")
            return self._build_error_response("File does not exist")

        try:
            logger.info(f"Starting fibrosis analysis for: {self.file_path}")
            
            # Use the new compute_fibrosis_ratio logic
            result = self.compute_fibrosis_ratio(
                self.file_path,
                threshold=self.threshold,
                visualize=visualize,
                save_overlay=visualize,
                overlay_suffix="_fibrosis.tiff"
            )

            return {
                "fibrosis_ratio": result.get("fibrosis_ratio"),
                "fibrotic_pixels": result.get("fibrotic_pixels"),
                "tissue_pixels": result.get("tissue_pixels"),
                "threshold": result.get("threshold"),
                "image_path": result.get("overlay_path"),
                "error": result.get("error")
            }

        except Exception as e:
            logger.exception(f"Error processing fibrosis for {self.file_path}")
            return self._build_error_response(str(e))

    def compute_fibrosis_ratio(
        self,
        image_path: str,
        mask_bool: np.ndarray | None = None,
        threshold: float = 0.4,
        visualize: bool = False,
        save_overlay: bool = True,
        overlay_suffix: str = "_fibrosis.tiff"
    ) -> dict:
        """
        Compute the fibrosis ratio based on LAB color space B-channel analysis.
        If mask_bool is provided, uses it directly.
        If mask_bool=None, loads the pre-generated mask from disk.
        """

        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Failed to load image: {image_path}")

        if mask_bool is None:
            mask_bool = load_mask_for_image(image_path)

        mask_bool = np.asarray(mask_bool)
        if mask_bool.ndim == 3 and mask_bool.shape[-1] == 1:
            mask_bool = mask_bool.squeeze(-1)

        # LAB color space → B channel
        tissue_only = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_bool.astype(np.uint8)*255)
        lab = cv2.cvtColor(tissue_only, cv2.COLOR_BGR2LAB)
        _, _, B = cv2.split(lab)
        B_norm = B.astype(np.float32) / 255.0

        # Fibrosis mask
        fibrotic_mask_bool = (B_norm < threshold) & mask_bool

        fibrotic_pixels = int(np.count_nonzero(fibrotic_mask_bool))
        tissue_pixels = int(np.count_nonzero(mask_bool))
        fibrosis_ratio = fibrotic_pixels / tissue_pixels if tissue_pixels > 0 else 0.0

        # Overlay visualization
        overlay_path = None
        if visualize or save_overlay:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            overlay = img_rgb.copy()
            overlay[~mask_bool] = 255
            overlay[fibrotic_mask_bool] = [0, 255, 0]

            overlay_path = str(Path(image_path).parent / (Path(image_path).stem + overlay_suffix))
            if save_overlay:
                cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            if visualize:
                plt.figure(figsize=(6,6))
                plt.imshow(overlay)
                plt.axis("off")
                plt.title(f"Fibrosis: {fibrosis_ratio:.2%}")
                plt.show()

        return {
            "fibrosis_ratio": float(fibrosis_ratio),
            "fibrotic_pixels": fibrotic_pixels,
            "tissue_pixels": tissue_pixels,
            "threshold": float(threshold),
            "overlay_path": overlay_path
        }

    def _build_error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Helper to construct a consistent error response.
        
        Parameters
        ----------
        error_message : str
            Description of the error.
        
        Returns
        -------
        dict
            Error response dictionary with default values.
        """
        return {
            "error": error_message,
            "fibrosis_ratio": -1.0,
            "fibrotic_pixels": 0,
            "tissue_pixels": 0,
            "threshold": float(self.threshold),
            "image_path": ""
        }