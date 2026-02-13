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
    
    # HSV tissue mask parameters
    LOWER_TISSUE_HSV = np.array([0, 10, 10], dtype=np.uint8)
    UPPER_TISSUE_HSV = np.array([179, 255, 240], dtype=np.uint8)

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
        self.output_dir = output_dir if output_dir else self._make_output_dir()
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
            
            # 1. Load image
            img_bgr = self._load_image()
            logger.debug(f"Image loaded successfully, shape: {img_bgr.shape}")
            
            # 2. Generate tissue mask
            tissue_mask = self._create_tissue_mask(img_bgr)
            tissue_pixels = int(np.count_nonzero(tissue_mask))
            logger.debug(f"Tissue mask created, tissue pixels: {tissue_pixels}")
            
            if tissue_pixels == 0:
                logger.warning("No tissue detected in the image")
                return self._build_error_response("No tissue detected in the image")
            
            # 3. Detect fibrotic areas
            fibrotic_mask = self._detect_fibrosis(img_bgr, tissue_mask)
            fibrotic_pixels = int(np.count_nonzero(fibrotic_mask))
            logger.debug(f"Fibrosis detection complete, fibrotic pixels: {fibrotic_pixels}")
            
            # 4. Calculate fibrosis ratio
            fibrosis_ratio = fibrotic_pixels / tissue_pixels
            logger.info(f"Fibrosis ratio calculated: {fibrosis_ratio:.4f} ({fibrosis_ratio:.2%})")
            
            # 5. Generate visualization if requested
            vis_path = ""
            if visualize:
                vis_filename = "fibrosis_overlay.tiff"
                vis_path = os.path.abspath(os.path.join(self.output_dir, vis_filename))
                self._save_visualization(img_bgr, tissue_mask, fibrotic_mask, fibrosis_ratio, vis_path)
                logger.info(f"Visualization saved to: {vis_path}")
            
            logger.info(f"Processing finished successfully for: {self.file_path}")
            
            return {
                "fibrosis_ratio": float(round(fibrosis_ratio, 6)),
                "fibrotic_pixels": fibrotic_pixels,
                "tissue_pixels": tissue_pixels,
                "threshold": float(self.threshold),
                "image_path": vis_path,
                "error": None
            }

        except Exception as e:
            logger.exception(f"Error processing fibrosis for {self.file_path}")
            return self._build_error_response(str(e))

    def _make_output_dir(self) -> str:
        """
        Creates and returns the output directory path.
        
        Uses Django settings if available, otherwise falls back
        to a temporary directory for standalone usage.
        """
        if DJANGO_AVAILABLE:
            try:
                base_dir = Path(settings.BASE_DIR)  # backend/cv
                out_dir = base_dir / "cv" / "result_analyze"
                out_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Output directory set to: {out_dir}")
                return str(out_dir)
            except Exception as e:
                logger.warning(f"Could not use Django settings for output dir: {e}")
        
        # Fallback to temp directory
        out_dir = Path(tempfile.gettempdir()) / "fibrosis_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Using fallback output directory: {out_dir}")
        return str(out_dir)

    def _load_image(self) -> np.ndarray:
        """
        Loads the image file into a BGR numpy array.
        
        Returns
        -------
        np.ndarray
            Image in BGR format.
        
        Raises
        ------
        ValueError
            If the image cannot be loaded.
        """
        img_bgr = cv2.imread(self.file_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Failed to load image: {self.file_path}")
        return img_bgr

    def _create_tissue_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Creates a binary mask separating tissue from background.
        
        Uses HSV color space to identify tissue areas.
        Background (white areas) and non-tissue regions are excluded.
        
        Parameters
        ----------
        img_bgr : np.ndarray
            Input image in BGR format.
        
        Returns
        -------
        np.ndarray
            Boolean mask where True indicates tissue pixels.
        """
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        tissue_mask = cv2.inRange(hsv, self.LOWER_TISSUE_HSV, self.UPPER_TISSUE_HSV)
        return tissue_mask > 0

    def _detect_fibrosis(self, img_bgr: np.ndarray, tissue_mask: np.ndarray) -> np.ndarray:
        """
        Detects fibrotic areas based on LAB color space analysis.
        
        Collagen fibers in trichrome-stained tissue appear blue,
        which corresponds to low B-channel values in LAB space.
        Pixels with B_normalized < threshold are classified as fibrotic.
        
        Parameters
        ----------
        img_bgr : np.ndarray
            Input image in BGR format.
        tissue_mask : np.ndarray
            Boolean mask indicating tissue areas.
        
        Returns
        -------
        np.ndarray
            Boolean mask where True indicates fibrotic pixels.
        """
        # Apply tissue mask to image
        tissue_mask_uint8 = tissue_mask.astype(np.uint8) * 255
        tissue_only = cv2.bitwise_and(img_bgr, img_bgr, mask=tissue_mask_uint8)
        
        # Convert to LAB color space
        lab = cv2.cvtColor(tissue_only, cv2.COLOR_BGR2LAB)
        
        # Extract and normalize B channel (0-255 -> 0-1)
        # In LAB: low B = blue (fibrosis), high B = yellow
        _, _, b_channel = cv2.split(lab)
        b_normalized = b_channel.astype(np.float32) / 255.0
        
        # Classify fibrotic areas: B < threshold AND within tissue
        fibrotic_mask = (b_normalized < self.threshold) & tissue_mask
        
        return fibrotic_mask

    def _save_visualization(self, img_bgr: np.ndarray, tissue_mask: np.ndarray,
                           fibrotic_mask: np.ndarray, fibrosis_ratio: float, 
                           out_path: str) -> None:
        """
        Creates and saves a visualization of the fibrosis analysis.
        
        Generates a 3-panel figure showing:
        1. Original image
        2. Tissue mask
        3. Overlay with fibrotic areas highlighted in green
        
        Parameters
        ----------
        img_bgr : np.ndarray
            Original image in BGR format.
        tissue_mask : np.ndarray
            Boolean mask of tissue areas.
        fibrotic_mask : np.ndarray
            Boolean mask of fibrotic areas.
        fibrosis_ratio : float
            Calculated fibrosis ratio for display.
        out_path : str
            Path to save the visualization.
        """
        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Create overlay
        overlay = img_rgb.copy()
        overlay[~tissue_mask] = [255, 255, 255]   # Background - white
        overlay[fibrotic_mask] = [0, 255, 0]      # Fibrosis - green

        # Create figure with 3 panels
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(img_rgb)
        axes[0].set_title("Original", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(tissue_mask, cmap="gray")
        axes[1].set_title("Tissue Mask", fontsize=12)
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(f"Fibrosis: {fibrosis_ratio:.2%} (threshold={self.threshold})", fontsize=12)
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(out_path, format='tiff', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.debug(f"Visualization saved to: {out_path}")

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
