import os
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any
import openslide
from django.conf import settings
import numpy as np
import matplotlib
# Set backend to Agg to prevent GUI errors in server environment
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from skimage import color, filters, morphology, measure
# Importing MCP here to avoid potential circular imports in some environments,
# though top-level is usually preferred if environment allows.
from skimage.graph import MCP_Geometric


# Setup Logging
logger = logging.getLogger('all_loggs')


class TissueLengthProcessor:
    """
    Processor class for calculating tissue length from slide images.
    It generates a skeleton of the tissue and finds the longest path.
    """

    DEFAULT_MPP = 0.23
    THUMBNAIL_SIZE = 2048
    BLACK_ARTIFACT_THRESHOLD = 0.05 
    MIN_OBJECT_SIZE = 500

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.output_dir = self._make_output_dir()

    def process_image(self) -> Dict[str, Any]:
        """
        Main method to execute the tissue length calculation pipeline.
        Returns a dictionary with length and visualization path.
        """
        if not os.path.exists(self.file_path):
            logger.error(f"File not found: {self.file_path}")
            return self._build_error_response("File does not exist")

        try:
            logger.info(f"Starting processing for: {self.file_path}")
            
            # 1. Load Image
            image, mpp = self._load_image_thumbnail()
            
            # 2. Analyze Image
            skeleton, best_path, length_px = self._generate_skeleton_and_path(image)
            
            # 3. Create Visualization
            vis_filename = "visual_overlay.tiff"
            vis_path = os.path.abspath(os.path.join(self.output_dir, vis_filename))
            self._save_visualization(image, skeleton, best_path, vis_path)
            
            # 4. Calculate Physics Length
            length_mm = self._calculate_physical_length(length_px, mpp)
            
            logger.info(f"Processing finished. Length: {length_mm}mm")
            
            return {
                "length": float(round(length_mm, 4)),
                "image_path": vis_path,
                "error": None
            }

        except Exception as e:
            logger.exception(f"Error processing tissue length for {self.file_path}")
            return self._build_error_response(str(e))

    def _make_output_dir(self) -> str:
        """Creates and returns the output directory path based on the filename."""
        base_name = Path(self.file_path).stem
        out_dir = Path(".") / base_name
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir)

    def _load_image_thumbnail(self) -> Tuple[Image.Image, float]:
        """
        Loads the image (OpenSlide or PIL) and returns the thumbnail and MPP.
        """
        if openslide:
            try:
                slide = openslide.OpenSlide(self.file_path)
                mpp_x = float(slide.properties.get("openslide.mpp-x", self.DEFAULT_MPP))
                mpp_y = float(slide.properties.get("openslide.mpp-y", self.DEFAULT_MPP))
                avg_mpp = (mpp_x + mpp_y) / 2.0
                
                thumbnail = slide.get_thumbnail((self.THUMBNAIL_SIZE, self.THUMBNAIL_SIZE))
                slide.close()
                return thumbnail, avg_mpp
            except Exception as e:
                pass

        # Fallback to PIL
        try:
            img = Image.open(self.file_path)
            img.thumbnail((self.THUMBNAIL_SIZE, self.THUMBNAIL_SIZE))
            return img, self.DEFAULT_MPP
        except Exception as e:
            raise ValueError(f"Failed to open image with PIL: {e}")

    def _generate_skeleton_and_path(self, img: Image.Image) -> Tuple[np.ndarray, List[Tuple[int, int]], int]:
        """
        Performs image processing: preprocessing, thresholding, skeletonization,
        and finding the longest path (Double Sweep algorithm).
        """
        # Convert to grayscale numpy array
        img_arr = np.array(img)
        if img_arr.ndim == 3:
            gray = color.rgb2gray(img_arr)
        else:
            gray = img_arr / 255.0 if img_arr.max() > 1 else img_arr

        # Remove black artifacts (microscope stitching borders)
        # Background is white (1.0), Tissue is dark, Artifacts are black (0.0)
        # We force artifacts to be white so they are treated as background.
        gray[gray < self.BLACK_ARTIFACT_THRESHOLD] = 1.0

        # Otsu Thresholding
        try:
            threshold = filters.threshold_otsu(gray)
        except ValueError:
            threshold = 0.5
        
        # Create mask (objects are darker than threshold)
        mask = gray < threshold
        
        # Morphological operations
        mask = morphology.dilation(mask, morphology.disk(3))
        mask = morphology.closing(mask, morphology.disk(5))
        mask = morphology.remove_small_objects(mask, min_size=self.MIN_OBJECT_SIZE)
        
        # Skeletonize
        skeleton = morphology.skeletonize(mask)
        
        # Find longest path
        return self._find_longest_path_in_skeleton(skeleton)

    def _find_longest_path_in_skeleton(self, skeleton: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]], int]:
        """
        Analyzes the skeleton graph components to find the longest path.
        """
        labels = measure.label(skeleton, connectivity=2)
        max_len = 0
        best_path_coords = []
        
        # Iterate over each connected component
        for region_label in np.unique(labels)[1:]:
            comp = labels == region_label
            
            # Find endpoints
            endpoints = self._find_endpoints(comp)
            if not endpoints:
                continue

            # Invert component for MCP (foreground = 1, background = inf)
            comp_inv = np.where(comp, 1.0, np.inf)
            mcp = MCP_Geometric(comp_inv)

            # Double Sweep Algorithm
            # 1. Find farthest point from an arbitrary start
            start_point = endpoints[0]
            costs, _ = mcp.find_costs([start_point])
            
            valid_costs = costs.copy()
            valid_costs[np.isinf(valid_costs)] = -1
            
            farthest_idx = np.argmax(valid_costs)
            farthest_pos = np.unravel_index(farthest_idx, costs.shape)
            
            # 2. Find diameter from that farthest point
            costs_2, traceback = mcp.find_costs([farthest_pos])
            valid_costs_2 = costs_2.copy()
            valid_costs_2[np.isinf(valid_costs_2)] = -1
            
            end_idx = np.argmax(valid_costs_2)
            end_pos = np.unravel_index(end_idx, costs_2.shape)
            
            dist = valid_costs_2[end_pos]
            
            if dist > max_len:
                max_len = dist
                best_path_coords = mcp.traceback(end_pos)

        return skeleton, best_path_coords, max_len

    def _find_endpoints(self, comp: np.ndarray) -> List[Tuple[int, int]]:
        """Identifies endpoints in a skeleton component."""
        padded = np.pad(comp, 1, mode='constant')
        neighbors = (padded[:-2, 1:-1] + padded[2:, 1:-1] + 
                     padded[1:-1, :-2] + padded[1:-1, 2:] +
                     padded[:-2, :-2] + padded[:-2, 2:] + 
                     padded[2:, :-2] + padded[2:, 2:])
        
        endpoints_mask = (neighbors == 1) & comp
        ey, ex = np.where(endpoints_mask)
        return list(zip(ey, ex))

    def _save_visualization(self, original_img: Image.Image, skeleton: np.ndarray, 
                          path_coords: List[Tuple[int, int]], out_path: str):
        """
        Overlays the skeleton and longest path on the original image and saves it.
        """
        original_gray = ImageOps.grayscale(original_img)
        
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
        
        ax.imshow(original_gray, cmap="gray")
        ax.contour(skeleton, [0.5], colors='red', linewidths=0.8, alpha=0.7)
        
        if path_coords:
            y, x = zip(*path_coords) 
            ax.plot(x, y, color='cyan', linewidth=2.5, label='Longest Path')
            
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(out_path, format='tiff', dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _calculate_physical_length(self, length_px: float, mpp: float) -> float:
        """Converts pixel length to millimeters."""
        return (length_px * mpp) / 1000.0

    def _build_error_response(self, error_message: str) -> Dict[str, Any]:
        """Helper to construct a consistent error response."""
        return {
            "error": error_message,
            "length": -1.0,
            "image_path": ""
        }

