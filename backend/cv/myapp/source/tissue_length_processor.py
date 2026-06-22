import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any
import os
import numpy as np
import openslide
from PIL import Image, ImageOps
from skimage import color, filters, morphology, measure
from skimage.graph import MCP_Geometric

Image.MAX_IMAGE_PIXELS = None

from .mask import _visualize_and_save, load_mask_for_image

# Setup Logging
logger = logging.getLogger(__name__)


class TissueLengthProcessor:
    """
    Processor class for calculating tissue length from slide images.
    It generates a skeleton of the tissue and finds the longest path.
    """

    DEFAULT_MPP = 0.23
    THUMBNAIL_SIZE = 2048
    BLACK_ARTIFACT_THRESHOLD = 0.05 
    MIN_OBJECT_SIZE = 500

    def __init__(self, file_path: str, output_dir: str | Path | None = None):
        self.file_path = file_path

        if output_dir is None:
            # Default: input file directory, i.e. slides/<job_id>
            self.output_dir = str(Path(file_path).parent)
        else:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            self.output_dir = str(out_dir)

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

            # 2. Load mask if available
            h, w = np.array(image).shape[:2]
            mask_bool = load_mask_for_image(self.file_path, (h, w))

            # 3. Analyze Image
            skeleton, best_paths, length_px = self._generate_skeleton_and_path(image)

            # 4. Create Visualization
            target_tiff = Path(self.file_path)
            vis_path = target_tiff.with_name(target_tiff.stem + "_length.tiff")
            vis_path = str(vis_path)
            self._save_visualization(image, skeleton, best_paths, vis_path, mask_bool)

            # 5. Calculate Physics Length
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

    def _get_image_thumbnail(self, path: str, size: int = 2048) -> Tuple[Image.Image, float]:
        """
        Universal image opener.
        Returns: (PIL_image, average_mpp)
        """
        if openslide:
            try:
                slide = openslide.OpenSlide(path)
                mpp_x = float(slide.properties.get("openslide.mpp-x", self.DEFAULT_MPP))
                mpp_y = float(slide.properties.get("openslide.mpp-y", self.DEFAULT_MPP))
                avg_mpp = (mpp_x + mpp_y) / 2.0

                thumb = slide.get_thumbnail((size, size))
                slide.close()
                return thumb, avg_mpp
            except Exception:
                pass

        try:
            img = Image.open(path)
            img.thumbnail((size, size))
            return img, self.DEFAULT_MPP
        except Exception as e:
            raise ValueError(f"Failed to open file. Error: {e}")

    def _load_image_thumbnail(self) -> Tuple[Image.Image, float]:
        """
        Loads the image (OpenSlide or PIL) and returns the thumbnail and MPP.
        """
        return self._get_image_thumbnail(self.file_path, self.THUMBNAIL_SIZE)

    def _generate_skeleton_and_path(
        self,
        img: Image.Image,
        mask_bool: np.ndarray | None = None
    ) -> Tuple[np.ndarray, List[List[Tuple[int, int]]], float]:
        """
        Performs image processing: preprocessing, thresholding, skeletonization,
        and finding the longest paths in all connected components.
        Returns skeleton, list of all longest paths, and total length.
        """
        # Convert to grayscale numpy array
        img_arr = np.array(img)
        if img_arr.ndim == 3:
            gray = color.rgb2gray(img_arr)
        else:
            gray = img_arr / 255.0 if img_arr.max() > 1 else img_arr

        h, w = gray.shape

        # Use provided mask or check for mask file
        if mask_bool is None:
            mask_bool = load_mask_for_image(self.file_path, (h, w))

        # Remove black artifacts (microscope stitching borders)
        # Background is white (1.0), Tissue is dark, Artifacts are black (0.0)
        # We force artifacts to be white so they are treated as background.
        gray[gray < self.BLACK_ARTIFACT_THRESHOLD] = 1.0

        # Otsu Thresholding (only in tissue if mask available)
        if mask_bool is not None:
            try:
                threshold = filters.threshold_otsu(gray[mask_bool])
            except ValueError:
                threshold = 0.5
        else:
            try:
                threshold = filters.threshold_otsu(gray)
            except ValueError:
                threshold = 0.5

        # Create mask (objects are darker than threshold)
        mask = gray < threshold
        if mask_bool is not None:
            mask = mask & mask_bool  # Apply tissue mask

        # Morphological operations
        mask = morphology.dilation(mask, morphology.disk(3))
        mask = morphology.closing(mask, morphology.disk(5))
        mask = morphology.remove_small_objects(mask, min_size=self.MIN_OBJECT_SIZE)

        # Skeletonize
        skeleton = morphology.skeletonize(mask)

        # Find longest paths in all components
        return self._find_longest_path_in_skeleton(skeleton)

    def _find_longest_path_in_skeleton(self, skeleton: np.ndarray) -> Tuple[np.ndarray, List[List[Tuple[int, int]]], float]:
        """
        Find the longest paths in all connected components of the skeleton.
        Returns all longest paths (one per component) and their total length.
        """
        labels = measure.label(skeleton, connectivity=2)

        all_paths = []
        total_length = 0.0

        for region_label in np.unique(labels)[1:]:
            comp = labels == region_label

            padded = np.pad(comp, 1, mode='constant')
            neighbors = (
                padded[:-2, 1:-1] + padded[2:, 1:-1] +
                padded[1:-1, :-2] + padded[1:-1, 2:] +
                padded[:-2, :-2] + padded[:-2, 2:] +
                padded[2:, :-2] + padded[2:, 2:]
            )

            endpoints_mask = (neighbors == 1) & comp
            ey, ex = np.where(endpoints_mask)
            endpoints = list(zip(ey, ex))

            if not endpoints:
                continue

            comp_inv = np.where(comp, 1.0, np.inf)

            # First sweep — find the farthest endpoint
            start_point = endpoints[0]
            mcp = MCP_Geometric(comp_inv)
            costs, _ = mcp.find_costs([start_point])

            costs[np.isinf(costs)] = -1
            farthest_idx = np.argmax(costs)
            farthest_pos = np.unravel_index(farthest_idx, costs.shape)

            # Second sweep — find the true longest path
            costs2, traceback = mcp.find_costs([farthest_pos])
            costs2[np.isinf(costs2)] = -1

            end_idx = np.argmax(costs2)
            end_pos = np.unravel_index(end_idx, costs2.shape)
            max_dist = costs2[end_pos]

            if max_dist > 0:  # Only add if path exists
                best_path_coords = mcp.traceback(end_pos)
                all_paths.append(best_path_coords)
                total_length += max_dist

        return skeleton, all_paths, float(total_length)

    def _build_error_response(self, error_message: str) -> Dict[str, Any]:
        """Helper to construct a consistent error response."""
        return {
            "error": error_message,
            "length": -1.0,
            "image_path": ""
        }

    def _save_visualization(self, original_img: Image.Image, skeleton: np.ndarray,
                          path_coords: List[List[Tuple[int, int]]], out_path: str, mask_bool: np.ndarray | None = None):
        """
        Overlays the skeleton and longest path on the original image and saves it.
        Uses _visualize_and_save from mask.py.
        """
        img_array = np.array(ImageOps.grayscale(original_img))

        _visualize_and_save(
            img=img_array,
            skeleton=skeleton,
            path_coords=path_coords,
            mask_bool=mask_bool,
            out_path=out_path,
            show=False
        )


    def _calculate_physical_length(self, length_px: float, mpp: float) -> float:
        """Converts pixel length to millimeters."""
        return (length_px * mpp) / 1000.0

