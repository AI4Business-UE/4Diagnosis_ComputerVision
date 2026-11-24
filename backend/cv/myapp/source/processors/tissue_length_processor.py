import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import openslide
from django.conf import settings
from PIL import Image, ImageOps
from skimage import color, filters, measure, morphology
from skimage.graph import MCP_Geometric

Image.MAX_IMAGE_PIXELS = None

from .mask import _visualize_and_save, load_mask_for_image

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
            raise FileNotFoundError(f"File not found: {self.file_path}")

        logger.info(f"Starting processing for: {self.file_path}")

        image, mpp = self._load_image_thumbnail()
        w, h = image.size
        mask_bool = load_mask_for_image(self.file_path, (h, w))

        skeleton, best_paths, length_px = self._generate_skeleton_and_path(image, mask_bool, mpp)

        target_tiff = Path(self.file_path)
        vis_path = str(Path(self.output_dir) / (target_tiff.stem + "_length.tiff"))
        self._save_visualization(image, skeleton, best_paths, vis_path, mask_bool)

        length_mm = self._calculate_physical_length(length_px, mpp)
        logger.info(f"Processing finished. Length: {length_mm}mm")

        return {
            "length": float(round(length_mm, 4)),
            "image_path": vis_path,
            "error": None,
        }

    def _get_image_thumbnail(self, path: str, size: int = 2048) -> Tuple[Image.Image, float]:
        """
        Universal image opener.
        Returns: (PIL_image, average_mpp)
        """
        mpp_x = self.DEFAULT_MPP
        mpp_y = self.DEFAULT_MPP

        try:
            from .metadata import SlideMetadata
            meta = SlideMetadata.load(Path(path).parent)
            mpp_x = meta.calibration.mpp_x
            mpp_y = meta.calibration.mpp_y
            logger.info(f"Loaded calibration from SlideMetadata: mpp_x={mpp_x}, mpp_y={mpp_y}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Calibration metadata file (.json) was not found in job directory: {e}")
        except Exception as e:
            logger.warning(f"Could not load SlideMetadata, falling back to openslide/defaults: {e}")
            if openslide:
                try:
                    with openslide.OpenSlide(path) as slide:
                        mpp_x = float(slide.properties.get("openslide.mpp-x", self.DEFAULT_MPP))
                        mpp_y = float(slide.properties.get("openslide.mpp-y", self.DEFAULT_MPP))
                except Exception:
                    pass

        try:
            with Image.open(path) as opened:
                original_w, original_h = opened.size
                opened.thumbnail((size, size))
                img = opened.copy()

            thumb_w, thumb_h = img.size
            thumb_mpp_x = mpp_x * (original_w / thumb_w)
            thumb_mpp_y = mpp_y * (original_h / thumb_h)

            avg_mpp = (thumb_mpp_x + thumb_mpp_y) / 2.0
            return img, avg_mpp
        except Exception as e:
            raise ValueError(f"Failed to open file. Error: {e}")

    def _load_image_thumbnail(self) -> Tuple[Image.Image, float]:
        return self._get_image_thumbnail(self.file_path, self.THUMBNAIL_SIZE)

    def _um_to_px(self, value_um: float, mpp: float | None, minimum: int = 1) -> int:
        if not mpp or mpp <= 0:
            return minimum
        return max(minimum, int(round(value_um / mpp)))

    def _generate_skeleton_and_path(
        self,
        img: Image.Image,
        mask_bool: np.ndarray | None = None,
        mpp: float | None = None,
    ) -> Tuple[np.ndarray, List[List[Tuple[int, int]]], float]:
        img_arr = np.array(img)
        if img_arr.ndim == 3:
            gray = color.rgb2gray(img_arr)
        else:
            gray = img_arr / 255.0 if img_arr.max() > 1 else img_arr.astype(np.float64)

        h, w = gray.shape

        if mask_bool is None:
            mask_bool = load_mask_for_image(self.file_path, (h, w))

        gray[gray < self.BLACK_ARTIFACT_THRESHOLD] = 1.0

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

        mask = gray < threshold
        if mask_bool is not None:
            mask = mask & mask_bool

        dilation_px = self._um_to_px(settings.TISSUE_LENGTH_DILATION_UM, mpp)
        closing_px = self._um_to_px(settings.TISSUE_LENGTH_CLOSING_UM, mpp)
        if mpp and mpp > 0:
            min_object_size = max(
                1,
                int(round(settings.TISSUE_LENGTH_MIN_OBJECT_AREA_UM2 / (mpp * mpp))),
            )
        else:
            min_object_size = self.MIN_OBJECT_SIZE

        mask = morphology.dilation(mask, morphology.disk(dilation_px))
        mask = morphology.closing(mask, morphology.disk(closing_px))
        mask = morphology.remove_small_objects(mask, min_size=min_object_size)

        skeleton = morphology.skeletonize(mask)
        return self._find_longest_path_in_skeleton(skeleton)

    def _find_longest_path_in_skeleton(
        self,
        skeleton: np.ndarray,
    ) -> Tuple[np.ndarray, List[List[Tuple[int, int]]], float]:
        """
        Find the longest paths in all connected components of the skeleton.
        Operates on each component's tight bounding box for speed and handles
        closed-loop components without endpoints.
        """
        labels = measure.label(skeleton, connectivity=2)
        all_paths: List[List[Tuple[int, int]]] = []
        total_length = 0.0

        for prop in measure.regionprops(labels):
            r0, c0, r1, c1 = prop.bbox
            comp = (labels[r0:r1, c0:c1] == prop.label)

            padded = np.pad(comp, 1, mode="constant")
            neighbors = (
                padded[:-2, 1:-1] + padded[2:, 1:-1] +
                padded[1:-1, :-2] + padded[1:-1, 2:] +
                padded[:-2, :-2] + padded[:-2, 2:] +
                padded[2:, :-2] + padded[2:, 2:]
            )

            endpoints_mask = (neighbors == 1) & comp
            ey, ex = np.where(endpoints_mask)
            if len(ey) > 0:
                start_point = (int(ey[0]), int(ex[0]))
            else:
                cy, cx = np.where(comp)
                if len(cy) == 0:
                    continue
                start_point = (int(cy[0]), int(cx[0]))

            comp_inv = np.where(comp, 1.0, np.inf)

            mcp = MCP_Geometric(comp_inv)
            costs, _ = mcp.find_costs([start_point])
            costs[np.isinf(costs)] = -1
            farthest_pos = np.unravel_index(int(np.argmax(costs)), costs.shape)

            costs2, _ = mcp.find_costs([farthest_pos])
            costs2[np.isinf(costs2)] = -1
            end_pos = np.unravel_index(int(np.argmax(costs2)), costs2.shape)
            max_dist = float(costs2[end_pos])

            if max_dist > 0:
                local_path = mcp.traceback(end_pos)
                global_path = [(int(y + r0), int(x + c0)) for y, x in local_path]
                all_paths.append(global_path)
                total_length += max_dist

        return skeleton, all_paths, float(total_length)

    def _save_visualization(
        self,
        original_img: Image.Image,
        skeleton: np.ndarray,
        path_coords: List[List[Tuple[int, int]]],
        out_path: str,
        mask_bool: np.ndarray | None = None,
    ):
        img_array = np.array(ImageOps.grayscale(original_img))

        _visualize_and_save(
            img=img_array,
            skeleton=skeleton,
            path_coords=path_coords,
            mask_bool=mask_bool,
            out_path=out_path,
            show=False,
        )

    def _calculate_physical_length(self, length_px: float, mpp: float) -> float:
        return (length_px * mpp) / 1000.0
