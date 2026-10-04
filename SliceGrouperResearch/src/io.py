"""
src/io.py — Image I/O and Connected Component Detection
=========================================================

This module is the entry point for every experiment.  It is responsible for:

1. Loading TIFF images (single-level and multi-resolution pyramids).
2. Loading binary tissue masks.
3. Detecting connected components and building ``ComponentData`` objects that
   bundle all raw data needed by downstream feature extractors.

Design notes
------------
- We deliberately separate I/O from all feature extraction.
  ``ComponentData`` is a pure data container: no algorithms live here.
- OpenCV is used for connected-component labelling because it is already
  present in the existing production stack (see ARCHITECTURE.md).
- tifffile is used for TIFF loading as it handles multi-page, pyramidal,
  and BigTIFF files that PIL/Pillow may not decode correctly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

_logger = logging.getLogger("SliceGrouper.io")


# ---------------------------------------------------------------------------
# ComponentData dataclass
# ---------------------------------------------------------------------------


@dataclass
class ComponentData:
    """
    Container for a single connected tissue component.

    All coordinates are in the space of the loaded (possibly downsampled)
    TIFF image.  No normalisation or feature values are stored here.

    Attributes
    ----------
    component_id : int
        Zero-based index of this component (stable within a single run).
    bbox : tuple of int
        Bounding box (x, y, width, height) in pixel coordinates.
    area : int
        Number of tissue pixels in the component mask.
    centroid : tuple of float
        (cx, cy) centroid of the component mass in image coordinates.
    mask_crop : np.ndarray, shape (H, W), dtype uint8
        Binary mask cropped to the bounding box (0 = background, 255 = tissue).
    rgb_crop : np.ndarray, shape (H, W, 3), dtype uint8
        Original BGR/RGB image cropped to the bounding box.
        Only valid pixels (mask == 255) represent tissue.
    contour : np.ndarray, shape (N, 1, 2), dtype int32
        Largest contour extracted by cv2.findContours for the component.
    convex_hull : np.ndarray
        Points of the convex hull of the component contour.
    label_id : int
        Raw label from ``cv2.connectedComponentsWithStats`` (1-based).
    """

    component_id: int
    bbox: Tuple[int, int, int, int]          # (x, y, w, h)
    area: int
    centroid: Tuple[float, float]            # (cx, cy)
    mask_crop: np.ndarray = field(repr=False)
    rgb_crop: np.ndarray = field(repr=False)
    contour: np.ndarray = field(repr=False)
    convex_hull: np.ndarray = field(repr=False)
    label_id: int = 0

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        """Bounding box width in pixels."""
        return self.bbox[2]

    @property
    def height(self) -> int:
        """Bounding box height in pixels."""
        return self.bbox[3]

    @property
    def aspect_ratio(self) -> float:
        """Width / height ratio of the bounding box."""
        return self.width / max(self.height, 1)

    @property
    def mask_gray(self) -> np.ndarray:
        """Return mask_crop as a uint8 grayscale image (0 or 255)."""
        return self.mask_crop

    @property
    def gray_crop(self) -> np.ndarray:
        """Return rgb_crop converted to grayscale."""
        return cv2.cvtColor(self.rgb_crop, cv2.COLOR_BGR2GRAY)

    @property
    def masked_rgb(self) -> np.ndarray:
        """
        RGB crop with background set to white (255, 255, 255).
        Useful for visualisation and for deep learning extractors that
        expect a white-background tissue image.
        """
        out = self.rgb_crop.copy()
        bg = self.mask_crop == 0
        out[bg] = 255
        return out

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"ComponentData(id={self.component_id}, "
            f"area={self.area}, bbox={self.bbox}, centroid=({self.centroid[0]:.1f}, {self.centroid[1]:.1f}))"
        )


# ---------------------------------------------------------------------------
# TIFF loading
# ---------------------------------------------------------------------------


def load_tiff(
    path: Union[str, Path],
    level: int = 0,
    as_rgb: bool = True,
) -> np.ndarray:
    """
    Load a TIFF image from disk.

    Supports:
    - Standard single-page TIFFs (uint8, uint16).
    - Multi-page pyramidal TIFFs (selects the page corresponding to ``level``).
    - BigTIFF format.

    The function uses ``tifffile`` as the primary reader and falls back to
    ``cv2.imread`` if tifffile is unavailable.

    Parameters
    ----------
    path : str or Path
        Path to the TIFF file.
    level : int
        Pyramid level to read. 0 = full resolution.  Higher levels = lower
        resolution (faster but less detail).  If the file is not pyramidal
        or the requested level does not exist, level 0 is returned.
    as_rgb : bool
        If True, convert BGR→RGB so the returned array uses RGB channel order.
        Set to False if you intend to pass the array to OpenCV functions.

    Returns
    -------
    np.ndarray, shape (H, W, 3) or (H, W), dtype uint8
        Image array.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the image could not be decoded.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TIFF not found: {path}")

    _logger.info(f"Loading TIFF: {path.name} (level={level})")

    image = _load_tifffile(path, level)
    if image is None:
        image = _load_opencv(path)

    if image is None:
        raise ValueError(f"Could not decode TIFF: {path}")

    # Ensure uint8
    if image.dtype != np.uint8:
        if image.dtype in (np.uint16,):
            image = (image / 256).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    # Ensure 3-channel
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if as_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    _logger.info(f"Loaded image: shape={image.shape}, dtype={image.dtype}")
    return image


def _load_tifffile(path: Path, level: int) -> Optional[np.ndarray]:
    """Try loading with tifffile (handles pyramidal TIFFs)."""
    try:
        import tifffile

        with tifffile.TiffFile(str(path)) as tif:
            # Check for pyramidal series
            if tif.series:
                series = tif.series[0]
                n_levels = len(series.levels)
                lvl = min(level, n_levels - 1)
                if lvl != level:
                    _logger.warning(
                        f"Requested level {level} does not exist (max={n_levels - 1}). "
                        f"Using level {lvl}."
                    )
                img = series.levels[lvl].asarray()
            else:
                img = tif.asarray()

            # Handle multi-band arrays (C, H, W) → (H, W, C)
            if img.ndim == 3 and img.shape[0] in (1, 3, 4):
                img = np.moveaxis(img, 0, -1)
            if img.ndim == 3 and img.shape[2] == 1:
                img = img[:, :, 0]

            return img
    except Exception as exc:
        _logger.debug(f"tifffile failed ({exc}), falling back to OpenCV.")
        return None


def _load_opencv(path: Path) -> Optional[np.ndarray]:
    """Fallback loader using OpenCV."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return img  # may be None if file is unsupported


# ---------------------------------------------------------------------------
# Mask loading
# ---------------------------------------------------------------------------


def load_mask(
    path: Union[str, Path],
    threshold: int = 1,
) -> np.ndarray:
    """
    Load a binary tissue mask from a TIFF file.

    Mask convention (as described in the problem statement):
      0 = background
      1 (or any positive value) = tissue

    The function binarises the mask at ``threshold`` and returns a uint8
    array with values 0 (background) or 255 (tissue) to be compatible with
    OpenCV morphological operations.

    Parameters
    ----------
    path : str or Path
        Path to the mask TIFF file.
    threshold : int
        Pixel values >= threshold are treated as tissue.

    Returns
    -------
    np.ndarray, shape (H, W), dtype uint8
        Binary mask with values in {0, 255}.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mask not found: {path}")

    _logger.info(f"Loading mask: {path.name}")

    # Use single-channel loading
    raw = _load_tifffile(path, level=0)
    if raw is None:
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"Could not decode mask: {path}")

    # If RGB was returned, collapse to grayscale
    if raw.ndim == 3:
        raw = raw[:, :, 0]

    # Binarise
    binary = (raw >= threshold).astype(np.uint8) * 255
    _logger.info(
        f"Mask loaded: shape={binary.shape}, "
        f"tissue_pixels={int(binary.sum() / 255):,}, "
        f"coverage={binary.mean() / 255 * 100:.1f}%"
    )
    return binary


# ---------------------------------------------------------------------------
# Connected component detection
# ---------------------------------------------------------------------------


def detect_components(
    mask: np.ndarray,
    image: Optional[np.ndarray] = None,
    min_area_px: int = 500,
    max_components: int = 500,
    margin_px: int = 20,
) -> List[ComponentData]:
    """
    Detect connected tissue components in a binary mask and build
    ``ComponentData`` objects for each.

    The algorithm uses OpenCV's ``connectedComponentsWithStats`` which runs
    a two-pass labelling algorithm in O(pixels) time.

    Parameters
    ----------
    mask : np.ndarray, shape (H, W), dtype uint8
        Binary mask with values in {0, 255} (as returned by ``load_mask``).
    image : np.ndarray or None, shape (H, W, 3)
        Original RGB/BGR image.  If None, ``rgb_crop`` in the returned
        ``ComponentData`` objects will be filled with zeros.
    min_area_px : int
        Components with area < min_area_px are discarded as noise.
    max_components : int
        Return at most this many components (largest first by area).
    margin_px : int
        Padding added to each bounding box when cropping rgb and mask.

    Returns
    -------
    list of ComponentData
        One entry per detected component, sorted by descending area.

    Notes
    -----
    Label 0 from ``connectedComponentsWithStats`` is always the background
    and is never included in the returned list.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    _logger.info("Running connected component analysis …")

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    H, W = mask.shape
    components: List[ComponentData] = []

    for lbl in range(1, n_labels):  # skip background label 0
        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        w = int(stats[lbl, cv2.CC_STAT_WIDTH])
        h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        cx, cy = float(centroids[lbl, 0]), float(centroids[lbl, 1])

        if area < min_area_px:
            continue

        # Crop mask with margin
        x1 = max(0, x - margin_px)
        y1 = max(0, y - margin_px)
        x2 = min(W, x + w + margin_px)
        y2 = min(H, y + h + margin_px)

        comp_mask_full = (labels[y1:y2, x1:x2] == lbl).astype(np.uint8) * 255

        # Crop image
        if image is not None:
            comp_rgb = image[y1:y2, x1:x2].copy()
        else:
            comp_rgb = np.zeros((*comp_mask_full.shape, 3), dtype=np.uint8)

        # Find contour of this component
        contours, _ = cv2.findContours(
            comp_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            _logger.debug(f"Component {lbl}: no contour found, skipping.")
            continue
        # Take the largest contour (handles small holes)
        main_contour = max(contours, key=cv2.contourArea)

        # Convex hull
        hull = cv2.convexHull(main_contour)

        comp = ComponentData(
            component_id=len(components),
            bbox=(x, y, w, h),
            area=area,
            centroid=(cx, cy),
            mask_crop=comp_mask_full,
            rgb_crop=comp_rgb,
            contour=main_contour,
            convex_hull=hull,
            label_id=lbl,
        )
        components.append(comp)

    # Sort by area descending and cap
    components.sort(key=lambda c: c.area, reverse=True)
    components = components[:max_components]

    # Re-assign sequential IDs after filtering/sorting
    for idx, comp in enumerate(components):
        comp.component_id = idx

    _logger.info(f"Detected {len(components)} components (min_area={min_area_px}px)")
    return components


# ---------------------------------------------------------------------------
# Convenience: load everything at once
# ---------------------------------------------------------------------------


def load_slide(
    tiff_path: Union[str, Path],
    mask_path: Union[str, Path],
    tiff_level: int = 0,
    min_area_px: int = 500,
    max_components: int = 500,
    margin_px: int = 20,
    mask_threshold: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[ComponentData]]:
    """
    High-level convenience function: load image + mask and detect components.

    Equivalent to calling ``load_tiff``, ``load_mask``, and
    ``detect_components`` in sequence.

    Parameters
    ----------
    tiff_path : str or Path
    mask_path : str or Path
    tiff_level : int
        Pyramid level for TIFF loading.
    min_area_px : int
    max_components : int
    margin_px : int
    mask_threshold : int

    Returns
    -------
    image : np.ndarray, shape (H, W, 3), dtype uint8, RGB
    mask  : np.ndarray, shape (H, W), dtype uint8, values {0, 255}
    components : list of ComponentData
    """
    image = load_tiff(tiff_path, level=tiff_level, as_rgb=True)
    mask = load_mask(mask_path, threshold=mask_threshold)

    if image.shape[:2] != mask.shape:
        _logger.warning(
            f"Image shape {image.shape[:2]} ≠ mask shape {mask.shape}. "
            "Resizing mask to match image."
        )
        mask = cv2.resize(
            mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    components = detect_components(
        mask,
        image=image,
        min_area_px=min_area_px,
        max_components=max_components,
        margin_px=margin_px,
    )
    return image, mask, components
