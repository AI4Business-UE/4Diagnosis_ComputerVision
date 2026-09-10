"""
src/geometry.py — Geometric Feature Extraction
===============================================

This module computes classical shape/geometry descriptors for each tissue
component.  All descriptors return numerical vectors that can be combined
into a feature matrix for similarity computation.

Theory
------
Geometric features describe the **overall form** of a tissue fragment without
looking at its internal pixel intensities.  They are invariant to staining
differences and scanner calibration, making them robust first-pass indicators
of whether two fragments could come from the same biopsy core.

Key metrics and their intuitions:
  • Area          — absolute size. Fragments from the same biopsy core tend to
                    have similar areas (same tissue volume per serial section).
  • Perimeter     — total boundary length.  Smooth vs. jagged boundaries
                    indicate processing quality differences.
  • Circularity   — 4πA / P².  A perfect disc = 1.0. Elongated cores ≈ 0.
  • Solidity      — area / convex-hull area.  Values close to 1 indicate compact,
                    solid fragments; low values indicate concave, spiky shapes.
  • Aspect ratio  — width / height of the bounding box.
  • Eccentricity  — how elongated the equivalent ellipse is. 0 = circle, 1 = line.
  • Orientation   — angle of the major axis.  Consistent orientation across
                    the slide can indicate alignment of biopsy cores.
  • PCA axes      — principal directions of the binary mask point cloud.
  • Skeleton len  — length of the medial axis.  For biopsy cores, this closely
                    approximates the core's physical length.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize

_logger = logging.getLogger("SliceGrouper.geometry")


# ---------------------------------------------------------------------------
# GeometryFeatures dataclass
# ---------------------------------------------------------------------------


@dataclass
class GeometryFeatures:
    """
    Bundle of all geometric descriptors for a single component.

    All fields default to 0.0 so that callers that cannot compute a particular
    metric (e.g., too few contour points for ellipse fitting) still get a
    well-defined object.

    Attributes
    ----------
    area : float
        Number of tissue pixels.
    perimeter : float
        Contour perimeter in pixels.
    hull_area : float
        Area of the convex hull in pixels.
    solidity : float
        area / hull_area. Range [0, 1].
    circularity : float
        4 * π * area / perimeter². Range [0, 1].
    compactness : float
        perimeter² / area.  Inverse of circularity (unnormalized).
    aspect_ratio : float
        bbox_width / bbox_height.
    eccentricity : float
        Eccentricity of the fitted ellipse. 0 = circle, 1 = line.
    orientation_deg : float
        Angle (degrees) of the major axis relative to horizontal.
    pca_axis1 : np.ndarray, shape (2,)
        First principal direction of the mask pixel cloud.
    pca_axis2 : np.ndarray, shape (2,)
        Second principal direction.
    pca_variance_ratio : np.ndarray, shape (2,)
        Fraction of variance explained by each PCA axis.
    skeleton_length : float
        Total length of the medial axis skeleton in pixels.
    """

    area: float = 0.0
    perimeter: float = 0.0
    hull_area: float = 0.0
    solidity: float = 0.0
    circularity: float = 0.0
    compactness: float = 0.0
    aspect_ratio: float = 0.0
    eccentricity: float = 0.0
    orientation_deg: float = 0.0
    pca_axis1: np.ndarray = field(default_factory=lambda: np.zeros(2))
    pca_axis2: np.ndarray = field(default_factory=lambda: np.zeros(2))
    pca_variance_ratio: np.ndarray = field(default_factory=lambda: np.zeros(2))
    skeleton_length: float = 0.0

    def to_vector(self) -> np.ndarray:
        """
        Flatten all scalar features into a 1-D numpy array.

        Included features (in order):
        area, perimeter, hull_area, solidity, circularity, compactness,
        aspect_ratio, eccentricity, orientation_deg, skeleton_length,
        pca_variance_ratio[0], pca_variance_ratio[1].

        Returns
        -------
        np.ndarray, shape (12,)
        """
        return np.array([
            self.area,
            self.perimeter,
            self.hull_area,
            self.solidity,
            self.circularity,
            self.compactness,
            self.aspect_ratio,
            self.eccentricity,
            self.orientation_deg,
            self.skeleton_length,
            self.pca_variance_ratio[0],
            self.pca_variance_ratio[1],
        ], dtype=np.float64)

    @staticmethod
    def feature_names() -> list[str]:
        """Return feature names in the same order as ``to_vector``."""
        return [
            "area", "perimeter", "hull_area", "solidity",
            "circularity", "compactness", "aspect_ratio",
            "eccentricity", "orientation_deg", "skeleton_length",
            "pca_var_ratio_0", "pca_var_ratio_1",
        ]


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def compute_area(mask: np.ndarray) -> float:
    """
    Count the number of tissue pixels in a binary mask.

    Parameters
    ----------
    mask : np.ndarray, shape (H, W), dtype uint8
        Binary mask with tissue pixels == 255.

    Returns
    -------
    float
        Total tissue area in pixels.
    """
    return float(np.count_nonzero(mask))


def compute_perimeter(contour: np.ndarray, closed: bool = True) -> float:
    """
    Compute the perimeter of a contour.

    Uses ``cv2.arcLength`` which sums Euclidean distances between
    consecutive contour vertices.

    Parameters
    ----------
    contour : np.ndarray, shape (N, 1, 2)
        Contour as returned by ``cv2.findContours``.
    closed : bool
        Whether to treat the contour as closed.

    Returns
    -------
    float
        Perimeter in pixels.
    """
    if contour is None or len(contour) < 2:
        return 0.0
    return float(cv2.arcLength(contour, closed))


def compute_convex_hull(
    contour: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Compute the convex hull of a contour and its enclosed area.

    Parameters
    ----------
    contour : np.ndarray, shape (N, 1, 2)

    Returns
    -------
    hull_points : np.ndarray
        Convex hull point array (from cv2.convexHull).
    hull_area : float
        Area enclosed by the convex hull in pixels.
    """
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    return hull, hull_area


def compute_solidity(area: float, hull_area: float) -> float:
    """
    Ratio of component area to its convex hull area.

    Solidity = area / hull_area.

    A value of 1.0 indicates a perfectly convex shape.
    Lower values indicate concavities (e.g. C-shapes, fragmented edges).

    Parameters
    ----------
    area : float
    hull_area : float

    Returns
    -------
    float in [0, 1].
    """
    if hull_area < 1e-6:
        return 0.0
    return min(1.0, area / hull_area)


def compute_circularity(area: float, perimeter: float) -> float:
    """
    Measure how circle-like a shape is.

    Formula: C = 4π * A / P²

    For a perfect circle: C = 1.0.
    For any other shape: C < 1.0.

    This is the standard ISO circularity definition used in image analysis.

    Parameters
    ----------
    area : float
        Component area in pixels.
    perimeter : float
        Contour perimeter in pixels.

    Returns
    -------
    float in [0, 1].
    """
    if perimeter < 1e-6:
        return 0.0
    return min(1.0, 4.0 * math.pi * area / (perimeter ** 2))


def compute_compactness(area: float, perimeter: float) -> float:
    """
    Inverse circularity measure: P² / A.

    Unlike circularity, compactness is unbounded above (a circle has the
    minimum value 4π ≈ 12.57).  High compactness indicates complex boundaries.

    Parameters
    ----------
    area : float
    perimeter : float

    Returns
    -------
    float
    """
    if area < 1e-6:
        return 0.0
    return (perimeter ** 2) / area


def compute_aspect_ratio(bbox: Tuple[int, int, int, int]) -> float:
    """
    Bounding box width-to-height ratio.

    Values > 1 indicate wider-than-tall components (landscape orientation).
    Values < 1 indicate taller-than-wide (portrait).

    Parameters
    ----------
    bbox : (x, y, w, h)

    Returns
    -------
    float
    """
    _, _, w, h = bbox
    if h == 0:
        return 0.0
    return w / h


def compute_eccentricity_and_orientation(
    contour: np.ndarray,
    min_points: int = 5,
) -> Tuple[float, float]:
    """
    Fit an ellipse to the contour and extract eccentricity and orientation.

    **Eccentricity** describes how elongated the fitted ellipse is:
      e = sqrt(1 - (b/a)²)  where a ≥ b are semi-axes.
    e = 0 → circle, e → 1 → infinitely thin ellipse.

    **Orientation** is the angle of the major axis measured in degrees
    counter-clockwise from the positive x-axis, in the range [-90°, 90°].

    Parameters
    ----------
    contour : np.ndarray, shape (N, 1, 2)
    min_points : int
        Minimum number of contour points needed for ellipse fitting.

    Returns
    -------
    eccentricity : float in [0, 1]
    orientation_deg : float in [-90, 90]
    """
    if contour is None or len(contour) < min_points:
        return 0.0, 0.0
    try:
        (_, _), (ma, Mi), angle = cv2.fitEllipse(contour)
        # ma = major axis length, Mi = minor axis length
        a = max(ma, Mi) / 2.0
        b = min(ma, Mi) / 2.0
        if a < 1e-6:
            return 0.0, float(angle)
        e = math.sqrt(max(0.0, 1.0 - (b / a) ** 2))
        return float(e), float(angle)
    except cv2.error:
        return 0.0, 0.0


def compute_pca_axes(
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply PCA to the binary mask pixel cloud to find principal axes.

    The mask pixel positions (y, x) are treated as a 2-D point cloud.
    PCA finds the two orthogonal directions that capture the most variance,
    which correspond to the major and minor axes of the tissue fragment.

    Parameters
    ----------
    mask : np.ndarray, shape (H, W), dtype uint8
        Binary mask with tissue pixels == 255.

    Returns
    -------
    axis1 : np.ndarray, shape (2,)
        Direction of maximum variance (normalised).
    axis2 : np.ndarray, shape (2,)
        Direction of minimum variance (normalised), perpendicular to axis1.
    variance_ratio : np.ndarray, shape (2,)
        Fraction of total variance in each principal direction.
    """
    # Collect tissue pixel coordinates
    ys, xs = np.where(mask > 0)
    if len(xs) < 3:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5, 0.5])

    points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    mean = points.mean(axis=0)
    centered = points - mean

    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # eigh returns ascending order → flip to descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    total_var = eigenvalues.sum()
    if total_var < 1e-12:
        var_ratio = np.array([0.5, 0.5])
    else:
        var_ratio = eigenvalues / total_var

    axis1 = eigenvectors[:, 0]
    axis2 = eigenvectors[:, 1]

    return axis1, axis2, var_ratio


def compute_skeleton_length(
    mask: np.ndarray,
    method: str = "lee",
) -> float:
    """
    Compute the length of the medial axis skeleton.

    The skeleton (thin version of the binary shape) approximates the
    centreline of the tissue fragment.  For biopsy cores, skeleton length
    corresponds closely to the physical core length.

    Algorithm
    ---------
    1. Skeletonize the binary mask using the Lee (1994) thinning algorithm.
    2. Count the number of skeleton pixels (one pixel = one pixel of path).

    Note: This gives the number of 8-connected skeleton pixels, which is
    a close approximation of the true medial axis length.

    Parameters
    ----------
    mask : np.ndarray, shape (H, W), dtype uint8
        Binary mask with tissue pixels == 255.
    method : str
        Skeletonization method: 'lee' (default, slower but more accurate)
        or 'zhang' (faster).

    Returns
    -------
    float
        Skeleton length in pixels.
    """
    if mask is None or mask.sum() == 0:
        return 0.0
    binary = (mask > 0).astype(np.uint8)
    skeleton = skeletonize(binary, method=method)
    return float(skeleton.sum())


# ---------------------------------------------------------------------------
# High-level extractor
# ---------------------------------------------------------------------------


def extract_geometry_features(
    component: "ComponentData",  # type: ignore[name-defined]  # forward reference
    skeleton_method: str = "lee",
) -> GeometryFeatures:
    """
    Compute all geometric features for a single ``ComponentData`` object.

    This is the primary entry point for geometry feature extraction.
    It calls all individual metric functions and bundles results into a
    ``GeometryFeatures`` dataclass.

    Parameters
    ----------
    component : ComponentData
        A component produced by ``src.io.detect_components``.
    skeleton_method : str
        Skeletonization method ('lee' or 'zhang').

    Returns
    -------
    GeometryFeatures
    """
    mask = component.mask_crop
    contour = component.contour
    bbox = component.bbox

    area = compute_area(mask)
    perimeter = compute_perimeter(contour)
    _, hull_area = compute_convex_hull(contour)
    solidity = compute_solidity(area, hull_area)
    circularity = compute_circularity(area, perimeter)
    compactness = compute_compactness(area, perimeter)
    aspect_ratio = compute_aspect_ratio(bbox)
    eccentricity, orientation_deg = compute_eccentricity_and_orientation(contour)
    axis1, axis2, var_ratio = compute_pca_axes(mask)
    skeleton_len = compute_skeleton_length(mask, method=skeleton_method)

    return GeometryFeatures(
        area=area,
        perimeter=perimeter,
        hull_area=hull_area,
        solidity=solidity,
        circularity=circularity,
        compactness=compactness,
        aspect_ratio=aspect_ratio,
        eccentricity=eccentricity,
        orientation_deg=orientation_deg,
        pca_axis1=axis1,
        pca_axis2=axis2,
        pca_variance_ratio=var_ratio,
        skeleton_length=skeleton_len,
    )
