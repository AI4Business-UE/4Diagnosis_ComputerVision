"""
src/shape_features.py — Shape Descriptor Extraction
=====================================================

This module implements global shape descriptors that capture the
**silhouette geometry** of a tissue component beyond simple area/perimeter.

Theory
------
Unlike geometric features (which summarise the region as a whole), shape
descriptors encode the **boundary or contour structure** and are designed
to be invariant to translation, rotation, and scale.

Descriptors implemented here:

1. **Hu Moments** (Hu, 1962)
   Seven algebraically independent moment invariants derived from
   central moments.  They are invariant to translation, scale, and rotation
   (the last one changes sign under reflection).
   The logarithm is taken to compress the dynamic range.

2. **Fourier Descriptors**
   The contour (x, y) is treated as a complex signal z(t) = x(t) + iy(t).
   Its Discrete Fourier Transform captures the frequency content of the
   boundary.  Low-frequency coefficients describe the global shape;
   high-frequency coefficients describe fine details.
   Normalisation for translation, scale, and rotation invariance follows
   the method of Granlund (1972).

3. **Shape Context** (Belongie, Malik, Puzicha 2002)
   For each sampled contour point, compute a log-polar histogram of the
   directions to all other points.  The result captures the spatial
   arrangement of the boundary around each sample point.
   Matching two shapes = finding the minimum-cost assignment of histograms.

4. **Hausdorff Distance** (Huttenlocher et al., 1993)
   The maximum of the minimum distances from every point of set A to the
   nearest point of set B, and vice versa.
   H(A, B) = max(h(A,B), h(B,A))
   Measures the worst-case mismatch between two point sets.

5. **Chamfer Distance**
   Average (rather than maximum) of the minimum distances.
   Less sensitive to outliers than Hausdorff.

6. **Contour Matching** via cv2.matchShapes
   Uses Hu moments internally.  Three matching methods are available:
   I1 (Hu), I2 (correlation), I3 (chi-square).
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from skimage.morphology import skeletonize

_logger = logging.getLogger("SliceGrouper.shape")


# ---------------------------------------------------------------------------
# Hu Moments
# ---------------------------------------------------------------------------


def compute_hu_moments(
    contour: np.ndarray,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """
    Compute the seven log-transformed Hu moment invariants.

    Hu moments are invariant to translation, scale, and rotation.
    The log transform is applied to make the values more numerically
    comparable across components of very different sizes.

    Formula: huᵢ = -sign(mᵢ) * log10(|mᵢ| + ε)

    Parameters
    ----------
    contour : np.ndarray, shape (N, 1, 2)
        Component contour.
    epsilon : float
        Small constant to avoid log(0).

    Returns
    -------
    np.ndarray, shape (7,)
        Log-transformed Hu moment invariants.
    """
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()
    log_hu = -np.sign(hu) * np.log10(np.abs(hu) + epsilon)
    return log_hu.astype(np.float64)


# ---------------------------------------------------------------------------
# Fourier Descriptors
# ---------------------------------------------------------------------------


def _resample_contour(
    contour: np.ndarray,
    n_points: int = 128,
) -> np.ndarray:
    """
    Resample a contour to exactly ``n_points`` points using linear interpolation.

    Parameters
    ----------
    contour : np.ndarray, shape (N, 1, 2)
    n_points : int

    Returns
    -------
    np.ndarray, shape (n_points, 2)
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    N = len(pts)
    if N == 0:
        return np.zeros((n_points, 2))
    # Compute cumulative arc-length
    diff = np.diff(pts, axis=0)
    dist = np.sqrt((diff ** 2).sum(axis=1))
    cumlen = np.concatenate([[0], np.cumsum(dist)])
    total_len = cumlen[-1]
    if total_len < 1e-6:
        return np.zeros((n_points, 2))
    target = np.linspace(0, total_len, n_points, endpoint=False)
    xs = np.interp(target, cumlen, pts[:, 0])
    ys = np.interp(target, cumlen, pts[:, 1])
    return np.column_stack([xs, ys])


def compute_fourier_descriptors(
    contour: np.ndarray,
    n_descriptors: int = 64,
    n_resample: int = 128,
) -> np.ndarray:
    """
    Compute normalised Fourier descriptors from a contour.

    Steps
    -----
    1. Resample contour to ``n_resample`` equally-spaced points.
    2. Represent as complex z(t) = x(t) + j*y(t).
    3. Compute DFT: Z = FFT(z).
    4. Normalise for translation (Z[0] = 0), scale (|Z[1]| = 1),
       and rotation (arg(Z[1]) = 0).
    5. Return the first ``n_descriptors`` normalised coefficients
       (excluding DC component).

    Parameters
    ----------
    contour : np.ndarray, shape (N, 1, 2)
    n_descriptors : int
        Number of Fourier coefficients to retain.
    n_resample : int
        Number of contour points before FFT.

    Returns
    -------
    np.ndarray, shape (n_descriptors * 2,)
        Interleaved real and imaginary parts of Fourier coefficients.
    """
    pts = _resample_contour(contour, n_resample)
    z = pts[:, 0] + 1j * pts[:, 1]
    Z = np.fft.fft(z)

    # Translation invariance: remove DC component
    Z[0] = 0.0

    # Scale + rotation invariance: normalise by first harmonic
    if abs(Z[1]) > 1e-10:
        Z = Z / abs(Z[1])
        Z = Z * np.exp(-1j * np.angle(Z[1]))

    # Retain first n_descriptors (skip index 0)
    coeffs = Z[1 : n_descriptors + 1]
    if len(coeffs) < n_descriptors:
        coeffs = np.pad(coeffs, (0, n_descriptors - len(coeffs)))

    return np.concatenate([coeffs.real, coeffs.imag]).astype(np.float64)


# ---------------------------------------------------------------------------
# Shape Context
# ---------------------------------------------------------------------------


def compute_shape_context(
    contour: np.ndarray,
    n_bins_r: int = 5,
    n_bins_theta: int = 12,
    n_resample: int = 64,
) -> np.ndarray:
    """
    Compute a global shape context descriptor (mean over all sample points).

    The shape context descriptor for a point p is a log-polar histogram of
    directions and distances to all other contour points.  For matching two
    shapes, one would compare histograms point-by-point and solve an
    assignment problem.  Here we compute the **mean histogram** to obtain a
    single descriptor per component.

    Parameters
    ----------
    contour : np.ndarray, shape (N, 1, 2)
    n_bins_r : int
        Number of radial bins (log-spaced).
    n_bins_theta : int
        Number of angular bins (uniform, 2π / n_bins_theta each).
    n_resample : int
        Number of contour sample points.

    Returns
    -------
    np.ndarray, shape (n_bins_r * n_bins_theta,)
        Normalised mean shape context histogram.
    """
    pts = _resample_contour(contour, n_resample)  # (n_resample, 2)
    n = len(pts)

    # Compute all pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 2)
    r = np.sqrt((diff ** 2).sum(axis=-1))  # (n, n)
    theta = np.arctan2(diff[:, :, 1], diff[:, :, 0])  # (n, n)

    # Normalise r by mean distance
    mean_r = r[r > 0].mean() if (r > 0).any() else 1.0
    r = r / (mean_r + 1e-10)

    # Log-spaced radial bins
    r_bins = np.logspace(-1, 2, n_bins_r + 1)

    # Angular bins: [0, 2π)
    theta_shifted = (theta + math.pi) / (2 * math.pi)  # [0, 1)
    theta_bins = np.linspace(0, 1, n_bins_theta + 1)

    # Build histogram for each reference point, then average
    histograms = np.zeros((n, n_bins_r * n_bins_theta))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r_val = r[i, j]
            t_val = theta_shifted[i, j] % 1.0
            r_idx = np.searchsorted(r_bins, r_val) - 1
            t_idx = int(t_val * n_bins_theta)
            r_idx = np.clip(r_idx, 0, n_bins_r - 1)
            t_idx = np.clip(t_idx, 0, n_bins_theta - 1)
            flat_idx = r_idx * n_bins_theta + t_idx
            histograms[i, flat_idx] += 1

    # Normalise each histogram
    row_sums = histograms.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-10, 1.0, row_sums)
    histograms /= row_sums

    return histograms.mean(axis=0).astype(np.float64)


# ---------------------------------------------------------------------------
# Hausdorff & Chamfer Distances
# ---------------------------------------------------------------------------


def hausdorff_distance(
    contour_a: np.ndarray,
    contour_b: np.ndarray,
    n_resample: int = 128,
) -> float:
    """
    Compute the directed Hausdorff distance between two contours.

    H(A, B) = max(h(A, B), h(B, A))
    where h(A, B) = max_{a ∈ A} min_{b ∈ B} ||a - b||

    The Hausdorff distance equals the longest minimum distance any point
    on one contour must travel to reach the other contour.

    Parameters
    ----------
    contour_a, contour_b : np.ndarray, shape (N, 1, 2)
    n_resample : int
        Both contours are resampled to this many points for comparable sampling.

    Returns
    -------
    float
        Hausdorff distance in pixels.
    """
    pts_a = _resample_contour(contour_a, n_resample)
    pts_b = _resample_contour(contour_b, n_resample)
    d_ab = directed_hausdorff(pts_a, pts_b)[0]
    d_ba = directed_hausdorff(pts_b, pts_a)[0]
    return float(max(d_ab, d_ba))


def chamfer_distance(
    contour_a: np.ndarray,
    contour_b: np.ndarray,
    n_resample: int = 128,
) -> float:
    """
    Compute the symmetric mean Chamfer distance between two contours.

    Chamfer(A, B) = (mean_{a ∈ A} min_{b ∈ B} ||a-b|| +
                     mean_{b ∈ B} min_{a ∈ A} ||b-a||) / 2

    Less sensitive to outliers than Hausdorff because it averages rather
    than takes the maximum.

    Parameters
    ----------
    contour_a, contour_b : np.ndarray, shape (N, 1, 2)
    n_resample : int

    Returns
    -------
    float
        Symmetric Chamfer distance in pixels.
    """
    from scipy.spatial import cKDTree

    pts_a = _resample_contour(contour_a, n_resample)
    pts_b = _resample_contour(contour_b, n_resample)

    tree_b = cKDTree(pts_b)
    tree_a = cKDTree(pts_a)

    dist_ab, _ = tree_b.query(pts_a)
    dist_ba, _ = tree_a.query(pts_b)

    return float((dist_ab.mean() + dist_ba.mean()) / 2.0)


# ---------------------------------------------------------------------------
# Skeleton comparison
# ---------------------------------------------------------------------------


def compare_skeletons(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> float:
    """
    Compare two binary masks by computing the Chamfer distance between
    their skeletons.

    Skeleton comparison captures the structural/topological similarity
    of tissue fragments: two biopsy cores from the same specimen will
    tend to have similarly shaped medial axes.

    Parameters
    ----------
    mask_a, mask_b : np.ndarray, shape (H, W), dtype uint8

    Returns
    -------
    float
        Chamfer distance between skeleton pixel clouds (pixels).
    """
    from scipy.spatial import cKDTree

    def _skeleton_points(mask: np.ndarray) -> np.ndarray:
        binary = (mask > 0).astype(np.uint8)
        skel = skeletonize(binary)
        ys, xs = np.where(skel)
        return np.column_stack([xs, ys]).astype(np.float64)

    pts_a = _skeleton_points(mask_a)
    pts_b = _skeleton_points(mask_b)

    if len(pts_a) == 0 or len(pts_b) == 0:
        return float("inf")

    tree_b = cKDTree(pts_b)
    tree_a = cKDTree(pts_a)
    d_ab, _ = tree_b.query(pts_a)
    d_ba, _ = tree_a.query(pts_b)
    return float((d_ab.mean() + d_ba.mean()) / 2.0)


# ---------------------------------------------------------------------------
# Contour matching via Hu moments
# ---------------------------------------------------------------------------


def contour_matching_score(
    contour_a: np.ndarray,
    contour_b: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compare two contours using OpenCV's Hu-moment-based shape matching.

    Three methods are available (see OpenCV documentation for formulas):
    - Method 1 (I1): Sum of reciprocals of products.
    - Method 2 (I2): Sum of squared differences.
    - Method 3 (I3): Maximum of relative differences.

    Lower values indicate greater similarity.  A value of 0 means identical
    shapes (within floating-point precision).

    Parameters
    ----------
    contour_a, contour_b : np.ndarray, shape (N, 1, 2)

    Returns
    -------
    tuple of float
        (score_I1, score_I2, score_I3)
    """
    s1 = cv2.matchShapes(contour_a, contour_b, cv2.CONTOURS_MATCH_I1, 0.0)
    s2 = cv2.matchShapes(contour_a, contour_b, cv2.CONTOURS_MATCH_I2, 0.0)
    s3 = cv2.matchShapes(contour_a, contour_b, cv2.CONTOURS_MATCH_I3, 0.0)
    return float(s1), float(s2), float(s3)


# ---------------------------------------------------------------------------
# High-level extractor
# ---------------------------------------------------------------------------


def extract_shape_features(
    component: "ComponentData",  # type: ignore[name-defined]
    n_fourier: int = 64,
    n_shape_context_r: int = 5,
    n_shape_context_theta: int = 12,
    n_resample: int = 128,
    hu_epsilon: float = 1e-10,
) -> dict:
    """
    Compute all shape features for a single component and return them
    as a plain dictionary.

    Parameters
    ----------
    component : ComponentData
    n_fourier : int
        Number of Fourier coefficients.
    n_shape_context_r : int
        Radial bins for shape context.
    n_shape_context_theta : int
        Angular bins for shape context.
    n_resample : int
        Contour resampling resolution.
    hu_epsilon : float
        Epsilon for log-Hu computation.

    Returns
    -------
    dict with keys:
        'hu_moments'         → np.ndarray shape (7,)
        'fourier_descriptors'→ np.ndarray shape (n_fourier * 2,)
        'shape_context'      → np.ndarray shape (n_shape_context_r * n_shape_context_theta,)
    """
    contour = component.contour
    return {
        "hu_moments": compute_hu_moments(contour, epsilon=hu_epsilon),
        "fourier_descriptors": compute_fourier_descriptors(
            contour, n_descriptors=n_fourier, n_resample=n_resample
        ),
        "shape_context": compute_shape_context(
            contour,
            n_bins_r=n_shape_context_r,
            n_bins_theta=n_shape_context_theta,
            n_resample=min(n_resample, 64),
        ),
    }


def shape_features_to_vector(shape_dict: dict) -> np.ndarray:
    """
    Flatten the shape feature dictionary into a single 1-D vector.

    Parameters
    ----------
    shape_dict : dict
        As returned by ``extract_shape_features``.

    Returns
    -------
    np.ndarray, shape (7 + n_fourier*2 + n_r*n_theta,)
    """
    parts = [
        shape_dict["hu_moments"],
        shape_dict["fourier_descriptors"],
        shape_dict["shape_context"],
    ]
    return np.concatenate(parts).astype(np.float64)
