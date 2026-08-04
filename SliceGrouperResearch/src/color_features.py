"""
src/color_features.py — Color Feature Extraction
=================================================

Color is one of the strongest cues for grouping tissue fragments from
the same histopathological specimen.  Fragments from the same biopsy
core processed together in the same paraffin block will share:

  • Staining intensity and hue (same dye batch, same protocol)
  • Background staining patterns
  • Tissue-type color distributions (e.g., Masson's trichrome gives
    collagen a distinct blue; haematoxylin stains nuclei purple)

This module extracts:
  1. Per-channel statistics (mean, std, skewness) in RGB, HSV, and Lab.
  2. Normalised histograms in each color space.
  3. Histogram comparison metrics (correlation, Chi-squared, intersection,
     Bhattacharyya distance).
  4. Stain normalisation (Macenko method) to remove scanner / batch effects
     before feature comparison.

Color Space Notes
-----------------
RGB   — device-dependent but fast. Useful for simple comparisons.
HSV   — Hue separates color from intensity; useful for ignoring brightness.
Lab   — Perceptually uniform. L = lightness, a = green↔red, b = blue↔yellow.
        Ideal for comparing perceived color appearance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import stats as scipy_stats

_logger = logging.getLogger("SliceGrouper.color")


# ---------------------------------------------------------------------------
# ColorFeatures dataclass
# ---------------------------------------------------------------------------


@dataclass
class ColorFeatures:
    """
    Bundle of all color descriptors for a single component.

    Attributes
    ----------
    rgb_stats : np.ndarray, shape (9,)
        [mean_R, mean_G, mean_B, std_R, std_G, std_B, skew_R, skew_G, skew_B]
    hsv_stats : np.ndarray, shape (9,)
        Same structure in HSV space.
    lab_stats : np.ndarray, shape (9,)
        Same structure in Lab space.
    rgb_hist : np.ndarray, shape (bins * 3,)
        Concatenated per-channel RGB histograms.
    hsv_hist : np.ndarray, shape (bins * 3,)
        Concatenated per-channel HSV histograms.
    lab_hist : np.ndarray, shape (bins * 3,)
        Concatenated per-channel Lab histograms.
    """
    rgb_stats: np.ndarray = field(default_factory=lambda: np.zeros(9))
    hsv_stats: np.ndarray = field(default_factory=lambda: np.zeros(9))
    lab_stats: np.ndarray = field(default_factory=lambda: np.zeros(9))
    rgb_hist: np.ndarray = field(default_factory=lambda: np.zeros(96))
    hsv_hist: np.ndarray = field(default_factory=lambda: np.zeros(96))
    lab_hist: np.ndarray = field(default_factory=lambda: np.zeros(96))

    def to_vector(self) -> np.ndarray:
        """
        Flatten all color features into a single 1-D array.

        Returns
        -------
        np.ndarray
        """
        return np.concatenate([
            self.rgb_stats,
            self.hsv_stats,
            self.lab_stats,
            self.rgb_hist,
            self.hsv_hist,
            self.lab_hist,
        ]).astype(np.float64)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _channel_stats(pixels: np.ndarray) -> np.ndarray:
    """
    Compute mean, std, and skewness for a 1-D pixel array.

    Parameters
    ----------
    pixels : np.ndarray, shape (N,)
        Tissue pixel values for one channel.

    Returns
    -------
    np.ndarray, shape (3,)  [mean, std, skewness]
    """
    if len(pixels) == 0:
        return np.zeros(3)
    m = float(np.mean(pixels))
    s = float(np.std(pixels))
    sk = float(scipy_stats.skew(pixels))
    return np.array([m, s, sk])


def _extract_stats(
    rgb: np.ndarray,
    mask: np.ndarray,
    color_space: str,
) -> np.ndarray:
    """
    Extract per-channel statistics for a given color space.

    Parameters
    ----------
    rgb : np.ndarray, shape (H, W, 3), dtype uint8
        RGB image of the component region.
    mask : np.ndarray, shape (H, W), dtype uint8
        Binary mask (255 = tissue).
    color_space : str
        'rgb', 'hsv', or 'lab'.

    Returns
    -------
    np.ndarray, shape (9,)
        [mean_c1, mean_c2, mean_c3, std_c1, ..., skew_c3]
    """
    if color_space == "rgb":
        img = rgb.astype(np.float32)
    elif color_space == "hsv":
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    elif color_space == "lab":
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab).astype(np.float32)
    else:
        raise ValueError(f"Unknown color_space: {color_space!r}")

    tissue = mask > 0
    stats = []
    for ch in range(3):
        pixels = img[:, :, ch][tissue]
        stats.append(_channel_stats(pixels))

    # stats is list of 3 arrays of shape (3,) → flatten to (9,)
    means = np.array([s[0] for s in stats])
    stds  = np.array([s[1] for s in stats])
    skews = np.array([s[2] for s in stats])
    return np.concatenate([means, stds, skews])


def extract_rgb_stats(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return mean/std/skewness for each RGB channel over tissue pixels."""
    return _extract_stats(rgb, mask, "rgb")


def extract_hsv_stats(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return mean/std/skewness for each HSV channel over tissue pixels."""
    return _extract_stats(rgb, mask, "hsv")


def extract_lab_stats(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return mean/std/skewness for each Lab channel over tissue pixels."""
    return _extract_stats(rgb, mask, "lab")


# ---------------------------------------------------------------------------
# Histogram extraction
# ---------------------------------------------------------------------------


def build_color_histogram(
    rgb: np.ndarray,
    mask: np.ndarray,
    color_space: str = "rgb",
    bins: int = 32,
) -> np.ndarray:
    """
    Build a normalised per-channel color histogram for tissue pixels.

    Parameters
    ----------
    rgb : np.ndarray, shape (H, W, 3), dtype uint8
    mask : np.ndarray, shape (H, W), dtype uint8
        Tissue mask (255 = tissue).
    color_space : str
        'rgb', 'hsv', or 'lab'.
    bins : int
        Number of bins per channel.

    Returns
    -------
    np.ndarray, shape (bins * 3,)
        Concatenated normalised per-channel histograms.
    """
    if color_space == "rgb":
        img = rgb
        ranges = [(0, 256)] * 3
    elif color_space == "hsv":
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        ranges = [(0, 181), (0, 256), (0, 256)]
    elif color_space == "lab":
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
        ranges = [(0, 256)] * 3
    else:
        raise ValueError(f"Unknown color_space: {color_space!r}")

    tissue_mask_uint8 = (mask > 0).astype(np.uint8) * 255
    hists = []
    for ch in range(3):
        lo, hi = ranges[ch]
        h = cv2.calcHist(
            [img],
            [ch],
            tissue_mask_uint8,
            [bins],
            [lo, hi],
        )
        h = h.flatten().astype(np.float64)
        total = h.sum()
        if total > 0:
            h /= total
        hists.append(h)
    return np.concatenate(hists)


# ---------------------------------------------------------------------------
# Histogram comparison
# ---------------------------------------------------------------------------


def compare_histograms(
    h1: np.ndarray,
    h2: np.ndarray,
    method: str = "bhattacharyya",
) -> float:
    """
    Compare two histograms using an OpenCV comparison method.

    Methods
    -------
    'correlation'     : Pearson correlation coefficient. Range [-1, 1].
                        Higher = more similar.
    'chi_squared'     : χ² distance. Range [0, ∞). Lower = more similar.
    'intersection'    : Sum of minimum values. Range [0, 1]. Higher = more similar.
    'bhattacharyya'   : Bhattacharyya distance. Range [0, 1]. Lower = more similar.

    Parameters
    ----------
    h1, h2 : np.ndarray
        Histograms to compare (must have the same shape).
    method : str
        Comparison method name.

    Returns
    -------
    float
        Similarity or distance value (interpretation depends on method).
    """
    method_map = {
        "correlation":   cv2.HISTCMP_CORREL,
        "chi_squared":   cv2.HISTCMP_CHISQR,
        "intersection":  cv2.HISTCMP_INTERSECT,
        "bhattacharyya": cv2.HISTCMP_BHATTACHARYYA,
    }
    if method not in method_map:
        raise ValueError(
            f"Unknown method {method!r}. Choose from {list(method_map.keys())}"
        )
    cv_method = method_map[method]
    h1f = h1.astype(np.float32).reshape(-1, 1)
    h2f = h2.astype(np.float32).reshape(-1, 1)
    return float(cv2.compareHist(h1f, h2f, cv_method))


# ---------------------------------------------------------------------------
# Stain normalisation — Macenko method
# ---------------------------------------------------------------------------


def _od_to_rgb(od: np.ndarray) -> np.ndarray:
    """Convert optical density to RGB. od = -log(I/I0), I0=255."""
    return (np.exp(-od) * 255).clip(0, 255).astype(np.uint8)


def _rgb_to_od(rgb: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Convert RGB to optical density (OD). Uses I0 = 255."""
    rgb_f = rgb.astype(np.float64) + 1e-6  # avoid log(0)
    od = -np.log(rgb_f / 255.0)
    return od


def normalize_stain_macenko(
    rgb: np.ndarray,
    mask: Optional[np.ndarray] = None,
    Io: int = 240,
    alpha: float = 1.0,
    beta: float = 0.15,
) -> np.ndarray:
    """
    Macenko stain normalisation for H&E histopathology images.

    Reference: Macenko et al. (2009), "A method for normalizing histology
    slides for quantitative analysis".

    Algorithm
    ---------
    1. Convert RGB to optical density (OD) space.
    2. Remove low-OD pixels (background).
    3. Compute SVD of the OD matrix → find stain directions.
    4. Project OD vectors onto stain plane and find extreme angles
       (α percentile / (100-α percentile) → stain vectors H and E).
    5. Project pixels onto H/E and rescale to reference concentrations.

    Parameters
    ----------
    rgb : np.ndarray, shape (H, W, 3), dtype uint8
        Input RGB image (tissue region).
    mask : np.ndarray or None, shape (H, W), dtype uint8
        Tissue mask. If None, all non-background pixels are used.
    Io : int
        Reference maximum intensity (default 240 for H&E slides).
    alpha : float
        Percentile for angle computation (default 1.0).
    beta : float
        OD threshold below which pixels are considered background.

    Returns
    -------
    np.ndarray, shape (H, W, 3), dtype uint8
        Stain-normalised image.
    """
    try:
        return _macenko_impl(rgb, mask, Io, alpha, beta)
    except Exception as exc:
        _logger.warning(f"Macenko normalisation failed ({exc}), returning original.")
        return rgb


def _macenko_impl(
    rgb: np.ndarray,
    mask: Optional[np.ndarray],
    Io: int,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Internal Macenko implementation."""
    h, w, _ = rgb.shape

    # Reference stain matrix for H&E (from Macenko 2009)
    # Row 0 = Haematoxylin, Row 1 = Eosin
    HE_REF = np.array([
        [0.5626, 0.2159],
        [0.7201, 0.8012],
        [0.4062, 0.5581],
    ])
    MAX_C_REF = np.array([1.9705, 1.0308])

    # Flatten to (N, 3)
    img = rgb.reshape(-1, 3).astype(np.float64)

    # Optional mask
    if mask is not None:
        tissue_pixels = img[mask.reshape(-1) > 0]
    else:
        tissue_pixels = img

    # Convert to OD
    OD = -np.log((tissue_pixels + 1) / Io)

    # Remove background (low OD)
    OD_mask = (OD > beta).all(axis=1)
    OD_hat = OD[OD_mask]

    if len(OD_hat) < 10:
        return rgb

    # SVD
    _, _, Vt = np.linalg.svd(OD_hat, full_matrices=False)
    plane = Vt[:2].T  # (3, 2) — two principal stain directions

    # Project onto stain plane
    proj = OD_hat @ plane  # (N, 2)

    # Find angle of each projected point
    phi = np.arctan2(proj[:, 1], proj[:, 0])

    # Stain vectors from percentile angles
    min_phi = np.percentile(phi, alpha)
    max_phi = np.percentile(phi, 100 - alpha)

    v1 = plane @ np.array([np.cos(min_phi), np.sin(min_phi)])
    v2 = plane @ np.array([np.cos(max_phi), np.sin(max_phi)])

    # Ensure haematoxylin has the larger blue component
    if v1[2] < v2[2]:
        HE = np.column_stack([v1, v2])
    else:
        HE = np.column_stack([v2, v1])

    # Solve for concentrations of each stain in all pixels
    OD_all = -np.log((img + 1) / Io)
    C, _, _, _ = np.linalg.lstsq(HE, OD_all.T, rcond=None)  # (2, N)

    # Normalise concentrations
    max_C = np.percentile(C, 99, axis=1)
    C = C / (max_C[:, np.newaxis] + 1e-6) * MAX_C_REF[:, np.newaxis]

    # Reconstruct using reference stain matrix
    OD_norm = HE_REF @ C
    img_norm = (Io * np.exp(-OD_norm.T)).clip(0, 255).astype(np.uint8)
    return img_norm.reshape(h, w, 3)


# ---------------------------------------------------------------------------
# Vahadane normalisation (SPAMS-based, optional)
# ---------------------------------------------------------------------------


def normalize_stain_vahadane(
    rgb: np.ndarray,
    reference_rgb: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Vahadane stain normalisation using structure-preserving sparse NMF.

    Requires ``staintools`` package: pip install staintools

    Falls back to Macenko if staintools is unavailable.

    Parameters
    ----------
    rgb : np.ndarray, shape (H, W, 3), dtype uint8
    reference_rgb : np.ndarray or None
        Reference image. If None, Macenko is used as fallback.

    Returns
    -------
    np.ndarray, shape (H, W, 3), dtype uint8
    """
    try:
        import staintools  # type: ignore[import]
        normalizer = staintools.StainNormalizer(method="vahadane")
        if reference_rgb is not None:
            normalizer.fit(reference_rgb)
            return normalizer.transform(rgb)
        else:
            return normalize_stain_macenko(rgb)
    except ImportError:
        _logger.debug("staintools not available; falling back to Macenko.")
        return normalize_stain_macenko(rgb)
    except Exception as exc:
        _logger.warning(f"Vahadane failed ({exc}); falling back to Macenko.")
        return normalize_stain_macenko(rgb)


# ---------------------------------------------------------------------------
# High-level extractor
# ---------------------------------------------------------------------------


def extract_color_features(
    component: "ComponentData",  # type: ignore[name-defined]
    histogram_bins: int = 32,
    normalize: bool = False,
    reference_rgb: Optional[np.ndarray] = None,
) -> ColorFeatures:
    """
    Compute all color features for a single ``ComponentData`` object.

    Parameters
    ----------
    component : ComponentData
    histogram_bins : int
        Number of histogram bins per channel.
    normalize : bool
        If True, apply Macenko stain normalisation before feature extraction.
    reference_rgb : np.ndarray or None
        Reference image for stain normalisation (only used if normalize=True).

    Returns
    -------
    ColorFeatures
    """
    rgb = component.rgb_crop
    mask = component.mask_crop

    if normalize:
        rgb = normalize_stain_macenko(rgb, mask=mask)

    rgb_stats = extract_rgb_stats(rgb, mask)
    hsv_stats = extract_hsv_stats(rgb, mask)
    lab_stats = extract_lab_stats(rgb, mask)

    rgb_hist = build_color_histogram(rgb, mask, "rgb", histogram_bins)
    hsv_hist = build_color_histogram(rgb, mask, "hsv", histogram_bins)
    lab_hist = build_color_histogram(rgb, mask, "lab", histogram_bins)

    return ColorFeatures(
        rgb_stats=rgb_stats,
        hsv_stats=hsv_stats,
        lab_stats=lab_stats,
        rgb_hist=rgb_hist,
        hsv_hist=hsv_hist,
        lab_hist=lab_hist,
    )
