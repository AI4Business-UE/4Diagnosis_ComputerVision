"""
src/texture_features.py — Texture Feature Extraction
=====================================================

Texture features capture the micro-structural patterns within tissue
that are not described by color or shape alone.  In histopathology,
texture encodes:

  • Cellular density and arrangement
  • Nuclear chromatin patterns
  • Collagen fibre orientation
  • Gland architecture

This module implements six complementary texture analysis methods:

1. **LBP** (Local Binary Patterns, Ojala et al., 1994)
   Encodes the local neighbourhood of each pixel as a binary code by
   thresholding against the centre pixel.  The uniform LBP variant
   is rotation-invariant and produces histograms of local patterns.

2. **GLCM** (Gray Level Co-occurrence Matrix, Haralick et al., 1973)
   Captures second-order statistics of pixel-pair relationships.
   Four Haralick properties are extracted: contrast, correlation,
   energy, and homogeneity.

3. **Haralick Features**
   13 statistical measures derived from the GLCM.  Computed at multiple
   distances and angles, then averaged.

4. **Entropy**
   Shannon entropy of the grayscale histogram: H = -Σ p log₂ p.
   High entropy = complex/disordered texture; low = uniform.

5. **Local Variance**
   Mean of local pixel variance computed with a sliding window.
   Captures the average roughness of the tissue texture.

6. **Gabor Filter Bank**
   A bank of oriented sinusoidal Gaussian filters tuned to specific
   frequencies and orientations.  The mean and standard deviation of
   each filter's response describe the dominant texture orientation
   and frequency content — well-suited to fibre/collagen analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

_logger = logging.getLogger("SliceGrouper.texture")


# ---------------------------------------------------------------------------
# TextureFeatures dataclass
# ---------------------------------------------------------------------------


@dataclass
class TextureFeatures:
    """
    Bundle of all texture descriptors for a single component.

    Attributes
    ----------
    lbp_hist : np.ndarray
        Normalised LBP histogram.
    glcm_contrast : float
        Mean GLCM contrast across distances and angles.
    glcm_correlation : float
        Mean GLCM correlation.
    glcm_energy : float
        Mean GLCM energy (Angular Second Moment).
    glcm_homogeneity : float
        Mean GLCM homogeneity (Inverse Difference Moment).
    haralick : np.ndarray, shape (13,)
        Haralick feature vector (averaged over all GLCM configurations).
    entropy : float
        Shannon entropy of the tissue pixel grayscale distribution.
    variance : float
        Mean local variance across the tissue region.
    gabor_features : np.ndarray, shape (n_filters * 2,)
        Concatenated [mean, std] for each Gabor filter response.
    """
    lbp_hist: np.ndarray = field(default_factory=lambda: np.zeros(26))
    glcm_contrast: float = 0.0
    glcm_correlation: float = 0.0
    glcm_energy: float = 0.0
    glcm_homogeneity: float = 0.0
    haralick: np.ndarray = field(default_factory=lambda: np.zeros(13))
    entropy: float = 0.0
    variance: float = 0.0
    gabor_features: np.ndarray = field(default_factory=lambda: np.zeros(16))

    def to_vector(self) -> np.ndarray:
        """Flatten all texture features into a 1-D array."""
        return np.concatenate([
            self.lbp_hist,
            [self.glcm_contrast, self.glcm_correlation,
             self.glcm_energy, self.glcm_homogeneity],
            self.haralick,
            [self.entropy, self.variance],
            self.gabor_features,
        ]).astype(np.float64)

    @staticmethod
    def feature_names(lbp_bins: int = 26, gabor_n: int = 16) -> list[str]:
        names = [f"lbp_{i}" for i in range(lbp_bins)]
        names += ["glcm_contrast", "glcm_correlation", "glcm_energy", "glcm_homogeneity"]
        names += [f"haralick_{i}" for i in range(13)]
        names += ["entropy", "variance"]
        names += [f"gabor_{i}" for i in range(gabor_n)]
        return names


# ---------------------------------------------------------------------------
# LBP
# ---------------------------------------------------------------------------


def compute_lbp(
    gray: np.ndarray,
    mask: np.ndarray,
    radius: int = 3,
    n_points: int = 24,
    method: str = "uniform",
) -> np.ndarray:
    """
    Compute a normalised LBP (Local Binary Pattern) histogram.

    LBP encodes the local texture around each pixel by comparing each
    of ``n_points`` neighbours (sampled on a circle of ``radius`` pixels)
    to the centre pixel value.  The resulting binary code is quantised
    into a histogram that is invariant to monotonic intensity changes.

    Parameters
    ----------
    gray : np.ndarray, shape (H, W), dtype uint8
        Grayscale image.
    mask : np.ndarray, shape (H, W), dtype uint8
        Tissue mask (255 = tissue).
    radius : int
        Radius of the circular LBP neighbourhood.
    n_points : int
        Number of sampling points on the circle (usually 8 * radius).
    method : str
        LBP variant: 'default', 'ror', 'uniform', or 'var'.
        'uniform' is the most common choice for texture classification.

    Returns
    -------
    np.ndarray
        Normalised LBP histogram.
    """
    lbp = local_binary_pattern(gray, n_points, radius, method)
    tissue = mask > 0
    lbp_vals = lbp[tissue]

    n_bins = n_points + 2 if method == "uniform" else 256
    hist, _ = np.histogram(lbp_vals, bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float64)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


# ---------------------------------------------------------------------------
# GLCM and Haralick
# ---------------------------------------------------------------------------


def compute_glcm(
    gray: np.ndarray,
    mask: np.ndarray,
    distances: List[int] = (1, 2, 4),
    angles: List[float] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    levels: int = 64,
) -> Tuple[float, float, float, float]:
    """
    Compute GLCM (Gray Level Co-occurrence Matrix) properties.

    The GLCM P(i, j, d, θ) counts how often intensity i appears at
    distance d and angle θ from intensity j.  From this matrix, four
    classic Haralick properties are derived:

    - **Contrast**: Measures local intensity variation. High = rough texture.
    - **Correlation**: Measures linear intensity dependencies. High = regular texture.
    - **Energy**: Sum of squared GLCM elements. High = uniform texture.
    - **Homogeneity** (Inverse Difference Moment): Measures closeness to diagonal.

    Parameters
    ----------
    gray : np.ndarray, shape (H, W), dtype uint8
    mask : np.ndarray, shape (H, W), dtype uint8
    distances : list of int
        Pixel distances for the GLCM.
    angles : list of float
        Angles in radians.
    levels : int
        Number of gray levels to quantise to.

    Returns
    -------
    (contrast, correlation, energy, homogeneity) : tuple of float
        Mean values across all distance and angle combinations.
    """
    # Extract tissue pixels and normalise to [0, levels)
    tissue_region = gray.copy()
    tissue_region[mask == 0] = 0
    quantised = (tissue_region / 256.0 * levels).clip(0, levels - 1).astype(np.uint8)

    try:
        glcm = graycomatrix(
            quantised,
            distances=list(distances),
            angles=list(angles),
            levels=levels,
            symmetric=True,
            normed=True,
        )
        contrast    = float(graycoprops(glcm, "contrast").mean())
        correlation = float(graycoprops(glcm, "correlation").mean())
        energy      = float(graycoprops(glcm, "energy").mean())
        homogeneity = float(graycoprops(glcm, "homogeneity").mean())
    except Exception as exc:
        _logger.debug(f"GLCM computation failed: {exc}")
        contrast = correlation = energy = homogeneity = 0.0

    return contrast, correlation, energy, homogeneity


def compute_haralick(
    gray: np.ndarray,
    mask: np.ndarray,
    distances: List[int] = (1, 2),
    angles: List[float] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    levels: int = 64,
) -> np.ndarray:
    """
    Compute all 13 Haralick texture features from the GLCM.

    The 13 features are (following Haralick 1973 + Soh & Tsatsoulis 1999):
    1.  Angular Second Moment (Energy)
    2.  Contrast
    3.  Correlation
    4.  Sum of Squares (Variance)
    5.  Inverse Difference Moment (Homogeneity)
    6.  Sum Average
    7.  Sum Variance
    8.  Sum Entropy
    9.  Entropy
    10. Difference Variance
    11. Difference Entropy
    12. Information Measure of Correlation 1
    13. Information Measure of Correlation 2

    Since scikit-image's graycoprops only implements 4 standard features,
    additional Haralick features are computed manually from the GLCM.

    Parameters
    ----------
    gray : np.ndarray, shape (H, W), dtype uint8
    mask : np.ndarray, shape (H, W), dtype uint8
    distances, angles, levels : see compute_glcm

    Returns
    -------
    np.ndarray, shape (13,)
        Mean Haralick features across all GLCM configurations.
    """
    tissue_region = gray.copy()
    tissue_region[mask == 0] = 0
    quantised = (tissue_region / 256.0 * levels).clip(0, levels - 1).astype(np.uint8)

    try:
        glcm = graycomatrix(
            quantised,
            distances=list(distances),
            angles=list(angles),
            levels=levels,
            symmetric=True,
            normed=True,
        )
    except Exception as exc:
        _logger.debug(f"Haralick GLCM failed: {exc}")
        return np.zeros(13)

    # Average across distances and angles → (levels, levels) matrix
    P = glcm.mean(axis=(2, 3))  # shape (levels, levels)
    P = P / (P.sum() + 1e-12)

    I, J = np.ogrid[0:levels, 0:levels]
    mu_i = (I * P).sum()
    mu_j = (J * P).sum()
    sigma_i = np.sqrt(((I - mu_i) ** 2 * P).sum())
    sigma_j = np.sqrt(((J - mu_j) ** 2 * P).sum())

    # 1. Angular Second Moment (Energy)
    f1 = (P ** 2).sum()
    # 2. Contrast
    f2 = ((I - J) ** 2 * P).sum()
    # 3. Correlation
    if sigma_i * sigma_j > 1e-10:
        f3 = (((I - mu_i) * (J - mu_j) * P).sum()) / (sigma_i * sigma_j)
    else:
        f3 = 0.0
    # 4. Variance (Sum of Squares)
    f4 = ((I - mu_i) ** 2 * P).sum()
    # 5. Inverse Difference Moment (Homogeneity)
    f5 = (P / (1 + (I - J) ** 2)).sum()
    # 6. Sum Average
    k = np.arange(0, 2 * levels)
    p_sum = np.array([P[I + J == k_val].sum() for k_val in k])
    f6 = (k * p_sum).sum()
    # 7. Sum Variance
    f7 = ((k - f6) ** 2 * p_sum).sum()
    # 8. Sum Entropy
    eps = 1e-12
    f8 = -(p_sum * np.log2(p_sum + eps)).sum()
    # 9. Entropy
    f9 = -(P * np.log2(P + eps)).sum()
    # 10. Difference Variance
    k_diff = np.arange(0, levels)
    p_diff = np.array([P[np.abs(I - J) == k_val].sum() for k_val in k_diff])
    f10 = p_diff.var()
    # 11. Difference Entropy
    f11 = -(p_diff * np.log2(p_diff + eps)).sum()
    # 12 & 13. Information Measures of Correlation
    Px = P.sum(axis=1)
    Py = P.sum(axis=0)
    HX = -(Px * np.log2(Px + eps)).sum()
    HY = -(Py * np.log2(Py + eps)).sum()
    HXY1 = -(P * np.log2(np.outer(Px, Py) + eps)).sum()
    HXY2 = -(np.outer(Px, Py) * np.log2(np.outer(Px, Py) + eps)).sum()
    HXY = f9
    denom = max(HX, HY)
    f12 = (HXY - HXY1) / denom if denom > eps else 0.0
    f13 = np.sqrt(max(0, 1 - np.exp(-2 * (HXY2 - HXY))))

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13])


# ---------------------------------------------------------------------------
# Entropy and Variance
# ---------------------------------------------------------------------------


def compute_entropy(
    gray: np.ndarray,
    mask: np.ndarray,
    bins: int = 256,
) -> float:
    """
    Compute Shannon entropy of the tissue pixel intensity distribution.

    H = -Σ p(i) * log₂(p(i))

    High entropy indicates a complex, disordered texture.
    Low entropy indicates a uniform or repetitive pattern.

    Parameters
    ----------
    gray : np.ndarray, shape (H, W), dtype uint8
    mask : np.ndarray, shape (H, W), dtype uint8
    bins : int

    Returns
    -------
    float
        Entropy in bits.
    """
    tissue_px = gray[mask > 0]
    if len(tissue_px) == 0:
        return 0.0
    hist, _ = np.histogram(tissue_px, bins=bins, range=(0, 256))
    hist = hist.astype(np.float64)
    hist /= hist.sum() + 1e-12
    eps = 1e-12
    return float(-np.sum(hist * np.log2(hist + eps)))


def compute_local_variance(
    gray: np.ndarray,
    mask: np.ndarray,
    window_size: int = 15,
) -> float:
    """
    Compute the mean local variance of tissue pixels.

    A sliding window computes the variance of pixel intensities in each
    neighbourhood.  The mean across all tissue-centre windows gives a
    measure of the average texture roughness.

    Parameters
    ----------
    gray : np.ndarray, shape (H, W), dtype uint8
    mask : np.ndarray, shape (H, W), dtype uint8
    window_size : int
        Side length of the square averaging window.

    Returns
    -------
    float
        Mean local variance (pixels²).
    """
    gray_f = gray.astype(np.float64)
    kernel = np.ones((window_size, window_size), dtype=np.float64) / (window_size ** 2)

    mean_img = cv2.filter2D(gray_f, -1, kernel)
    mean_sq_img = cv2.filter2D(gray_f ** 2, -1, kernel)
    var_img = mean_sq_img - mean_img ** 2
    var_img = np.maximum(var_img, 0)

    tissue = mask > 0
    if not tissue.any():
        return 0.0
    return float(var_img[tissue].mean())


# ---------------------------------------------------------------------------
# Gabor Filters
# ---------------------------------------------------------------------------


def compute_gabor_responses(
    gray: np.ndarray,
    mask: np.ndarray,
    frequencies: List[float] = (0.1, 0.2, 0.3, 0.4),
    n_orientations: int = 4,
    kernel_size: int = 21,
    sigma: float = 4.0,
) -> np.ndarray:
    """
    Apply a bank of Gabor filters and extract mean and std of responses.

    A Gabor filter is a Gaussian kernel modulated by a sinusoidal wave.
    It is sensitive to edges and texture at a specific frequency and
    orientation.  A bank of filters covering multiple frequencies and
    orientations provides a multi-scale, multi-direction texture descriptor.

    g(x, y; λ, θ, σ) = exp(-( x'² + γ²y'² ) / (2σ²)) * cos(2πx'/λ + ψ)

    Parameters
    ----------
    gray : np.ndarray, shape (H, W), dtype uint8
    mask : np.ndarray, shape (H, W), dtype uint8
    frequencies : list of float
        Spatial frequencies in cycles/pixel. E.g. 0.1 = 10 pixels/cycle.
    n_orientations : int
        Number of orientations uniformly spaced in [0, π).
    kernel_size : int
        Size of the Gabor kernel (must be odd).
    sigma : float
        Standard deviation of the Gaussian envelope.

    Returns
    -------
    np.ndarray, shape (n_frequencies * n_orientations * 2,)
        Concatenated [mean, std] for each filter response over tissue.
    """
    gray_f = gray.astype(np.float32)
    tissue = mask > 0
    features = []

    orientations = [np.pi * i / n_orientations for i in range(n_orientations)]

    for freq in frequencies:
        wavelength = 1.0 / (freq + 1e-6)
        for theta in orientations:
            # Build Gabor kernel
            kernel_real = cv2.getGaborKernel(
                (kernel_size, kernel_size),
                sigma,
                theta,
                wavelength,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F,
            )
            # Apply filter
            response = cv2.filter2D(gray_f, cv2.CV_32F, kernel_real)
            response_abs = np.abs(response)

            if tissue.any():
                mean_resp = float(response_abs[tissue].mean())
                std_resp = float(response_abs[tissue].std())
            else:
                mean_resp = std_resp = 0.0

            features.extend([mean_resp, std_resp])

    return np.array(features, dtype=np.float64)


# ---------------------------------------------------------------------------
# High-level extractor
# ---------------------------------------------------------------------------


def extract_texture_features(
    component: "ComponentData",  # type: ignore[name-defined]
    lbp_radius: int = 3,
    lbp_n_points: int = 24,
    lbp_method: str = "uniform",
    glcm_distances: List[int] = (1, 2, 4),
    glcm_angles: List[float] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    gabor_frequencies: List[float] = (0.1, 0.2, 0.3, 0.4),
    gabor_orientations: int = 4,
) -> TextureFeatures:
    """
    Compute all texture features for a single ``ComponentData`` object.

    Parameters
    ----------
    component : ComponentData
    lbp_radius, lbp_n_points, lbp_method : LBP parameters.
    glcm_distances, glcm_angles : GLCM parameters.
    gabor_frequencies, gabor_orientations : Gabor parameters.

    Returns
    -------
    TextureFeatures
    """
    gray = component.gray_crop
    mask = component.mask_crop

    lbp_hist = compute_lbp(gray, mask, lbp_radius, lbp_n_points, lbp_method)
    contrast, correlation, energy, homogeneity = compute_glcm(
        gray, mask, glcm_distances, glcm_angles
    )
    haralick = compute_haralick(gray, mask, list(glcm_distances), list(glcm_angles))
    entropy = compute_entropy(gray, mask)
    variance = compute_local_variance(gray, mask)
    gabor_feat = compute_gabor_responses(
        gray, mask,
        frequencies=list(gabor_frequencies),
        n_orientations=gabor_orientations,
    )

    return TextureFeatures(
        lbp_hist=lbp_hist,
        glcm_contrast=contrast,
        glcm_correlation=correlation,
        glcm_energy=energy,
        glcm_homogeneity=homogeneity,
        haralick=haralick,
        entropy=entropy,
        variance=variance,
        gabor_features=gabor_feat,
    )
