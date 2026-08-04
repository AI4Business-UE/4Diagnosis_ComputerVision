"""
src/similarity.py — Feature Weighting and Similarity Matrix Computation
=======================================================================

The central challenge in the SliceGrouper problem is defining a single,
informative similarity measure between any two tissue components, given
that multiple heterogeneous feature types are available.

This module implements:

1. **FeatureWeights** — a dataclass that defines the contribution of each
   feature group to the combined similarity.

2. **Feature-level similarity functions** — convert raw feature vectors into
   normalised pairwise similarity matrices in [0, 1].

3. **Weighted combination** — merge individual similarity matrices into one
   composite matrix using user-defined weights.

4. **Distance matrix** — convert similarity to distance (1 - similarity).

Mathematical Formulation
-------------------------
Given N components, each with feature vectors from K groups:

  S_k(i, j) ∈ [0, 1]  — similarity between components i and j, feature group k
  w_k ∈ [0, 1], Σ w_k = 1  — user-defined weight for group k

  S_combined(i, j) = Σ_k w_k * S_k(i, j)
  D_combined(i, j) = 1 - S_combined(i, j)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.distance import cdist

_logger = logging.getLogger("SliceGrouper.similarity")


# ---------------------------------------------------------------------------
# FeatureWeights dataclass
# ---------------------------------------------------------------------------


@dataclass
class FeatureWeights:
    """
    Defines the relative importance of each feature group in the combined
    similarity computation.

    All weights must be non-negative and should sum to 1.0.  Setting a weight
    to 0.0 effectively disables that feature group.

    Attributes
    ----------
    geometry : float
        Weight for geometric features (area, perimeter, solidity, …).
    shape : float
        Weight for shape descriptors (Hu moments, Fourier, shape context).
    color : float
        Weight for color features (RGB/HSV/Lab statistics and histograms).
    texture : float
        Weight for texture features (LBP, GLCM, Haralick, Gabor).
    deep : float
        Weight for deep embedding similarity (ResNet50, DINOv2, etc.).
    """
    geometry: float = 0.15
    shape: float = 0.20
    color: float = 0.25
    texture: float = 0.20
    deep: float = 0.20

    def validate(self) -> None:
        """Raise ValueError if weights don't sum to 1.0."""
        total = self.geometry + self.shape + self.color + self.texture + self.deep
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Feature weights must sum to 1.0 (got {total:.4f}). "
                "Please adjust geometry, shape, color, texture, and deep."
            )

    def as_dict(self) -> Dict[str, float]:
        """Return weights as a plain dictionary."""
        return {
            "geometry": self.geometry,
            "shape": self.shape,
            "color": self.color,
            "texture": self.texture,
            "deep": self.deep,
        }

    def normalise(self) -> "FeatureWeights":
        """Return a new FeatureWeights with values scaled to sum to 1.0."""
        total = self.geometry + self.shape + self.color + self.texture + self.deep
        if total < 1e-9:
            raise ValueError("All weights are zero; cannot normalise.")
        return FeatureWeights(
            geometry=self.geometry / total,
            shape=self.shape / total,
            color=self.color / total,
            texture=self.texture / total,
            deep=self.deep / total,
        )


# ---------------------------------------------------------------------------
# Feature matrix normalisation
# ---------------------------------------------------------------------------


def normalize_feature_matrix(
    features: np.ndarray,
    method: str = "minmax",
) -> np.ndarray:
    """
    Normalise a feature matrix so columns are on comparable scales.

    Parameters
    ----------
    features : np.ndarray, shape (N, D)
        Raw feature matrix.
    method : str
        'minmax' : scale each column to [0, 1].
        'zscore' : standardise each column to mean=0, std=1.
        'l2'     : L2-normalise each row (sample).
        'none'   : return a copy unchanged.

    Returns
    -------
    np.ndarray, shape (N, D)
    """
    from src.utils import normalize_array
    return normalize_array(features, method)


# ---------------------------------------------------------------------------
# Feature-level similarity matrices
# ---------------------------------------------------------------------------


def cosine_similarity_matrix(features: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity from a feature matrix.

    Values in [-1, 1], shifted to [0, 1] via (1 + cos) / 2.

    Parameters
    ----------
    features : np.ndarray, shape (N, D)

    Returns
    -------
    np.ndarray, shape (N, N), values in [0, 1]
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    normed = features / norms
    raw = normed @ normed.T
    return ((raw + 1.0) / 2.0).clip(0.0, 1.0)


def rbf_similarity_matrix(
    features: np.ndarray,
    gamma: Optional[float] = None,
) -> np.ndarray:
    """
    Compute pairwise RBF (Radial Basis Function / Gaussian) similarity.

    S(i, j) = exp(-γ * ||fᵢ - fⱼ||²)

    This maps Euclidean distances to [0, 1] similarity values.
    gamma = 1 / (2 * σ²) where σ is the bandwidth.

    Parameters
    ----------
    features : np.ndarray, shape (N, D)
    gamma : float or None
        RBF kernel bandwidth parameter.
        If None, uses 1 / (2 * mean_pairwise_distance²).

    Returns
    -------
    np.ndarray, shape (N, N), values in [0, 1]
    """
    D2 = cdist(features, features, metric="sqeuclidean")
    if gamma is None:
        mean_d2 = D2[D2 > 0].mean() if (D2 > 0).any() else 1.0
        gamma = 1.0 / (2.0 * mean_d2 + 1e-12)
    return np.exp(-gamma * D2)


def histogram_similarity_matrix(
    histograms: np.ndarray,
) -> np.ndarray:
    """
    Compute pairwise histogram intersection similarity.

    Intersection(H1, H2) = Σᵢ min(H1ᵢ, H2ᵢ)

    For normalised histograms (sums to 1), values are in [0, 1].

    Parameters
    ----------
    histograms : np.ndarray, shape (N, bins)
        Row-normalised histograms.

    Returns
    -------
    np.ndarray, shape (N, N), values in [0, 1]
    """
    n = histograms.shape[0]
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            val = np.minimum(histograms[i], histograms[j]).sum()
            S[i, j] = S[j, i] = val
    return S.clip(0.0, 1.0)


def hausdorff_similarity_matrix(
    components: List["ComponentData"],  # type: ignore[name-defined]
    max_distance: float = 500.0,
) -> np.ndarray:
    """
    Compute a pairwise similarity matrix based on Hausdorff contour distance.

    Converts Hausdorff distance to similarity: S = 1 - min(d / max_d, 1).

    Parameters
    ----------
    components : list of ComponentData
    max_distance : float
        Distance value corresponding to similarity = 0.

    Returns
    -------
    np.ndarray, shape (N, N), values in [0, 1]
    """
    from src.shape_features import hausdorff_distance

    n = len(components)
    S = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            d = hausdorff_distance(components[i].contour, components[j].contour)
            s = 1.0 - min(d / max_distance, 1.0)
            S[i, j] = S[j, i] = s
    return S


# ---------------------------------------------------------------------------
# Combined similarity matrix
# ---------------------------------------------------------------------------


def compute_pairwise_similarity(
    feature_matrices: Dict[str, np.ndarray],
    weights: FeatureWeights,
    normalise: str = "minmax",
    similarity_fn: str = "rbf",
) -> np.ndarray:
    """
    Compute a weighted combined pairwise similarity matrix.

    Parameters
    ----------
    feature_matrices : dict
        Keys: 'geometry', 'shape', 'color', 'texture', 'deep'.
        Values: np.ndarray, shape (N, D_k) — raw feature vectors.
    weights : FeatureWeights
        Relative weights for each feature group.
    normalise : str
        Feature normalisation method ('minmax', 'zscore', 'l2', 'none').
    similarity_fn : str
        How to convert feature distances to similarity:
        'rbf' → Gaussian kernel, 'cosine' → cosine similarity.

    Returns
    -------
    np.ndarray, shape (N, N)
        Weighted combined similarity matrix, values in [0, 1].
    """
    weights_dict = weights.as_dict()
    combined = None
    total_weight = 0.0

    for key, feat in feature_matrices.items():
        w = weights_dict.get(key, 0.0)
        if w <= 0.0 or feat is None or feat.size == 0:
            continue

        # Handle NaN/Inf in features
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalise
        feat_norm = normalize_feature_matrix(feat, method=normalise)

        # Compute similarity
        if similarity_fn == "cosine":
            S_k = cosine_similarity_matrix(feat_norm)
        elif similarity_fn == "rbf":
            S_k = rbf_similarity_matrix(feat_norm)
        else:
            raise ValueError(f"Unknown similarity_fn: {similarity_fn!r}")

        # Accumulate
        if combined is None:
            combined = w * S_k
        else:
            combined += w * S_k
        total_weight += w

    if combined is None or total_weight < 1e-9:
        n = next(iter(feature_matrices.values())).shape[0]
        _logger.warning("No active features found; returning identity similarity.")
        return np.eye(n)

    combined /= total_weight
    np.fill_diagonal(combined, 1.0)
    return combined.clip(0.0, 1.0)


def combine_similarity_matrices(
    matrices: Dict[str, np.ndarray],
    weights: Dict[str, float],
) -> np.ndarray:
    """
    Linearly combine pre-computed similarity matrices with given weights.

    Unlike ``compute_pairwise_similarity``, this function takes already-
    computed similarity matrices (not feature vectors) as input.
    Useful for combining matrices that were produced by different methods
    (e.g. RBF for color, histogram intersection for textures, cosine for deep).

    Parameters
    ----------
    matrices : dict
        {name: np.ndarray shape (N, N)} — pairwise similarity matrices.
    weights : dict
        {name: float} — weight for each matrix.

    Returns
    -------
    np.ndarray, shape (N, N)
        Weighted average. Values are clamped to [0, 1].
    """
    combined = None
    total_w = 0.0
    for name, S in matrices.items():
        w = weights.get(name, 0.0)
        if w <= 0.0:
            continue
        if combined is None:
            combined = w * S
        else:
            combined += w * S
        total_w += w

    if combined is None or total_w < 1e-9:
        n = next(iter(matrices.values())).shape[0]
        return np.eye(n)

    combined /= total_w
    np.fill_diagonal(combined, 1.0)
    return combined.clip(0.0, 1.0)


def compute_distance_matrix(similarity_matrix: np.ndarray) -> np.ndarray:
    """
    Convert a similarity matrix to a distance matrix.

    D(i, j) = 1 - S(i, j)

    Parameters
    ----------
    similarity_matrix : np.ndarray, shape (N, N)
        Values assumed in [0, 1].

    Returns
    -------
    np.ndarray, shape (N, N)
        Distance matrix with zeros on the diagonal.
    """
    D = 1.0 - similarity_matrix
    np.fill_diagonal(D, 0.0)
    return D.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Per-method similarity (convenience wrappers)
# ---------------------------------------------------------------------------


def geometry_similarity(
    components: List["ComponentData"],  # type: ignore[name-defined]
    normalise: str = "minmax",
) -> np.ndarray:
    """Compute RBF similarity from geometry feature vectors."""
    from src.geometry import extract_geometry_features
    feats = np.vstack([
        extract_geometry_features(c).to_vector() for c in components
    ])
    feats_norm = normalize_feature_matrix(feats, normalise)
    return rbf_similarity_matrix(feats_norm)


def color_similarity(
    components: List["ComponentData"],  # type: ignore[name-defined]
    normalise: str = "minmax",
    histogram_bins: int = 32,
) -> np.ndarray:
    """Compute similarity from color feature vectors."""
    from src.color_features import extract_color_features
    feats = np.vstack([
        extract_color_features(c, histogram_bins=histogram_bins).to_vector()
        for c in components
    ])
    feats_norm = normalize_feature_matrix(feats, normalise)
    return rbf_similarity_matrix(feats_norm)


def texture_similarity(
    components: List["ComponentData"],  # type: ignore[name-defined]
    normalise: str = "minmax",
) -> np.ndarray:
    """Compute similarity from texture feature vectors."""
    from src.texture_features import extract_texture_features
    feats = np.vstack([
        extract_texture_features(c).to_vector() for c in components
    ])
    feats_norm = normalize_feature_matrix(feats, normalise)
    return rbf_similarity_matrix(feats_norm)


def deep_similarity(
    embeddings: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity from pre-computed deep embeddings."""
    return cosine_similarity_matrix(embeddings)
