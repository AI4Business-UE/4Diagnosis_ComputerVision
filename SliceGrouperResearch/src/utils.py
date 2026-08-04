"""
src/utils.py — General Utilities for SliceGrouperResearch
==========================================================

Shared helper functions used across all modules.
These are intentionally small, focused, and dependency-light.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
_logger = logging.getLogger("SliceGrouper")


def log_step(name: str, level: str = "info") -> None:
    """
    Emit a clearly formatted progress log message.

    Parameters
    ----------
    name : str
        Human-readable description of the step being executed.
    level : str
        Logging level string: 'debug', 'info', 'warning', 'error'.
    """
    fn = getattr(_logger, level.lower(), _logger.info)
    fn(f"▶  {name}")


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------


class Timer:
    """
    Context manager that measures wall-clock elapsed time.

    Example
    -------
    >>> with Timer("Feature extraction") as t:
    ...     some_expensive_call()
    >>> print(t.elapsed)   # seconds as float
    """

    def __init__(self, label: str = "", verbose: bool = True) -> None:
        self.label = label
        self.verbose = verbose
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed = time.perf_counter() - self._start
        if self.verbose and self.label:
            _logger.info(f"⏱  {self.label}: {self.elapsed:.3f}s")

    def __repr__(self) -> str:
        return f"Timer(label={self.label!r}, elapsed={self.elapsed:.3f}s)"


# ---------------------------------------------------------------------------
# File / directory helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Create a directory (including parents) if it does not exist.

    Parameters
    ----------
    path : str or Path
        Directory path to create.

    Returns
    -------
    Path
        Resolved absolute Path of the directory.
    """
    p = Path(path).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_figure(
    fig: Any,
    path: Union[str, Path],
    dpi: int = 150,
    tight: bool = True,
) -> Path:
    """
    Save a Matplotlib figure to disk and return the resolved path.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    path : str or Path
        Destination file path. Parent directories are created automatically.
    dpi : int
        Output resolution in dots per inch.
    tight : bool
        Whether to call ``tight_layout()`` before saving.

    Returns
    -------
    Path
        Resolved path where the figure was saved.
    """
    p = Path(path).resolve()
    ensure_dir(p.parent)
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    _logger.debug(f"Saved figure → {p}")
    return p


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def resize_to_thumbnail(
    image: np.ndarray,
    max_size: int = 256,
) -> np.ndarray:
    """
    Resize an image so its largest dimension equals ``max_size``,
    preserving aspect ratio.

    Parameters
    ----------
    image : np.ndarray
        Input image (H, W) or (H, W, C).
    max_size : int
        Maximum size of the largest dimension in pixels.

    Returns
    -------
    np.ndarray
        Resized image.
    """
    import cv2  # local import to keep top-level utils light

    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image
    scale = max_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def crop_with_margin(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin_px: int = 20,
) -> np.ndarray:
    """
    Crop a region from an image with an optional margin, clamped to image bounds.

    Parameters
    ----------
    image : np.ndarray
        Source image (H, W) or (H, W, C).
    bbox : tuple of int
        Bounding box as (x, y, w, h) in pixel coordinates.
    margin_px : int
        Additional border to include on all four sides.

    Returns
    -------
    np.ndarray
        Cropped image region.
    """
    x, y, w, h = bbox
    H, W = image.shape[:2]
    x1 = max(0, x - margin_px)
    y1 = max(0, y - margin_px)
    x2 = min(W, x + w + margin_px)
    y2 = min(H, y + h + margin_px)
    return image[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Embedding / dimensionality reduction wrappers
# ---------------------------------------------------------------------------


def compute_pca_embedding(
    features: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, "sklearn.decomposition.PCA"]:  # type: ignore[name-defined]
    """
    Apply PCA to reduce feature dimensionality.

    Parameters
    ----------
    features : np.ndarray, shape (n_samples, n_features)
        Feature matrix. Must not contain NaN values.
    n_components : int
        Number of principal components to retain.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    embedding : np.ndarray, shape (n_samples, n_components)
        Projected coordinates.
    pca : sklearn.decomposition.PCA
        Fitted PCA object (useful for inspecting explained_variance_ratio_).
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, random_state=random_state)
    embedding = pca.fit_transform(features)
    return embedding, pca


def compute_tsne_embedding(
    features: np.ndarray,
    n_components: int = 2,
    perplexity: float = 5.0,
    n_iter: int = 1000,
    random_state: int = 42,
    **kwargs: Any,
) -> np.ndarray:
    """
    Apply t-SNE to produce a 2-D (or 3-D) embedding.

    t-SNE (t-Distributed Stochastic Neighbour Embedding) preserves local
    structure better than PCA but does not support out-of-sample projection.

    Parameters
    ----------
    features : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    n_components : int
        Output dimensionality (typically 2).
    perplexity : float
        Balances local vs. global structure.  A good value is usually
        between 5 and 50 but must be less than n_samples.
    n_iter : int
        Maximum number of optimisation iterations.
    random_state : int
        Seed for reproducibility.
    **kwargs
        Additional arguments forwarded to ``sklearn.manifold.TSNE``.

    Returns
    -------
    np.ndarray, shape (n_samples, n_components)
    """
    from sklearn.manifold import TSNE

    # perplexity must be < n_samples
    n = features.shape[0]
    effective_perplexity = min(perplexity, max(1.0, n - 1))

    tsne = TSNE(
        n_components=n_components,
        perplexity=effective_perplexity,
        max_iter=n_iter,
        random_state=random_state,
        **kwargs,
    )
    return tsne.fit_transform(features)


def compute_umap_embedding(
    features: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 5,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
    **kwargs: Any,
) -> np.ndarray:
    """
    Apply UMAP for dimensionality reduction.

    UMAP (Uniform Manifold Approximation and Projection) generally produces
    cleaner global structure than t-SNE and is faster for large datasets.

    Parameters
    ----------
    features : np.ndarray, shape (n_samples, n_features)
    n_components : int
        Output dimensionality (typically 2).
    n_neighbors : int
        Number of nearest neighbours considered for local structure.
        Lower = more local, higher = more global.
    min_dist : float
        Minimum distance between embedded points.
        Lower values = tighter clusters.
    metric : str
        Distance metric: 'cosine', 'euclidean', 'correlation', etc.
    random_state : int
        Seed for reproducibility.
    **kwargs
        Additional arguments forwarded to ``umap.UMAP``.

    Returns
    -------
    np.ndarray, shape (n_samples, n_components)

    Raises
    ------
    ImportError
        If the ``umap-learn`` package is not installed.
    """
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "umap-learn is required for UMAP embeddings. "
            "Install it with:  pip install umap-learn"
        ) from exc

    n = features.shape[0]
    effective_n_neighbors = min(n_neighbors, max(2, n - 1))

    init = kwargs.pop("init", "spectral")

    def _build_reducer(init_method: str) -> "umap.UMAP":  # type: ignore[name-defined]
        return umap.UMAP(
            n_components=n_components,
            n_neighbors=effective_n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            init=init_method,
            **kwargs,
        )

    try:
        return _build_reducer(init).fit_transform(features)
    except TypeError as exc:
        # Spectral init failed (e.g. "sparse A with k >= N" from scipy eigensolver).
        # Retry with random — always safe regardless of n.
        if init != "random":
            _logger.warning(
                f"UMAP spectral init failed ({exc}); retrying with init='random'."
            )
            return _build_reducer("random").fit_transform(features)
        raise



# ---------------------------------------------------------------------------
# Array / numeric helpers
# ---------------------------------------------------------------------------


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Return numerator / denominator, or *default* if denominator is zero."""
    if abs(denominator) < 1e-12:
        return default
    return numerator / denominator


def normalize_array(
    arr: np.ndarray,
    method: str = "minmax",
) -> np.ndarray:
    """
    Normalise a 1-D or 2-D array along axis=0 (per feature).

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (n,) or (n_samples, n_features).
    method : str
        Normalisation method:
        - 'minmax' : scale each feature to [0, 1].
        - 'zscore' : standardise to zero mean and unit variance.
        - 'l2'     : divide each row by its L2 norm.
        - 'none'   : return unchanged copy.

    Returns
    -------
    np.ndarray
        Normalised array with same shape as input.
    """
    arr = arr.copy().astype(np.float64)
    if method == "none":
        return arr
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
        scalar = True
    else:
        scalar = False

    if method == "minmax":
        mn = arr.min(axis=0)
        mx = arr.max(axis=0)
        rng = np.where(mx - mn < 1e-12, 1.0, mx - mn)
        arr = (arr - mn) / rng
    elif method == "zscore":
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        arr = (arr - mean) / std
    elif method == "l2":
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        arr = arr / norms
    else:
        raise ValueError(f"Unknown normalisation method: {method!r}")

    if scalar:
        arr = arr.ravel()
    return arr
