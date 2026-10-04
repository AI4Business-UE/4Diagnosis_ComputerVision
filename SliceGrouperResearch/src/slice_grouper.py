"""
src/slice_grouper.py — Stage 2: Spatial Slice Regrouping
=========================================================

After Stage 1 clusters components by visual appearance (tissue type),
Stage 2 re-groups those components into SLICES: sets of components
that physically belong to the same cross-sectional cut of the specimen.

Background
----------
In histopathology, a specimen block (preparat) is physically cut into
multiple thin sections (plastry/slices).  All sections are placed on
the same glass slide.  Because all sections are cut from the same block
at the same time, they are spatially co-located on the slide in a regular
pattern (e.g. left→right in order of cutting).

Stage 1 identifies TISSUE TYPES (appearance clusters).
Stage 2 identifies SECTIONS (which components were cut together).

Example
-------
Specimen: 3 tissue elements [A, B, C], sliced 4 times.
Slide contains 12 components total.

After Stage 1:
  Cluster A: [A1, A2, A3, A4]
  Cluster B: [B1, B2, B3, B4]
  Cluster C: [C1, C2, C3, C4]

After Stage 2:
  Slice 0: {A1, B1, C1}  ← leftmost section
  Slice 1: {A2, B2, C2}
  Slice 2: {A3, B3, C3}
  Slice 3: {A4, B4, C4}  ← rightmost section

Algorithm
---------
1. Find the dominant spatial axis (auto=PCA, or explicit x/y) of centroids.
2. Within each appearance cluster, sort components by projection onto that axis.
3. Components at rank i across ALL clusters → slice i.
4. K (number of slices) = median cluster size from Stage 1 (automatic).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_logger = logging.getLogger("SliceGrouper.slice_grouper")


# ---------------------------------------------------------------------------
# SliceGroupResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class SliceGroupResult:
    """
    Result of Stage 2 spatial slice regrouping.

    Attributes
    ----------
    slice_labels : np.ndarray, shape (N,)
        Slice index for each component.  -1 = unassigned (noise from Stage 1).
    n_slices : int
        Number of detected slices (= K).
    appearance_labels : np.ndarray, shape (N,)
        Stage 1 appearance labels that were used as input.
    sort_axis : str
        Axis used for sorting: 'x', 'y', or 'pca (≈x)' / 'pca (≈y)'.
    sort_scores : np.ndarray, shape (N,)
        Scalar projection value used for ordering (NaN for noise points).
    cluster_size_balance : float
        Std-dev of appearance cluster sizes (0 = all equal = perfect).
    intra_slice_dist : float
        Mean pixel distance between centroids of components in the same slice.
    inter_slice_dist : float
        Mean pixel distance between centroids of components in different slices.
    cohesion_ratio : float
        intra / inter.  Lower = slices are spatially compact.  Good: < 1.
    runtime : float
        Wall-clock time in seconds.
    method_name : str
    """
    slice_labels: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    n_slices: int = 0
    appearance_labels: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    sort_axis: str = "pca"
    sort_scores: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    cluster_size_balance: float = float("nan")
    intra_slice_dist: float = float("nan")
    inter_slice_dist: float = float("nan")
    cohesion_ratio: float = float("nan")
    runtime: float = 0.0
    method_name: str = "spatial_slice_grouper"


# ---------------------------------------------------------------------------
# Dominant axis detection
# ---------------------------------------------------------------------------


def detect_sort_axis(
    centroids: np.ndarray,
    method: str = "auto",
) -> Tuple[str, np.ndarray]:
    """
    Find the dominant spatial axis and compute per-component sort scores.

    Parameters
    ----------
    centroids : np.ndarray, shape (N, 2)
        (cx, cy) centroid for each component.
    method : str
        'auto' or 'pca' — PCA of centroid positions (detects any orientation).
        'x'             — sort strictly by X pixel coordinate.
        'y'             — sort strictly by Y pixel coordinate.

    Returns
    -------
    axis_name : str
        Human-readable label for the chosen axis.
    scores : np.ndarray, shape (N,)
        Projection value.  Higher = further along axis = later slice.
    """
    if method in ("auto", "pca"):
        if len(centroids) < 2:
            return "x", centroids[:, 0].copy()
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            scores = pca.fit_transform(centroids).ravel()
            direction = pca.components_[0]   # unit vector (dx, dy)
            axis_name = (
                "pca (≈x)" if abs(direction[0]) >= abs(direction[1]) else "pca (≈y)"
            )
            return axis_name, scores
        except Exception as exc:
            _logger.warning(f"PCA axis detection failed ({exc}), falling back to X.")
            return "x", centroids[:, 0].copy()

    elif method == "x":
        return "x", centroids[:, 0].copy()

    elif method == "y":
        return "y", centroids[:, 1].copy()

    else:
        raise ValueError(
            f"Unknown sort_axis: {method!r}.  Use 'auto', 'pca', 'x', or 'y'."
        )


# ---------------------------------------------------------------------------
# Main Stage 2 function
# ---------------------------------------------------------------------------


def group_slices_by_position(
    components: List[Any],
    appearance_labels: np.ndarray,
    sort_axis: str = "auto",
) -> SliceGroupResult:
    """
    Stage 2: Re-group components into slices based on spatial position.

    Given Stage 1 appearance clusters (tissue types), assigns each component
    a *slice index* — which physical section of the specimen it came from —
    by sorting components within each cluster along the dominant spatial axis.

    Parameters
    ----------
    components : list of ComponentData
        All components detected on the slide.
    appearance_labels : np.ndarray, shape (N,)
        Cluster labels from Stage 1.
        Noise points (-1) remain unassigned (slice_label = -1).
    sort_axis : str
        Axis for sorting: 'auto' (recommended), 'x', 'y', or 'pca'.

    Returns
    -------
    SliceGroupResult
    """
    t0 = time.perf_counter()
    n = len(components)
    appearance_labels = np.asarray(appearance_labels, dtype=int)

    if n == 0:
        _logger.warning("group_slices_by_position called with 0 components.")
        return SliceGroupResult(runtime=time.perf_counter() - t0)

    # --- Extract centroids ---------------------------------------------------
    centroids = np.array([c.centroid for c in components], dtype=float)  # (N, 2)

    # --- Detect dominant axis (on non-noise components only) -----------------
    valid_mask = appearance_labels >= 0
    n_valid = int(valid_mask.sum())

    if n_valid < 2:
        _logger.warning("Fewer than 2 valid (non-noise) components; cannot group slices.")
        slice_labels = np.full(n, -1, dtype=int)
        return SliceGroupResult(
            slice_labels=slice_labels,
            n_slices=0,
            appearance_labels=appearance_labels,
            sort_axis=sort_axis,
            sort_scores=centroids[:, 0],
            runtime=time.perf_counter() - t0,
        )

    axis_name, valid_scores = detect_sort_axis(centroids[valid_mask], sort_axis)
    _logger.info(f"Sort axis detected: {axis_name}")

    # Map scores back to full array (noise → NaN)
    full_scores = np.full(n, np.nan)
    full_scores[valid_mask] = valid_scores

    # --- Group by appearance cluster -----------------------------------------
    unique_clusters = sorted(set(int(lbl) for lbl in appearance_labels if lbl >= 0))
    n_appearance_clusters = len(unique_clusters)

    cluster_to_sorted: Dict[int, List[int]] = {}
    for cid in unique_clusters:
        idx = np.where(appearance_labels == cid)[0]
        order = np.argsort(full_scores[idx])   # ascending = left→right / top→bottom
        cluster_to_sorted[cid] = idx[order].tolist()

    cluster_sizes_list = [len(v) for v in cluster_to_sorted.values()]
    K = int(np.median(cluster_sizes_list)) if cluster_sizes_list else 0
    cluster_size_balance = float(np.std(cluster_sizes_list)) if cluster_sizes_list else float("nan")

    _logger.info(
        f"Appearance clusters: {n_appearance_clusters}, "
        f"cluster sizes: {cluster_sizes_list}, "
        f"K (slices) = {K}"
    )

    if K == 0:
        return SliceGroupResult(
            slice_labels=np.full(n, -1, dtype=int),
            n_slices=0,
            appearance_labels=appearance_labels,
            sort_axis=axis_name,
            sort_scores=full_scores,
            cluster_size_balance=cluster_size_balance,
            runtime=time.perf_counter() - t0,
        )

    # --- Assign slice indices ------------------------------------------------
    slice_labels = np.full(n, -1, dtype=int)

    for cid, sorted_indices in cluster_to_sorted.items():
        for rank, comp_idx in enumerate(sorted_indices):
            # Clamp: if a cluster has more components than K, extras go to last slice
            slice_idx = min(rank, K - 1)
            slice_labels[comp_idx] = slice_idx

    # --- Spatial evaluation --------------------------------------------------
    intra_dist, inter_dist, cohesion = _compute_spatial_metrics(centroids, slice_labels)

    runtime = time.perf_counter() - t0
    _logger.info(
        f"Slice grouping done in {runtime:.3f}s: "
        f"{K} slices, cohesion ratio={cohesion:.3f} "
        f"(intra={intra_dist:.1f}px, inter={inter_dist:.1f}px)"
    )

    return SliceGroupResult(
        slice_labels=slice_labels,
        n_slices=K,
        appearance_labels=appearance_labels,
        sort_axis=axis_name,
        sort_scores=full_scores,
        cluster_size_balance=cluster_size_balance,
        intra_slice_dist=intra_dist,
        inter_slice_dist=inter_dist,
        cohesion_ratio=cohesion,
        runtime=runtime,
        method_name="spatial_slice_grouper",
    )


# ---------------------------------------------------------------------------
# Spatial evaluation
# ---------------------------------------------------------------------------


def _compute_spatial_metrics(
    centroids: np.ndarray,
    slice_labels: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Mean intra-slice and inter-slice centroid distances.

    Returns
    -------
    (intra_mean, inter_mean, cohesion_ratio)
    cohesion_ratio = intra / inter  (lower = more spatially compact slices)
    """
    valid = slice_labels >= 0
    if valid.sum() < 2:
        return float("nan"), float("nan"), float("nan")

    c = centroids[valid]
    s = slice_labels[valid]
    unique = np.unique(s)

    if len(unique) < 2:
        return float("nan"), float("nan"), float("nan")

    intra_dists: List[float] = []
    inter_dists: List[float] = []

    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            d = float(np.linalg.norm(c[i] - c[j]))
            if s[i] == s[j]:
                intra_dists.append(d)
            else:
                inter_dists.append(d)

    intra = float(np.mean(intra_dists)) if intra_dists else float("nan")
    inter = float(np.mean(inter_dists)) if inter_dists else float("nan")
    ratio = (intra / inter) if (inter > 1e-9 and not np.isnan(intra)) else float("nan")
    return intra, inter, ratio


# ---------------------------------------------------------------------------
# Diagnostics: slice composition table
# ---------------------------------------------------------------------------


def slice_composition_table(
    result: SliceGroupResult,
    components: List[Any],
) -> List[Dict[str, Any]]:
    """
    Return a list of dicts (one per slice) describing its composition.

    Each row contains:
    - slice           : slice index
    - n_components    : number of components in this slice
    - component_ids   : list of component IDs
    - appearance_lbls : list of appearance cluster labels
    - centroids       : list of (cx, cy) tuples
    - total_area      : combined tissue area in pixels

    Useful for display (pd.DataFrame) or further analysis.
    """
    rows = []
    for s_idx in range(result.n_slices):
        members = [
            i for i, lbl in enumerate(result.slice_labels)
            if lbl == s_idx
        ]
        rows.append({
            "slice": s_idx,
            "n_components": len(members),
            "component_ids": [components[i].component_id for i in members],
            "appearance_lbls": [int(result.appearance_labels[i]) for i in members],
            "centroids": [components[i].centroid for i in members],
            "total_area": sum(components[i].area for i in members),
        })
    return rows


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def print_slice_summary(result: SliceGroupResult, components: List[Any]) -> None:
    """Print a human-readable summary of the slice grouping result."""
    sep = "─" * 60
    print(sep)
    print("  SliceGrouper — Stage 2 Summary")
    print(sep)
    print(f"  Sort axis        : {result.sort_axis}")
    print(f"  N slices (K)     : {result.n_slices}")
    print(f"  Cluster balance  : σ = {result.cluster_size_balance:.2f} components")
    print(f"  Intra-slice dist : {result.intra_slice_dist:.1f} px")
    print(f"  Inter-slice dist : {result.inter_slice_dist:.1f} px")
    print(f"  Cohesion ratio   : {result.cohesion_ratio:.3f}  (< 1 = good)")
    print(f"  Runtime          : {result.runtime:.3f}s")
    print(sep)

    table = slice_composition_table(result, components)
    for row in table:
        app = row["appearance_lbls"]
        cids = row["component_ids"]
        print(
            f"  Slice {row['slice']}  →  "
            f"{row['n_components']} components  "
            f"[comp_ids={cids}, tissue_types={app}]  "
            f"area={row['total_area']:,}px"
        )
    print(sep)
