"""
src/clustering.py — Clustering Algorithm Implementations
=========================================================

This module wraps eight clustering algorithms behind a uniform interface.
Each algorithm receives a distance or similarity matrix (or embedding matrix
for algorithms that support arbitrary metrics) and returns a ``ClusterResult``.

Algorithm Overview
------------------

1. **Connected Components** (via threshold graph)
   Simplest approach: build a graph with edges where similarity > threshold,
   then extract connected components.  Fast, interpretable, no hyperparameter
   search needed.  Baseline for all comparisons.

2. **Agglomerative Hierarchical Clustering** (Ward, Complete, Average, Single)
   Bottom-up merging of clusters.  Uses a precomputed distance matrix.
   Produces a dendrogram — supports variable number of clusters.

3. **Spectral Clustering**
   Eigenvector decomposition of the graph Laplacian → cluster in the
   embedding space.  Excellent for non-convex cluster shapes.
   Requires knowing the number of clusters k.

4. **HDBSCAN** (Campello et al., 2013)
   Hierarchical DBSCAN that finds clusters of varying density.
   Marks noise points with label -1.  No need to specify k.
   Ideal for slides with a variable number of specimens.

5. **DBSCAN**
   Density-based spatial clustering.  Connects core points within eps.
   Sensitive to the eps parameter but very robust to outlier noise.

6. **Louvain Community Detection** (Blondel et al., 2008)
   Optimises modularity Q = (edges within clusters) - (expected).
   Resolution parameter controls granularity.

7. **Leiden Community Detection** (Traag et al., 2019)
   Improved version of Louvain that guarantees well-connected communities.
   Faster and more stable than Louvain for large graphs.

8. **K-Means** (comparison baseline)
   Classic centroid-based clustering.  Requires knowing k.
   Does not leverage graph structure — useful as a lower bound comparison.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

_logger = logging.getLogger("SliceGrouper.clustering")


# ---------------------------------------------------------------------------
# ClusterResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class ClusterResult:
    """
    Result of a clustering run.

    Attributes
    ----------
    labels : np.ndarray, shape (N,)
        Cluster label for each component.  -1 indicates noise (HDBSCAN/DBSCAN).
    n_clusters : int
        Number of distinct clusters (excluding noise label -1).
    n_noise : int
        Number of points labelled as noise (-1).
    runtime : float
        Wall-clock time in seconds.
    method_name : str
        Name of the clustering method.
    params : dict
        Algorithm-specific parameters used.
    """
    labels: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    n_clusters: int = 0
    n_noise: int = 0
    runtime: float = 0.0
    method_name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.labels) > 0:
            unique = set(self.labels[self.labels >= 0])
            self.n_clusters = len(unique)
            self.n_noise = int((self.labels == -1).sum())


def _make_result(labels: np.ndarray, method: str, params: dict, t: float) -> ClusterResult:
    return ClusterResult(labels=labels, method_name=method, params=params, runtime=t)


# ---------------------------------------------------------------------------
# 1. Connected Components
# ---------------------------------------------------------------------------


def run_connected_components(
    similarity_matrix: np.ndarray,
    threshold: float = 0.5,
) -> ClusterResult:
    """
    Cluster by finding connected components in a threshold graph.

    Edges are added between component pairs with similarity >= threshold.
    Each connected subgraph becomes one cluster.

    Parameters
    ----------
    similarity_matrix : np.ndarray, shape (N, N)
    threshold : float
        Minimum similarity to create an edge.

    Returns
    -------
    ClusterResult
    """
    import networkx as nx

    t0 = time.perf_counter()
    n = similarity_matrix.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] >= threshold:
                G.add_edge(i, j)

    labels = np.full(n, -1, dtype=int)
    for cluster_id, component in enumerate(nx.connected_components(G)):
        for node in component:
            labels[node] = cluster_id

    return _make_result(
        labels, "connected_components",
        {"threshold": threshold},
        time.perf_counter() - t0,
    )


# ---------------------------------------------------------------------------
# 2. Agglomerative Hierarchical Clustering
# ---------------------------------------------------------------------------


def run_agglomerative(
    distance_matrix: np.ndarray,
    n_clusters: Optional[int] = None,
    linkage: str = "average",
    distance_threshold: Optional[float] = 0.5,
) -> ClusterResult:
    """
    Agglomerative hierarchical clustering on a precomputed distance matrix.

    Parameters
    ----------
    distance_matrix : np.ndarray, shape (N, N)
        Symmetric pairwise distance matrix (diagonal = 0).
    n_clusters : int or None
        Target number of clusters.  If None, uses distance_threshold.
    linkage : str
        Linkage criterion: 'ward' (requires Euclidean), 'complete',
        'average', or 'single'.
        Note: 'ward' ignores the precomputed matrix and uses Euclidean
        distances — use 'average' or 'complete' with precomputed distances.
    distance_threshold : float or None
        Stop merging when distance exceeds this threshold.
        Only used when n_clusters is None.

    Returns
    -------
    ClusterResult
    """
    from sklearn.cluster import AgglomerativeClustering

    t0 = time.perf_counter()
    n = distance_matrix.shape[0]

    params: Dict[str, Any] = {
        "linkage": linkage,
        "n_clusters": n_clusters,
        "distance_threshold": distance_threshold,
    }

    if n_clusters is not None:
        clf = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage=linkage if linkage != "ward" else "average",
        )
    else:
        clf = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="precomputed",
            linkage=linkage if linkage != "ward" else "average",
        )

    labels = clf.fit_predict(distance_matrix).astype(int)
    return _make_result(labels, "agglomerative", params, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# 3. Spectral Clustering
# ---------------------------------------------------------------------------


def run_spectral(
    similarity_matrix: np.ndarray,
    n_clusters: int = 4,
    n_init: int = 10,
    random_state: int = 42,
) -> ClusterResult:
    """
    Spectral clustering using the precomputed similarity matrix as affinity.

    Spectral clustering embeds data into the eigenspace of the graph
    Laplacian before performing K-Means.  This allows it to find
    non-convex clusters that are globally connected but locally variable.

    Parameters
    ----------
    similarity_matrix : np.ndarray, shape (N, N)
    n_clusters : int
        Number of clusters.
    n_init : int
        Number of K-Means initialisations.
    random_state : int

    Returns
    -------
    ClusterResult
    """
    from sklearn.cluster import SpectralClustering

    t0 = time.perf_counter()
    n = similarity_matrix.shape[0]
    n_clusters = min(n_clusters, n)

    params = {"n_clusters": n_clusters}
    clf = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        n_init=n_init,
        random_state=random_state,
    )
    # Ensure non-negative affinity (fix numerical issues)
    S = similarity_matrix.clip(0.0, 1.0)
    np.fill_diagonal(S, 1.0)

    try:
        labels = clf.fit_predict(S).astype(int)
    except Exception as exc:
        _logger.warning(f"Spectral clustering failed: {exc}; returning trivial labels.")
        labels = np.zeros(n, dtype=int)

    return _make_result(labels, "spectral", params, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# 4. HDBSCAN
# ---------------------------------------------------------------------------


def run_hdbscan(
    distance_matrix: np.ndarray,
    min_cluster_size: int = 2,
    min_samples: int = 1,
) -> ClusterResult:
    """
    HDBSCAN (Hierarchical DBSCAN) on a precomputed distance matrix.

    HDBSCAN extracts the flat clustering from the hierarchy that maximises
    the overall persistence of clusters.  It automatically determines the
    number of clusters and labels outliers as -1.

    Parameters
    ----------
    distance_matrix : np.ndarray, shape (N, N)
    min_cluster_size : int
        Minimum number of points to form a cluster.
    min_samples : int
        Number of samples in the neighbourhood for a point to be core.

    Returns
    -------
    ClusterResult
    """
    t0 = time.perf_counter()
    n = distance_matrix.shape[0]
    if n <= 1:
        labels = np.zeros(n, dtype=int)
        params = {"min_cluster_size": min_cluster_size, "min_samples": min_samples}
        return _make_result(labels, "hdbscan", params, time.perf_counter() - t0)

    # Bound parameters to prevent crashes on very few components
    min_cluster_size = max(2, min(min_cluster_size, n))
    min_samples = max(1, min(min_samples, n - 1))
    
    params = {"min_cluster_size": min_cluster_size, "min_samples": min_samples}

    try:
        import hdbscan  # type: ignore
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="precomputed",
        )
        labels = clusterer.fit_predict(distance_matrix).astype(int)
    except ImportError:
        _logger.info("hdbscan not found; trying sklearn.cluster.HDBSCAN …")
        try:
            from sklearn.cluster import HDBSCAN as SKLEARNHDBSCAN
            clusterer = SKLEARNHDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="precomputed",
            )
            labels = clusterer.fit_predict(distance_matrix).astype(int)
        except Exception as exc:
            _logger.warning(f"HDBSCAN failed: {exc}")
            labels = np.zeros(distance_matrix.shape[0], dtype=int)

    return _make_result(labels, "hdbscan", params, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# 5. DBSCAN
# ---------------------------------------------------------------------------


def run_dbscan(
    distance_matrix: np.ndarray,
    eps: float = 0.4,
    min_samples: int = 2,
) -> ClusterResult:
    """
    DBSCAN on a precomputed distance matrix.

    DBSCAN connects all points within distance eps to the same cluster if
    at least min_samples points are in the neighbourhood.  Points with fewer
    than min_samples neighbours are classified as noise (-1).

    Parameters
    ----------
    distance_matrix : np.ndarray, shape (N, N)
    eps : float
        Maximum distance for neighbourhood membership.  Range [0, 1] for
        normalised distance matrices.
    min_samples : int

    Returns
    -------
    ClusterResult
    """
    from sklearn.cluster import DBSCAN

    t0 = time.perf_counter()
    n = distance_matrix.shape[0]
    if n <= 1:
        labels = np.zeros(n, dtype=int)
        params = {"eps": eps, "min_samples": min_samples}
        return _make_result(labels, "dbscan", params, time.perf_counter() - t0)

    min_samples = max(1, min(min_samples, n))
    params = {"eps": eps, "min_samples": min_samples}

    clf = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = clf.fit_predict(distance_matrix).astype(int)
    return _make_result(labels, "dbscan", params, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# 6. Louvain Community Detection
# ---------------------------------------------------------------------------


def run_louvain(
    graph: "networkx.Graph",  # type: ignore[name-defined]
    resolution: float = 1.0,
    seed: int = 42,
) -> ClusterResult:
    """
    Louvain community detection on a weighted networkx graph.

    The Louvain algorithm greedily optimises the modularity function:
      Q = Σ_c [e_c / m - (d_c / 2m)²]
    where e_c = intra-cluster edges, d_c = degree sum, m = total edge weight.

    A higher resolution parameter finds smaller communities.

    Parameters
    ----------
    graph : networkx.Graph
        Weighted graph with 'weight' edge attribute.
    resolution : float
        Controls community granularity.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    ClusterResult
    """
    t0 = time.perf_counter()
    n = graph.number_of_nodes()
    params = {"resolution": resolution}

    try:
        import community as community_louvain  # python-louvain
        partition = community_louvain.best_partition(
            graph, weight="weight", resolution=resolution, random_state=seed
        )
        labels = np.array([partition[i] for i in range(n)], dtype=int)
    except ImportError:
        try:
            import networkx.algorithms.community as nx_comm
            parts = nx_comm.louvain_communities(
                graph, weight="weight", resolution=resolution, seed=seed
            )
            labels = np.full(n, 0, dtype=int)
            for cid, part in enumerate(parts):
                for node in part:
                    labels[node] = cid
        except Exception as exc:
            _logger.warning(f"Louvain failed: {exc}")
            labels = np.zeros(n, dtype=int)

    return _make_result(labels, "louvain", params, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# 7. Leiden Community Detection
# ---------------------------------------------------------------------------


def run_leiden(
    graph: "networkx.Graph",  # type: ignore[name-defined]
    resolution: float = 1.0,
    seed: int = 42,
) -> ClusterResult:
    """
    Leiden community detection (improved Louvain).

    Leiden guarantees that all communities are subpartition-stable,
    avoiding the 'poorly connected communities' issue in Louvain.

    Requires: pip install leidenalg igraph

    Parameters
    ----------
    graph : networkx.Graph
    resolution : float
    seed : int

    Returns
    -------
    ClusterResult
    """
    import networkx as nx

    t0 = time.perf_counter()
    n = graph.number_of_nodes()
    params = {"resolution": resolution}

    try:
        import leidenalg  # type: ignore
        import igraph as ig  # type: ignore

        # Convert networkx → igraph
        g = ig.Graph.from_networkx(graph)
        weights = [graph[u][v].get("weight", 1.0) for u, v in graph.edges()]
        g.es["weight"] = weights

        part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            seed=seed,
            resolution_parameter=resolution,
        )
        labels = np.array(part.membership, dtype=int)
    except ImportError:
        _logger.warning("leidenalg not installed; falling back to Louvain.")
        result = run_louvain(graph, resolution, seed)
        result.method_name = "leiden_fallback_louvain"
        return result
    except Exception as exc:
        _logger.warning(f"Leiden failed: {exc}")
        labels = np.zeros(n, dtype=int)

    return _make_result(labels, "leiden", params, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# 8. K-Means (comparison baseline)
# ---------------------------------------------------------------------------


def run_kmeans(
    features: np.ndarray,
    n_clusters: int = 4,
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int = 42,
) -> ClusterResult:
    """
    K-Means clustering on feature vectors.

    K-Means minimises intra-cluster variance (sum of squared distances
    to centroids).  Requires specifying k and assumes convex clusters.

    Note: K-Means does not leverage the graph structure or precomputed
    distance matrix.  It operates directly on feature vectors and is
    included as a classical baseline for comparison.

    Parameters
    ----------
    features : np.ndarray, shape (N, D)
        Normalised feature matrix.
    n_clusters : int
    n_init : int
        Number of K-Means restarts.
    max_iter : int
    random_state : int

    Returns
    -------
    ClusterResult
    """
    from sklearn.cluster import KMeans

    t0 = time.perf_counter()
    n = features.shape[0]
    if n <= 1:
        labels = np.zeros(n, dtype=int)
        params = {"n_clusters": n_clusters}
        return _make_result(labels, "kmeans", params, time.perf_counter() - t0)

    n_clusters = max(1, min(n_clusters, n))
    params = {"n_clusters": n_clusters, "n_init": n_init}

    clf = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    labels = clf.fit_predict(features).astype(int)
    return _make_result(labels, "kmeans", params, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Cluster label utilities
# ---------------------------------------------------------------------------


def relabel_from_zero(labels: np.ndarray) -> np.ndarray:
    """
    Relabel cluster IDs so they are contiguous starting from 0.
    Noise points (-1) remain -1.

    Parameters
    ----------
    labels : np.ndarray

    Returns
    -------
    np.ndarray
    """
    result = labels.copy()
    current_id = 0
    mapping: Dict[int, int] = {}
    for lbl in labels:
        if lbl >= 0 and lbl not in mapping:
            mapping[lbl] = current_id
            current_id += 1
    for i, lbl in enumerate(labels):
        if lbl >= 0:
            result[i] = mapping[lbl]
    return result


def cluster_sizes(labels: np.ndarray) -> Dict[int, int]:
    """Return a dict {cluster_id: size} for all non-noise clusters."""
    sizes = {}
    for lbl in labels:
        if lbl >= 0:
            sizes[int(lbl)] = sizes.get(int(lbl), 0) + 1
    return sizes
