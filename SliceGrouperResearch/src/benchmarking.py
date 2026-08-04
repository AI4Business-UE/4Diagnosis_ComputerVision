"""
src/benchmarking.py — Algorithm Benchmarking and Reporting
===========================================================

This module systematically evaluates all clustering configurations and
produces structured comparison tables, plots, and exportable reports.

Evaluation Metrics
------------------

Since we rarely have ground-truth labels for tissue grouping, we rely on
**internal validation metrics** that measure cluster quality without labels:

1. **Silhouette Score** (Rousseeuw, 1987)
   For each sample i: s(i) = (b(i) - a(i)) / max(a(i), b(i))
   where a(i) = mean intra-cluster distance,
         b(i) = mean distance to nearest other cluster.
   Range: [-1, 1]. Higher = better separation. -1 = wrong assignment.

2. **Davies-Bouldin Index** (Davies & Bouldin, 1979)
   DB = (1/k) Σ_k max_{k≠l} [(σ_k + σ_l) / d(c_k, c_l)]
   where σ = mean distance to centroid, d(c_k, c_l) = distance between centroids.
   Range: [0, ∞). Lower = better. 0 = perfect separation.

3. **Calinski-Harabasz Score** (Calinski & Harabasz, 1974)
   Also known as Variance Ratio Criterion.
   CH = [trace(B_k) / (k-1)] / [trace(W_k) / (n-k)]
   where B_k = between-cluster scatter, W_k = within-cluster scatter.
   Range: [0, ∞). Higher = better defined clusters.

4. **Runtime** — wall-clock time in seconds for each method.

5. **N Clusters** — number of discovered clusters.

6. **Cluster Sizes** — distribution of items per cluster.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_logger = logging.getLogger("SliceGrouper.benchmarking")


# ---------------------------------------------------------------------------
# BenchmarkResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """
    Evaluation result for a single clustering method run.

    Attributes
    ----------
    method_name : str
    n_clusters : int
    n_noise : int
    cluster_sizes : list of int
    silhouette : float
    davies_bouldin : float
    calinski_harabasz : float
    runtime : float
    params : dict
    labels : np.ndarray
    """
    method_name: str = ""
    n_clusters: int = 0
    n_noise: int = 0
    cluster_sizes: List[int] = field(default_factory=list)
    silhouette: float = float("nan")
    davies_bouldin: float = float("nan")
    calinski_harabasz: float = float("nan")
    runtime: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)
    labels: np.ndarray = field(default_factory=lambda: np.array([], dtype=int), repr=False)

    def to_dict(self) -> dict:
        """Serialisable dictionary (excludes labels array)."""
        return {
            "method_name": self.method_name,
            "n_clusters": self.n_clusters,
            "n_noise": self.n_noise,
            "cluster_sizes": self.cluster_sizes,
            "silhouette": round(self.silhouette, 4) if not np.isnan(self.silhouette) else None,
            "davies_bouldin": round(self.davies_bouldin, 4) if not np.isnan(self.davies_bouldin) else None,
            "calinski_harabasz": round(self.calinski_harabasz, 4) if not np.isnan(self.calinski_harabasz) else None,
            "runtime_s": round(self.runtime, 4),
            "params": self.params,
        }


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_silhouette(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute Silhouette Score from a precomputed distance matrix.

    Parameters
    ----------
    distance_matrix : np.ndarray, shape (N, N)
    labels : np.ndarray, shape (N,)
        Cluster labels. Points with label -1 are excluded.

    Returns
    -------
    float
        Mean Silhouette Score in [-1, 1]. Returns NaN if < 2 clusters.
    """
    from sklearn.metrics import silhouette_score

    valid = labels >= 0
    if valid.sum() < 4:
        return float("nan")
    n_unique = len(np.unique(labels[valid]))
    if n_unique < 2:
        return float("nan")
    try:
        return float(silhouette_score(
            distance_matrix[np.ix_(valid, valid)],
            labels[valid],
            metric="precomputed",
        ))
    except Exception as exc:
        _logger.debug(f"Silhouette computation failed: {exc}")
        return float("nan")


def compute_davies_bouldin(
    features: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute Davies-Bouldin Index from feature vectors.

    Parameters
    ----------
    features : np.ndarray, shape (N, D)
    labels : np.ndarray, shape (N,)

    Returns
    -------
    float
        Davies-Bouldin Index. Lower = better.
    """
    from sklearn.metrics import davies_bouldin_score

    valid = labels >= 0
    if valid.sum() < 4:
        return float("nan")
    n_unique = len(np.unique(labels[valid]))
    if n_unique < 2:
        return float("nan")
    try:
        return float(davies_bouldin_score(features[valid], labels[valid]))
    except Exception as exc:
        _logger.debug(f"Davies-Bouldin failed: {exc}")
        return float("nan")


def compute_calinski_harabasz(
    features: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute Calinski-Harabasz Score from feature vectors.

    Parameters
    ----------
    features : np.ndarray, shape (N, D)
    labels : np.ndarray, shape (N,)

    Returns
    -------
    float
        Calinski-Harabasz Score. Higher = better.
    """
    from sklearn.metrics import calinski_harabasz_score

    valid = labels >= 0
    if valid.sum() < 4:
        return float("nan")
    n_unique = len(np.unique(labels[valid]))
    if n_unique < 2:
        return float("nan")
    try:
        return float(calinski_harabasz_score(features[valid], labels[valid]))
    except Exception as exc:
        _logger.debug(f"Calinski-Harabasz failed: {exc}")
        return float("nan")


# ---------------------------------------------------------------------------
# Run single method benchmark
# ---------------------------------------------------------------------------


def benchmark_method(
    cluster_result: "ClusterResult",  # type: ignore[name-defined]
    distance_matrix: np.ndarray,
    features: np.ndarray,
) -> BenchmarkResult:
    """
    Evaluate a single ClusterResult with all internal metrics.

    Parameters
    ----------
    cluster_result : ClusterResult
        As returned by any run_* function from src.clustering.
    distance_matrix : np.ndarray, shape (N, N)
        Symmetric distance matrix for Silhouette computation.
    features : np.ndarray, shape (N, D)
        Feature matrix for Davies-Bouldin and Calinski-Harabasz.

    Returns
    -------
    BenchmarkResult
    """
    from src.clustering import cluster_sizes

    labels = cluster_result.labels
    sizes = cluster_sizes(labels)

    sil = compute_silhouette(distance_matrix, labels)
    db = compute_davies_bouldin(features, labels)
    ch = compute_calinski_harabasz(features, labels)

    return BenchmarkResult(
        method_name=cluster_result.method_name,
        n_clusters=cluster_result.n_clusters,
        n_noise=cluster_result.n_noise,
        cluster_sizes=sorted(sizes.values(), reverse=True),
        silhouette=sil,
        davies_bouldin=db,
        calinski_harabasz=ch,
        runtime=cluster_result.runtime,
        params=cluster_result.params,
        labels=labels,
    )


# ---------------------------------------------------------------------------
# Full benchmarking pipeline
# ---------------------------------------------------------------------------


def benchmark_all_methods(
    components: List[Any],
    similarity_matrix: np.ndarray,
    features: np.ndarray,
    graph: Optional[Any] = None,
    methods: Optional[List[str]] = None,
    threshold: float = 0.5,
    n_clusters: int = 4,
    min_cluster_size: int = 2,
    hdbscan_eps: float = 0.4,
) -> List[BenchmarkResult]:
    """
    Run all clustering methods and return a list of BenchmarkResult objects.

    Parameters
    ----------
    components : list of ComponentData
    similarity_matrix : np.ndarray, shape (N, N)
    features : np.ndarray, shape (N, D)
        Normalised combined feature matrix for metric computation.
    graph : networkx.Graph or None
        Graph for community detection methods. If None, a threshold graph
        is built automatically.
    methods : list of str or None
        Subset of methods to run. None = run all.
    threshold : float
        Similarity threshold for connected components and graph construction.
    n_clusters : int
        Target number of clusters for methods that require it.
    min_cluster_size : int
    hdbscan_eps : float

    Returns
    -------
    list of BenchmarkResult
    """
    from src.clustering import (
        run_agglomerative,
        run_connected_components,
        run_dbscan,
        run_hdbscan,
        run_kmeans,
        run_leiden,
        run_louvain,
        run_spectral,
    )
    from src.graph_builder import build_threshold_graph
    from src.similarity import compute_distance_matrix

    distance_matrix = compute_distance_matrix(similarity_matrix)

    if graph is None:
        graph_data = build_threshold_graph(similarity_matrix, threshold, components)
        G = graph_data.graph
    else:
        G = graph

    all_methods = {
        "connected_components": lambda: run_connected_components(similarity_matrix, threshold),
        "agglomerative": lambda: run_agglomerative(distance_matrix, n_clusters=n_clusters),
        "spectral": lambda: run_spectral(similarity_matrix, n_clusters=n_clusters),
        "hdbscan": lambda: run_hdbscan(distance_matrix, min_cluster_size=min_cluster_size),
        "dbscan": lambda: run_dbscan(distance_matrix, eps=hdbscan_eps),
        "louvain": lambda: run_louvain(G),
        "leiden": lambda: run_leiden(G),
        "kmeans": lambda: run_kmeans(features, n_clusters=n_clusters),
    }

    if methods is None:
        methods = list(all_methods.keys())

    results = []
    for method_name in methods:
        if method_name not in all_methods:
            _logger.warning(f"Unknown method: {method_name!r}, skipping.")
            continue
        _logger.info(f"Benchmarking: {method_name} …")
        try:
            result = all_methods[method_name]()
            bm = benchmark_method(result, distance_matrix, features)
            results.append(bm)
        except Exception as exc:
            _logger.error(f"Method {method_name!r} raised: {exc}")

    return results


def results_to_dataframe(results: List[BenchmarkResult]) -> pd.DataFrame:
    """
    Convert a list of BenchmarkResult objects to a pandas DataFrame.

    Returns
    -------
    pd.DataFrame
        Sorted by Silhouette Score descending (best first).
    """
    rows = []
    for r in results:
        rows.append({
            "Method": r.method_name,
            "N Clusters": r.n_clusters,
            "N Noise": r.n_noise,
            "Silhouette ↑": round(r.silhouette, 4) if not np.isnan(r.silhouette) else None,
            "Davies-Bouldin ↓": round(r.davies_bouldin, 4) if not np.isnan(r.davies_bouldin) else None,
            "Calinski-Harabasz ↑": round(r.calinski_harabasz, 4) if not np.isnan(r.calinski_harabasz) else None,
            "Runtime (s)": round(r.runtime, 4),
            "Cluster Sizes": str(r.cluster_sizes),
        })
    df = pd.DataFrame(rows)
    if not df.empty and "Silhouette ↑" in df.columns:
        df = df.sort_values("Silhouette ↑", ascending=False, na_position="last")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_benchmark_report(
    results: List[BenchmarkResult],
    output_dir: Optional[Path] = None,
    experiment_name: str = "benchmark",
) -> Dict[str, Path]:
    """
    Export benchmark results as CSV, JSON, and Markdown.

    Parameters
    ----------
    results : list of BenchmarkResult
    output_dir : Path or None
        Directory to write files. Defaults to ./outputs/.
    experiment_name : str
        Used in file names.

    Returns
    -------
    dict
        {format: Path} for each created file.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "outputs"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = results_to_dataframe(results)
    paths = {}

    # --- CSV ---
    csv_path = output_dir / f"{experiment_name}.csv"
    df.to_csv(csv_path, index=False)
    paths["csv"] = csv_path
    _logger.info(f"Saved benchmark CSV: {csv_path}")

    # --- JSON ---
    json_path = output_dir / f"{experiment_name}.json"
    json_data = [r.to_dict() for r in results]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    paths["json"] = json_path
    _logger.info(f"Saved benchmark JSON: {json_path}")

    # --- Markdown ---
    md_path = output_dir / f"{experiment_name}.md"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# SliceGrouper Benchmark Report\n\n")
        f.write(f"Generated: {timestamp}\n\n")
        f.write(f"## Results Table\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Metric Interpretation\n\n")
        f.write("| Metric | Direction | Interpretation |\n")
        f.write("|--------|-----------|----------------|\n")
        f.write("| Silhouette ↑ | Higher is better | Range [-1, 1]; 1 = perfect separation |\n")
        f.write("| Davies-Bouldin ↓ | Lower is better | 0 = perfect; higher = more overlap |\n")
        f.write("| Calinski-Harabasz ↑ | Higher is better | Unbounded above; more variance ratio = better |\n")
    paths["markdown"] = md_path
    _logger.info(f"Saved benchmark Markdown: {md_path}")

    return paths
