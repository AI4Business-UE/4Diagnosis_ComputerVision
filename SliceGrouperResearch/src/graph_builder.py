"""
src/graph_builder.py — Graph Construction for Tissue Components
================================================================

Graph-based representations are a natural fit for the tissue grouping
problem.  Each tissue component becomes a **node**, and edges encode the
degree of similarity between components.

Graph structure strongly influences downstream clustering:
  - Dense graphs → community detection algorithms work well.
  - Sparse graphs → connected-component analysis is effective.
  - Delaunay/Gabriel graphs → preserve spatial topology.

This module implements four complementary graph construction strategies:

1. **KNN Graph**
   Connect each node to its K most similar neighbours.
   Result: sparse graph, bounded degree per node.

2. **Threshold Graph**
   Add an edge between every pair with similarity > threshold.
   Result: variable density; reveals natural clusters as dense subgraphs.

3. **Delaunay Triangulation Graph**
   Connect nodes whose Voronoi regions share a face (spatial proximity).
   Result: planar graph that reflects the spatial layout of fragments.
   Edges are then weighted by similarity, allowing spatial + visual filtering.

4. **Gabriel Graph**
   A subgraph of Delaunay: keep edge (i, j) only if no other node k lies
   inside the circle with diameter [i, j].
   Result: sparser than Delaunay, better for locally-aware clustering.

All graphs are returned as weighted ``networkx.Graph`` objects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

_logger = logging.getLogger("SliceGrouper.graph")


# ---------------------------------------------------------------------------
# GraphData dataclass
# ---------------------------------------------------------------------------


@dataclass
class GraphData:
    """
    Container for a constructed graph and its metadata.

    Attributes
    ----------
    graph : networkx.Graph
        Weighted graph. Node attribute ``pos`` = (x, y) centroid.
        Edge attribute ``weight`` = similarity value.
    node_labels : list of str
        Display labels for each node (e.g. "C0", "C1", …).
    node_positions : dict
        {node_id: (x, y)} mapping for layout rendering.
    method : str
        Construction method name ('knn', 'threshold', 'delaunay', 'gabriel').
    threshold : float
        Similarity threshold used (0 for KNN).
    k : int
        K value used for KNN (0 for others).
    """
    graph: nx.Graph = field(default_factory=nx.Graph)
    node_labels: List[str] = field(default_factory=list)
    node_positions: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    method: str = ""
    threshold: float = 0.0
    k: int = 0


# ---------------------------------------------------------------------------
# Helper: centroid positions
# ---------------------------------------------------------------------------


def _build_positions(
    components: Optional[List["ComponentData"]],  # type: ignore[name-defined]
    n_nodes: int,
) -> Dict[int, Tuple[float, float]]:
    """Build a {node_id: (cx, cy)} dict from component centroids or a grid."""
    if components is not None and len(components) == n_nodes:
        return {i: comp.centroid for i, comp in enumerate(components)}
    # Fall back to a circular layout
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    return {i: (float(np.cos(a)), float(np.sin(a))) for i, a in enumerate(angles)}


# ---------------------------------------------------------------------------
# KNN Graph
# ---------------------------------------------------------------------------


def build_knn_graph(
    similarity_matrix: np.ndarray,
    k: int = 5,
    components: Optional[List] = None,
) -> GraphData:
    """
    Build a K-Nearest-Neighbours graph from a pairwise similarity matrix.

    For each node i, add edges to the K nodes with the highest similarity
    (excluding self-similarity on the diagonal).  The graph is made symmetric:
    if (i → j) is a KNN edge, (j → i) is also added.

    Parameters
    ----------
    similarity_matrix : np.ndarray, shape (N, N)
        Pairwise similarity matrix. Higher = more similar.
    k : int
        Number of neighbours per node.
    components : list or None
        Used for spatial positions in visualisation.

    Returns
    -------
    GraphData
    """
    n = similarity_matrix.shape[0]
    k = min(k, n - 1)
    G = nx.Graph()

    # Add all nodes
    positions = _build_positions(components, n)
    for i in range(n):
        G.add_node(i, pos=positions[i], label=f"C{i}")

    # Add KNN edges
    for i in range(n):
        # Similarities to all other nodes
        sims = similarity_matrix[i].copy()
        sims[i] = -np.inf  # exclude self
        # Top-k neighbours
        neighbours = np.argsort(sims)[::-1][:k]
        for j in neighbours:
            sim_val = float(similarity_matrix[i, j])
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=sim_val)

    _logger.info(f"KNN graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (k={k})")
    return GraphData(
        graph=G,
        node_labels=[f"C{i}" for i in range(n)],
        node_positions=positions,
        method="knn",
        k=k,
    )


# ---------------------------------------------------------------------------
# Threshold Graph
# ---------------------------------------------------------------------------


def build_threshold_graph(
    similarity_matrix: np.ndarray,
    threshold: float = 0.5,
    components: Optional[List] = None,
) -> GraphData:
    """
    Build a graph by adding edges between all pairs with similarity > threshold.

    This is the simplest approach and produces a graph where connected
    components correspond directly to potential tissue groups.

    Parameters
    ----------
    similarity_matrix : np.ndarray, shape (N, N)
    threshold : float
        Minimum similarity to include an edge. Range [0, 1].
    components : list or None

    Returns
    -------
    GraphData
    """
    n = similarity_matrix.shape[0]
    G = nx.Graph()

    positions = _build_positions(components, n)
    for i in range(n):
        G.add_node(i, pos=positions[i], label=f"C{i}")

    for i in range(n):
        for j in range(i + 1, n):
            sim = float(similarity_matrix[i, j])
            if sim >= threshold:
                G.add_edge(i, j, weight=sim)

    _logger.info(
        f"Threshold graph (thr={threshold:.2f}): "
        f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )
    return GraphData(
        graph=G,
        node_labels=[f"C{i}" for i in range(n)],
        node_positions=positions,
        method="threshold",
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Delaunay Graph
# ---------------------------------------------------------------------------


def build_delaunay_graph(
    components: List["ComponentData"],  # type: ignore[name-defined]
    similarity_matrix: np.ndarray,
) -> GraphData:
    """
    Build a Delaunay triangulation graph from component centroids, weighted
    by pairwise similarity.

    The Delaunay triangulation of a point set P creates triangles such that
    no point lies inside the circumcircle of any triangle (maximises minimum
    angles).  This provides a natural spatial neighbourhood graph that
    connects spatially close fragments.

    Spatial proximity alone is not enough for grouping (as stated in the
    problem requirements), so each edge is additionally weighted by the
    visual similarity of the connected components.

    Parameters
    ----------
    components : list of ComponentData
        Components with centroid coordinates.
    similarity_matrix : np.ndarray, shape (N, N)
        Pairwise similarity weights.

    Returns
    -------
    GraphData
    """
    from scipy.spatial import Delaunay

    n = len(components)
    G = nx.Graph()
    positions = _build_positions(components, n)

    for i in range(n):
        G.add_node(i, pos=positions[i], label=f"C{i}")

    if n < 3:
        # Delaunay requires at least 3 non-collinear points
        _logger.warning("Too few components for Delaunay; building complete graph.")
        for i in range(n):
            for j in range(i + 1, n):
                G.add_edge(i, j, weight=float(similarity_matrix[i, j]))
    else:
        centroids = np.array([c.centroid for c in components])
        tri = Delaunay(centroids)

        edges_added = set()
        for simplex in tri.simplices:
            for a, b in [(0, 1), (1, 2), (0, 2)]:
                i, j = int(simplex[a]), int(simplex[b])
                key = (min(i, j), max(i, j))
                if key not in edges_added:
                    G.add_edge(i, j, weight=float(similarity_matrix[i, j]))
                    edges_added.add(key)

    _logger.info(
        f"Delaunay graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )
    return GraphData(
        graph=G,
        node_labels=[f"C{i}" for i in range(n)],
        node_positions=positions,
        method="delaunay",
    )


# ---------------------------------------------------------------------------
# Gabriel Graph
# ---------------------------------------------------------------------------


def build_gabriel_graph(
    components: List["ComponentData"],  # type: ignore[name-defined]
    similarity_matrix: np.ndarray,
) -> GraphData:
    """
    Build a Gabriel graph: a subgraph of the Delaunay triangulation where
    edge (i, j) is retained only if no other node k lies strictly inside
    the open disc with diameter [pᵢ, pⱼ].

    The Gabriel graph is sparser than Delaunay, reducing spurious
    cross-fragment connections in dense slide regions.

    Parameters
    ----------
    components : list of ComponentData
    similarity_matrix : np.ndarray, shape (N, N)

    Returns
    -------
    GraphData
    """
    # Start from Delaunay
    delaunay_data = build_delaunay_graph(components, similarity_matrix)
    G_del = delaunay_data.graph

    n = len(components)
    centroids = np.array([c.centroid for c in components])
    G = nx.Graph()
    positions = delaunay_data.node_positions

    for i in range(n):
        G.add_node(i, pos=positions[i], label=f"C{i}")

    for i, j in G_del.edges():
        # Midpoint of edge (i, j)
        mid = (centroids[i] + centroids[j]) / 2.0
        # Radius = half of edge length
        r_sq = ((centroids[i] - centroids[j]) ** 2).sum() / 4.0

        is_gabriel = True
        for k in range(n):
            if k == i or k == j:
                continue
            dist_sq = ((centroids[k] - mid) ** 2).sum()
            if dist_sq < r_sq:
                is_gabriel = False
                break

        if is_gabriel:
            G.add_edge(i, j, weight=float(similarity_matrix[i, j]))

    _logger.info(
        f"Gabriel graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )
    return GraphData(
        graph=G,
        node_labels=[f"C{i}" for i in range(n)],
        node_positions=positions,
        method="gabriel",
    )


# ---------------------------------------------------------------------------
# Edge weight operations
# ---------------------------------------------------------------------------


def add_edge_weights(
    graph_data: GraphData,
    similarity_matrix: np.ndarray,
    weight_attr: str = "weight",
) -> GraphData:
    """
    Update edge weights from a new similarity matrix.

    Useful for re-weighting a graph without rebuilding its topology.

    Parameters
    ----------
    graph_data : GraphData
    similarity_matrix : np.ndarray, shape (N, N)
    weight_attr : str
        Name of the edge weight attribute.

    Returns
    -------
    GraphData (modified in place, also returned for chaining)
    """
    G = graph_data.graph
    for i, j in G.edges():
        G[i][j][weight_attr] = float(similarity_matrix[i, j])
    return graph_data


def filter_edges_by_weight(
    graph_data: GraphData,
    min_weight: float,
) -> GraphData:
    """
    Return a new GraphData with all edges below min_weight removed.

    Parameters
    ----------
    graph_data : GraphData
    min_weight : float
        Minimum edge weight to retain.

    Returns
    -------
    GraphData (new object with filtered graph)
    """
    G = graph_data.graph
    filtered = nx.Graph()
    for node, data in G.nodes(data=True):
        filtered.add_node(node, **data)
    for i, j, data in G.edges(data=True):
        if data.get("weight", 0.0) >= min_weight:
            filtered.add_edge(i, j, **data)

    _logger.info(
        f"Filtered graph: {filtered.number_of_edges()} edges remain "
        f"(min_weight={min_weight:.3f})"
    )
    return GraphData(
        graph=filtered,
        node_labels=graph_data.node_labels,
        node_positions=graph_data.node_positions,
        method=graph_data.method + f"_filtered_{min_weight:.2f}",
        threshold=min_weight,
        k=graph_data.k,
    )


def graph_to_adjacency_matrix(graph_data: GraphData, n: int) -> np.ndarray:
    """
    Convert a GraphData's graph to a dense adjacency / weight matrix.

    Parameters
    ----------
    graph_data : GraphData
    n : int
        Number of nodes (used to define matrix size).

    Returns
    -------
    np.ndarray, shape (n, n)
    """
    A = np.zeros((n, n), dtype=np.float64)
    for i, j, data in graph_data.graph.edges(data=True):
        w = data.get("weight", 1.0)
        A[i, j] = w
        A[j, i] = w
    return A
