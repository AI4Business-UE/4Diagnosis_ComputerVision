"""
src/visualization.py — Research Visualisations
===============================================

This module contains all plotting and visualisation functions used in the
research notebook.  Every function returns a Matplotlib Figure or Plotly
Figure so the caller controls display and export.

Design principles:
  - Every function is standalone (no global state).
  - Matplotlib is used for static publication-quality figures.
  - Plotly is used for interactive (hover/click) visualisations.
  - PyVis is used for interactive network graphs.
  - All figures can be saved to disk via src.utils.save_figure.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.figure import Figure

_logger = logging.getLogger("SliceGrouper.viz")


# ---------------------------------------------------------------------------
# Color palette helper
# ---------------------------------------------------------------------------


def _cluster_palette(n: int, cmap_name: str = "tab20") -> List[Tuple[float, float, float]]:
    """Return n RGBA colors from a Matplotlib colormap."""
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


# ---------------------------------------------------------------------------
# 1. Component Gallery
# ---------------------------------------------------------------------------


def plot_component_gallery(
    components: List[Any],
    n_cols: int = 6,
    thumbnail_size: int = 128,
    labels: Optional[np.ndarray] = None,
    title: str = "Component Gallery",
    cmap_name: str = "tab20",
) -> Figure:
    """
    Display a grid of component thumbnail images, optionally colour-coded
    by cluster assignment.

    Parameters
    ----------
    components : list of ComponentData
    n_cols : int
        Number of columns in the gallery grid.
    thumbnail_size : int
        Side length of each thumbnail in pixels.
    labels : np.ndarray or None
        Cluster labels for colour-coded borders.
    title : str
    cmap_name : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(components)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2.2))
    fig.suptitle(title, fontsize=14, y=1.01)
    axes = np.array(axes).flatten()

    if labels is not None:
        unique_labels = sorted(set(labels[labels >= 0]))
        palette = _cluster_palette(max(len(unique_labels), 1), cmap_name)
        label_color = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}
        label_color[-1] = (0.7, 0.7, 0.7, 1.0)  # noise = grey

    for i, comp in enumerate(components):
        ax = axes[i]
        thumb = cv2.resize(
            comp.masked_rgb,
            (thumbnail_size, thumbnail_size),
            interpolation=cv2.INTER_AREA,
        )
        ax.imshow(thumb)
        ax.set_title(f"C{comp.component_id}\nA={comp.area:,}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

        if labels is not None:
            lbl = int(labels[i])
            color = label_color.get(lbl, (0.5, 0.5, 0.5, 1.0))
            for spine in ax.spines.values():
                spine.set_edgecolor(color[:3])
                spine.set_linewidth(3)

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Similarity / Distance Matrix Heatmap
# ---------------------------------------------------------------------------


def plot_similarity_matrix(
    matrix: np.ndarray,
    title: str = "Similarity Matrix",
    labels: Optional[List[str]] = None,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (8, 7),
    cluster_labels: Optional[np.ndarray] = None,
) -> Figure:
    """
    Plot a pairwise similarity (or distance) matrix as a heatmap.

    Parameters
    ----------
    matrix : np.ndarray, shape (N, N)
    title : str
    labels : list of str or None
        Axis tick labels.
    cmap : str
        Matplotlib colormap.
    figsize : tuple
    cluster_labels : np.ndarray or None
        If provided, components are reordered so clusters are contiguous
        (makes block structure visible).

    Returns
    -------
    Figure
    """
    import matplotlib.ticker as ticker

    n = matrix.shape[0]
    m = matrix.copy()

    if cluster_labels is not None and len(cluster_labels) == n:
        order = np.argsort(cluster_labels)
        m = m[np.ix_(order, order)]
        if labels is not None:
            labels = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(m, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=13)

    if labels is not None and n <= 50:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
    elif n <= 30:
        ticks = list(range(n))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks], rotation=90, fontsize=7)
        ax.set_yticklabels([str(t) for t in ticks], fontsize=7)

    plt.tight_layout()
    return fig


def plot_distance_matrix(
    matrix: np.ndarray,
    title: str = "Distance Matrix",
    **kwargs: Any,
) -> Figure:
    """Plot a pairwise distance matrix (inverts colormap from similarity)."""
    kw = dict(cmap="magma_r", **kwargs)
    return plot_similarity_matrix(matrix, title=title, **kw)


# ---------------------------------------------------------------------------
# 3. Embedding Scatter Plot (Plotly interactive)
# ---------------------------------------------------------------------------


def plot_embedding_2d(
    embeddings_2d: np.ndarray,
    labels: Optional[np.ndarray] = None,
    component_ids: Optional[List[int]] = None,
    title: str = "2D Embedding",
    method_name: str = "PCA",
    figsize: Tuple[int, int] = (800, 600),
) -> "plotly.graph_objects.Figure":  # type: ignore
    """
    Interactive 2-D scatter plot of embeddings using Plotly.

    Hovering over a point shows the component ID and cluster label.

    Parameters
    ----------
    embeddings_2d : np.ndarray, shape (N, 2)
    labels : np.ndarray or None
        Cluster labels for colour coding.
    component_ids : list of int or None
    title : str
    method_name : str
        Dimensionality reduction method name for axis labels.
    figsize : (width, height) in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    import plotly.express as px

    n = len(embeddings_2d)
    ids = component_ids if component_ids is not None else list(range(n))
    hover_text = [f"Component {cid}" for cid in ids]

    if labels is not None:
        color_vals = labels.astype(str)
        hover_text = [
            f"Component {cid}<br>Cluster {lbl}"
            for cid, lbl in zip(ids, labels)
        ]
    else:
        color_vals = ["#636EFA"] * n

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        mode="markers+text",
        text=[f"C{cid}" for cid in ids],
        textposition="top center",
        textfont=dict(size=8),
        hovertext=hover_text,
        hoverinfo="text",
        marker=dict(
            size=12,
            color=labels.astype(int) if labels is not None else None,
            colorscale="Viridis",
            showscale=labels is not None,
            line=dict(width=1, color="white"),
        ),
    ))
    fig.update_layout(
        title=title,
        xaxis_title=f"{method_name} Dimension 1",
        yaxis_title=f"{method_name} Dimension 2",
        width=figsize[0],
        height=figsize[1],
        hovermode="closest",
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Cluster Overlay on Image
# ---------------------------------------------------------------------------


def plot_cluster_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    components: List[Any],
    title: str = "Cluster Overlay",
    alpha: float = 0.45,
    cmap_name: str = "tab20",
) -> Figure:
    """
    Overlay coloured cluster masks on the original tissue image.

    Parameters
    ----------
    image : np.ndarray, shape (H, W, 3), dtype uint8
    mask : np.ndarray, shape (H, W), dtype uint8
    labels : np.ndarray, shape (N,)
    components : list of ComponentData
    title, alpha, cmap_name : see module docstring.

    Returns
    -------
    Figure
    """
    overlay = image.copy().astype(np.float32)
    unique = sorted(set(int(l) for l in labels if l >= 0))
    palette = _cluster_palette(max(len(unique), 1), cmap_name)
    label_color = {lbl: (np.array(palette[i][:3]) * 255).astype(np.uint8)
                   for i, lbl in enumerate(unique)}

    color_map = np.zeros((*mask.shape, 3), dtype=np.float32)
    for comp, lbl in zip(components, labels):
        if lbl < 0:
            continue
        x, y, w, h = comp.bbox
        color = label_color.get(int(lbl), np.array([128, 128, 128], dtype=np.uint8))
        # Get component mask in image coordinates
        comp_mask = comp.mask_crop > 0
        # Determine crop slice in the full image (with margin)
        margin = 20
        y1 = max(0, y - margin)
        x1 = max(0, x - margin)
        y2 = min(image.shape[0], y + h + margin)
        x2 = min(image.shape[1], x + w + margin)
        # Align sizes (comp_mask might be slightly different due to clamping)
        cm = comp_mask[:y2 - y1, :x2 - x1]
        region = color_map[y1:y1 + cm.shape[0], x1:x1 + cm.shape[1]]
        region[cm] = color.astype(np.float32)

    blended = (1 - alpha) * overlay + alpha * color_map
    blended = blended.clip(0, 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(blended)
    ax.set_title(title, fontsize=13)
    ax.axis("off")

    patches = [
        mpatches.Patch(
            color=np.array(label_color[lbl]) / 255,
            label=f"Cluster {lbl}",
        )
        for lbl in unique
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=9, ncol=2)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Radar Chart
# ---------------------------------------------------------------------------


def plot_radar_chart(
    feature_dict: Dict[str, np.ndarray],
    component_ids: List[int],
    title: str = "Feature Radar",
) -> Figure:
    """
    Radar (spider) chart comparing normalised feature groups across components.

    Parameters
    ----------
    feature_dict : dict
        {feature_group_name: np.ndarray, shape (N,)} — one scalar per component.
    component_ids : list of int
    title : str

    Returns
    -------
    Figure
    """
    categories = list(feature_dict.keys())
    n_cat = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cat, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = plt.get_cmap("tab10")

    for ci, cid in enumerate(component_ids):
        values = []
        for key in categories:
            v = float(feature_dict[key][cid])
            values.append(v)
        values += values[:1]
        ax.plot(angles, values, color=colors(ci % 10), linewidth=2, label=f"C{cid}")
        ax.fill(angles, values, color=colors(ci % 10), alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    ax.set_title(title, size=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Histogram Comparison
# ---------------------------------------------------------------------------


def plot_histogram_comparison(
    components: List[Any],
    comp_ids: List[int],
    color_space: str = "hsv",
    bins: int = 32,
    title: Optional[str] = None,
) -> Figure:
    """
    Side-by-side histogram comparison for selected components.

    Parameters
    ----------
    components : list of ComponentData
    comp_ids : list of int
        Component indices to compare.
    color_space : str
        'rgb', 'hsv', or 'lab'.
    bins : int
    title : str or None

    Returns
    -------
    Figure
    """
    from src.color_features import build_color_histogram

    channel_names = {
        "rgb": ["R", "G", "B"],
        "hsv": ["H", "S", "V"],
        "lab": ["L", "a", "b"],
    }.get(color_space, ["C1", "C2", "C3"])

    n_comp = len(comp_ids)
    n_ch = 3
    fig, axes = plt.subplots(
        n_comp, n_ch,
        figsize=(n_ch * 4, n_comp * 2.5),
        squeeze=False,
    )
    fig.suptitle(title or f"Histogram Comparison ({color_space.upper()})", fontsize=13)

    colors_per_ch = {
        "rgb": ["#e74c3c", "#2ecc71", "#3498db"],
        "hsv": ["#9b59b6", "#1abc9c", "#f39c12"],
        "lab": ["#95a5a6", "#e74c3c", "#3498db"],
    }.get(color_space, ["grey", "grey", "grey"])

    for row, cid in enumerate(comp_ids):
        comp = components[cid]
        rgb = comp.rgb_crop
        mask = comp.mask_crop
        hist_all = build_color_histogram(rgb, mask, color_space, bins)
        per_ch = np.split(hist_all, 3)

        for ch in range(n_ch):
            ax = axes[row][ch]
            ax.bar(range(bins), per_ch[ch], color=colors_per_ch[ch], alpha=0.8, width=0.9)
            if row == 0:
                ax.set_title(f"Channel {channel_names[ch]}", fontsize=10)
            if ch == 0:
                ax.set_ylabel(f"C{cid}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Graph Visualisation (NetworkX)
# ---------------------------------------------------------------------------


def plot_graph_networkx(
    graph_data: Any,
    labels: Optional[np.ndarray] = None,
    layout: str = "spring",
    title: str = "Component Graph",
    figsize: Tuple[int, int] = (10, 8),
    cmap_name: str = "tab20",
) -> Figure:
    """
    Plot a weighted graph using NetworkX and Matplotlib.

    Parameters
    ----------
    graph_data : GraphData
    labels : np.ndarray or None
        Cluster labels for node colour coding.
    layout : str
        'spring', 'kamada_kawai', 'spectral', 'circular', or 'spatial'.
        'spatial' uses centroid coordinates.
    title : str
    figsize : tuple

    Returns
    -------
    Figure
    """
    import networkx as nx

    G = graph_data.graph
    n = G.number_of_nodes()

    if layout == "spatial" and graph_data.node_positions:
        pos = graph_data.node_positions
    elif layout == "spring":
        pos = nx.spring_layout(G, weight="weight", seed=42, k=1.5 / max(n ** 0.5, 1))
    elif layout == "kamada_kawai":
        try:
            pos = nx.kamada_kawai_layout(G, weight="weight")
        except Exception:
            pos = nx.spring_layout(G, seed=42)
    elif layout == "spectral":
        pos = nx.spectral_layout(G, weight="weight")
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Node colours
    if labels is not None and len(labels) == n:
        unique = sorted(set(int(l) for l in labels if l >= 0))
        palette = _cluster_palette(max(len(unique), 1), cmap_name)
        lc = {lbl: palette[i][:3] for i, lbl in enumerate(unique)}
        node_colors = [lc.get(int(labels[i]), (0.5, 0.5, 0.5)) for i in range(n)]
    else:
        node_colors = ["#3498db"] * n

    # Edge widths from weights
    edges = G.edges(data=True)
    edge_weights = [d.get("weight", 0.5) for _, _, d in edges]
    max_w = max(edge_weights) if edge_weights else 1.0
    edge_widths = [3 * w / max(max_w, 1e-6) for w in edge_weights]

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=400, ax=ax)
    nx.draw_networkx_labels(G, pos, {n: f"C{n}" for n in G.nodes()},
                             font_size=8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, ax=ax,
                            edge_color="grey")

    ax.set_title(title, fontsize=13)
    ax.axis("off")

    if labels is not None:
        unique_lbls = sorted(set(int(l) for l in labels if l >= 0))
        patches = [
            mpatches.Patch(color=lc[l], label=f"Cluster {l}")
            for l in unique_lbls
        ]
        ax.legend(handles=patches, loc="upper left", fontsize=9)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 8. PyVis Interactive Graph
# ---------------------------------------------------------------------------


def export_pyvis_graph(
    graph_data: Any,
    labels: Optional[np.ndarray] = None,
    output_path: str = "graph.html",
    height: str = "600px",
    cmap_name: str = "tab20",
) -> str:
    """
    Export an interactive HTML graph visualisation using PyVis.

    Requires: pip install pyvis

    Parameters
    ----------
    graph_data : GraphData
    labels : np.ndarray or None
    output_path : str
    height : str

    Returns
    -------
    str
        Absolute path to the saved HTML file.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        _logger.warning("pyvis not installed. Install with: pip install pyvis")
        return ""

    G = graph_data.graph
    n = G.number_of_nodes()

    net = Network(height=height, width="100%", bgcolor="#1a1a2e", font_color="white")

    if labels is not None and len(labels) == n:
        unique = sorted(set(int(l) for l in labels if l >= 0))
        palette = _cluster_palette(max(len(unique), 1), cmap_name)
        lc = {lbl: "#{:02x}{:02x}{:02x}".format(
            int(palette[i][0] * 255),
            int(palette[i][1] * 255),
            int(palette[i][2] * 255),
        ) for i, lbl in enumerate(unique)}
    else:
        lc = {}

    for node_id in G.nodes():
        color = lc.get(int(labels[node_id]) if labels is not None else 0, "#3498db")
        pos = graph_data.node_positions.get(node_id, (0.0, 0.0))
        net.add_node(
            node_id,
            label=f"C{node_id}",
            color=color,
            title=f"Component {node_id}",
            x=float(pos[0]) * 500,
            y=float(pos[1]) * 500,
        )

    for i, j, data in G.edges(data=True):
        w = data.get("weight", 0.5)
        net.add_edge(i, j, value=float(w), title=f"sim={w:.3f}")

    net.set_options("""
    {
      "physics": {
        "barnesHut": { "gravitationalConstant": -3000 },
        "stabilization": { "iterations": 100 }
      },
      "edges": { "smooth": false }
    }
    """)
    net.save_graph(output_path)
    _logger.info(f"PyVis graph saved: {output_path}")
    return str(Path(output_path).resolve())


# ---------------------------------------------------------------------------
# 9. Keypoint Visualisation
# ---------------------------------------------------------------------------


def plot_keypoints(
    rgb: np.ndarray,
    keypoints: List[Any],
    title: str = "Keypoints",
    max_kp: int = 100,
    figsize: Tuple[int, int] = (8, 6),
) -> Figure:
    """
    Visualise detected keypoints on a component image.

    Parameters
    ----------
    rgb : np.ndarray, shape (H, W, 3)
    keypoints : list of cv2.KeyPoint
    title : str
    max_kp : int
        Maximum number of keypoints to display.
    figsize : tuple

    Returns
    -------
    Figure
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    kp_limited = keypoints[:max_kp]
    out = cv2.drawKeypoints(
        bgr, kp_limited, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(out_rgb)
    ax.set_title(f"{title} ({len(kp_limited)} shown of {len(keypoints)})", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 10. Performance Comparison Bar Charts
# ---------------------------------------------------------------------------


def plot_performance_charts(
    benchmark_df: "pd.DataFrame",  # type: ignore
    figsize: Tuple[int, int] = (14, 10),
) -> Figure:
    """
    Generate comparison bar charts for all benchmark metrics.

    Parameters
    ----------
    benchmark_df : pd.DataFrame
        As returned by src.benchmarking.results_to_dataframe.

    Returns
    -------
    Figure
    """
    metrics = [
        ("Silhouette ↑", True, "#2ecc71"),
        ("Davies-Bouldin ↓", False, "#e74c3c"),
        ("Calinski-Harabasz ↑", True, "#3498db"),
        ("Runtime (s)", False, "#f39c12"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Clustering Algorithm Performance Comparison", fontsize=14)
    axes = axes.flatten()

    for ax, (col, higher_is_better, color) in zip(axes, metrics):
        if col not in benchmark_df.columns:
            ax.set_visible(False)
            continue
        df_sorted = benchmark_df.dropna(subset=[col]).sort_values(
            col, ascending=not higher_is_better
        )
        ax.barh(
            df_sorted["Method"],
            df_sorted[col].values,
            color=color,
            alpha=0.8,
            edgecolor="white",
        )
        ax.set_xlabel(col, fontsize=10)
        ax.set_title(col, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.invert_yaxis()

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 11. Clustering Comparison Grid
# ---------------------------------------------------------------------------


def plot_clustering_comparison(
    image: np.ndarray,
    mask: np.ndarray,
    components: List[Any],
    all_labels: Dict[str, np.ndarray],
    n_cols: int = 3,
    figsize_per_cell: Tuple[int, int] = (6, 4),
) -> Figure:
    """
    Display side-by-side cluster overlays for multiple clustering results.

    Parameters
    ----------
    image : np.ndarray
    mask : np.ndarray
    components : list of ComponentData
    all_labels : dict {method_name: labels_array}
    n_cols : int
    figsize_per_cell : tuple

    Returns
    -------
    Figure
    """
    methods = list(all_labels.keys())
    n = len(methods)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
    )
    axes = np.array(axes).flatten()

    for i, method in enumerate(methods):
        ax = axes[i]
        labels = all_labels[method]
        overlay = image.copy().astype(np.float32)

        unique = sorted(set(int(l) for l in labels if l >= 0))
        palette = _cluster_palette(max(len(unique), 1))
        lc = {lbl: (np.array(palette[idx][:3]) * 255).astype(np.uint8)
              for idx, lbl in enumerate(unique)}

        for comp, lbl in zip(components, labels):
            if lbl < 0:
                continue
            color = lc.get(int(lbl), np.array([128, 128, 128], dtype=np.uint8))
            x, y, w, h = comp.bbox
            margin = 20
            y1 = max(0, y - margin)
            x1 = max(0, x - margin)
            y2 = min(image.shape[0], y + h + margin)
            x2 = min(image.shape[1], x + w + margin)
            cm = comp.mask_crop[:y2 - y1, :x2 - x1] > 0
            region = overlay[y1:y1 + cm.shape[0], x1:x1 + cm.shape[1]]
            region[cm] = 0.5 * region[cm] + 0.5 * color.astype(np.float32)

        ax.imshow(overlay.clip(0, 255).astype(np.uint8))
        n_clusters = len(unique)
        ax.set_title(f"{method}\n({n_clusters} clusters)", fontsize=9)
        ax.axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Clustering Algorithm Comparison", fontsize=13, y=1.01)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 12. Feature Importance Bar Chart
# ---------------------------------------------------------------------------


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    title: str = "Feature Importance",
    top_k: int = 20,
    figsize: Tuple[int, int] = (10, 6),
) -> Figure:
    """
    Horizontal bar chart of feature importances.

    Parameters
    ----------
    feature_names : list of str
    importances : np.ndarray, shape (D,)
        Importance values (e.g. variance, silhouette gain).
    title : str
    top_k : int
        Show only the top-k most important features.
    figsize : tuple

    Returns
    -------
    Figure
    """
    order = np.argsort(importances)[::-1][:top_k]
    selected_names = [feature_names[i] for i in order]
    selected_vals = importances[order]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(selected_names[::-1], selected_vals[::-1],
                   color="#3498db", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 13. Stage 2 — Spatial Scatter (centroids coloured by slice)
# ---------------------------------------------------------------------------


def plot_slice_groups_spatial(
    components: List[Any],
    slice_result: "SliceGroupResult",  # type: ignore[name-defined]
    appearance_labels: Optional[np.ndarray] = None,
    title: str = "Slice Groups — Spatial Layout",
    figsize: Tuple[int, int] = (13, 7),
    cmap_slices: str = "tab10",
    cmap_types: str = "Set2",
) -> Figure:
    """
    2-D scatter plot of component centroids.

    - Colour   = slice group (Stage 2 output)
    - Marker   = appearance cluster / tissue type (Stage 1)
    - Size     = component area (log-scaled)

    Parameters
    ----------
    components : list of ComponentData
    slice_result : SliceGroupResult
    appearance_labels : np.ndarray or None
        Stage 1 labels.  If None, all markers are the same shape.
    title : str
    figsize : tuple
    cmap_slices : str
        Colormap for slice index.
    cmap_types : str
        Colormap for appearance type markers.

    Returns
    -------
    Figure
    """
    import matplotlib.patches as mpatches

    n = len(components)
    centroids = np.array([c.centroid for c in components], dtype=float)
    slice_lbl = slice_result.slice_labels
    areas = np.array([c.area for c in components], dtype=float)
    sizes = 80 + 400 * (np.log1p(areas) / (np.log1p(areas.max()) + 1e-9))

    # Colour per slice
    unique_slices = sorted(set(int(s) for s in slice_lbl if s >= 0))
    n_slices = max(len(unique_slices), 1)
    slice_cmap = plt.get_cmap(cmap_slices)
    slice_colors = {s: slice_cmap(i / max(n_slices - 1, 1)) for i, s in enumerate(unique_slices)}
    slice_colors[-1] = (0.7, 0.7, 0.7, 0.5)   # noise = grey

    # Marker per appearance type
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p"]
    if appearance_labels is not None:
        unique_app = sorted(set(int(a) for a in appearance_labels if a >= 0))
        app_marker = {a: markers[i % len(markers)] for i, a in enumerate(unique_app)}
        app_marker[-1] = "x"
    else:
        app_marker = None

    fig, ax = plt.subplots(figsize=figsize)

    # Plot per (slice, appearance_type) combination for clean legend
    if app_marker is not None:
        plotted = set()
        for i, comp in enumerate(components):
            s = int(slice_lbl[i])
            a = int(appearance_labels[i]) if appearance_labels is not None else 0
            color = slice_colors.get(s, (0.5, 0.5, 0.5, 0.5))
            marker = app_marker.get(a, "o")
            label_key = (s, a)
            lbl = (
                f"Slice {s} / Type {a}" if label_key not in plotted else None
            )
            plotted.add(label_key)
            ax.scatter(
                centroids[i, 0], centroids[i, 1],
                s=sizes[i],
                c=[color],
                marker=marker,
                edgecolors="white",
                linewidths=0.8,
                label=lbl,
                zorder=3,
            )
            ax.annotate(
                f"C{comp.component_id}",
                centroids[i],
                fontsize=7,
                ha="center",
                va="bottom",
                xytext=(0, 6),
                textcoords="offset points",
                color="black",
            )
    else:
        colors = [slice_colors.get(int(s), (0.5, 0.5, 0.5, 0.5)) for s in slice_lbl]
        ax.scatter(
            centroids[:, 0], centroids[:, 1],
            s=sizes, c=colors, edgecolors="white", linewidths=0.8, zorder=3,
        )

    # Sort-axis arrow (if sort_scores available and not all NaN)
    sort_scores = slice_result.sort_scores
    if len(sort_scores) == n and not np.all(np.isnan(sort_scores)):
        valid = ~np.isnan(sort_scores)
        min_idx = int(np.argmin(np.where(valid, sort_scores, np.inf)))
        max_idx = int(np.argmax(np.where(valid, sort_scores, -np.inf)))
        ax.annotate(
            "",
            xy=(centroids[max_idx, 0], centroids[max_idx, 1]),
            xytext=(centroids[min_idx, 0], centroids[min_idx, 1]),
            arrowprops=dict(
                arrowstyle="->", color="black", lw=1.5, connectionstyle="arc3,rad=0.0"
            ),
        )
        mid = (centroids[min_idx] + centroids[max_idx]) / 2
        ax.text(
            mid[0], mid[1] - 20,
            f"sort axis: {slice_result.sort_axis}",
            fontsize=8, ha="center", color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
        )

    # Legend — slice colours
    slice_patches = [
        mpatches.Patch(color=slice_colors[s], label=f"Slice {s}")
        for s in unique_slices
    ]
    if -1 in slice_lbl:
        slice_patches.append(mpatches.Patch(color=slice_colors[-1], label="Noise"))

    legend1 = ax.legend(
        handles=slice_patches, loc="upper right",
        title="Slice group", fontsize=8, title_fontsize=9,
    )
    ax.add_artist(legend1)

    # Legend — appearance marker types
    if app_marker is not None and appearance_labels is not None:
        type_patches = [
            plt.Line2D(
                [0], [0],
                marker=app_marker[a],
                color="w",
                markerfacecolor="grey",
                markeredgecolor="white",
                markersize=8,
                label=f"Type {a}",
            )
            for a in unique_app
        ]
        ax.legend(
            handles=type_patches, loc="upper left",
            title="Tissue type", fontsize=8, title_fontsize=9,
        )

    # Invert Y axis to match image coordinate system (0,0 = top-left)
    ax.invert_yaxis()
    ax.set_xlabel("X (pixels)", fontsize=10)
    ax.set_ylabel("Y (pixels)", fontsize=10)
    ax.set_title(title, fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 14. Stage 2 — Slice Groups on Slide Image
# ---------------------------------------------------------------------------


def plot_slice_groups_on_slide(
    image: np.ndarray,
    components: List[Any],
    slice_result: "SliceGroupResult",  # type: ignore[name-defined]
    alpha: float = 0.50,
    title: str = "Slice Groups on Slide",
    cmap_name: str = "tab10",
    max_display_px: int = 1024,
) -> Figure:
    """
    Overlay slice group colours on the full slide image.

    Each slice has a distinct colour; within a slice, all tissue types
    are rendered in the same colour (since they belong together).

    Parameters
    ----------
    image : np.ndarray, shape (H, W, 3)
    components : list of ComponentData
    slice_result : SliceGroupResult
    alpha : float
        Blending factor for colour overlay.
    title : str
    cmap_name : str
    max_display_px : int
        Downsample longest edge to this for fast rendering.

    Returns
    -------
    Figure
    """
    # Downsample for display
    h_orig, w_orig = image.shape[:2]
    scale = min(max_display_px / max(h_orig, w_orig, 1), 1.0)
    if scale < 1.0:
        img_small = cv2.resize(
            image, (max(1, int(w_orig * scale)), max(1, int(h_orig * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        img_small = image
        scale = 1.0

    # Build slice colour palette
    unique_slices = sorted(set(int(s) for s in slice_result.slice_labels if s >= 0))
    n_slices = max(len(unique_slices), 1)
    scmap = plt.get_cmap(cmap_name)
    slice_color = {
        s: (np.array(scmap(i / max(n_slices - 1, 1))[:3]) * 255).astype(np.uint8)
        for i, s in enumerate(unique_slices)
    }

    overlay = img_small.copy().astype(np.float32)

    for comp, s_lbl in zip(components, slice_result.slice_labels):
        if int(s_lbl) < 0:
            continue
        color = slice_color.get(int(s_lbl), np.array([128, 128, 128], dtype=np.uint8))
        x, y, w, h = comp.bbox
        margin = 20
        y1 = max(0, int((y - margin) * scale))
        x1 = max(0, int((x - margin) * scale))
        y2 = min(img_small.shape[0], int((y + h + margin) * scale))
        x2 = min(img_small.shape[1], int((x + w + margin) * scale))
        rh, rw = y2 - y1, x2 - x1
        if rh <= 0 or rw <= 0:
            continue
        cm = cv2.resize(
            comp.mask_crop.astype(np.uint8),
            (rw, rh),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
        region = overlay[y1:y1 + cm.shape[0], x1:x1 + cm.shape[1]]
        region[cm] = (1 - alpha) * region[cm] + alpha * color.astype(np.float32)

    # Draw centroid labels
    result_img = overlay.clip(0, 255).astype(np.uint8)
    for comp, s_lbl in zip(components, slice_result.slice_labels):
        cx, cy = comp.centroid
        cx_s, cy_s = int(cx * scale), int(cy * scale)
        cv2.putText(
            result_img,
            f"S{s_lbl}",
            (cx_s, cy_s),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.4, scale * 0.8),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(result_img)
    ax.set_title(title, fontsize=13)
    ax.axis("off")

    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(
            color=np.array(slice_color[s]) / 255,
            label=f"Slice {s}",
        )
        for s in unique_slices
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=9, ncol=2)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 15. Stage 2 — Slice Group Gallery (rows=slices, cols=tissue types)
# ---------------------------------------------------------------------------


def plot_slice_group_gallery(
    components: List[Any],
    slice_result: "SliceGroupResult",  # type: ignore[name-defined]
    thumbnail_size: int = 128,
    title: str = "Slice Group Gallery",
    cmap_slices: str = "tab10",
    cmap_types: str = "Set2",
) -> Figure:
    """
    Grid showing each slice as a ROW and each tissue type as a COLUMN.

    Empty cells appear for slices where a tissue type is missing.

    Parameters
    ----------
    components : list of ComponentData
    slice_result : SliceGroupResult
    thumbnail_size : int
        Side length (px) of each thumbnail.
    title : str

    Returns
    -------
    Figure
    """
    n_slices = slice_result.n_slices
    if n_slices == 0:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "No slices detected", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    appearance_labels = slice_result.appearance_labels
    unique_types = sorted(set(int(a) for a in appearance_labels if a >= 0))
    n_types = len(unique_types)
    type_idx = {t: i for i, t in enumerate(unique_types)}

    # Build lookup: (slice_idx, type_idx) → component index
    cell: Dict[Tuple[int, int], int] = {}
    for i, (s, a) in enumerate(zip(slice_result.slice_labels, appearance_labels)):
        if int(s) >= 0 and int(a) >= 0:
            key = (int(s), type_idx[int(a)])
            cell[key] = i   # last one wins (shouldn't conflict)

    # Color palettes
    scmap = plt.get_cmap(cmap_slices)
    tcmap = plt.get_cmap(cmap_types)
    slice_colors = [scmap(i / max(n_slices - 1, 1)) for i in range(n_slices)]
    type_colors  = [tcmap(i / max(n_types  - 1, 1)) for i in range(n_types)]

    fig, axes = plt.subplots(
        n_slices, n_types,
        figsize=(n_types * 2.2, n_slices * 2.4),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=13, y=1.01)

    # Column headers (tissue type)
    for col, t in enumerate(unique_types):
        axes[0][col].set_title(
            f"Type {t}",
            fontsize=9,
            color=type_colors[col][:3],
            fontweight="bold",
        )

    for row in range(n_slices):
        for col in range(n_types):
            ax = axes[row][col]
            key = (row, col)

            if key in cell:
                comp = components[cell[key]]
                thumb = cv2.resize(
                    comp.masked_rgb,
                    (thumbnail_size, thumbnail_size),
                    interpolation=cv2.INTER_AREA,
                )
                ax.imshow(thumb)
                ax.set_xticks([])
                ax.set_yticks([])
                # Border colour = slice colour
                for spine in ax.spines.values():
                    spine.set_edgecolor(slice_colors[row][:3])
                    spine.set_linewidth(3)
                ax.set_xlabel(
                    f"C{comp.component_id} | {comp.area:,}px",
                    fontsize=6,
                )
            else:
                # Empty cell
                ax.set_facecolor("#f0f0f0")
                ax.text(
                    0.5, 0.5, "—", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, color="#aaa",
                )
                ax.set_xticks([])
                ax.set_yticks([])

        # Row label (slice index)
        axes[row][0].set_ylabel(
            f"Slice {row}",
            fontsize=9,
            color=slice_colors[row][:3],
            fontweight="bold",
            rotation=0,
            labelpad=40,
            va="center",
        )

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 16. Stage 2 — Evaluation Bar Chart (intra vs inter distances)
# ---------------------------------------------------------------------------


def plot_slice_evaluation(
    slice_result: "SliceGroupResult",  # type: ignore[name-defined]
    components: Optional[List[Any]] = None,
    figsize: Tuple[int, int] = (10, 5),
    title: str = "Stage 2 — Slice Group Quality",
) -> Figure:
    """
    Bar chart comparing intra-slice and inter-slice spatial distances,
    plus cohesion ratio.

    Parameters
    ----------
    slice_result : SliceGroupResult
    components : list of ComponentData or None
        If provided, also shows per-slice internal distances as a secondary plot.
    figsize : tuple
    title : str

    Returns
    -------
    Figure
    """
    n_cols = 2 if components is not None and slice_result.n_slices > 1 else 1
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]

    # --- Left panel: intra vs inter bar chart --------------------------------
    ax = axes[0]
    values = [slice_result.intra_slice_dist, slice_result.inter_slice_dist]
    labels = ["Intra-slice\n(same section)", "Inter-slice\n(different sections)"]
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)
    for bar, v in zip(bars, values):
        if not np.isnan(v):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{v:.1f}px",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )
    ax.set_ylabel("Mean centroid distance (px)", fontsize=10)
    ax.set_title(
        f"Cohesion ratio = {slice_result.cohesion_ratio:.3f}"
        + (" ✓ good" if slice_result.cohesion_ratio < 1 else " ✗ poor"),
        fontsize=10,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Right panel: per-slice internal distance ----------------------------
    if n_cols == 2 and components is not None:
        ax2 = axes[1]
        centroids = np.array([c.centroid for c in components], dtype=float)
        per_slice = []
        for s_idx in range(slice_result.n_slices):
            idx = [i for i, s in enumerate(slice_result.slice_labels) if s == s_idx]
            if len(idx) < 2:
                per_slice.append(0.0)
                continue
            c = centroids[idx]
            dists = [
                float(np.linalg.norm(c[i] - c[j]))
                for i in range(len(c))
                for j in range(i + 1, len(c))
            ]
            per_slice.append(float(np.mean(dists)))

        slices = list(range(slice_result.n_slices))
        scmap = plt.get_cmap("tab10")
        bar_colors = [scmap(i / max(slice_result.n_slices - 1, 1)) for i in slices]
        ax2.bar(
            [f"Slice {s}" for s in slices],
            per_slice,
            color=bar_colors,
            edgecolor="white",
        )
        ax2.set_ylabel("Mean intra-slice distance (px)", fontsize=10)
        ax2.set_title("Per-slice spatial compactness", fontsize=10)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    return fig

