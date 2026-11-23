"""
src/slice_matching_viz.py — Visualizations for Constellation Slice Matching
===========================================================================

Produces clear, publication-quality figures for:
1. Constellation Map: Shows each slice set as a connected polygon on the slide,
   with lines connecting cores within the same slice and arrows showing the slice sequence.
2. Cross-Slice Grid: Thumbnail gallery aligning corresponding cores (columns) across slices (rows).
3. Benchmark Performance: Bar charts of ARI, NMI, and perfect slice percentages across datasets.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Polygon


def plot_constellation_slices_on_slide(
    image: np.ndarray,
    components: List[Any],
    slice_set_result: Any,
    alpha: float = 0.45,
    figsize: Tuple[int, int] = (14, 8),
    title: str = "Constellation Slice Sets & Core Matching",
) -> Figure:
    """
    Draw connected constellations on the whole slide image.
    Each slice set is connected by a dashed polygon and color-coded.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)

    centroids = np.array([c.centroid for c in components], dtype=float)
    slice_labels = slice_set_result.slice_labels
    tissue_labels = slice_set_result.tissue_labels
    n_slices = slice_set_result.n_slices

    slice_cmap = plt.get_cmap("tab10")
    tissue_markers = ["o", "s", "^", "D", "v", "P"]
    tissue_names = ["A", "B", "C", "D", "E", "F"]

    # 1. Connect components in the same slice with lines/polygon
    for s_idx in range(n_slices):
        members = [i for i, s in enumerate(slice_labels) if s == s_idx]
        if len(members) >= 2:
            pts = centroids[members]
            color = slice_cmap(s_idx % 10)
            # Draw polygon or convex hull outline
            if len(members) >= 3:
                hull = cv2.convexHull(pts.astype(np.float32))
                poly = Polygon(
                    hull.reshape(-1, 2),
                    closed=True,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.18,
                    linewidth=2.5,
                    linestyle="--",
                )
                ax.add_patch(poly)
            else:
                ax.plot(pts[:, 0], pts[:, 1], color=color, linestyle="--", linewidth=2.5, alpha=0.6)

    # 2. Plot individual components with badges and labels
    for i, c in enumerate(components):
        s_idx = slice_labels[i]
        t_idx = tissue_labels[i]
        cx, cy = c.centroid

        color = slice_cmap(s_idx % 10) if s_idx >= 0 else (0.5, 0.5, 0.5)
        marker = tissue_markers[t_idx % len(tissue_markers)] if t_idx >= 0 else "x"
        tissue_name = tissue_names[t_idx % len(tissue_names)] if t_idx >= 0 else "?"

        ax.scatter(cx, cy, s=280, color=color, edgecolors="white", linewidth=1.5, marker=marker, zorder=5)
        label_text = f"S{s_idx}:{tissue_name} (#{c.component_id})" if s_idx >= 0 else f"Noise (#{c.component_id})"
        ax.annotate(
            label_text,
            (cx, cy),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor=color),
            zorder=6,
        )

    # 3. Draw inter-slice trajectory arrow
    slice_centroids = []
    for s_idx in range(n_slices):
        members = [i for i, s in enumerate(slice_labels) if s == s_idx]
        if members:
            slice_centroids.append(centroids[members].mean(axis=0))

    if len(slice_centroids) >= 2:
        sc = np.array(slice_centroids)
        for i in range(len(sc) - 1):
            ax.annotate(
                "",
                xy=(sc[i + 1, 0], sc[i + 1, 1]),
                xytext=(sc[i, 0], sc[i, 1]),
                arrowprops=dict(arrowstyle="->", color="#e67e22", lw=3.0, mutation_scale=18),
                zorder=7,
            )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    return fig


def plot_cross_slice_alignment_grid(
    image: np.ndarray,
    components: List[Any],
    slice_set_result: Any,
    thumbnail_size: int = 120,
    figsize: Optional[Tuple[int, int]] = None,
) -> Figure:
    """
    Grid gallery:
    Rows = Slices (Slice 0, Slice 1, Slice 2...)
    Columns = Tissue Cores (Core A, Core B, Core C...)
    Demonstrates exact cross-slice matching!
    """
    n_slices = max(1, slice_set_result.n_slices)
    n_tissues = max(1, slice_set_result.n_tissues)
    tissue_names = ["Core A", "Core B", "Core C", "Core D", "Core E"]

    if figsize is None:
        figsize = (max(6, n_tissues * 3), max(4, n_slices * 2.5))

    fig, axes = plt.subplots(n_slices, n_tissues, figsize=figsize, squeeze=False)

    # Map (slice_id, tissue_id) -> component
    grid_map: dict = {}
    for i, c in enumerate(components):
        s = slice_set_result.slice_labels[i]
        t = slice_set_result.tissue_labels[i]
        if s >= 0 and t >= 0:
            grid_map[(s, t)] = c

    for s in range(n_slices):
        for t in range(n_tissues):
            ax = axes[s, t]
            comp = grid_map.get((s, t))
            if comp is not None:
                x, y, w, h = comp.bbox
                margin = 15
                h_img, w_img = image.shape[:2]
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(w_img, x + w + margin)
                y2 = min(h_img, y + h + margin)
                crop = image[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_resized = cv2.resize(crop, (thumbnail_size, thumbnail_size))
                    ax.imshow(crop_resized)
                ax.set_title(f"Comp #{comp.component_id}", fontsize=9)
            else:
                # Missing core placeholder
                blank = np.ones((thumbnail_size, thumbnail_size, 3), dtype=np.uint8) * 240
                ax.imshow(blank)
                ax.text(
                    thumbnail_size // 2,
                    thumbnail_size // 2,
                    "Missing Core",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#999",
                )

            ax.set_xticks([])
            ax.set_yticks([])

            if t == 0:
                ax.set_ylabel(f"Slice {s}", fontsize=11, fontweight="bold")
            if s == 0:
                col_name = tissue_names[t] if t < len(tissue_names) else f"Core {t}"
                ax.set_xlabel(col_name, fontsize=11, fontweight="bold")
                ax.xaxis.set_label_position("top")

    fig.suptitle("Cross-Slice Tissue Core Alignment Grid", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig
