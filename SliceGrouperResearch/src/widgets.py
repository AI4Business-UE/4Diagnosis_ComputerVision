"""
src/widgets.py — Interactive ipywidgets Dashboard
=================================================

This module builds all interactive widgets used in the notebook.
ipywidgets enables reactive parameter controls without requiring any
server-side code — everything runs in the browser/kernel.

Architecture
------------
Each widget class wraps one or more ipywidgets controls and exposes:
  - ``widget`` property: the renderable widget tree (call display(w.widget))
  - Observe callbacks via ipywidgets' `observe` pattern

The ``DashboardWidget`` assembles all sub-widgets into a single coordinated
interface.  When any parameter changes, the relevant downstream computation
is automatically re-triggered via registered callbacks.

Design note: All widgets are stateless with respect to computed results —
they only store UI parameters.  The notebook cells are responsible for
connecting widget value changes to feature extraction and clustering.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

_logger = logging.getLogger("SliceGrouper.widgets")


# ---------------------------------------------------------------------------
# Lazy imports (avoid importing ipywidgets at module load time)
# ---------------------------------------------------------------------------

def _ipw():
    """Lazy import of ipywidgets."""
    try:
        import ipywidgets as widgets
        return widgets
    except ImportError:
        raise ImportError(
            "ipywidgets is required for interactive widgets. "
            "Install with:  pip install ipywidgets"
        )


def _display():
    """Lazy import of IPython.display."""
    from IPython.display import display
    return display


# ---------------------------------------------------------------------------
# 1. Feature Selection Widget
# ---------------------------------------------------------------------------


class FeatureSelectionWidget:
    """
    A group of checkboxes that lets the user select which feature groups
    are active in the similarity computation.

    Usage
    -----
    >>> fsw = FeatureSelectionWidget()
    >>> display(fsw.widget)
    >>> fsw.selected  # → dict of {name: bool}
    """

    FEATURES = ["geometry", "shape", "color", "texture", "deep"]

    def __init__(self, defaults: Optional[Dict[str, bool]] = None) -> None:
        widgets = _ipw()
        if defaults is None:
            defaults = {f: True for f in self.FEATURES}

        self._boxes = {
            f: widgets.Checkbox(
                value=defaults.get(f, True),
                description=f.capitalize(),
                style={"description_width": "initial"},
            )
            for f in self.FEATURES
        }
        self.widget = widgets.VBox(
            [widgets.Label("Active Feature Groups:")] + list(self._boxes.values()),
            layout=widgets.Layout(border="1px solid #ddd", padding="8px"),
        )

    @property
    def selected(self) -> Dict[str, bool]:
        """Return {feature_name: is_active} for all features."""
        return {name: box.value for name, box in self._boxes.items()}

    def observe(self, callback: Callable, names: str = "value") -> None:
        """Register a callback on any checkbox change."""
        for box in self._boxes.values():
            box.observe(callback, names=names)


# ---------------------------------------------------------------------------
# 2. Weight Slider Widget
# ---------------------------------------------------------------------------


class WeightSliderWidget:
    """
    Five linked sliders for feature group weights.

    The sliders are kept normalised so their sum stays close to 1.0
    (automatic normalisation on update is optional).

    Usage
    -----
    >>> wsw = WeightSliderWidget()
    >>> display(wsw.widget)
    >>> wsw.weights  # → {'geometry': 0.15, 'shape': 0.20, ...}
    """

    DEFAULTS = {
        "geometry": 0.15,
        "shape": 0.20,
        "color": 0.25,
        "texture": 0.20,
        "deep": 0.20,
    }

    def __init__(self, defaults: Optional[Dict[str, float]] = None) -> None:
        widgets = _ipw()
        vals = defaults if defaults is not None else self.DEFAULTS

        self._sliders = {
            name: widgets.FloatSlider(
                value=vals[name],
                min=0.0,
                max=1.0,
                step=0.05,
                description=name.capitalize(),
                style={"description_width": "80px"},
                layout=widgets.Layout(width="350px"),
            )
            for name in self.DEFAULTS
        }
        self._total_label = widgets.Label(self._total_str())
        self.widget = widgets.VBox(
            [widgets.Label("Feature Weights (should sum to 1.0):")]
            + list(self._sliders.values())
            + [self._total_label],
            layout=widgets.Layout(border="1px solid #ddd", padding="8px"),
        )
        for sl in self._sliders.values():
            sl.observe(self._update_label, names="value")

    def _total_str(self) -> str:
        total = sum(sl.value for sl in self._sliders.values()) if hasattr(self, "_sliders") else 1.0
        ok = "✅" if abs(total - 1.0) < 0.05 else "⚠️"
        return f"Total: {total:.2f}  {ok}"

    def _update_label(self, _: Any = None) -> None:
        self._total_label.value = self._total_str()

    @property
    def weights(self) -> Dict[str, float]:
        """Return current slider values as a dict."""
        return {name: sl.value for name, sl in self._sliders.items()}

    def observe(self, callback: Callable, names: str = "value") -> None:
        for sl in self._sliders.values():
            sl.observe(callback, names=names)


# ---------------------------------------------------------------------------
# 3. Threshold Slider Widget
# ---------------------------------------------------------------------------


class ThresholdSliderWidget:
    """
    Single slider for the graph edge similarity threshold.

    Usage
    -----
    >>> tsw = ThresholdSliderWidget()
    >>> display(tsw.widget)
    >>> tsw.value  # → float
    """

    def __init__(self, initial: float = 0.5) -> None:
        widgets = _ipw()
        self._slider = widgets.FloatSlider(
            value=initial,
            min=0.0,
            max=1.0,
            step=0.05,
            description="Edge threshold:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="400px"),
        )
        self.widget = widgets.VBox(
            [widgets.Label("Graph Edge Similarity Threshold:"), self._slider],
            layout=widgets.Layout(border="1px solid #ddd", padding="8px"),
        )

    @property
    def value(self) -> float:
        return float(self._slider.value)

    def observe(self, callback: Callable, names: str = "value") -> None:
        self._slider.observe(callback, names=names)


# ---------------------------------------------------------------------------
# 4. Clustering Control Widget
# ---------------------------------------------------------------------------


class ClusteringControlWidget:
    """
    Dropdown selector for clustering algorithm with dynamic secondary controls.

    The secondary controls update automatically to match the selected algorithm:
      - connected_components: threshold slider
      - agglomerative: n_clusters + linkage
      - spectral: n_clusters
      - hdbscan: min_cluster_size
      - dbscan: eps + min_samples
      - louvain / leiden: resolution
      - kmeans: n_clusters

    Usage
    -----
    >>> ccw = ClusteringControlWidget(n_components=10)
    >>> display(ccw.widget)
    >>> ccw.algorithm       # → str
    >>> ccw.params          # → dict
    """

    ALGORITHMS = [
        "connected_components",
        "agglomerative",
        "spectral",
        "hdbscan",
        "dbscan",
        "louvain",
        "leiden",
        "kmeans",
    ]

    def __init__(self, n_components: int = 10) -> None:
        widgets = _ipw()
        self._n = n_components

        self._algo_dd = widgets.Dropdown(
            options=self.ALGORITHMS,
            value="connected_components",
            description="Algorithm:",
            style={"description_width": "initial"},
        )

        # Secondary controls
        self._n_clusters = widgets.IntSlider(
            value=min(4, n_components),
            min=2,
            max=max(2, n_components),
            description="n_clusters:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="350px"),
        )
        self._threshold_sl = widgets.FloatSlider(
            value=0.5, min=0.0, max=1.0, step=0.05,
            description="Threshold:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="350px"),
        )
        self._linkage_dd = widgets.Dropdown(
            options=["average", "complete", "single"],
            value="average",
            description="Linkage:",
            style={"description_width": "initial"},
        )
        self._eps_sl = widgets.FloatSlider(
            value=0.4, min=0.05, max=1.0, step=0.05,
            description="eps:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="350px"),
        )
        self._min_cluster_sl = widgets.IntSlider(
            value=2, min=2, max=max(2, n_components // 2),
            description="min_cluster_size:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="350px"),
        )
        self._resolution_sl = widgets.FloatSlider(
            value=1.0, min=0.1, max=5.0, step=0.1,
            description="Resolution:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="350px"),
        )

        self._controls_box = widgets.VBox([])
        self._update_controls(None)
        self._algo_dd.observe(self._update_controls, names="value")

        self.widget = widgets.VBox([
            widgets.Label("Clustering Algorithm & Parameters:"),
            self._algo_dd,
            self._controls_box,
        ], layout=widgets.Layout(border="1px solid #ddd", padding="8px"))

    def _update_controls(self, _: Any) -> None:
        widgets = _ipw()
        algo = self._algo_dd.value
        ctrl_map = {
            "connected_components": [self._threshold_sl],
            "agglomerative": [self._n_clusters, self._linkage_dd, self._threshold_sl],
            "spectral": [self._n_clusters],
            "hdbscan": [self._min_cluster_sl],
            "dbscan": [self._eps_sl, self._min_cluster_sl],
            "louvain": [self._resolution_sl],
            "leiden": [self._resolution_sl],
            "kmeans": [self._n_clusters],
        }
        self._controls_box.children = ctrl_map.get(algo, [])

    @property
    def algorithm(self) -> str:
        return str(self._algo_dd.value)

    @property
    def params(self) -> Dict[str, Any]:
        algo = self.algorithm
        if algo == "connected_components":
            return {"threshold": float(self._threshold_sl.value)}
        if algo == "agglomerative":
            return {
                "n_clusters": int(self._n_clusters.value),
                "linkage": str(self._linkage_dd.value),
                "distance_threshold": float(self._threshold_sl.value),
            }
        if algo == "spectral":
            return {"n_clusters": int(self._n_clusters.value)}
        if algo == "hdbscan":
            return {"min_cluster_size": int(self._min_cluster_sl.value)}
        if algo == "dbscan":
            return {
                "eps": float(self._eps_sl.value),
                "min_samples": int(self._min_cluster_sl.value),
            }
        if algo in ("louvain", "leiden"):
            return {"resolution": float(self._resolution_sl.value)}
        if algo == "kmeans":
            return {"n_clusters": int(self._n_clusters.value)}
        return {}

    def observe(self, callback: Callable, names: str = "value") -> None:
        for widget in [
            self._algo_dd,
            self._n_clusters,
            self._threshold_sl,
            self._linkage_dd,
            self._eps_sl,
            self._min_cluster_sl,
            self._resolution_sl,
        ]:
            widget.observe(callback, names=names)


# ---------------------------------------------------------------------------
# 5. Embedding Control Widget
# ---------------------------------------------------------------------------


class EmbeddingWidget:
    """
    Controls for dimensionality reduction method and parameters.

    Usage
    -----
    >>> ew = EmbeddingWidget()
    >>> display(ew.widget)
    >>> ew.method, ew.params
    """

    METHODS = ["PCA", "t-SNE", "UMAP"]

    def __init__(self) -> None:
        widgets = _ipw()
        self._method_dd = widgets.Dropdown(
            options=self.METHODS,
            value="PCA",
            description="Method:",
            style={"description_width": "initial"},
        )
        self._perplexity = widgets.FloatSlider(
            value=5.0, min=1.0, max=50.0, step=1.0,
            description="Perplexity (t-SNE):",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="380px"),
        )
        self._n_neighbors = widgets.IntSlider(
            value=5, min=2, max=50,
            description="N neighbors (UMAP):",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="380px"),
        )
        self._min_dist = widgets.FloatSlider(
            value=0.1, min=0.0, max=1.0, step=0.05,
            description="Min dist (UMAP):",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="380px"),
        )
        self._n_components = widgets.IntSlider(
            value=2, min=2, max=3,
            description="Dimensions:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="350px"),
        )

        self._extra_box = widgets.VBox([])
        self._update_extra(None)
        self._method_dd.observe(self._update_extra, names="value")

        self.widget = widgets.VBox([
            widgets.Label("Embedding Visualisation:"),
            self._method_dd,
            self._n_components,
            self._extra_box,
        ], layout=widgets.Layout(border="1px solid #ddd", padding="8px"))

    def _update_extra(self, _: Any) -> None:
        m = self._method_dd.value
        if m == "t-SNE":
            self._extra_box.children = [self._perplexity]
        elif m == "UMAP":
            self._extra_box.children = [self._n_neighbors, self._min_dist]
        else:
            self._extra_box.children = []

    @property
    def method(self) -> str:
        return str(self._method_dd.value)

    @property
    def params(self) -> Dict[str, Any]:
        m = self.method
        base = {"n_components": int(self._n_components.value)}
        if m == "t-SNE":
            base["perplexity"] = float(self._perplexity.value)
        elif m == "UMAP":
            base["n_neighbors"] = int(self._n_neighbors.value)
            base["min_dist"] = float(self._min_dist.value)
        return base

    def observe(self, callback: Callable, names: str = "value") -> None:
        for w in [self._method_dd, self._perplexity, self._n_neighbors,
                  self._min_dist, self._n_components]:
            w.observe(callback, names=names)


# ---------------------------------------------------------------------------
# 6. Component Inspector Widget
# ---------------------------------------------------------------------------


class ComponentInspectorWidget:
    """
    An interactive component browser: select a component ID and view
    its visual properties side by side.

    Usage
    -----
    >>> ciw = ComponentInspectorWidget(components)
    >>> display(ciw.widget)
    """

    def __init__(self, components: List[Any]) -> None:
        self._components = components
        widgets = _ipw()

        self._id_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=max(0, len(components) - 1),
            description="Component ID:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="400px"),
        )
        self._out = widgets.Output()

        self._id_slider.observe(self._update, names="value")
        self._update(None)

        self.widget = widgets.VBox([
            widgets.Label("Component Inspector"),
            self._id_slider,
            self._out,
        ])

    def _update(self, _: Any) -> None:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from IPython.display import display as ipy_display

        cid = int(self._id_slider.value)
        if cid >= len(self._components):
            return
        comp = self._components[cid]

        with self._out:
            self._out.clear_output(wait=True)
            fig = plt.figure(figsize=(16, 4))
            gs = gridspec.GridSpec(1, 6, figure=fig)

            def show(ax, img, title, cmap=None):
                ax.imshow(img, cmap=cmap)
                ax.set_title(title, fontsize=9)
                ax.axis("off")

            # RGB
            ax = fig.add_subplot(gs[0, 0])
            show(ax, comp.masked_rgb, "RGB")

            # HSV
            ax = fig.add_subplot(gs[0, 1])
            import cv2
            hsv = cv2.cvtColor(comp.rgb_crop, cv2.COLOR_RGB2HSV)
            show(ax, hsv, "HSV")

            # Lab
            ax = fig.add_subplot(gs[0, 2])
            lab = cv2.cvtColor(comp.rgb_crop, cv2.COLOR_RGB2Lab)
            show(ax, lab, "Lab")

            # Mask
            ax = fig.add_subplot(gs[0, 3])
            show(ax, comp.mask_crop, "Mask", cmap="gray")

            # Contour
            ax = fig.add_subplot(gs[0, 4])
            cont_img = np.zeros((*comp.mask_crop.shape, 3), dtype=np.uint8)
            cv2.drawContours(cont_img, [comp.contour], -1, (0, 255, 0), 2)
            cv2.drawContours(cont_img, [comp.convex_hull], -1, (255, 100, 0), 1)
            show(ax, cont_img, "Contour + Hull")

            # Skeleton
            ax = fig.add_subplot(gs[0, 5])
            mask_bin = (comp.mask_crop > 0).astype(np.uint8)
            # Resize if too large to avoid freezing (max 1000px)
            if mask_bin.shape[0] > 1000 or mask_bin.shape[1] > 1000:
                scale = 1000.0 / max(mask_bin.shape)
                new_size = (int(mask_bin.shape[1] * scale), int(mask_bin.shape[0] * scale))
                mask_bin = cv2.resize(mask_bin, new_size, interpolation=cv2.INTER_NEAREST)
            from skimage.morphology import skeletonize
            skel = skeletonize(mask_bin)
            show(ax, skel.astype(np.uint8) * 255, "Skeleton", cmap="hot")

            # Stats annotation
            cx, cy = comp.centroid
            bbox = comp.bbox
            fig.text(
                0.5, -0.02,
                f"ID={comp.component_id}  |  Area={comp.area:,}px  |  "
                f"Centroid=({cx:.0f},{cy:.0f})  |  BBox={bbox}",
                ha="center", fontsize=9,
            )
            plt.tight_layout()
            ipy_display(fig)
            plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Graph Control Widget
# ---------------------------------------------------------------------------


class GraphControlWidget:
    """Controls for graph construction parameters."""

    GRAPH_TYPES = ["knn", "threshold", "delaunay", "gabriel"]
    LAYOUTS = ["spring", "kamada_kawai", "spectral", "circular", "spatial"]

    def __init__(self, n_components: int = 10) -> None:
        widgets = _ipw()

        self._graph_type = widgets.Dropdown(
            options=self.GRAPH_TYPES,
            value="threshold",
            description="Graph type:",
            style={"description_width": "initial"},
        )
        self._k = widgets.IntSlider(
            value=min(5, n_components - 1),
            min=1,
            max=max(1, n_components - 1),
            description="K (KNN):",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="350px"),
        )
        self._threshold = widgets.FloatSlider(
            value=0.5, min=0.0, max=1.0, step=0.05,
            description="Threshold:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="350px"),
        )
        self._layout = widgets.Dropdown(
            options=self.LAYOUTS,
            value="spring",
            description="Layout:",
            style={"description_width": "initial"},
        )

        self.widget = widgets.VBox([
            widgets.Label("Graph Construction:"),
            self._graph_type,
            self._k,
            self._threshold,
            self._layout,
        ], layout=widgets.Layout(border="1px solid #ddd", padding="8px"))

    @property
    def graph_type(self) -> str:
        return str(self._graph_type.value)

    @property
    def k(self) -> int:
        return int(self._k.value)

    @property
    def threshold(self) -> float:
        return float(self._threshold.value)

    @property
    def layout(self) -> str:
        return str(self._layout.value)

    def observe(self, callback: Callable, names: str = "value") -> None:
        for w in [self._graph_type, self._k, self._threshold, self._layout]:
            w.observe(callback, names=names)


# ---------------------------------------------------------------------------
# 8. Full Dashboard Widget
# ---------------------------------------------------------------------------


class DashboardWidget:
    """
    Master dashboard that combines all sub-widgets into a tabbed interface.

    Usage
    -----
    >>> dashboard = DashboardWidget(components)
    >>> display(dashboard.widget)

    The dashboard exposes `on_run` to register a callback that is invoked
    whenever the user clicks "Run Experiment".

    Parameters
    ----------
    components : list of ComponentData
    """

    def __init__(self, components: List[Any]) -> None:
        widgets = _ipw()
        self._components = components
        n = len(components)

        # Sub-widgets
        self.feature_selection = FeatureSelectionWidget()
        self.weights = WeightSliderWidget()
        self.threshold = ThresholdSliderWidget()
        self.clustering = ClusteringControlWidget(n_components=n)
        self.embedding = EmbeddingWidget()
        self.graph_control = GraphControlWidget(n_components=n)
        self.inspector = ComponentInspectorWidget(components)

        # Run button and output
        self._run_btn = widgets.Button(
            description="▶ Run Experiment",
            button_style="success",
            layout=widgets.Layout(width="200px", height="40px"),
        )
        self._status = widgets.HTML("<i>Ready</i>")
        self._output = widgets.Output()
        self._run_callbacks: List[Callable] = []
        self._run_btn.on_click(self._on_run_clicked)

        # Assemble tabs
        tab1 = widgets.VBox([
            self.feature_selection.widget,
            self.weights.widget,
            self.threshold.widget,
        ])
        tab2 = widgets.VBox([
            self.clustering.widget,
            self.graph_control.widget,
        ])
        tab3 = self.embedding.widget
        tab4 = self.inspector.widget

        tabs = widgets.Tab(children=[tab1, tab2, tab3, tab4])
        for i, name in enumerate(["Features & Weights", "Clustering", "Embedding", "Inspector"]):
            tabs.set_title(i, name)

        controls = widgets.VBox([
            widgets.HBox([self._run_btn, self._status]),
        ])

        self.widget = widgets.VBox([
            widgets.HTML("<h3>SliceGrouper Research Dashboard</h3>"),
            tabs,
            controls,
            self._output,
        ])

    def _on_run_clicked(self, _: Any) -> None:
        widgets = _ipw()
        self._status.value = "<i>⏳ Running …</i>"
        with self._output:
            self._output.clear_output(wait=True)
            for callback in self._run_callbacks:
                try:
                    callback(self)
                except Exception as exc:
                    print(f"Experiment error: {exc}")
        self._status.value = "<i>✅ Done</i>"

    def on_run(self, callback: Callable) -> None:
        """Register a callback to be called when Run is clicked."""
        self._run_callbacks.append(callback)

    # Convenience accessors
    @property
    def active_features(self) -> Dict[str, bool]:
        return self.feature_selection.selected

    @property
    def feature_weights(self) -> Dict[str, float]:
        return self.weights.weights

    @property
    def graph_threshold(self) -> float:
        return self.threshold.value

    @property
    def clustering_algorithm(self) -> str:
        return self.clustering.algorithm

    @property
    def clustering_params(self) -> Dict[str, Any]:
        return self.clustering.params

    @property
    def embedding_method(self) -> str:
        return self.embedding.method

    @property
    def embedding_params(self) -> Dict[str, Any]:
        return self.embedding.params
