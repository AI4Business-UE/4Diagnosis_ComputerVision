"""
config.py — Global Configuration for SliceGrouperResearch
==========================================================

All tuneable parameters for the research environment are centralised here.
Modify this file to change default behaviour across all modules and the notebook.

Design philosophy:
  - One source of truth for every threshold, model name, path, and default weight.
  - Every parameter is documented with its purpose and valid range.
  - The Config dataclass can be instantiated with defaults or overridden per-experiment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------

# Root of the SliceGrouperResearch project
PROJECT_ROOT: Path = Path(__file__).parent.resolve()
SRC_DIR: Path = PROJECT_ROOT / "src"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
SAMPLES_DIR: Path = PROJECT_ROOT.parent / "backend" / "cv" / "slides"


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class IOConfig:
    """
    Parameters controlling how input images and masks are loaded.

    Attributes
    ----------
    tiff_level : int
        Pyramid level to read from a multi-resolution TIFF.
        0 = full resolution, higher numbers = lower resolution.
        Default 0 reads at full working resolution.
    max_components : int
        Safety cap on the number of connected components to process.
        Prevents memory issues on very fragmented slides.
    min_component_area_px : int
        Connected components smaller than this (in pixels) are discarded
        as noise or debris. Tune per slide magnification.
    mask_binary_threshold : int
        Pixel value threshold for binarising the mask (0-255).
        Values >= threshold are treated as tissue.
    margin_px : int
        Extra margin added around each component bounding box when cropping
        the RGB image for feature extraction. Prevents edge artefacts.
    """
    tiff_level: int = 0
    max_components: int = 500
    min_component_area_px: int = 500
    mask_binary_threshold: int = 1
    margin_px: int = 20
    
    # Sample ID (UUID folder) and base name for the slide
    sample_id: str = "97f817df-4652-4059-8a97-9f6c62cde589"
    sample_name: str = "59311 Mallory"

    @property
    def slide_path(self) -> Path:
        return SAMPLES_DIR / self.sample_id / f"{self.sample_name}.tiff"

    @property
    def mask_path(self) -> Path:
        return SAMPLES_DIR / self.sample_id / f"{self.sample_name}_mask.tiff"


@dataclass
class GeometryConfig:
    """Parameters for geometric feature extraction."""
    skeleton_method: str = "lee"          # 'lee' or 'zhang' (skimage skeletonize methods)
    pca_n_components: int = 2             # PCA components for orientation analysis
    ellipse_min_points: int = 5           # Minimum contour points to fit ellipse


@dataclass
class ShapeConfig:
    """Parameters for shape descriptor extraction."""
    fourier_n_descriptors: int = 64       # Number of Fourier coefficients to retain
    shape_context_n_bins_r: int = 5       # Radial bins in shape context histogram
    shape_context_n_bins_theta: int = 12  # Angular bins in shape context histogram
    contour_resample_n: int = 128         # Resample contour to this many points
    hu_log_epsilon: float = 1e-10         # Epsilon to avoid log(0) in Hu moments


@dataclass
class ColorConfig:
    """Parameters for color feature extraction."""
    histogram_bins: int = 32              # Bins per channel in color histograms
    stain_normalisation: str = "macenko"  # 'macenko', 'vahadane', or 'none'
    # Reference image path for stain normalisation (None = use first component)
    stain_reference_path: Optional[str] = None


@dataclass
class TextureConfig:
    """Parameters for texture feature extraction."""
    lbp_radius: int = 3                           # LBP sampling radius (pixels)
    lbp_n_points: int = 24                        # LBP sampling points (8 * radius recommended)
    lbp_method: str = "uniform"                   # 'default', 'ror', 'uniform', 'var'
    glcm_distances: List[int] = field(default_factory=lambda: [1, 2, 4])
    glcm_angles: List[float] = field(default_factory=lambda: [0.0, 0.785, 1.571, 2.356])
    gabor_frequencies: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4])
    gabor_orientations: int = 4                   # Number of orientations (0, π/n, …)


@dataclass
class LocalFeatureConfig:
    """Parameters for local keypoint descriptor extraction (ORB/SIFT/AKAZE)."""
    orb_n_keypoints: int = 500
    sift_n_features: int = 0              # 0 = retain all
    akaze_descriptor_type: int = 5        # cv2.AKAZE_DESCRIPTOR_MLDB
    max_display_keypoints: int = 100      # Limit keypoints shown in visualisations


@dataclass
class DeepConfig:
    """Parameters for deep learning feature extraction."""
    resnet_model: str = "resnet50"        # torchvision model name
    resnet_weights: str = "IMAGENET1K_V2"
    dino_model: str = "dinov2_vitb14"     # torch.hub model identifier
    input_size: Tuple[int, int] = (224, 224)
    batch_size: int = 8
    device: str = "auto"                  # 'auto', 'cpu', 'cuda', 'mps'
    # Optional HuggingFace model IDs for pathology foundation models
    uni_model_id: Optional[str] = "MahmoodLab/UNI"
    conch_model_id: Optional[str] = "MahmoodLab/CONCH"
    plip_model_id: Optional[str] = "vinid/plip"
    normalize_embeddings: bool = True     # L2-normalise all deep embeddings


@dataclass
class GraphConfig:
    """Parameters for graph construction."""
    knn_k: int = 5                        # K nearest neighbours for KNN graph
    similarity_threshold: float = 0.5    # Edge inclusion threshold for threshold graph
    delaunay_weight_attr: str = "weight"  # Edge attribute name for weights
    layout_algorithm: str = "spring"      # 'spring', 'kamada_kawai', 'spectral', 'circular'
    spring_k: Optional[float] = None      # Spring layout k param (None = auto)
    spring_iterations: int = 100


@dataclass
class SimilarityConfig:
    """Parameters for similarity/distance matrix computation."""
    # Default feature weights (must sum to 1.0)
    weight_geometry: float = 0.15
    weight_shape: float = 0.20
    weight_color: float = 0.25
    weight_texture: float = 0.20
    weight_deep: float = 0.20
    # Normalisation strategy before combining features
    normalise_features: str = "minmax"    # 'minmax', 'zscore', 'l2', 'none'


@dataclass
class ClusteringConfig:
    """Parameters for all clustering algorithms."""
    # Agglomerative
    agglomerative_linkage: str = "average"  # 'ward', 'complete', 'average', 'single'
    agglomerative_n_clusters: Optional[int] = None  # None = use distance_threshold
    agglomerative_distance_threshold: float = 0.5

    # Spectral
    spectral_n_clusters: int = 4
    spectral_affinity: str = "precomputed"

    # HDBSCAN
    hdbscan_min_cluster_size: int = 2
    hdbscan_min_samples: int = 1
    hdbscan_metric: str = "precomputed"

    # DBSCAN
    dbscan_eps: float = 0.4
    dbscan_min_samples: int = 2
    dbscan_metric: str = "precomputed"

    # Louvain / Leiden
    louvain_resolution: float = 1.0
    leiden_resolution: float = 1.0

    # KMeans
    kmeans_n_clusters: int = 4
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 300
    random_state: int = 42

    # Graph threshold for connected-components clustering
    graph_threshold: float = 0.5



@dataclass
class EmbeddingConfig:
    """Parameters for dimensionality reduction visualisations."""
    pca_n_components: int = 2
    tsne_n_components: int = 2
    tsne_perplexity: float = 5.0
    tsne_n_iter: int = 1000
    tsne_random_state: int = 42
    umap_n_components: int = 2
    umap_n_neighbors: int = 5
    umap_min_dist: float = 0.1
    umap_random_state: int = 42
    umap_metric: str = "cosine"


@dataclass
class VisualizationConfig:
    """Parameters controlling plot aesthetics and output."""
    figure_dpi: int = 150
    figure_format: str = "png"
    colormap_clusters: str = "tab20"      # Matplotlib colormap for cluster colours
    colormap_similarity: str = "viridis"  # Heatmap colormap
    gallery_thumbnail_size: int = 128     # Pixels for component gallery thumbnails
    gallery_n_cols: int = 6
    pyvis_height: str = "600px"
    pyvis_width: str = "100%"
    interactive_plot_height: int = 600
    save_figures: bool = True
    output_dir: Path = OUTPUTS_DIR


@dataclass
class SliceGrouperConfig:
    """
    Parameters for Stage 2: spatial slice regrouping.

    Stage 2 takes the appearance clusters from Stage 1 and re-groups
    components into SLICES — sets of components that physically belong
    to the same cross-sectional cut of the specimen.

    Attributes
    ----------
    sort_axis : str
        Axis for ordering components within each appearance cluster.
        'auto' (or 'pca') uses PCA on centroid positions to detect the
        dominant cutting direction automatically.
        'x' = sort by X pixel coordinate (left → right).
        'y' = sort by Y pixel coordinate (top → bottom).
    """
    sort_axis: str = "auto"   # 'auto', 'pca', 'x', or 'y'


@dataclass
class BenchmarkingConfig:
    """Parameters for the benchmarking pipeline."""
    methods_to_benchmark: List[str] = field(default_factory=lambda: [
        "connected_components",
        "agglomerative",
        "spectral",
        "hdbscan",
        "dbscan",
        "louvain",
        "leiden",
        "kmeans",
    ])
    save_csv: bool = True
    save_json: bool = True
    save_markdown: bool = True
    output_dir: Path = OUTPUTS_DIR


@dataclass
class Config:
    """
    Master configuration object for the SliceGrouperResearch environment.

    Usage
    -----
    >>> from config import Config
    >>> cfg = Config()                   # All defaults
    >>> cfg.io.min_component_area_px = 1000  # Override one parameter

    All sub-configs are dataclasses and can be overridden individually or
    passed as constructor arguments.
    """
    io: IOConfig = field(default_factory=IOConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    shape: ShapeConfig = field(default_factory=ShapeConfig)
    color: ColorConfig = field(default_factory=ColorConfig)
    texture: TextureConfig = field(default_factory=TextureConfig)
    local_features: LocalFeatureConfig = field(default_factory=LocalFeatureConfig)
    deep: DeepConfig = field(default_factory=DeepConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    slice_grouper: SliceGrouperConfig = field(default_factory=SliceGrouperConfig)
    benchmarking: BenchmarkingConfig = field(default_factory=BenchmarkingConfig)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_feature_weights(self) -> Dict[str, float]:
        """Return the current feature weights as a plain dict."""
        s = self.similarity
        return {
            "geometry": s.weight_geometry,
            "shape": s.weight_shape,
            "color": s.weight_color,
            "texture": s.weight_texture,
            "deep": s.weight_deep,
        }

    def set_feature_weights(
        self,
        geometry: float = 0.15,
        shape: float = 0.20,
        color: float = 0.25,
        texture: float = 0.20,
        deep: float = 0.20,
    ) -> None:
        """
        Set feature weights and validate that they sum to 1.0.

        Raises
        ------
        ValueError
            If weights do not sum to approximately 1.0.
        """
        total = geometry + shape + color + texture + deep
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Feature weights must sum to 1.0, got {total:.4f}. "
                "Adjust the values so geometry + shape + color + texture + deep = 1.0."
            )
        self.similarity.weight_geometry = geometry
        self.similarity.weight_shape = shape
        self.similarity.weight_color = color
        self.similarity.weight_texture = texture
        self.similarity.weight_deep = deep

    def validate(self) -> None:
        """Run basic sanity checks across all sub-configs."""
        w = self.get_feature_weights()
        total = sum(w.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Feature weights sum to {total:.4f}, expected 1.0")
        if self.io.min_component_area_px < 0:
            raise ValueError("min_component_area_px must be non-negative")
        if not (0.0 <= self.graph.similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be in [0, 1]")


# ---------------------------------------------------------------------------
# Default singleton — import and use directly in the notebook
# ---------------------------------------------------------------------------

# Usage:  from config import cfg
cfg = Config()
