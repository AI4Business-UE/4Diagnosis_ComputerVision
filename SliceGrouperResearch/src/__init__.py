"""
src/__init__.py — SliceGrouperResearch Package
===============================================

Re-exports the most commonly used symbols from every submodule
so the notebook can do:

    from src import ComponentData, Config, ...

without knowing the exact module layout.
"""

from src.io import (
    load_tiff,
    load_mask,
    detect_components,
    ComponentData,
)
from src.utils import Timer, ensure_dir, log_step
from src.slice_grouper import (
    SliceGroupResult,
    group_slices_by_position,
    slice_composition_table,
    print_slice_summary,
)
from src.clustering import cluster_components, ClusterResult
from src.slice_matching import (
    ConstellationSliceMatcher,
    SliceSetResult,
)
from src.evaluation import (
    evaluate_slice_grouping,
    evaluate_dataset,
    load_ground_truth,
    SlideEvaluationResult,
)
from src.labeler import SlideLabelerWidget
from src.slice_matching_viz import (
    plot_constellation_slices_on_slide,
    plot_cross_slice_alignment_grid,
)
from src.method_benchmarker import (
    BaseSliceGrouperStrategy,
    BaselineSpatialMethod,
    AppearanceOnlyMethod,
    ConstellationMethod,
    AdaptiveConstellationMethod,
    MLAdaptiveConstellationMethod,
    HybridSafetySpatialMethod,
    benchmark_all_methods,
    generate_method_summary_table,
)

__all__ = [
    # io
    "load_tiff",
    "load_mask",
    "detect_components",
    "ComponentData",
    # utils
    "Timer",
    "ensure_dir",
    "log_step",
    # slice_grouper (Stage 2)
    "SliceGroupResult",
    "group_slices_by_position",
    "slice_composition_table",
    "print_slice_summary",
    # slice_matching (Constellation)
    "ConstellationSliceMatcher",
    "SliceSetResult",
    # evaluation
    "evaluate_slice_grouping",
    "evaluate_dataset",
    "load_ground_truth",
    "SlideEvaluationResult",
    # labeler
    "SlideLabelerWidget",
    # viz
    "plot_constellation_slices_on_slide",
    "plot_cross_slice_alignment_grid",
    # method benchmarker
    "BaseSliceGrouperStrategy",
    "BaselineSpatialMethod",
    "AppearanceOnlyMethod",
    "ConstellationMethod",
    "AdaptiveConstellationMethod",
    "benchmark_all_methods",
    "generate_method_summary_table",
]

