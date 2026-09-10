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
]
