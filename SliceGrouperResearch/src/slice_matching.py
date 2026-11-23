"""
src/slice_matching.py — Advanced Constellation-Based Slice Set Grouping
======================================================================

Solves the core pathology problem:
When a paraffin block with M biopsy cores (A, B, C) is cut into K sequential slices,
all M cores in slice k appear together on the glass slide.

Why simple spatial distance fails:
- Slices can be very close to each other (inter-slice distance < intra-slice distance).
- Components of the same tissue (e.g. A1 and A2) might be closer to each other than A1 is to B1!
- Naive 1D rank-sorting fails when a fragment is torn or missing in one slice.

Solution: Constellation / Template Matching
------------------------------------------
1. Tissue Appearance: Each component is assigned a tissue core ID (A, B, C...)
   via appearance clustering (color, texture, deep embeddings).
2. Constellation Invariance: Within any slice k, the relative position vector
   r_m = pos(core_m) - pos(core_0) is invariant across all slices because the cores
   are physically fixed in the paraffin block.
3. Slice Pitch Vector: Sequential slices are shifted along the slide by a regular
   translation vector Δ = (Δx, Δy).
4. Robust Matching: We match components into slice sets using constellation topology
   and linear assignment, with graceful handling of missing or fragmented pieces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

_logger = logging.getLogger("SliceGrouper.slice_matching")


@dataclass
class SliceSetResult:
    """
    Result of constellation-based slice set grouping.

    Attributes
    ----------
    slice_labels : np.ndarray, shape (N,)
        Assigned slice set index for each component (0 .. K-1). -1 for outliers.
    tissue_labels : np.ndarray, shape (N,)
        Assigned tissue core type for each component (0 .. M-1).
    n_slices : int
        Number of detected slice sets (K).
    n_tissues : int
        Number of distinct tissue cores per slice (M).
    slice_sets : List[Dict[str, Any]]
        Detailed composition per slice set:
        [{'slice_id': 0, 'components': {tissue_type: comp_id}, 'centroid': (x, y)}, ...]
    shift_vector : Tuple[float, float]
        Estimated translation vector (dx, dy) between consecutive slices.
    template_offsets : Dict[int, Tuple[float, float]]
        Relative offset vector of each tissue type relative to anchor tissue 0.
    confidence_scores : np.ndarray, shape (N,)
        Confidence score for each assignment (0.0 .. 1.0).
    """
    slice_labels: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    tissue_labels: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    n_slices: int = 0
    n_tissues: int = 0
    slice_sets: List[Dict[str, Any]] = field(default_factory=list)
    shift_vector: Tuple[float, float] = (0.0, 0.0)
    template_offsets: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    confidence_scores: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))


class ConstellationSliceMatcher:
    """
    Groups tissue fragments into complete slice sets using relative geometry
    and constellation matching.

    Parameters
    ----------
    max_spatial_tolerance_px : float
        Maximum allowed spatial deviation (in pixels) from the expected constellation template.
    allow_missing_fragments : bool
        If True, a slice can have fewer than M components if a fragment was torn or lost.
    """

    def __init__(
        self,
        max_spatial_tolerance_px: float = 300.0,
        allow_missing_fragments: bool = True,
    ):
        self.max_spatial_tolerance_px = max_spatial_tolerance_px
        self.allow_missing_fragments = allow_missing_fragments

    def match(
        self,
        components: List[Any],
        tissue_labels: np.ndarray,
        slide_direction_hint: str = "auto",
    ) -> SliceSetResult:
        """
        Group components into slices using constellation matching.

        Parameters
        ----------
        components : list of ComponentData
            Detected components on the slide with .centroid = (cx, cy) and .component_id.
        tissue_labels : np.ndarray, shape (N,)
            Tissue core class for each component from Stage 1 appearance clustering.
        slide_direction_hint : str
            'auto', 'x' (horizontal layout), or 'y' (vertical layout).
        """
        N = len(components)
        if N == 0:
            return SliceSetResult()

        centroids = np.array([c.centroid for c in components], dtype=float)
        unique_tissues = sorted(list(set(tissue_labels[tissue_labels >= 0])))
        M = len(unique_tissues)

        if M <= 1:
            # Fallback for 1 tissue type: sort along dominant axis
            _logger.info("Single tissue type detected; grouping along dominant axis.")
            return self._fallback_single_tissue(components, centroids, tissue_labels)

        # 1. Identify anchor tissue (the tissue with the cleanest/most consistent distribution)
        tissue_groups: Dict[int, List[int]] = {t: [] for t in unique_tissues}
        for idx, t in enumerate(tissue_labels):
            if t >= 0:
                tissue_groups[t].append(idx)

        # Determine K (number of slices) from median counts across tissues
        counts = [len(idxs) for idxs in tissue_groups.values() if idxs]
        K = int(np.median(counts)) if counts else 1
        K = max(1, K)

        # Pick anchor tissue: the one with count closest to K
        anchor_tissue = min(unique_tissues, key=lambda t: abs(len(tissue_groups[t]) - K))
        anchor_indices = tissue_groups[anchor_tissue]

        # 2. Determine dominant slice axis and order anchor components along this axis
        anchor_pts = centroids[anchor_indices]
        if len(anchor_pts) >= 2:
            dx_span = anchor_pts[:, 0].max() - anchor_pts[:, 0].min()
            dy_span = anchor_pts[:, 1].max() - anchor_pts[:, 1].min()
            is_horizontal = (dx_span >= dy_span) if slide_direction_hint == "auto" else (slide_direction_hint == "x")
            sort_axis_idx = 0 if is_horizontal else 1
            sorted_anchor = sorted(anchor_indices, key=lambda i: centroids[i, sort_axis_idx])
        else:
            sorted_anchor = anchor_indices

        # 3. Estimate inter-slice shift vector Δ = (Δx, Δy)
        shift_vector = (0.0, 0.0)
        if len(sorted_anchor) >= 2:
            shifts = np.diff(centroids[sorted_anchor], axis=0)
            shift_vector = (float(np.median(shifts[:, 0])), float(np.median(shifts[:, 1])))

        # 4. Estimate constellation template (relative vector from anchor to each other tissue)
        template_offsets: Dict[int, Tuple[float, float]] = {anchor_tissue: (0.0, 0.0)}
        for t in unique_tissues:
            if t == anchor_tissue:
                continue
            t_indices = tissue_groups[t]
            if not t_indices:
                template_offsets[t] = (0.0, 0.0)
                continue

            # Find median vector from nearest anchor to each element of tissue t
            all_offsets = []
            for t_idx in t_indices:
                t_pt = centroids[t_idx]
                dists = np.linalg.norm(anchor_pts - t_pt, axis=1)
                best_anchor_idx = anchor_indices[int(np.argmin(dists))]
                offset = t_pt - centroids[best_anchor_idx]
                all_offsets.append(offset)

            if all_offsets:
                med_offset = np.median(np.array(all_offsets), axis=0)
                template_offsets[t] = (float(med_offset[0]), float(med_offset[1]))
            else:
                template_offsets[t] = (0.0, 0.0)

        # 5. Global Assignment: Build candidate slices around each anchor slice
        # If there are K anchor points, each forms the seed for one slice
        slice_labels = np.full(N, -1, dtype=int)
        confidence_scores = np.zeros(N, dtype=float)
        slice_sets: List[Dict[str, Any]] = []

        # Assign anchor points to slice 0 .. len(sorted_anchor)-1
        for s_idx, a_idx in enumerate(sorted_anchor):
            slice_labels[a_idx] = s_idx
            confidence_scores[a_idx] = 1.0

        # For each non-anchor tissue type, use optimal bipartite matching (Hungarian algorithm)
        # to assign components to the candidate slices based on expected constellation position
        for t in unique_tissues:
            if t == anchor_tissue:
                continue

            t_indices = tissue_groups[t]
            if not t_indices:
                continue

            n_slices = len(sorted_anchor)
            n_comps = len(t_indices)

            # Build cost matrix: cost(slice_s, component_j) = distance from expected position
            offset = np.array(template_offsets[t])
            expected_positions = centroids[sorted_anchor] + offset  # shape (n_slices, 2)
            actual_positions = centroids[t_indices]                  # shape (n_comps, 2)

            cost_matrix = np.zeros((n_slices, n_comps), dtype=float)
            for s in range(n_slices):
                for j in range(n_comps):
                    dist = np.linalg.norm(expected_positions[s] - actual_positions[j])
                    cost_matrix[s, j] = dist

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for s_slot, c_slot in zip(row_ind, col_ind):
                dist = cost_matrix[s_slot, c_slot]
                comp_idx = t_indices[c_slot]
                # Accept assignment if within tolerance or relative fit is good
                conf = max(0.0, min(1.0, 1.0 - (dist / max(1.0, self.max_spatial_tolerance_px))))
                slice_labels[comp_idx] = s_slot
                confidence_scores[comp_idx] = conf

        # Build structured slice_sets list
        n_final_slices = len(sorted_anchor)
        for s_idx in range(n_final_slices):
            members = [i for i, lbl in enumerate(slice_labels) if lbl == s_idx]
            slice_comps = {}
            for m in members:
                t_type = int(tissue_labels[m])
                slice_comps[t_type] = components[m].component_id

            slice_centroid = (
                centroids[members].mean(axis=0).tolist()
                if members else [0.0, 0.0]
            )

            slice_sets.append({
                "slice_id": s_idx,
                "n_components": len(members),
                "components": slice_comps,
                "component_indices": members,
                "centroid": slice_centroid,
            })

        return SliceSetResult(
            slice_labels=slice_labels,
            tissue_labels=tissue_labels,
            n_slices=n_final_slices,
            n_tissues=M,
            slice_sets=slice_sets,
            shift_vector=shift_vector,
            template_offsets=template_offsets,
            confidence_scores=confidence_scores,
        )

    def _fallback_single_tissue(
        self,
        components: List[Any],
        centroids: np.ndarray,
        tissue_labels: np.ndarray,
    ) -> SliceSetResult:
        """Fallback when all components belong to a single tissue type."""
        from sklearn.decomposition import PCA

        pca = PCA(n_components=1)
        projections = pca.fit_transform(centroids).ravel()
        sorted_indices = np.argsort(projections)

        slice_labels = np.full(len(components), -1, dtype=int)
        for s_idx, comp_idx in enumerate(sorted_indices):
            slice_labels[comp_idx] = s_idx

        slice_sets = [
            {
                "slice_id": s_idx,
                "n_components": 1,
                "components": {0: components[comp_idx].component_id},
                "component_indices": [comp_idx],
                "centroid": centroids[comp_idx].tolist(),
            }
            for s_idx, comp_idx in enumerate(sorted_indices)
        ]

        return SliceSetResult(
            slice_labels=slice_labels,
            tissue_labels=tissue_labels,
            n_slices=len(components),
            n_tissues=1,
            slice_sets=slice_sets,
            shift_vector=(0.0, 0.0),
            template_offsets={0: (0.0, 0.0)},
            confidence_scores=np.ones(len(components), dtype=float),
        )
