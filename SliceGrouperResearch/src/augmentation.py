"""
src/augmentation.py — Histopathology Serial Section Augmentation Module
=======================================================================

Generates realistic synthetic 3- and 4-slice serial sections exclusively from
the TRAINING set. Guarantees that the TEST set is never touched.

Histological Realism:
---------------------
When microtome technicians cut consecutive serial sections from a formalin-fixed
paraffin-embedded (FFPE) tissue block:
1. Geometry: Cores maintain the same constellation geometry with slight micro-rotation (1-3 deg).
2. Spacing: Consecutive slices are laid out sequentially along the dominant slide axis.
3. Micro-deformation: Slight non-uniform stretching/compression from the water bath (scale +/- 4%).
4. Staining: Slight optical density / color variation between consecutive sections (+/- 3% RGB).
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from src.dataset_split import load_dataset_split
from src.io import ComponentData, load_slide

_logger = logging.getLogger("SliceGrouper.augmentation")

AUG_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "augmented_data"
AUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SyntheticSlide:
    """Container for an augmented multi-slice synthetic slide."""
    slide_name: str
    sample_id: str
    image: np.ndarray
    mask: np.ndarray
    components: List[ComponentData]
    slice_labels: np.ndarray
    tissue_labels: np.ndarray
    n_slices: int


class SerialSectionAugmentor:
    """
    Augmentor that produces realistic 3- and 4-slice slides from training samples.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        random_state: int = 42,
    ):
        self.output_dir = Path(output_dir) if output_dir else AUG_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.RandomState(random_state)
        
        # Load verified frozen train/test split
        train_names, test_names = load_dataset_split()
        self.train_slide_names = set(train_names)
        self.test_slide_names = set(test_names)

    def is_train_slide(self, slide_name: str) -> bool:
        """Enforces zero data leakage: returns True only if slide is in Train set."""
        return slide_name in self.train_slide_names and slide_name not in self.test_slide_names

    def augment_to_multi_slice(
        self,
        base_image: np.ndarray,
        base_mask: np.ndarray,
        base_components: List[ComponentData],
        base_slice_labels: np.ndarray,
        base_tissue_labels: np.ndarray,
        target_n_slices: int = 3,
        slide_name_prefix: str = "aug",
    ) -> SyntheticSlide:
        """
        Takes a base slide (with 1 or 2 slices) and expands it into a synthetic
        target_n_slices (3 or 4) slide layout along the primary slide axis.
        """
        assert target_n_slices in (3, 4), "Target slices must be 3 or 4."
        H, W = base_image.shape[:2]

        # 1. Identify representative core template(s) from slice 0
        s0_indices = [i for i, s in enumerate(base_slice_labels) if s == 0]
        if not s0_indices:
            s0_indices = list(range(len(base_components)))

        template_comps = [base_components[i] for i in s0_indices]
        template_tissues = [base_tissue_labels[i] for i in s0_indices]

        # Determine dominant axis for serial alignment (usually vertical in histopathology)
        # Vertical = 0 (top-to-bottom), Horizontal = 1 (left-to-right)
        layout_axis = 0 if H >= W else 1
        canvas_h = max(H, int(H * 1.2)) if layout_axis == 0 else H
        canvas_w = max(W, int(W * 1.2)) if layout_axis == 1 else W

        aug_image = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)  # Clean glass background
        aug_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

        new_components: List[ComponentData] = []
        new_slice_labels: List[int] = []
        new_tissue_labels: List[Any] = []

        # Compute centroid center of the template constellation
        t_cents = np.array([c.centroid for c in template_comps])
        template_center = np.mean(t_cents, axis=0)

        # Spacing step along layout axis
        total_span = (canvas_h if layout_axis == 0 else canvas_w) * 0.85
        margin = (canvas_h if layout_axis == 0 else canvas_w) * 0.08
        step = total_span / target_n_slices

        global_comp_id = 0

        for slice_idx in range(target_n_slices):
            target_pos = margin + (slice_idx + 0.5) * step
            
            # Microtome physical perturbations
            angle_deg = float(self.rng.uniform(-3.5, 3.5))
            scale_factor = float(self.rng.uniform(0.96, 1.04))
            color_shift = self.rng.uniform(-7, 7, size=(1, 1, 3))
            
            # Position offset
            if layout_axis == 0:
                dy = target_pos - template_center[1] + float(self.rng.uniform(-15, 15))
                dx = float(self.rng.uniform(-25, 25))
            else:
                dx = target_pos - template_center[0] + float(self.rng.uniform(-15, 15))
                dy = float(self.rng.uniform(-25, 25))

            for comp_idx, comp in enumerate(template_comps):
                # 1. Transform RGB and Mask crop
                crop_rgb = comp.rgb_crop.astype(np.float32)
                crop_mask = comp.mask_crop.copy()
                
                # Apply staining variation
                crop_rgb = np.clip(crop_rgb + color_shift, 0, 255).astype(np.uint8)

                # Affine transformation matrix
                ch, cw = crop_rgb.shape[:2]
                M_rot = cv2.getRotationMatrix2D((cw / 2, ch / 2), angle_deg, scale_factor)
                
                # Compute new bounding box size after rotation
                cos = np.abs(M_rot[0, 0])
                sin = np.abs(M_rot[0, 1])
                new_w = int((ch * sin) + (cw * cos))
                new_h = int((ch * cos) + (cw * sin))
                M_rot[0, 2] += (new_w / 2) - (cw / 2)
                M_rot[1, 2] += (new_h / 2) - (ch / 2)

                rot_rgb = cv2.warpAffine(
                    crop_rgb, M_rot, (new_w, new_h),
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(245, 245, 245)
                )
                rot_mask = cv2.warpAffine(
                    crop_mask, M_rot, (new_w, new_h),
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )

                # 2. Place on canvas
                orig_cx, orig_cy = comp.centroid
                rel_cx = orig_cx - template_center[0]
                rel_cy = orig_cy - template_center[1]
                
                # New centroid coordinates
                new_cx = template_center[0] + dx + rel_cx
                new_cy = template_center[1] + dy + rel_cy
                
                top_left_x = int(round(new_cx - new_w / 2))
                top_left_y = int(round(new_cy - new_h / 2))

                # Bounds clamping
                if top_left_x < 0:
                    top_left_x = 0
                if top_left_y < 0:
                    top_left_y = 0
                if top_left_x + new_w >= canvas_w:
                    top_left_x = canvas_w - new_w - 1
                if top_left_y + new_h >= canvas_h:
                    top_left_y = canvas_h - new_h - 1

                # Blend into canvas
                tissue_pix = rot_mask > 128
                target_roi = aug_image[top_left_y : top_left_y + new_h, top_left_x : top_left_x + new_w]
                target_mask_roi = aug_mask[top_left_y : top_left_y + new_h, top_left_x : top_left_x + new_w]

                target_roi[tissue_pix] = rot_rgb[tissue_pix]
                target_mask_roi[tissue_pix] = 255

                cnts, _ = cv2.findContours(rot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour = cnts[0] if cnts else np.zeros((1, 1, 2), dtype=np.int32)
                convex_hull = cv2.convexHull(contour) if len(contour) >= 3 else contour

                # Construct new ComponentData
                new_comp = ComponentData(
                    component_id=global_comp_id,
                    bbox=(top_left_x, top_left_y, new_w, new_h),
                    area=int(np.sum(tissue_pix)),
                    centroid=(float(top_left_x + new_w / 2), float(top_left_y + new_h / 2)),
                    mask_crop=rot_mask,
                    rgb_crop=rot_rgb,
                    contour=contour,
                    convex_hull=convex_hull,
                    label_id=global_comp_id + 1,
                )
                new_components.append(new_comp)
                new_slice_labels.append(slice_idx)
                new_tissue_labels.append(template_tissues[comp_idx])
                global_comp_id += 1

        synthetic = SyntheticSlide(
            slide_name=f"{slide_name_prefix}_slices_{target_n_slices}",
            sample_id=f"aug_{target_n_slices}_slices",
            image=aug_image,
            mask=aug_mask,
            components=new_components,
            slice_labels=np.array(new_slice_labels, dtype=int),
            tissue_labels=np.array(new_tissue_labels),
            n_slices=target_n_slices,
        )
        return synthetic

    def generate_training_augmentation_batch(
        self,
        slides_info: List[Dict[str, Any]],
        num_3_slice_samples: int = 30,
        num_4_slice_samples: int = 25,
    ) -> List[SyntheticSlide]:
        """
        Takes candidate slides from the training set and produces a balanced
        batch of 3-slice and 4-slice synthetic slides.
        """
        # STRICT LEAKAGE GUARD: filter to only verified train slides
        valid_train_slides = [s for s in slides_info if self.is_train_slide(s["slide_name"])]
        _logger.info(f"Augmentation: {len(valid_train_slides)} verified training slides eligible.")

        if not valid_train_slides:
            raise ValueError("No eligible training slides found! Ensure train_test_split.json exists.")

        synthetic_slides: List[SyntheticSlide] = []

        # Generate 3-slice samples
        for i in range(num_3_slice_samples):
            base_info = valid_train_slides[i % len(valid_train_slides)]
            image, mask, comps = load_slide(
                tiff_path=base_info["tiff_path"],
                mask_path=base_info["mask_path"],
                min_area_px=6000,
            )
            with open(base_info["labels_path"], "r", encoding="utf-8") as f:
                gt_data = json.load(f)
            gt_by_id = {c["id"]: c for c in gt_data.get("components", [])}
            s_labels = np.array([gt_by_id.get(k, {}).get("slice_id", 0) for k in range(len(comps))])
            t_labels = np.array([gt_by_id.get(k, {}).get("tissue_id", "A") for k in range(len(comps))])

            syn = self.augment_to_multi_slice(
                base_image=image,
                base_mask=mask,
                base_components=comps,
                base_slice_labels=s_labels,
                base_tissue_labels=t_labels,
                target_n_slices=3,
                slide_name_prefix=f"syn3_{i+1:02d}_{base_info['slide_name']}",
            )
            synthetic_slides.append(syn)

        # Generate 4-slice samples
        for i in range(num_4_slice_samples):
            base_info = valid_train_slides[i % len(valid_train_slides)]
            image, mask, comps = load_slide(
                tiff_path=base_info["tiff_path"],
                mask_path=base_info["mask_path"],
                min_area_px=6000,
            )
            with open(base_info["labels_path"], "r", encoding="utf-8") as f:
                gt_data = json.load(f)
            gt_by_id = {c["id"]: c for c in gt_data.get("components", [])}
            s_labels = np.array([gt_by_id.get(k, {}).get("slice_id", 0) for k in range(len(comps))])
            t_labels = np.array([gt_by_id.get(k, {}).get("tissue_id", "A") for k in range(len(comps))])

            syn = self.augment_to_multi_slice(
                base_image=image,
                base_mask=mask,
                base_components=comps,
                base_slice_labels=s_labels,
                base_tissue_labels=t_labels,
                target_n_slices=4,
                slide_name_prefix=f"syn4_{i+1:02d}_{base_info['slide_name']}",
            )
            synthetic_slides.append(syn)

        _logger.info(
            f"✅ Generated {len(synthetic_slides)} synthetic slides "
            f"({num_3_slice_samples} x 3-slices, {num_4_slice_samples} x 4-slices)."
        )
        return synthetic_slides
