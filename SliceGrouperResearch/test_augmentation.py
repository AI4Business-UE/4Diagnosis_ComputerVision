"""
test_augmentation.py — Verifies Augmentation Module and Generates Demo Plot
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.augmentation import SerialSectionAugmentor
from src.labeler import SlideLabelerWidget

labeler = SlideLabelerWidget()
labeled_slides = [s for s in labeler.slides if s["labels_path"].exists()]

augmentor = SerialSectionAugmentor(random_state=42)

# Pick first verified train slide
train_candidates = [s for s in labeled_slides if augmentor.is_train_slide(s["slide_name"])]
print(f"Verified train candidates: {len(train_candidates)}")
base_slide = train_candidates[0]
print(f"Using base slide: {base_slide['slide_name']}")

# Load base slide
from src.io import load_slide
image, mask, comps = load_slide(
    tiff_path=base_slide["tiff_path"],
    mask_path=base_slide["mask_path"],
    min_area_px=6000,
)

import json
with open(base_slide["labels_path"], "r", encoding="utf-8") as f:
    gt_data = json.load(f)
gt_by_id = {c["id"]: c for c in gt_data.get("components", [])}
s_labels = np.array([gt_by_id.get(k, {}).get("slice_id", 0) for k in range(len(comps))])
t_labels = np.array([gt_by_id.get(k, {}).get("tissue_id", "A") for k in range(len(comps))])

# Generate 3-slice and 4-slice synthetic slides
syn3 = augmentor.augment_to_multi_slice(image, mask, comps, s_labels, t_labels, target_n_slices=3)
syn4 = augmentor.augment_to_multi_slice(image, mask, comps, s_labels, t_labels, target_n_slices=4)

print(f"Synthetic 3-slice components: {len(syn3.components)}, slices: {len(np.unique(syn3.slice_labels))}")
print(f"Synthetic 4-slice components: {len(syn4.components)}, slices: {len(np.unique(syn4.slice_labels))}")

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 8), facecolor="#1e1e1e")
cmap = plt.get_cmap("tab10")

# 1. Base slide
axes[0].imshow(image)
axes[0].set_title(f"Original Train Slide\n({base_slide['slide_name']})", color="white", fontsize=12, fontweight="bold")
axes[0].axis("off")

# 2. Synthetic 3-slice
axes[1].imshow(syn3.image)
for c, s in zip(syn3.components, syn3.slice_labels):
    bx, by, bw, bh = c.bbox
    rect = plt.Rectangle((bx, by), bw, bh, fill=False, edgecolor=cmap(s % 10), linewidth=2)
    axes[1].add_patch(rect)
    axes[1].text(c.centroid[0], c.centroid[1], f"S{s}", color="white", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor=cmap(s % 10), alpha=0.8),
                 ha="center", va="center")
axes[1].set_title(f"Augmented: 3 Serial Slices\n({len(syn3.components)} Cores)", color="#2ecc71", fontsize=12, fontweight="bold")
axes[1].axis("off")

# 3. Synthetic 4-slice
axes[2].imshow(syn4.image)
for c, s in zip(syn4.components, syn4.slice_labels):
    bx, by, bw, bh = c.bbox
    rect = plt.Rectangle((bx, by), bw, bh, fill=False, edgecolor=cmap(s % 10), linewidth=2)
    axes[2].add_patch(rect)
    axes[2].text(c.centroid[0], c.centroid[1], f"S{s}", color="white", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor=cmap(s % 10), alpha=0.8),
                 ha="center", va="center")
axes[2].set_title(f"Augmented: 4 Serial Slices\n({len(syn4.components)} Cores)", color="#3498db", fontsize=12, fontweight="bold")
axes[2].axis("off")

plt.tight_layout()
out_fig = Path("outputs/figures/synthetic_augmented_example.png")
fig.savefig(out_fig, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)

print(f"Saved visualization to: {out_fig}")
