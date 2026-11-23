"""
src/labeler.py — Interactive Ground Truth Labeling Tool for Digital Pathology
=============================================================================

Enables fast, visual annotation of histopathology tissue components on WSI slides.
For each detected component on a slide, assigns:
- slice_id : Which microtome cut / physical slice set it belongs to (0, 1, 2, ...)
- tissue_id: Which biopsy core / tissue element it represents ('A', 'B', 'C', ...)

Features:
- Outside Badges: All labels and IDs are placed completely OUTSIDE the tissue boundary
  with clear pointer arrows targeting the component, guaranteeing zero occlusion.
- Reliable CLI / Notebook Loop: Works 100% reliably in any JupyterLab / terminal
  without failing JS widget dependencies (no VBox text dumps).
- One-Key Advance: Simply press [ENTER] to accept autolabeler proposal, save labels.json,
  and automatically display the next slide.
- Clear Feedback: Notifies whether labels.json was newly created or overwritten.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

_logger = logging.getLogger("SliceGrouper.labeler")

# Distinct, high-contrast palette for slice sets
SLICE_COLORS = [
    "#2ecc71",  # emerald green (Slice 0)
    "#3498db",  # peter river blue (Slice 1)
    "#e67e22",  # carrot orange (Slice 2)
    "#9b59b6",  # amethyst purple (Slice 3)
    "#1abc9c",  # turquoise (Slice 4)
    "#e74c3c",  # alizarin red (Slice 5)
    "#f39c12",  # orange (Slice 6)
    "#34495e",  # wet asphalt (Slice 7)
]


class SlideLabelerWidget:
    """
    Interactive tool for fast ground-truth annotation and verification of slides.

    Usage in Jupyter Notebook
    -------------------------
    >>> from src.labeler import SlideLabelerWidget
    >>> labeler = SlideLabelerWidget()

    # Continuous review loop across all slides (press [ENTER] to save & advance):
    >>> labeler.review(0)
    """

    def __init__(
        self,
        slides_root: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        if slides_root is None:
            self.slides_root = Path(__file__).resolve().parent.parent.parent / "backend" / "cv" / "slides"
        else:
            self.slides_root = Path(slides_root)

        if output_dir is None:
            self.output_dir = Path(__file__).resolve().parent.parent / "outputs" / "ground_truth"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.slides = self._discover_slides()
        self.current_idx = 0
        self.current_data: Optional[Dict[str, Any]] = None

        # State for active review flow
        self._active_idx: Optional[int] = None
        self._active_pred: Optional[Dict[int, Tuple[int, str]]] = None

    def _discover_slides(self) -> List[Dict[str, Any]]:
        """Find all valid slide directories."""
        found = []
        if not self.slides_root.exists():
            return found

        for uuid_dir in sorted(self.slides_root.iterdir()):
            if not uuid_dir.is_dir():
                continue
            for mask_path in sorted(uuid_dir.glob("*_mask.tiff")):
                if "_repr_crop_mask" in mask_path.name:
                    continue
                slide_name = mask_path.name.replace("_mask.tiff", "")
                tiff_path = mask_path.parent / f"{slide_name}.tiff"
                if tiff_path.exists():
                    found.append({
                        "sample_id": uuid_dir.name,
                        "slide_name": slide_name,
                        "tiff_path": tiff_path,
                        "mask_path": mask_path,
                        "labels_path": uuid_dir / "labels.json",
                    })
        return found

    def load_slide_data(self, idx: int, min_area_px: int = 6000) -> Dict[str, Any]:
        """
        Load clean TIFF image, mask, and detect components.
        Uses min_area_px=6000 matching the backend application's mask.py
        to filter out dust and noise artifacts.
        """
        from src.io import load_slide

        slide_info = self.slides[idx]
        image, mask, components = load_slide(
            tiff_path=slide_info["tiff_path"],
            mask_path=slide_info["mask_path"],
            min_area_px=min_area_px,
        )

        existing_gt = None
        if slide_info["labels_path"].exists():
            try:
                with open(slide_info["labels_path"], "r", encoding="utf-8") as f:
                    existing_gt = json.load(f)
            except Exception:
                pass

        return {
            "slide_info": slide_info,
            "image": image,
            "mask": mask,
            "components": components,
            "existing_gt": existing_gt,
        }

    def _find_outside_badge_position(
        self,
        comp: Any,
        img_h: int,
        img_w: int,
        all_comps: List[Any],
        offset: int = 50,
    ) -> Tuple[float, float, float, float]:
        """
        Find coordinates (bx, by) completely OUTSIDE the tissue boundary,
        returning (bx, by, tx, ty) where:
        - (bx, by) is the label center on clean background
        - (tx, ty) is the target connection point on the component's bbox edge.
        """
        x, y, w, h = comp.bbox
        cx, cy = comp.centroid

        candidates = [
            (cx, y - offset, cx, y),               # Top: arrow targets top edge
            (cx, y + h + offset, cx, y + h),       # Bottom: arrow targets bottom edge
            (x - offset * 1.5, cy, x, cy),         # Left: arrow targets left edge
            (x + w + offset * 1.5, cy, x + w, cy), # Right: arrow targets right edge
        ]

        def min_dist_to_all_tissues(px: float, py: float) -> float:
            min_d = float("inf")
            for other in all_comps:
                ox, oy, ow, oh = other.bbox
                dx = max(ox - px, 0, px - (ox + ow))
                dy = max(oy - py, 0, py - (oy + oh))
                d = (dx * dx + dy * dy) ** 0.5
                if d < min_d:
                    min_d = d
            return min_d

        best_cand = None
        best_dist = -1.0

        for bx, by, tx, ty in candidates:
            if 15 <= bx <= img_w - 15 and 15 <= by <= img_h - 15:
                d = min_dist_to_all_tissues(bx, by)
                if d > best_dist:
                    best_dist = d
                    best_cand = (bx, by, tx, ty)

        if best_cand is not None and best_dist > 5.0:
            return best_cand

        # Fallback: place above or below with safe margin
        if y >= offset + 15:
            return (cx, max(15, y - offset), cx, y)
        else:
            return (cx, min(img_h - 15, y + h + offset), cx, y + h)

    def plot_slide(
        self,
        idx: int = 0,
        assignments: Optional[Dict[int, Tuple[int, str]]] = None,
        show_labels: bool = True,
        show_bboxes: bool = True,
        show_crops: bool = True,
        label_size: int = 7,
        figsize: Optional[Tuple[int, int]] = None,
        title_suffix: str = "",
        return_fig: bool = False,
    ) -> Optional[plt.Figure]:
        """
        Plot slide with clean, non-obscuring labels AND a dedicated high-resolution
        crop gallery on the right so you see every tissue structure 100% clearly.

        Parameters
        ----------
        idx : int
            Slide index.
        assignments : Optional[Dict[int, Tuple[int, str]]]
            Mapping {comp_id: (slice_id, tissue_core)} to color-code components.
        show_labels : bool
            Whether to draw labels and pointer arrows on overview.
        show_bboxes : bool
            Whether to draw bounding box outlines.
        show_crops : bool
            If True, displays zoomed-in, 100% clean tissue crops alongside the slide.
        label_size : int
            Font size for badges.
        figsize : Optional[Tuple[int, int]]
            Figure dimensions.
        title_suffix : str
            Extra text for figure title.
        return_fig : bool
            If False (default), returns None so Jupyter displays the plot exactly once.
        """
        data = self.load_slide_data(idx)
        img = data["image"]
        comps = data["components"]
        slide_name = self.slides[idx]["slide_name"]

        # Fallback to existing GT if no assignments passed
        if assignments is None and data["existing_gt"]:
            assignments = {}
            for c in data["existing_gt"].get("components", []):
                assignments[c["id"]] = (c.get("slice_id", 0), c.get("tissue_id", "A"))

        n_comps = len(comps)
        img_h, img_w = img.shape[:2]

        if show_crops and n_comps > 0:
            # Multi-panel: Overview on left (60% width), clean crops gallery on right (40% width)
            calc_w = 16
            calc_h = max(7, min(14, 2.5 * n_comps))
            fig = plt.figure(figsize=figsize or (calc_w, calc_h))

            import matplotlib.gridspec as gridspec
            gs = gridspec.GridSpec(max(1, n_comps), 2, width_ratios=[1.2, 1.0], figure=fig)
            ax_main = fig.add_subplot(gs[:, 0])

            # Draw overview on ax_main
            ax_main.imshow(img)
            for c in comps:
                cid = c.component_id
                x, y, w, h = c.bbox
                if assignments and cid in assignments:
                    s_id, t_id = assignments[cid]
                    color = SLICE_COLORS[s_id % len(SLICE_COLORS)]
                    badge_text = f"#{cid} [S{s_id}:{t_id}]"
                else:
                    color = "#3498db"
                    badge_text = f"#{cid}"

                if show_bboxes:
                    rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=1.5, alpha=0.9)
                    ax_main.add_patch(rect)

                if show_labels:
                    bx, by, tx, ty = self._find_outside_badge_position(c, img_h, img_w, comps)
                    ax_main.annotate(
                        badge_text,
                        xy=(tx, ty),
                        xytext=(bx, by),
                        color="white",
                        fontsize=label_size,
                        fontweight="bold",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=color, edgecolor="white", linewidth=0.8, alpha=0.9),
                        arrowprops=dict(arrowstyle="->,head_width=0.2,head_length=0.3", color=color, lw=1.2, shrinkA=2, shrinkB=2),
                    )

                # Small unobtrusive ID in the top-left corner above bbox
                tag_y = max(12, y - 6)
                ax_main.text(x, tag_y, f"#{cid}", color=color, fontsize=8, fontweight="bold", ha="left", va="bottom")

            status_text = "GT: exists ✅" if data["existing_gt"] else "GT: not saved yet ⚠️"
            title = f"Slide [{idx}]: {slide_name} ({n_comps} tkanek) | {status_text}"
            if title_suffix:
                title += f" | {title_suffix}"
            ax_main.set_title(title, fontsize=11, fontweight="bold")
            ax_main.axis("off")

            # Draw clean zoomed-in crop for each component in right column
            for i, c in enumerate(comps):
                ax_crop = fig.add_subplot(gs[i, 1])
                cid = c.component_id
                if assignments and cid in assignments:
                    s_id, t_id = assignments[cid]
                    c_color = SLICE_COLORS[s_id % len(SLICE_COLORS)]
                    desc = f"Comp #{cid} ➔ Slice {s_id}, Tkanka {t_id}"
                else:
                    c_color = "#3498db"
                    desc = f"Comp #{cid}"

                # Show 100% clean crop: NO TEXT ON TOP OF TISSUE AT ALL!
                crop_img = c.rgb_crop if c.rgb_crop is not None else img[c.bbox[1]:c.bbox[1]+c.bbox[3], c.bbox[0]:c.bbox[0]+c.bbox[2]]
                ax_crop.imshow(crop_img)
                ax_crop.set_title(f"{desc} (Powierzchnia: {c.area:,} px)", fontsize=9, fontweight="bold", color=c_color, pad=4)

                # Colored border around crop indicating slice assignment
                for spine in ax_crop.spines.values():
                    spine.set_color(c_color)
                    spine.set_linewidth(2.5)
                ax_crop.set_xticks([])
                ax_crop.set_yticks([])

        else:
            # Single panel fallback
            fig, ax_main = plt.subplots(figsize=figsize or (12, 6))
            ax_main.imshow(img)
            for c in comps:
                cid = c.component_id
                x, y, w, h = c.bbox
                if assignments and cid in assignments:
                    s_id, t_id = assignments[cid]
                    color = SLICE_COLORS[s_id % len(SLICE_COLORS)]
                    badge_text = f"#{cid} [S{s_id}:{t_id}]"
                else:
                    color = "#3498db"
                    badge_text = f"#{cid}"

                if show_bboxes:
                    rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=1.5, alpha=0.9)
                    ax_main.add_patch(rect)

                tag_y = max(12, y - 6)
                ax_main.text(x, tag_y, badge_text, color=color, fontsize=8, fontweight="bold", ha="left", va="bottom")

            ax_main.set_title(f"Slide [{idx}]: {slide_name} | {n_comps} components", fontsize=11, fontweight="bold")
            ax_main.axis("off")

        plt.tight_layout()
        plt.show()

        if return_fig:
            return fig
        return None


    def predict(self, idx: int) -> Dict[int, Tuple[int, str]]:
        """
        Run the constellation-based autolabeler on slide `idx` and return
        suggested assignments {comp_id: (slice_id, tissue_core)} without saving.
        """
        data = self.load_slide_data(idx)
        comps = data["components"]
        if not comps:
            return {}
        if len(comps) == 1:
            return {comps[0].component_id: (0, "A")}

        from src.color_features import extract_color_features_batch
        from src.clustering import cluster_components
        from src.slice_matching import ConstellationSliceMatcher

        feats = extract_color_features_batch(comps)
        n_est_tissues = min(3, max(1, len(comps) // 2))
        clust_res = cluster_components(feats, method="agglomerative", n_clusters=n_est_tissues)
        matcher = ConstellationSliceMatcher()
        set_res = matcher.match(comps, clust_res.labels)

        tissue_letters = ["A", "B", "C", "D", "E", "F"]
        assignments = {}
        for i, c in enumerate(comps):
            pred_s = int(set_res.slice_labels[i])
            pred_t = int(set_res.tissue_labels[i])
            t_name = tissue_letters[pred_t] if pred_t < len(tissue_letters) else "A"
            assignments[c.component_id] = (max(0, pred_s), t_name)

        return assignments

    def _save_with_feedback(
        self,
        idx: int,
        assignments: Dict[int, Tuple[int, str]],
    ) -> Path:
        """Internal helper to save ground truth with rich feedback on overwrite/creation."""
        slide_info = self.slides[idx]
        data = self.load_slide_data(idx)

        components_gt = []
        for c in data["components"]:
            cid = c.component_id
            if cid in assignments:
                s_id, t_id = assignments[cid]
            else:
                s_id, t_id = 0, "A"
            components_gt.append({
                "id": cid,
                "slice_id": int(s_id),
                "tissue_id": str(t_id),
            })

        gt_payload = {
            "slide_name": slide_info["slide_name"],
            "sample_id": slide_info["sample_id"],
            "n_components": len(components_gt),
            "components": components_gt,
        }

        target_path = slide_info["labels_path"]
        already_existed = target_path.exists()

        # Save to slide folder
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(gt_payload, f, indent=2)

        # Save backup to outputs
        backup_path = self.output_dir / f"{slide_info['slide_name']}_labels.json"
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(gt_payload, f, indent=2)

        file_size = target_path.stat().st_size
        n_slices = len(set(c["slice_id"] for c in components_gt))

        if already_existed:
            print(f"⚠️  Plik labels.json już istniał — pomyślnie NADPISANO etykiety dla slajdu '{slide_info['slide_name']}'!")
        else:
            print(f"✅ Utworzono NOWY plik labels.json dla slajdu '{slide_info['slide_name']}'!")

        print(f"   📁 Zapisano do: {target_path}")
        print(f"   📊 Szczegóły: {len(components_gt)} komponentów w {n_slices} zestawach slice ({file_size} bajtów)")
        return target_path

    def quick_label(
        self,
        idx: int,
        assignments: Dict[int, Tuple[int, str]],
    ) -> Path:
        """Save ground truth directly via python dict {comp_id: (slice_id, tissue_core)}."""
        return self._save_with_feedback(idx, assignments)

    def auto_label(self, idx: int) -> Path:
        """Run auto-suggest algorithm and save ground truth directly."""
        assignments = self.predict(idx)
        return self._save_with_feedback(idx, assignments)

    def accept(self) -> Optional[Path]:
        """Accept and save the current autolabeler proposal."""
        if self._active_idx is None or self._active_pred is None:
            print("⚠️ Brak aktywnej propozycji do zapisania. Uruchom najpierw: labeler.review(idx)")
            return None
        return self._save_with_feedback(self._active_idx, self._active_pred)

    def edit(self, assignments: Dict[int, Tuple[int, str]]) -> Optional[Path]:
        """Apply manual assignments, show updated preview, and save."""
        if self._active_idx is None:
            print("⚠️ Brak aktywnego slajdu. Uruchom najpierw: labeler.review(idx)")
            return None
        self._active_pred = assignments
        print("🔄 Zaktualizowano grupy ręcznie. Nowy podgląd:")
        self.plot_slide(self._active_idx, assignments=assignments, label_size=8)
        return self._save_with_feedback(self._active_idx, assignments)

    def _parse_manual_input(
        self,
        text: str,
        comps: List[Any],
        base_assignments: Optional[Dict[int, Tuple[int, str]]] = None,
    ) -> Optional[Dict[int, Tuple[int, str]]]:
        """
        Robust regex-based parser for user manual inputs.
        Supports:
        - Full or partial list: '0:(0,A), 1:(0,B), 2:(1,A)' (with or without spaces!)
        - Delta updates: '4:(1, B)' or '4: 1 B' or '4: B'
        - Slice transfer: 's1: 4, 11'
        - Full specifications: 's0: 0, 1; s1: 2, 3'
        - Python dict: '{0: (0, "A"), 1: (0, "B")}'
        """
        import re
        text = text.strip()
        if not text:
            return None

        # Start from base assignments if provided (delta mode)
        res = dict(base_assignments) if base_assignments else {}

        # 1. Full assignment syntax: 's0: 0, 1; s1: 2, 3'
        if "s" in text.lower() and ";" in text:
            full_mode_res = {}
            tissue_letters = ["A", "B", "C", "D", "E", "F"]
            parts = text.split(";")
            for part in parts:
                if ":" not in part:
                    continue
                s_part, comps_part = part.split(":", 1)
                s_id = int("".join(filter(str.isdigit, s_part)) or 0)
                cid_list = [int(v.strip().replace("#", "")) for v in comps_part.split(",") if v.strip()]
                for i, cid in enumerate(cid_list):
                    full_mode_res[cid] = (s_id, tissue_letters[i % len(tissue_letters)])
            if full_mode_res:
                res.update(full_mode_res)
                return res

        # 2. Single slice group assignment: 's1: 4, 11' (e.g. move these to Slice 1)
        if text.lower().startswith("s") and ":" in text and ";" not in text:
            s_part, comps_part = text.split(":", 1)
            s_id = int("".join(filter(str.isdigit, s_part)) or 0)
            cid_list = [int(v.strip().replace("#", "")) for v in comps_part.split(",") if v.strip()]
            tissue_letters = ["A", "B", "C", "D", "E", "F"]
            for i, cid in enumerate(cid_list):
                cur_t = res.get(cid, (0, tissue_letters[i % len(tissue_letters)]))[1]
                res[cid] = (s_id, cur_t)
            return res

        # 3. Python dict: {0: (0, 'A'), ...}
        if text.startswith("{") and text.endswith("}"):
            try:
                import ast
                parsed = ast.literal_eval(text)
                if isinstance(parsed, dict):
                    res.update({int(k): (int(v[0]), str(v[1]).upper()) for k, v in parsed.items()})
                    return res
            except Exception:
                pass

        # 4. Regex parsing for 'cid:(slice_id, tissue_id)' or 'cid: slice_id tissue_id'
        # Handles any spacing, quotes, commas or brackets: e.g. "0:(0,A)", "0:(0, A)", "4:(1,B)", "4: 1 B"
        matches_full = re.findall(r"(\d+)\s*:\s*\(?\s*(\d+)\s*[\s,:]+\s*([A-Za-z]+)\s*\)?", text)
        if matches_full:
            for cid_str, sid_str, tid_str in matches_full:
                res[int(cid_str)] = (int(sid_str), tid_str.upper())
            return res

        # 5. Regex for changing only tissue letter: e.g. '4: B' or '4:B'
        matches_tissue = re.findall(r"(\d+)\s*:\s*\(?\s*([A-Za-z]+)\s*\)?", text)
        if matches_tissue:
            for cid_str, tid_str in matches_tissue:
                cid = int(cid_str)
                cur_s = res.get(cid, (0, "A"))[0]
                res[cid] = (cur_s, tid_str.upper())
            return res

        # 6. Regex for changing only slice id: e.g. '4: 1' or '4:(1)'
        matches_slice = re.findall(r"(\d+)\s*:\s*\(?\s*(\d+)\s*\)?", text)
        if matches_slice:
            for cid_str, sid_str in matches_slice:
                cid = int(cid_str)
                cur_t = res.get(cid, (0, "A"))[1]
                res[cid] = (int(sid_str), cur_t)
            return res

        return None


    def review(
        self,
        idx: int = 0,
        skip_labeled: bool = False,
    ):
        """
        Interactive continuous review workflow using standard notebook input.
        Works 100% reliably in any JupyterLab/terminal environment without widget dependencies.

        Controls:
        - Press [ENTER]  : Accept autolabeler proposal, save labels.json, and advance to next slide!
        - Type 'f'       : Mark whole slide as single uncut piece (Slice 0 for all).
        - Type 'z'       : Change groupings manually (typing only deltas/changes!).
        - Type 'p'       : Skip this slide without saving.
        - Type 'q'       : Stop review loop.
        """
        current_idx = idx
        n_total = len(self.slides)

        while current_idx < n_total:
            slide_info = self.slides[current_idx]
            if skip_labeled and slide_info["labels_path"].exists():
                current_idx += 1
                continue

            self.current_idx = current_idx
            self._active_idx = current_idx

            data = self.load_slide_data(current_idx)
            comps = data["components"]

            # Run prediction
            pred = self.predict(current_idx)
            self._active_pred = pred

            try:
                from IPython.display import clear_output
                clear_output(wait=True)
            except Exception:
                pass

            n_labeled = sum(1 for s in self.slides if s["labels_path"].exists())
            pct = int((n_labeled / n_total) * 100) if n_total > 0 else 0

            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"📌 Slajd [{current_idx + 1} / {n_total}] ({pct}% oznaczonych) | {slide_info['slide_name']}")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            self.plot_slide(
                idx=current_idx,
                assignments=pred,
                label_size=8,
                title_suffix="PROPOZYCJA AUTOLABELERA",
            )

            # Summary of groups
            by_slice: Dict[int, List[str]] = {}
            for cid, (s_id, t_id) in sorted(pred.items()):
                by_slice.setdefault(s_id, []).append(f"Comp #{cid} ({t_id})")

            # Ready-to-copy formatted string: 0:(0,A), 1:(0,B), 2:(1,A)...
            copyable_str = ", ".join(f"{cid}:({s_id},{t_id})" for cid, (s_id, t_id) in sorted(pred.items()))

            print("📋 Wykryte zestawy cięć (Slice Sets):")
            for s_id, items in sorted(by_slice.items()):
                print(f"   • Slice {s_id}: " + ", ".join(items))

            print("\n✂️  Gotowy tekst do skopiowania i edycji (Ctrl+C ➔ Ctrl+V):")
            print(f"   {copyable_str}")

            print("-" * 76)
            prompt = "👉 [ENTER] = Zapisz sugestię  |  [f] = Całość (nie tnij)  |  [z] = Zmień  |  [p] = Pomiń  |  [q] = Wyjdź: "
            try:
                choice = input(prompt).strip()
            except (KeyboardInterrupt, EOFError):
                print("\n⏹️ Zakończono pętlę.")
                break

            if choice.lower() == "q":
                print("⏹️ Zakończono oznaczanie. Możesz wznowić w dowolnej chwili: labeler.review()")
                break

            elif choice.lower() == "p":
                print(f"⏭️ Pominięto slajd [{current_idx + 1}]. Ładuję kolejny...")
                current_idx += 1
                continue

            elif choice.lower() == "f":
                # Fallback: assign all components to Slice 0 (full slide, do not slice)
                tissue_letters = ["A", "B", "C", "D", "E", "F"]
                fallback_assignments = {
                    c.component_id: (0, tissue_letters[i % len(tissue_letters)])
                    for i, c in enumerate(comps)
                }
                print(f"🛡️ Oznaczono cały slajd jako jeden nienacinany preparat (Slice 0 dla wszystkich {len(comps)} tkanek).")
                self._save_with_feedback(current_idx, fallback_assignments)
                current_idx += 1
                continue

            elif choice.lower() == "z":
                # Manual editing mode
                print(f"\n✏️ Wklej skopiowany tekst i popraw błędy (lub wpisz tylko zmiany, np. '4:(1,B)'):")
                print(f"   Sugestia: {copyable_str}")
                print("   (Wciśnij [ENTER], aby anulować zmianę i wrócić)")
                edit_input = input("Twoje przypisania: ").strip()
                if not edit_input:
                    continue
                parsed = self._parse_manual_input(edit_input, comps, base_assignments=pred)
                if parsed is None:
                    print("⚠️ Nie udało się sparsować formatu. Spróbuj np.: '4:(1,B)' lub wklej skopiowany tekst.")

                    input("Wciśnij [ENTER], aby kontynuować...")
                    continue

                # Show updated preview
                print("\n🔄 Zaktualizowane przypisania komponentów:")
                for cid, (s_id, t_id) in sorted(parsed.items()):
                    print(f"   • Comp #{cid} ➔ Slice {s_id}, Tkanka {t_id}")

                self.plot_slide(current_idx, assignments=parsed, label_size=8, title_suffix="TWOJE POPRAWKI")
                confirm = input("Zapisać te grupy i przejść dalej? [ENTER = Tak, n = Anuluj]: ").strip()
                if confirm.lower() != "n":
                    self._save_with_feedback(current_idx, parsed)
                    current_idx += 1
                continue

            else:
                # Default: Accept autolabeler suggestion!
                self._save_with_feedback(current_idx, pred)
                current_idx += 1


        if current_idx >= n_total:
            print("\n🎉 Gratulacje! Wszystkie slajdy w dataset zostały przejrzane!")
            print("Możesz teraz przejść do Sekcji 7, aby uruchomić pełny Benchmark (ARI, NMI).")

    def start_loop(self, start_idx: int = 0, skip_labeled: bool = False):
        """Convenient alias to start the sequential labeling workflow."""
        self.review(idx=start_idx, skip_labeled=skip_labeled)
