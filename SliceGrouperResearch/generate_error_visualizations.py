"""
generate_error_visualizations.py
=================================
Generates side-by-side Ground Truth vs Model Prediction comparison images
for the restored original architecture, clearly distinguishing:
- FROZEN TEST SET (22 slides)
- TRAIN SET (86 slides)
Includes labeler indices and copyable labeler.review(idx) commands.
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.dataset_split import load_dataset_split
from src.io import ComponentData, load_slide
from src.labeler import SlideLabelerWidget
from src.method_benchmarker import HybridSafetySpatialMethod

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "errors"

# Clean old error PNGs
if OUTPUT_DIR.exists():
    for f in OUTPUT_DIR.glob("*.png"):
        try:
            f.unlink()
        except Exception:
            pass
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

labeler = SlideLabelerWidget()
name_to_idx = {s["slide_name"]: i for i, s in enumerate(labeler.slides)}
slide_map = {s["slide_name"]: s for s in labeler.slides if s["labels_path"].exists()}

train_names, test_names = load_dataset_split()
train_set = set(train_names)
test_set = set(test_names)

model_path = PROJECT_ROOT / "outputs" / "models" / "fallback_classifier.joblib"
hybrid_model = HybridSafetySpatialMethod(fallback_model_path=model_path)
cmap_slices = plt.get_cmap("tab10")

# Load records from train and test benchmarks
records_files = [
    ("TEST", PROJECT_ROOT / "outputs" / "benchmarks" / "test_benchmark_records.csv"),
    ("TRAIN", PROJECT_ROOT / "outputs" / "benchmarks" / "train_benchmark_records.csv"),
]

all_error_cases = []

for split_type, csv_path in records_files:
    if not csv_path.exists():
        continue
    df = pd.read_csv(csv_path)
    hybrid_df = df[df["Method"].str.contains("Hybrid", na=False)]
    fails = hybrid_df[hybrid_df["Slice ARI"] < 1.0].sort_values(by="Slice ARI")
    
    for _, row in fails.iterrows():
        s_name = row["Slide"]
        all_error_cases.append({
            "split": split_type,
            "slide_name": s_name,
            "ari": float(row["Slice ARI"]),
            "true_slices": int(row["True Slices"]),
            "pred_slices": int(row["Pred Slices"]),
            "components": int(row["Components"]),
            "is_fallback": "Yes" in str(row.get("Fallback?", "")),
            "labeler_idx": name_to_idx.get(s_name, "?"),
        })

print(f"Total error cases across all benchmarks: {len(all_error_cases)}")
test_err_count = sum(1 for c in all_error_cases if c["split"] == "TEST")
train_err_count = sum(1 for c in all_error_cases if c["split"] == "TRAIN")
print(f" - TEST Set errors: {test_err_count}")
print(f" - TRAIN Set errors: {train_err_count}")

gallery_cards = []

for rank, item in enumerate(all_error_cases, 1):
    s_name = item["slide_name"]
    s_info = slide_map.get(s_name)
    if not s_info:
        continue
        
    try:
        image, mask, components = load_slide(
            tiff_path=s_info["tiff_path"],
            mask_path=s_info["mask_path"],
            min_area_px=6000,
        )
        with open(s_info["labels_path"], "r", encoding="utf-8") as f:
            gt_data = json.load(f)
            
        gt_comps = gt_data.get("components", [])
        gt_by_id = {c["id"]: c for c in gt_comps}
        gt_slices = np.array([gt_by_id.get(k, {}).get("slice_id", 0) for k in range(len(components))])
        gt_tissues = np.array([gt_by_id.get(k, {}).get("tissue_id", "A") for k in range(len(components))])

        pred_slices, pred_tissues, conf, info = hybrid_model.predict(image, components)
        ari = float(adjusted_rand_score(gt_slices, pred_slices))
    except Exception as e:
        print(f"Error on {s_name}: {e}")
        continue

    split_tag = item["split"]
    idx = item["labeler_idx"]
    is_fallback = info.get("is_fallback", False)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), facecolor="#1e1e1e")

    # Left: Ground Truth
    axes[0].imshow(image)
    axes[0].set_title(
        f"[{split_tag} SET] GROUND TRUTH — #{rank} (idx: {idx}) {s_name}\n"
        f"True Slices: {len(np.unique(gt_slices))} | Components: {len(components)}",
        color="white", fontsize=13, fontweight="bold", pad=12
    )
    axes[0].axis("off")

    # Right: Prediction
    axes[1].imshow(image)
    mode_text = "SAFE FALLBACK (NO SPLIT)" if is_fallback else "SPATIAL GROUPING"
    axes[1].set_title(
        f"MODEL PREDICTION — ARI: {ari:.3f}\n"
        f"Pred Slices: {len(np.unique(pred_slices))} | Mode: {mode_text}",
        color="#2ecc71" if ari == 1.0 else "#e74c3c", fontsize=13, fontweight="bold", pad=12
    )
    axes[1].axis("off")

    for c_i, comp in enumerate(components):
        bx, by, bw, bh = comp.bbox
        cx, cy = comp.centroid

        # Ground Truth
        gt_s = gt_slices[c_i] if c_i < len(gt_slices) else 0
        gt_t = gt_tissues[c_i] if c_i < len(gt_tissues) else "A"
        gt_color = cmap_slices(gt_s % 10)

        rect_gt = plt.Rectangle((bx, by), bw, bh, fill=False, edgecolor=gt_color, linewidth=2.5)
        axes[0].add_patch(rect_gt)
        gt_badge = chr(65 + (int(gt_t) % 26)) if str(gt_t).isdigit() else str(gt_t)
        axes[0].text(
            cx, cy, f"S{gt_s}:C{gt_badge}",
            color="white", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=gt_color, edgecolor="black", alpha=0.85),
            ha="center", va="center"
        )

        # Prediction
        pr_s = pred_slices[c_i] if c_i < len(pred_slices) else 0
        pr_color = cmap_slices(pr_s % 10)

        rect_pr = plt.Rectangle((bx, by), bw, bh, fill=False, edgecolor=pr_color, linewidth=2.5)
        axes[1].add_patch(rect_pr)
        axes[1].text(
            cx, cy, f"Slice {pr_s}",
            color="white", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=pr_color, edgecolor="black", alpha=0.85),
            ha="center", va="center"
        )

    plt.tight_layout()
    safe_name = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in s_name])
    filename = f"err_{rank:02d}_{split_tag}_idx{str(idx).zfill(3)}_{safe_name}_ARI_{ari:.2f}.png"
    save_path = OUTPUT_DIR / filename
    fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    gallery_cards.append({
        "rank": rank,
        "split": split_tag,
        "idx": idx,
        "slide_name": s_name,
        "ari": ari,
        "filename": filename,
        "true_slices": len(np.unique(gt_slices)),
        "pred_slices": len(np.unique(pred_slices)),
        "components": len(components),
        "is_fallback": is_fallback,
    })
    print(f"[{rank}/{len(all_error_cases)}] Saved: {filename}")

# Generate modern HTML gallery
html_path = OUTPUT_DIR / "error_gallery.html"
with open(html_path, "w", encoding="utf-8") as f:
    f.write(f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SliceGrouper Error Gallery (Original Model)</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #121212; color: #e0e0e0; margin: 0; padding: 24px; }}
h1 {{ color: #fff; border-bottom: 2px solid #333; padding-bottom: 12px; margin-bottom: 8px; }}
.subtitle {{ font-size: 15px; color: #bbb; margin-bottom: 24px; }}
.card {{ background: #1e1e1e; border: 1px solid #333; border-radius: 10px; margin-bottom: 35px; padding: 20px; box-shadow: 0 4px 16px rgba(0,0,0,0.6); }}
.header-row {{ display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; margin-bottom: 12px; gap: 10px; }}
.slide-title {{ font-size: 20px; font-weight: bold; color: #fff; margin: 0; }}
.cmd-box {{ background: #2c3e50; border: 1px solid #3498db; border-radius: 6px; padding: 6px 14px; font-family: monospace; font-size: 15px; color: #ecf0f1; font-weight: bold; }}
.badge {{ display: inline-block; padding: 4px 12px; border-radius: 4px; font-weight: bold; font-size: 14px; margin-right: 6px; }}
.badge-test {{ background: #e67e22; color: white; }}
.badge-train {{ background: #34495e; color: white; }}
.badge-idx {{ background: #8e44ad; color: white; }}
.badge-ari {{ background: #e74c3c; color: white; }}
.badge-safe {{ background: #16a085; color: white; }}
img {{ width: 100%; border-radius: 8px; display: block; margin-top: 10px; border: 1px solid #2a2a2a; }}
</style>
</head>
<body>
<h1>🔬 SliceGrouper — Galeria Błędów Oryginalnego Modelu</h1>
<div class="subtitle">
  <b>Zbiór TESTOWY (22 slajdy zamrożone):</b> tylko {test_err_count} błędy (ARI = 0.909, Perfect = 90.9%)<br>
  <b>Zbiór TRENINGOWY (86 slajdów):</b> {train_err_count} błędów (ARI = 0.855, Perfect = 84.9%)<br>
  Poniżej znajduje się podgląd wszystkich przypadków z gotowymi indeksami do <code>labeler.review(idx)</code>.
</div>
""")
    for item in gallery_cards:
        split_cls = "badge-test" if item["split"] == "TEST" else "badge-train"
        fb_badge = '<span class="badge badge-safe">🛡️ Safe Fallback</span>' if item["is_fallback"] else ''
        f.write(f"""
<div class="card">
  <div class="header-row">
    <div class="slide-title">
      <span class="badge {split_cls}">[{item['split']}]</span>
      #{item['rank']} Slajd: {item['slide_name']}
    </div>
    <div>
      <span class="badge badge-idx">Indeks: {item['idx']}</span>
      <span class="badge badge-ari">ARI: {item['ari']:.2f}</span>
      {fb_badge}
      <span class="cmd-box">labeler.review({item['idx']})</span>
    </div>
  </div>
  <div style="font-size:14px; color:#aaa; margin-bottom:8px;">
    Komponenty: {item['components']} | Prawdziwe Slices: {item['true_slices']} | Predykcja: {item['pred_slices']}
  </div>
  <img src="{item['filename']}" alt="{item['slide_name']}">
</div>
""")
    f.write("</body></html>")

print(f"\nDone! Updated gallery saved to: {html_path}")

print("\n" + "="*85)
print(f"{'SET':<6} | {'IDX':<5} | {'SLIDE NAME':<25} | {'ARI':<6} | {'KOMENDA DO REVIEW':<25}")
print("="*85)
for item in gallery_cards:
    cmd = f"labeler.review({item['idx']})"
    print(f"{item['split']:<6} | {item['idx']:<5} | {item['slide_name']:<25} | {item['ari']:<6.2f} | {cmd:<25}")
print("="*85)
