"""
update_gallery_with_indices.py
==============================
Updates error_gallery.html to include the exact `idx` and `labeler.review(idx)`
command for every error slide, making it trivial to review and edit in the notebook.
"""

from pathlib import Path
import json
import pandas as pd
from src.labeler import SlideLabelerWidget

PROJECT_ROOT = Path(__file__).resolve().parent
labeler = SlideLabelerWidget()
name_to_idx = {s["slide_name"]: i for i, s in enumerate(labeler.slides)}

records_path = PROJECT_ROOT / "outputs" / "benchmarks" / "benchmark_records.csv"
df = pd.read_csv(records_path)
hybrid_df = df[df["Method"].str.contains("Hybrid", na=False)]
error_df = hybrid_df[hybrid_df["Slice ARI"] < 1.0].sort_values(by="Slice ARI")

error_dir = PROJECT_ROOT / "outputs" / "figures" / "errors"
html_path = error_dir / "error_gallery.html"

# Load existing files in error directory
png_files = sorted(list(error_dir.glob("err_*.png")))

gallery_items = []
for p in png_files:
    # Example filename: err_01_99708_Mallory_ARI_0.00.png
    parts = p.stem.split("_")
    rank = parts[1]
    ari_str = parts[-1]
    slide_slug = "_".join(parts[2:-2])
    
    # Match to slide name
    matched_slide = None
    for s_name in name_to_idx:
        slug = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in s_name])
        if slug == slide_slug or s_name.replace(" ", "_") == slide_slug:
            matched_slide = s_name
            break
            
    if not matched_slide:
        # Fallback substring search
        for s_name in name_to_idx:
            if s_name.split()[0] in slide_slug:
                matched_slide = s_name
                break

    idx = name_to_idx.get(matched_slide, "?")
    gallery_items.append({
        "rank": rank,
        "filename": p.name,
        "slide_name": matched_slide if matched_slide else slide_slug,
        "labeler_idx": idx,
        "ari": ari_str,
    })

# Write rich HTML gallery with copyable review commands
with open(html_path, "w", encoding="utf-8") as f:
    f.write("""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SliceGrouper Model Error Gallery with Labeler Indices</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #121212; color: #e0e0e0; margin: 0; padding: 24px; }
h1 { color: #fff; border-bottom: 2px solid #333; padding-bottom: 12px; margin-bottom: 16px; }
.card { background: #1e1e1e; border: 1px solid #333; border-radius: 10px; margin-bottom: 35px; padding: 20px; box-shadow: 0 4px 16px rgba(0,0,0,0.6); }
.header-row { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; margin-bottom: 12px; gap: 10px; }
.slide-title { font-size: 20px; font-weight: bold; color: #fff; margin: 0; }
.cmd-box { background: #2c3e50; border: 1px solid #3498db; border-radius: 6px; padding: 6px 14px; font-family: monospace; font-size: 15px; color: #ecf0f1; font-weight: bold; }
.badge { display: inline-block; padding: 4px 12px; border-radius: 4px; font-weight: bold; font-size: 14px; }
.badge-idx { background: #8e44ad; color: white; }
.badge-ari { background: #e74c3c; color: white; }
img { width: 100%; border-radius: 8px; display: block; margin-top: 10px; border: 1px solid #2a2a2a; }
</style>
</head>
<body>
<h1>🔬 SliceGrouper — Galeria Błędów z Indeksami do Notebooka</h1>
<p style="font-size: 15px; color: #bbb;">
Każda karta poniżej zawiera dokładny indeks <code>idx</code> oraz gotową komendę <code>labeler.review(idx)</code>. 
Skopiuj komendę i wklej w notebooku, aby od razu edytować i zapisać poprawiony Ground Truth.
</p>
""")
    for item in gallery_items:
        f.write(f"""
<div class="card">
  <div class="header-row">
    <div class="slide-title">#{item['rank']} Slajd: {item['slide_name']}</div>
    <div>
      <span class="badge badge-idx">Indeks: {item['labeler_idx']}</span>
      <span class="badge badge-ari">ARI: {item['ari']}</span>
      <span class="cmd-box">labeler.review({item['labeler_idx']})</span>
    </div>
  </div>
  <img src="{item['filename']}" alt="{item['slide_name']}">
</div>
""")
    f.write("</body></html>")

print(f"Updated HTML gallery saved with labeler indices: {html_path}")

# Print list for console/user
print("\n" + "="*80)
print(f"{'IDX':<5} | {'SLIDE NAME':<25} | {'ARI':<6} | {'KOMENDA DO NOTEBOOKA':<25}")
print("="*80)
for item in gallery_items:
    cmd = f"labeler.review({item['labeler_idx']})"
    print(f"{str(item['labeler_idx']):<5} | {item['slide_name']:<25} | {item['ari']:<6} | {cmd:<25}")
print("="*80)
