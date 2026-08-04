# SliceGrouperResearch

A **complete research environment** for experimenting with grouping histopathological tissue fragments
that originate from the same specimen.

This environment is an R&D sandbox — it is not a production algorithm.
Its purpose is to evaluate feature extraction strategies, similarity metrics, graph construction
approaches, and clustering algorithms so that the best combination can be promoted into the
production `SliceGrouper` processor.

---

## Background

In digital pathology, tissue fragments from the same biopsy core are embedded together in a
paraffin block, serially sectioned, and scanned as Whole Slide Images (WSI).
After tissue detection and connected-component labelling, the question arises:
**which disconnected components on the slide belong to the same specimen?**

The production system uses a simple spatial-proximity heuristic (see `ARCHITECTURE.md`).
This notebook evaluates whether richer visual features can substantially improve grouping quality.

---

## Project Structure

```
SliceGrouperResearch/
│
├── SliceGrouperResearch.ipynb     ← Main research notebook (17 sections)
├── config.py                      ← All tuneable parameters
│
├── src/
│   ├── io.py                      ← TIFF/mask loading, component detection
│   ├── geometry.py                ← Area, perimeter, solidity, skeleton, PCA
│   ├── shape_features.py          ← Hu moments, Fourier, shape context, Hausdorff
│   ├── color_features.py          ← RGB/HSV/Lab stats, histograms, Macenko norm
│   ├── texture_features.py        ← LBP, GLCM, Haralick, Gabor
│   ├── deep_features.py           ← ResNet50, DINOv2, pathology foundation models
│   ├── graph_builder.py           ← KNN, threshold, Delaunay, Gabriel graphs
│   ├── similarity.py              ← Weighted feature combination → similarity matrix
│   ├── clustering.py              ← 8 clustering algorithms
│   ├── benchmarking.py            ← Silhouette/DB/CH metrics, report generation
│   ├── visualization.py           ← All Matplotlib/Plotly/PyVis figures
│   ├── widgets.py                 ← ipywidgets interactive dashboard
│   └── utils.py                   ← Timer, normalisation, embedding wrappers
│
├── outputs/                       ← Generated figures, CSVs, reports
└── examples/                      ← Example slides for testing
```

---

## Installation

### Setting up a Virtual Environment (Recommended)

To ensure these packages do not conflict with your main backend environment, create a separate virtual environment:

```powershell
# 1. Navigate to the research folder
cd c:\Users\sitko\dev\4Diagnosis_ComputerVision\SliceGrouperResearch

# 2. Create the virtual environment
py -m venv venv_research

# 3. Activate the virtual environment
.\venv_research\Scripts\activate
```

*(You should see `(venv_research)` appear at the beginning of your terminal prompt).*

### Minimum Requirements

Once the virtual environment is activated, install the base requirements:

```powershell
pip install numpy scipy matplotlib pandas scikit-learn scikit-image
pip install opencv-python tifffile networkx
pip install ipywidgets plotly jupyterlab ipykernel
```

### Linking the Environment to Jupyter Notebook

To use this isolated environment inside your Jupyter notebooks, you need to register it as a kernel:

```powershell
# With your venv_research still activated, run:
py -m ipykernel install --user --name=venv_research --display-name "Python (SliceGrouper Research)"
```

When you open the notebook, select **Kernel -> Change Kernel** and choose **Python (SliceGrouper Research)**.

### Recommended (full feature set)

```bash
# Deep features
pip install torch torchvision

# UMAP
pip install umap-learn

# Community detection
pip install python-louvain leidenalg igraph

# HDBSCAN (standalone, faster than sklearn's)
pip install hdbscan

# Interactive graph
pip install pyvis

# Stain normalisation
pip install staintools

# Benchmarking table formatting
pip install tabulate
```

---

## Usage

1. **Prepare input files**

   Place your files in the notebook directory (or update paths in `config.py`):
   ```
   slide.tiff    ← original WSI TIFF
   mask.tiff     ← binary tissue mask (0=background, 1=tissue)
   ```

2. **Start JupyterLab**
   ```bash
   jupyter lab SliceGrouperResearch.ipynb
   ```

3. **Run cells top-to-bottom**

   Each section builds on the previous. Run all cells to populate all
   feature matrices and build the dashboard.

4. **Experiment**

   Use the interactive dashboard in Section 14 to:
   - Select/deselect feature groups
   - Adjust feature weights
   - Choose clustering algorithms
   - Tune thresholds
   - Compare embedding visualisations

5. **Export Results**

   Section 16 automatically exports:
   - `outputs/benchmark.csv` — numeric comparison table
   - `outputs/benchmark.json` — machine-readable results
   - `outputs/benchmark.md` — Markdown report
   - `outputs/*.png` — all generated figures

---

## Notebook Sections

| # | Section | Contents |
|---|---------|----------|
| 1 | Introduction & Setup | Imports, config, data loading |
| 2 | Data Loading & Component Detection | TIFF/mask parsing, CC detection |
| 3 | Component Explorer | Interactive per-component inspection |
| 4 | Geometry Features | Area, perimeter, solidity, skeleton, PCA |
| 5 | Shape Features | Hu, Fourier, shape context, Hausdorff |
| 6 | Color Features | RGB/HSV/Lab stats, histograms, stain norm |
| 7 | Texture Features | LBP, GLCM, Haralick, Gabor |
| 8 | Local Features | ORB, SIFT, AKAZE keypoints |
| 9 | Deep Features | ResNet50, DINOv2, cosine similarity |
| 10 | Embedding Visualization | PCA, t-SNE, UMAP interactive scatter |
| 11 | Similarity & Distance Matrices | Feature weighting dashboard |
| 12 | Graph Construction | KNN, Delaunay, PyVis interactive graph |
| 13 | Clustering Experiments | All 8 methods, side-by-side comparison |
| 14 | Interactive Dashboard | Full widget dashboard |
| 15 | Benchmarking | Runtime + quality metrics comparison |
| 16 | Reporting | CSV, JSON, Markdown, PNG export |
| 17 | Conclusions & Next Steps | Productionisation guidance |

---

## Feature Groups

| Group | Features | Typical Dim |
|-------|----------|-------------|
| Geometry | Area, perimeter, solidity, circularity, aspect ratio, eccentricity, orientation, PCA, skeleton | 12 |
| Shape | Hu moments (7), Fourier descriptors (128), Shape context (60) | ~195 |
| Color | RGB/HSV/Lab stats (27), RGB/HSV/Lab histograms (96) | ~123 |
| Texture | LBP (26), GLCM (4), Haralick (13), entropy, variance, Gabor (32) | ~78 |
| Deep | ResNet50 (2048) or DINOv2 (768) | 768–2048 |

---

## Extending the Environment

To add a new feature extractor:

1. Add your function/class to the appropriate `src/` module.
2. Add it to the feature extraction loop in Section 4–9 of the notebook.
3. Add the feature vector to `similarity.py`'s `compute_pairwise_similarity`.
4. Add a slider in `widgets.py`'s `WeightSliderWidget`.

To add a new clustering algorithm:

1. Add `run_my_algorithm()` to `src/clustering.py`.
2. Register it in `benchmarking.py`'s `benchmark_all_methods`.
3. Add it to the dropdown in `widgets.py`'s `ClusteringControlWidget`.

---

## Reference Papers

- **Hu moments**: M.K. Hu (1962). Visual Pattern Recognition by Moment Invariants.
- **Shape context**: Belongie, Malik, Puzicha (2002). Shape Matching and Object Recognition Using Shape Contexts.
- **LBP**: Ojala, Pietikainen, Harwood (1994). Performance evaluation of texture measures.
- **GLCM**: Haralick, Shanmugam, Dinstein (1973). Textural Features for Image Classification.
- **Macenko**: Macenko et al. (2009). A method for normalizing histology slides.
- **ResNet50**: He, Zhang, Ren, Sun (2016). Deep Residual Learning for Image Recognition.
- **DINOv2**: Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision.
- **Louvain**: Blondel et al. (2008). Fast unfolding of communities in large networks.
- **Leiden**: Traag, Waltman, van Eck (2019). From Louvain to Leiden.
- **HDBSCAN**: Campello, Moulavi, Sander (2013). Density-Based Clustering Based on Hierarchical Density Estimates.
- **Silhouette**: Rousseeuw (1987). Silhouettes: A Graphical Aid to the Interpretation of Cluster Analysis.

---

*SliceGrouperResearch — R&D Environment for 4Diagnosis Computer Vision*
