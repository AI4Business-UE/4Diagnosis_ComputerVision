import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from PIL import Image, ImageOps

# ============================================================
# 1. Maska HSV + morfologia
# ============================================================

def compute_initial_mask(img_bgr, sat_min, val_max, open_ksize, close_ksize):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, sat_min, 0], dtype=np.uint8)
    upper = np.array([179, 255, val_max], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    if open_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    if close_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    return (mask > 0).astype(np.uint8)


# ============================================================
# 2. Ekstrakcja komponentów
# ============================================================

def extract_components(binary_mask, min_area):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    components = []
    for lab in range(1, num):
        if stats[lab, cv2.CC_STAT_AREA] >= min_area:
            comp = (labels == lab)
            components.append(comp)

    return components


# ============================================================
# 3. Funkcja główna
# ============================================================

def generate_mask(
    image_path: str,
    mode: str = "all",   # "largest" albo "all"
    visualize: bool = True,
    save_mask: bool = True,
    save_preview: bool = True,
    mask_suffix: str = "_mask.tiff",
    sat_min: int = 5,
    val_max: int = 250,
    open_ksize: int = 3,
    close_ksize: int = 3,
    min_component_area: int = 50000, # 20000
):
    """
    mode="largest" → zwraca największy komponent.
    mode="all"     → zwraca maskę łączoną wszystkich komponentów.
    Preview i visualize są wykonywane tylko raz.
    """

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Nie można wczytać obrazu: {image_path}")

    # 1. Maska wstępna
    binary = compute_initial_mask(img_bgr, sat_min, val_max, open_ksize, close_ksize)

    # 2. Komponenty
    components = extract_components(binary, min_component_area)

    # 3. Wybór trybu
    if mode == "largest":
        if components:
            mask_bool = max(components, key=lambda c: c.sum())
        else:
            mask_bool = np.zeros_like(binary, dtype=bool)

    elif mode == "all":
        if components:
            mask_bool = np.any(np.stack(components, axis=0), axis=0)
        else:
            mask_bool = np.zeros_like(binary, dtype=bool)

    else:
        raise ValueError("mode must be 'largest' or 'all'")

    # 4. Zapisywanie maski
    out_dir = Path(image_path).parent
    stem = Path(image_path).stem
    mask_path = None

    if save_mask:
        mask_path = out_dir / f"{stem}{mask_suffix}"
        cv2.imwrite(str(mask_path), mask_bool.astype(np.uint8) * 255)

    # 5. Preview (jedno)
    if save_preview:
        preview = img_bgr.copy()
        preview[~mask_bool] = (255, 255, 255)
        preview_path = out_dir / f"{stem}_preview{mask_suffix}"
        cv2.imwrite(str(preview_path), preview)

    # 6. Wizualizacja (jedna)
    if visualize:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        _visualize_and_save(img=img_rgb, mask_bool=mask_bool, show=True)

    return {
        "mask": mask_bool,
        "mask_path": str(mask_path) if save_mask else None,
        "preview_path": str(preview_path) if save_preview else None,
        "tissue_pixels": int(mask_bool.sum()),
        "params": {
            "sat_min": sat_min,
            "val_max": val_max,
            "open_ksize": open_ksize,
            "close_ksize": close_ksize,
            "min_component_area": min_component_area,
            "mode": mode,
        },
    }

def _visualize_and_save(
    img: np.ndarray,
    skeleton: np.ndarray | list[np.ndarray] | None = None,
    path_coords: list[tuple[int, int]] | list[list[tuple[int, int]]] | None = None,
    mask_bool: np.ndarray | None = None,
    out_path: str | None = None,
    overlay_title: str | None = None,
    show: bool = False
):
    """
    Unified visualization & saving helper.
    Obsługuje:
    - pojedynczy skeleton lub listę skeletonów
    - pojedynczą ścieżkę lub listę ścieżek
    """

    img_vis = img.copy()
    if mask_bool is not None:
        img_vis[~mask_bool] = 255

    if img_vis.ndim == 3:
        img_gray = cv2.cvtColor(img_vis, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_vis.copy()

    plt.figure(figsize=(12, 12))

    # --- 3. Wyświetlanie grayscale ---
    plt.imshow(img_gray, cmap="gray")

    # --- Skeletony ---
    if skeleton is not None:
        if isinstance(skeleton, list):
            for sk in skeleton:
                plt.contour(sk, [0.5], colors='red', linewidths=1)
        else:
            plt.contour(skeleton, [0.5], colors='red', linewidths=1)

    # --- Ścieżki ---
    if path_coords:
        if isinstance(path_coords, list) and isinstance(path_coords[0], list):
            for path in path_coords:
                if path:
                    y, x = zip(*path)
                    plt.plot(x, y, color='cyan', linewidth=3)
        else:
            y, x = zip(*path_coords)
            plt.plot(x, y, color='cyan', linewidth=3)

    if overlay_title:
        plt.title(overlay_title)

    plt.axis('off')
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, format='tiff', dpi=150, bbox_inches='tight')
        plt.close()
    elif show:
        plt.show()

def load_mask_for_image(image_path: str, target_shape: tuple[int, int] | None = None) -> np.ndarray | None:
    """
    Utility function to load tissue mask for an image.

    Parameters
    ----------
    image_path : str
        Path to the image file (e.g., TIFF).
    target_shape : tuple[int, int], optional
        Target shape (height, width) to resize mask to. If None, uses original mask size.

    Returns
    -------
    np.ndarray or None
        Boolean mask array where True indicates tissue pixels, or None if mask not found/failed to load.
    """
    mask_path = Path(image_path).with_name(Path(image_path).stem + "_mask.tiff")
    if not mask_path.exists():
        return None

    try:
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            return None

        mask_bool = (mask_img > 0).astype(bool)

        # Resize if target shape provided and different
        if target_shape is not None and mask_bool.shape != target_shape:
            mask_bool = cv2.resize(
                mask_bool.astype(np.uint8),
                (target_shape[1], target_shape[0]),  # (width, height)
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        return mask_bool
    except Exception:
        return None
