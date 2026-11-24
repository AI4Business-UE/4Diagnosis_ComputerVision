import threading
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

_PLT_LOCK = threading.Lock()

try:
    from django.conf import settings as django_settings
except Exception:
    django_settings = None


_DEFAULT_CLEANING_LEVEL_AREA_UM2 = {
    0: 0.0,
    1: 5_000.0,
    2: 15_000.0,
    3: 30_000.0,
    4: 55_000.0,
    5: 90_000.0,
    6: 160_000.0,
    7: 280_000.0,
    8: 500_000.0,
    9: 900_000.0,
    10: 1_500_000.0,
}


def cleaning_level_to_um2(cleaning_level: int | None) -> float:
    level_map = (
        getattr(django_settings, "CLEANING_LEVEL_AREA_UM2", _DEFAULT_CLEANING_LEVEL_AREA_UM2)
        if django_settings
        else _DEFAULT_CLEANING_LEVEL_AREA_UM2
    )
    if cleaning_level is None:
        default_lvl = getattr(django_settings, "CLEANING_DEFAULT_LEVEL", 5) if django_settings else 5
        return float(level_map.get(default_lvl, 90_000.0))
    clamped = max(0, min(10, int(cleaning_level)))
    return float(level_map.get(clamped, 90_000.0))


def compute_min_component_area_px(
    mpp_x: float = 0.0,
    mpp_y: float = 0.0,
    downsample: float = 32.0,
    min_area_um2: float | None = None,
    fallback_area_lvl5_px: int | None = None,
    cleaning_level: int | None = None,
) -> int:
    base_um2 = float(getattr(django_settings, "SLICE_MIN_AREA_UM2", 90_000.0)) if django_settings else 90_000.0
    if min_area_um2 is None:
        min_area_um2 = cleaning_level_to_um2(cleaning_level) if cleaning_level is not None else base_um2
    if fallback_area_lvl5_px is None:
        fallback_area_lvl5_px = int(getattr(django_settings, "SLICE_MIN_AREA", 6_000)) if django_settings else 6_000

    if min_area_um2 <= 0:
        return 1

    if mpp_x > 0 and mpp_y > 0:
        px_area_um2 = mpp_x * mpp_y
        return max(1, int(round(min_area_um2 / px_area_um2)))

    ds = max(float(downsample), 1.0)
    level_ratio = (min_area_um2 / base_um2) if base_um2 > 0 else 1.0
    scale_factor = ((32.0 / ds) ** 2) * level_ratio
    return max(1, int(round(fallback_area_lvl5_px * scale_factor)))


# ============================================================
# 1. HSV mask + interior-safe morphology
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


def _fill_enclosed_holes(binary_u8: np.ndarray, max_hole_area: int = 20_000) -> np.ndarray:
    """
    Fill enclosed background holes (not touching image border) up to max_hole_area px.
    Used to solidify tissue interiors before MORPH_OPEN so exterior opening trims only
    loose external strands/bridges and never punches holes into interior tissue/collagen.
    """
    inv = (binary_u8 == 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    if num <= 1:
        return binary_u8
    h, w = binary_u8.shape
    fill_lab = np.zeros(num, dtype=bool)
    for lab in range(1, num):
        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        bw = int(stats[lab, cv2.CC_STAT_WIDTH])
        bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if x == 0 or y == 0 or (x + bw) >= w or (y + bh) >= h:
            continue
        if area <= max_hole_area:
            fill_lab[lab] = True
    if not np.any(fill_lab):
        return binary_u8
    return np.where(fill_lab[labels], np.uint8(1), binary_u8)


def _segment_interior_safe(
    img_bgr: np.ndarray,
    mode: str,
    sat_min: int,
    val_max: int,
    open_ksize: int,
    close_ksize: int,
    min_component_area: int,
    max_hole_area: int = 20_000,
    max_components: int = 500,
):
    """
    Interior-preserving tissue segmentation:
    1. Computes pixel-exact `closed` tissue mask (HSV threshold + optional MORPH_CLOSE, zero erosion).
    2. When `open_ksize > 0`, solidifies tissue interiors (5x5 close + fill enclosed holes) BEFORE
       applying `MORPH_OPEN` so opening acts only on the exterior silhouette (trimming loose
       wispy ends and severing thin external bridges) without eroding interior tissue or collagen.
    3. Filters connected components by `min_component_area` (and `mode`), then intersects the
       kept silhouette with `closed` so interior tissue stays 100% intact while empty white
       lumens remain excluded.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, sat_min, 0], dtype=np.uint8)
    upper = np.array([179, 255, val_max], dtype=np.uint8)
    raw = (cv2.inRange(hsv, lower, upper) > 0).astype(np.uint8)
    del hsv

    if close_ksize > 0:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k_close)
    else:
        closed = raw

    if open_ksize > 0:
        solid_k = max(5, close_ksize)
        k_solid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (solid_k, solid_k))
        solid = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k_solid)
        solid = _fill_enclosed_holes(solid, max_hole_area=max_hole_area)
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        work = cv2.morphologyEx(solid, cv2.MORPH_OPEN, k_open)
        del solid
    else:
        work = closed
    del raw

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(work, connectivity=8)
    del work

    valid_labs = [
        lab for lab in range(1, num)
        if int(stats[lab, cv2.CC_STAT_AREA]) >= min_component_area
    ]
    if not valid_labs:
        return np.zeros(closed.shape, dtype=bool), [], labels, stats, centroids

    if mode == "largest":
        best_lab = max(valid_labs, key=lambda l: int(stats[l, cv2.CC_STAT_AREA]))
        valid_labs = [best_lab]

    keep = np.zeros(num, dtype=bool)
    keep[valid_labs] = True
    kept_region = keep[labels]
    if open_ksize > 0:
        kept_region = _fill_enclosed_holes(kept_region.astype(np.uint8), max_hole_area=max_hole_area) > 0

    mask_bool = kept_region & (closed > 0)
    del kept_region, closed

    comp_labs = valid_labs
    if len(comp_labs) > max_components:
        comp_labs = sorted(
            comp_labs,
            key=lambda l: int(stats[l, cv2.CC_STAT_AREA]),
            reverse=True,
        )[:max_components]

    return mask_bool, comp_labs, labels, stats, centroids


# ============================================================
# 2. Component extraction
# ============================================================

def _extract_components_with_labels(binary_mask, min_area):
    """
    Extract connected components storing only tight bounding-box boolean masks
    instead of N full-resolution (H, W) arrays.
    """
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    components = []
    valid_labels = []
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= min_area:
            x = int(stats[lab, cv2.CC_STAT_LEFT])
            y = int(stats[lab, cv2.CC_STAT_TOP])
            w = int(stats[lab, cv2.CC_STAT_WIDTH])
            h = int(stats[lab, cv2.CC_STAT_HEIGHT])
            components.append({
                "bbox": [x, y, w, h],
                "area": area,
                "centroid": (float(centroids[lab, 0]), float(centroids[lab, 1])),
                "crop_mask": (labels[y:y + h, x:x + w] == lab),
            })
            valid_labels.append(lab)

    return components, labels, valid_labels


def extract_components(binary_mask, min_area):
    components, _, _ = _extract_components_with_labels(binary_mask, min_area)
    return components


# ============================================================
# 3. Main function
# ============================================================

def _build_scale_invariant_mask(
    img_bgr: np.ndarray,
    mode: str,
    sat_min: int,
    val_max: int,
    open_ksize: int,
    close_ksize: int,
    min_component_area_px: int,
    avg_mpp: float,
    ref_mpp: float,
):
    """
    Run the exact calibrated level-5 (M, ~3.885 µm/px) interior-safe cleaning pipeline
    on a downscaled copy of the high-res image, then smoothly upscale the resulting
    mask and components back to full (H, W) resolution.

    This guarantees identical cleaning behavior across M, L, and XL while keeping
    interior tissue/collagen untouched and reducing RAM/CPU usage.
    """
    h, w = img_bgr.shape[:2]
    scale = avg_mpp / ref_mpp  # < 1.0 for L (~0.5) and XL (~0.25)
    small_w = max(1, int(round(w * scale)))
    small_h = max(1, int(round(h * scale)))

    small_bgr = cv2.resize(img_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
    sx = small_w / w
    sy = small_h / h
    min_area_small = 1 if min_component_area_px <= 1 else max(1, int(round(min_component_area_px * sx * sy)))

    small_mask_bool, comp_labs, small_labels, stats, _ = _segment_interior_safe(
        img_bgr=small_bgr,
        mode=mode,
        sat_min=sat_min,
        val_max=val_max,
        open_ksize=open_ksize,
        close_ksize=close_ksize,
        min_component_area=min_area_small,
        max_hole_area=20_000,
    )
    del small_bgr

    if not comp_labs:
        return np.zeros((h, w), dtype=bool), []

    # Upscale full mask in one vectorized pass (fast even at cleaning_level=0 with many specks)
    mask_bool = cv2.resize(
        small_mask_bool.astype(np.uint8) * 255,
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    ) >= 128

    components = []
    inv_sx = w / small_w
    inv_sy = h / small_h

    for lab in comp_labs:
        lx = int(stats[lab, cv2.CC_STAT_LEFT])
        ly = int(stats[lab, cv2.CC_STAT_TOP])
        lw = int(stats[lab, cv2.CC_STAT_WIDTH])
        lh = int(stats[lab, cv2.CC_STAT_HEIGHT])

        sx0 = max(0, lx - 1)
        sy0 = max(0, ly - 1)
        sx1 = min(small_w, lx + lw + 1)
        sy1 = min(small_h, ly + lh + 1)

        lab_small = (
            ((small_labels[sy0:sy1, sx0:sx1] == lab) & small_mask_bool[sy0:sy1, sx0:sx1]).astype(np.uint8)
            * 255
        )

        fx0 = max(0, int(np.floor(sx0 * inv_sx)))
        fy0 = max(0, int(np.floor(sy0 * inv_sy)))
        fx1 = min(w, int(np.ceil(sx1 * inv_sx)))
        fy1 = min(h, int(np.ceil(sy1 * inv_sy)))
        if fx1 <= fx0 or fy1 <= fy0:
            continue

        comp_full = cv2.resize(
            lab_small,
            (fx1 - fx0, fy1 - fy0),
            interpolation=cv2.INTER_LINEAR,
        ) >= 128
        comp_full &= mask_bool[fy0:fy1, fx0:fx1]

        ys, xs = np.where(comp_full)
        if len(ys) == 0:
            continue

        rx0, rx1 = int(xs.min()), int(xs.max()) + 1
        ry0, ry1 = int(ys.min()), int(ys.max()) + 1
        tight_crop = comp_full[ry0:ry1, rx0:rx1]

        gx0, gy0 = fx0 + rx0, fy0 + ry0
        gw, gh = rx1 - rx0, ry1 - ry0

        components.append({
            "bbox": [gx0, gy0, gw, gh],
            "area": int(len(ys)),
            "centroid": (float(xs.mean()) + fx0, float(ys.mean()) + fy0),
            "crop_mask": tight_crop,
        })

    return mask_bool, components


def generate_mask(
    image_path: str,
    mode: str = "all",   # "largest" or "all"
    visualize: bool = True,
    save_mask: bool = True,
    save_preview: bool = True,
    mask_suffix: str = "_mask.tiff",
    sat_min: int = 5,
    val_max: int = 250,
    open_ksize: int = 3,
    close_ksize: int = 3,
    min_component_area: int = 6000, # 20000
    img_bgr: np.ndarray | None = None,
    mpp_x: float = 0.0,
    mpp_y: float = 0.0,
):
    """
    mode="largest" → returns the largest component.
    mode="all"     → returns the combined mask of all components.
    """
    if mode not in ("largest", "all"):
        raise ValueError("mode must be 'largest' or 'all'")

    if img_bgr is None:
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Failed to load image: {image_path}")

    # For high-res presets (L/XL, avg_mpp < ref_mpp), run the exact calibrated level-5 (M)
    # mask pipeline on a downscaled copy and upscale back so L/XL match M and SliceGrouper.
    ref_mpp = float(getattr(django_settings, "MASK_REFERENCE_MPP", 3.885)) if django_settings else 3.885
    avg_mpp = (mpp_x + mpp_y) / 2.0 if (mpp_x > 0 and mpp_y > 0) else 0.0

    if avg_mpp > 0 and avg_mpp < ref_mpp * 0.85:
        mask_bool, components = _build_scale_invariant_mask(
            img_bgr=img_bgr,
            mode=mode,
            sat_min=sat_min,
            val_max=val_max,
            open_ksize=open_ksize,
            close_ksize=close_ksize,
            min_component_area_px=min_component_area,
            avg_mpp=avg_mpp,
            ref_mpp=ref_mpp,
        )
    else:
        if avg_mpp > 0:
            max_hole_area = max(100, int(round(20_000 * ((ref_mpp / avg_mpp) ** 2))))
        else:
            max_hole_area = 20_000

        mask_bool, comp_labs, labels, stats, centroids = _segment_interior_safe(
            img_bgr=img_bgr,
            mode=mode,
            sat_min=sat_min,
            val_max=val_max,
            open_ksize=open_ksize,
            close_ksize=close_ksize,
            min_component_area=min_component_area,
            max_hole_area=max_hole_area,
        )
        components = []
        for lab in comp_labs:
            x = int(stats[lab, cv2.CC_STAT_LEFT])
            y = int(stats[lab, cv2.CC_STAT_TOP])
            cw = int(stats[lab, cv2.CC_STAT_WIDTH])
            ch = int(stats[lab, cv2.CC_STAT_HEIGHT])
            crop_mask = (labels[y:y + ch, x:x + cw] == lab) & mask_bool[y:y + ch, x:x + cw]
            area = int(np.count_nonzero(crop_mask))
            if area == 0:
                continue
            components.append({
                "bbox": [x, y, cw, ch],
                "area": area,
                "centroid": (float(centroids[lab, 0]), float(centroids[lab, 1])),
                "crop_mask": crop_mask,
            })
        del labels

    # 4. Save mask
    out_dir = Path(image_path).parent
    stem = Path(image_path).stem
    mask_path = None

    if save_mask:
        mask_path = out_dir / f"{stem}{mask_suffix}"
        cv2.imwrite(str(mask_path), mask_bool.astype(np.uint8) * 255)

    # 5. Preview
    if save_preview:
        preview = img_bgr.copy()
        preview[~mask_bool] = (255, 255, 255)
        preview_path = out_dir / f"{stem}_preview{mask_suffix}"
        cv2.imwrite(str(preview_path), preview)

    # 6. Visualization
    if visualize:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        _visualize_and_save(img=img_rgb, mask_bool=mask_bool, show=True)

    return {
        "mask": mask_bool,
        "components": components, 
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
    Supports:
    - single skeleton or list of skeletons
    - single path or list of paths
    """

    img_vis = img.copy()
    if mask_bool is not None:
        img_vis[~mask_bool] = 255

    if img_vis.ndim == 3:
        img_gray = cv2.cvtColor(img_vis, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_vis.copy()

    with _PLT_LOCK:
        fig = plt.figure(figsize=(12, 12))
        try:
            # --- Display grayscale ---
            plt.imshow(img_gray, cmap="gray")

            # --- Skeletons ---
            if skeleton is not None:
                if isinstance(skeleton, list):
                    for sk in skeleton:
                        plt.contour(sk, [0.5], colors='red', linewidths=1)
                else:
                    plt.contour(skeleton, [0.5], colors='red', linewidths=1)

            # --- Paths ---
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
                fmt = Path(out_path).suffix.strip('.').lower()
                if fmt == 'jpg':
                    fmt = 'jpeg'
                plt.savefig(out_path, format=fmt, dpi=150, bbox_inches='tight', pad_inches=0)
            elif show:
                plt.show()
        finally:
            plt.close(fig)

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
        if mask_img.ndim == 3:
            mask_img = mask_img[:, :, 0]

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

def crop_to_mask(image: np.ndarray, mask: np.ndarray, margin: int = 50) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Crops an image to the bounding box of a mask with a margin.
    Returns the cropped image and the top-left corner (row, col) of the crop.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows):
        return image, (0, 0)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    rmin = max(0, rmin - margin)
    rmax = min(image.shape[0], rmax + margin)
    cmin = max(0, cmin - margin)
    cmax = min(image.shape[1], cmax + margin)

    cropped_image = image[rmin:rmax, cmin:cmax]
    
    return cropped_image, (rmin, cmin)
