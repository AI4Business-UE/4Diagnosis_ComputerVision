from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np

GLOMERULI_CLASSES = {
    0: "circle",  # kłębuszek niezwłókniony (zdrowy)
    1: "rect",    # kłębuszek zwłókniony — glomerulosclerosis
}


def _apply_tissue_mask(tile_rgb: np.ndarray) -> np.ndarray:
    """Zastępuje nieistotne tło (szkło, niebieskie barwniki itp.) bielą.
    Te same parametry co generate_mask: sat_min=5, val_max=250, kernel 3×3."""
    tile_bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([0, 5, 0], dtype=np.uint8),
        np.array([179, 255, 250], dtype=np.uint8),
    )
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    result = tile_rgb.copy()
    result[mask == 0] = 255  # tło → biały
    return result


class GlomeruliProcessor:
    def __init__(
        self,
        path_mrxs: str,
        model_path: str,
        conf: float = 0.15,
        iou: float = 0.3,
        imgsz: int = 1024,
        tile_size: int = 4000,
        overlap: int = 500,
        wsi_level: int = 0,
        batch_size: int = 16,
        mask_path: str | None = None,
    ):
        self.path = Path(path_mrxs)
        self.model_path = Path(model_path)
        self.mask_path = Path(mask_path) if mask_path else None
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.tile_size = tile_size
        self.overlap = overlap
        self.wsi_level = wsi_level
        self.batch_size = batch_size

        self.model = None
        self.glomeruli: List[Dict[str, Any]] = []

    def load_model(self):
        if self.model is None:
            import torch
            from ultralytics import YOLO
            _orig = torch.load
            torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
            try:
                self.model = YOLO(str(self.model_path))
            finally:
                torch.load = _orig
        return self.model

    def detect_glomeruli(self, on_batch=None, on_tile=None) -> List[Dict[str, Any]]:
        import openslide

        self.glomeruli = []
        model = self.load_model()

        slide = openslide.OpenSlide(str(self.path))
        W, H = slide.level_dimensions[self.wsi_level]

        scale = self.tile_size / self.imgsz
        step = self.tile_size - self.overlap

        # ---------------------------------------------------------------
        # Maska z TIFF — ta sama co podgląd "Oryginalny TIFF".
        # Używamy jej do filtrowania kafelków: kafelek biały w podglądzie
        # = pominięty przez YOLO. Wczytujemy raz (niskie zużycie RAM).
        # ---------------------------------------------------------------
        tiff_mask = None
        if self.mask_path and self.mask_path.exists():
            _m = cv2.imread(str(self.mask_path), cv2.IMREAD_GRAYSCALE)
            if _m is not None:
                tiff_mask = (_m > 0)  # bool array — True = tkanka

        def is_tissue_in_tiff_mask(x, y, w, h) -> bool:
            if tiff_mask is None:
                return True  # fallback: brak maski → skanuj wszystko
            mh, mw = tiff_mask.shape
            mx1 = int(x * mw / W); my1 = int(y * mh / H)
            mx2 = max(mx1 + 1, int((x + w) * mw / W))
            my2 = max(my1 + 1, int((y + h) * mh / H))
            region = tiff_mask[my1:my2, mx1:mx2]
            if region.size == 0:
                return False
            # ≥5% pikseli maski oznaczone jako tkanka → skanuj kafelek
            return region.sum() / region.size >= 0.05

        # ---------------------------------------------------------------
        # Thumbnail pre-filter: sprawdź które kafelki to tkanka BEZ
        # czytania level-0. OpenSlide pobiera thumbnail z gotowego
        # low-res levelu — zajmuje ułamek sekundy dla całego slajdu.
        # ---------------------------------------------------------------
        THUMB_SCALE = 32  # miniaturka 1:32, wystarczy do detekcji tkanki
        thumb_w = max(1, W // THUMB_SCALE)
        thumb_h = max(1, H // THUMB_SCALE)
        thumb = slide.get_thumbnail((thumb_w, thumb_h))
        thumb_gray = cv2.cvtColor(np.array(thumb.convert("RGB")), cv2.COLOR_RGB2GRAY)
        actual_tw, actual_th = thumb.size  # OpenSlide może zwrócić nieco inne wymiary

        sx = actual_tw / W  # współczynnik skalowania thumbnail → level-0
        sy = actual_th / H

        def is_tissue_thumb(x, y, w, h) -> bool:
            tx1 = int(x * sx)
            ty1 = int(y * sy)
            tx2 = max(tx1 + 1, int((x + w) * sx))
            ty2 = max(ty1 + 1, int((y + h) * sy))
            region = thumb_gray[ty1:ty2, tx1:tx2]
            if region.size == 0:
                return False
            # Liczymy piksele w zakresie tkanki (nie puste szkło >230, nie czerna pustka <15)
            # Niski próg 3% żeby łapać kafelki na brzegach tkanki
            tissue_px = int(np.sum((region > 15) & (region < 230)))
            return tissue_px >= max(1, int(region.size * 0.03))
        # ---------------------------------------------------------------

        batch_tiles = []
        batch_offsets = []

        def process_batch():
            if not batch_tiles:
                return
            results = model.predict(
                batch_tiles,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                verbose=False,
            )
            for i, result in enumerate(results):
                x_off, y_off = batch_offsets[i]
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                    cls_id = int(box.cls[0].cpu().item()) if box.cls is not None else 0
                    score = float(box.conf[0].cpu().item()) if box.conf is not None else 0.0
                    self.glomeruli.append({
                        "x1": int(x1 * scale + x_off),
                        "y1": int(y1 * scale + y_off),
                        "x2": int(x2 * scale + x_off),
                        "y2": int(y2 * scale + y_off),
                        "cls": cls_id,
                        "cls_name": GLOMERULI_CLASSES.get(cls_id, str(cls_id)),
                        "conf": score,
                    })
            if on_batch:
                on_batch(self.glomeruli)
            batch_tiles.clear()
            batch_offsets.clear()

        for y in range(0, H, step):
            for x in range(0, W, step):
                actual_w = min(self.tile_size, W - x)
                actual_h = min(self.tile_size, H - y)
                if actual_w < 50 or actual_h < 50:
                    continue

                # Krok 1: thumbnail — szybki check bez I/O level-0
                if not is_tissue_thumb(x, y, actual_w, actual_h):
                    if on_tile:
                        on_tile(x, y, actual_w, actual_h, False)
                    continue

                # Krok 2: maska z TIFF — spójne z podglądem (białe w podglądzie = pominięte)
                if not is_tissue_in_tiff_mask(x, y, actual_w, actual_h):
                    if on_tile:
                        on_tile(x, y, actual_w, actual_h, False)
                    continue

                # Krok 3: czytaj level-0 i zastosuj maskę HSV dla YOLO
                region = slide.read_region((x, y), self.wsi_level, (actual_w, actual_h))
                tile = np.array(region.convert("RGB"))
                tile = _apply_tissue_mask(tile)

                if on_tile:
                    on_tile(x, y, actual_w, actual_h, True)

                tile_resized = cv2.resize(tile, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
                batch_tiles.append(tile_resized)
                batch_offsets.append((x, y))

                if len(batch_tiles) >= self.batch_size:
                    process_batch()

        process_batch()
        slide.close()

        self.glomeruli = self.simple_global_merge(self.glomeruli, overlap_thresh=0.4)
        return self.glomeruli

    def count_glomeruli(self) -> int:
        return len(self.glomeruli or [])

    def simple_global_merge(self, detections, overlap_thresh=0.4):
        if not detections:
            return []

        # Sortuj malejąco po powierzchni — większe boxy mają pierwszeństwo
        detections = sorted(
            detections,
            key=lambda d: (d["x2"] - d["x1"]) * (d["y2"] - d["y1"]),
            reverse=True,
        )
        merged = []
        for det in detections:
            area_det = (det["x2"] - det["x1"]) * (det["y2"] - det["y1"])
            if area_det <= 0:
                continue
            suppress = False
            for kept in merged:
                x1 = max(det["x1"], kept["x1"])
                y1 = max(det["y1"], kept["y1"])
                x2 = min(det["x2"], kept["x2"])
                y2 = min(det["y2"], kept["y2"])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                # Jeśli ≥40% powierzchni kandydata pokrywa się z zatrzymanym boxem → usuń kandydata
                if inter / area_det >= overlap_thresh:
                    suppress = True
                    break
            if not suppress:
                merged.append(det)
        return merged
