from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image


class GlomeruliProcessor:
    def __init__(
        self,
        path_tiff: str,
        model_path: str,
        mask_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        conf: float = 0.5,
        iou: float = 0.45,
        imgsz: int = 1024,
        patch_size: int = 1024,
        overlap: int = 256,
        batch_size: int = 16,
    ):
        self.path = Path(path_tiff)
        self.model_path = Path(model_path)
        self.mask_path = Path(mask_path) if mask_path else None
        self.output_dir = Path(output_dir) if output_dir else self.path.parent
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.patch_size = patch_size
        self.overlap = overlap
        self.batch_size = batch_size

        self.model = None
        self.glomeruli: List[Dict[str, Any]] = []

    def load_model(self):
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
        return self.model

    def load_image(self) -> np.ndarray:
        try:
            Image.MAX_IMAGE_PIXELS = None
            img = Image.open(self.path).convert("RGB")
            return np.array(img)
        except Exception:
            img = cv2.imread(str(self.path))
            if img is None:
                raise ValueError(f"Failed to load image: {self.path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def load_mask(self, h: int, w: int) -> Optional[np.ndarray]:
        if not self.mask_path or not self.mask_path.exists():
            return None
        mask = cv2.imread(str(self.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask

    def predict_on_patch(self, patch: np.ndarray, x_offset: int = 0, y_offset: int = 0):
        model = self.load_model()
        results = model.predict(
            patch,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )

        detections = []
        if not results:
            return detections

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            cls_id = int(box.cls[0].cpu().item()) if box.cls is not None else -1
            score = float(box.conf[0].cpu().item()) if box.conf is not None else 0.0

            x1, y1, x2, y2 = xyxy
            detections.append(
                {
                    "x1": int(x1 + x_offset),
                    "y1": int(y1 + y_offset),
                    "x2": int(x2 + x_offset),
                    "y2": int(y2 + y_offset),
                    "cls": cls_id,
                    "conf": score,
                    "source": "ai",
                }
            )

        return detections

    def detect_glomeruli(self, save_patches: bool = False) -> List[Dict[str, Any]]:
        img = self.load_image()
        h, w = img.shape[:2]
        self.glomeruli = []

        mask = self.load_mask(h, w)

        patches_dir = self.output_dir / "patches"
        if save_patches:
            patches_dir.mkdir(parents=True, exist_ok=True)

        step = self.patch_size - self.overlap

        batch_patches = []
        batch_coords = []
        
        # Single model instance for all patches
        model = self.load_model()

        def process_batch():
            if not batch_patches:
                return

            results = model.predict(
                batch_patches,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                verbose=False,
            )

            for i, result in enumerate(results):
                x_offset, y_offset = batch_coords[i]
                patch = batch_patches[i]
                boxes = result.boxes
                
                detections_in_patch = []
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy().tolist()
                        cls_id = int(box.cls[0].cpu().item()) if box.cls is not None else -1
                        score = float(box.conf[0].cpu().item()) if box.conf is not None else 0.0

                        x1, y1, x2, y2 = xyxy
                        det = {
                            "x1": int(x1 + x_offset),
                            "y1": int(y1 + y_offset),
                            "x2": int(x2 + x_offset),
                            "y2": int(y2 + y_offset),
                            "cls": cls_id,
                            "conf": score,
                            "source": "ai",
                        }
                        self.glomeruli.append(det)
                        detections_in_patch.append(det)

                if save_patches and detections_in_patch:
                    patch_vis = patch.copy()
                    for det in detections_in_patch:
                        dx1 = det["x1"] - x_offset
                        dy1 = det["y1"] - y_offset
                        dx2 = det["x2"] - x_offset
                        dy2 = det["y2"] - y_offset
                        conf = det["conf"]

                        cv2.rectangle(patch_vis, (dx1, dy1), (dx2, dy2), (0, 255, 0), 3)
                        cv2.putText(
                            patch_vis,
                            f"{conf:.2f}",
                            (dx1, max(0, dy1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )

                    cv2.imwrite(
                        str(patches_dir / f"patch_x{x_offset}_y{y_offset}_det.jpg"),
                        cv2.cvtColor(patch_vis, cv2.COLOR_RGB2BGR)
                    )

            batch_patches.clear()
            batch_coords.clear()

        for y in range(0, h, step):
            for x in range(0, w, step):
                y_end = min(y + self.patch_size, h)
                x_end = min(x + self.patch_size, w)
                
                if mask is not None:
                    mask_patch = mask[y:y_end, x:x_end]
                    # Skip patch if it contains no tissue (assuming tissue > 0 in mask)
                    if not np.any(mask_patch):
                        continue
                else:
                    # Fallback: simple brightness thresholding to skip white background
                    patch_gray = cv2.cvtColor(img[y:y_end, x:x_end], cv2.COLOR_RGB2GRAY)
                    if np.mean(patch_gray) > 230:
                        continue

                patch = img[y:y_end, x:x_end]
                ph, pw = patch.shape[:2]
                if ph < 50 or pw < 50:
                    continue

                batch_patches.append(patch)
                batch_coords.append((x, y))

                if len(batch_patches) >= self.batch_size:
                    process_batch()

        # Process any remaining patches
        process_batch()

        self.glomeruli = self.simple_global_merge(self.glomeruli, iou_thresh=0.5)
        return self.glomeruli

    def count_glomeruli(self) -> int:
        return len(self.glomeruli or [])

    def save_annotated_image(self, out_path: Optional[str] = None) -> str:
        img = self.load_image()
        annotated = img.copy()

        if not self.glomeruli:
            self.detect_glomeruli()

        for det in self.glomeruli:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            conf = det["conf"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(
                annotated,
                f"glomerulus {conf:.2f}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

        save_dir = self.output_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_path or str(save_dir / f"{self.path.stem}_glomeruli.jpg")

        cv2.imwrite(out_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        return out_path
 
    def simple_global_merge(self, detections, iou_thresh=0.5):
        if not detections:
            return []

        def are_close_or_overlap(a, b):
            x1 = max(a["x1"], b["x1"])
            y1 = max(a["y1"], b["y1"])
            x2 = min(a["x2"], b["x2"])
            y2 = min(a["y2"], b["y2"])

            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
            area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])

            if area_a == 0 or area_b == 0:
                return False

            if inter > 0:
                iou = inter / (area_a + area_b - inter)
                io_min = inter / min(area_a, area_b)
                if iou > iou_thresh or io_min > 0.1:
                    return True

            cx_a = (a["x1"] + a["x2"]) / 2
            cy_a = (a["y1"] + a["y2"]) / 2
            cx_b = (b["x1"] + b["x2"]) / 2
            cy_b = (b["y1"] + b["y2"]) / 2
            
            dist_x = abs(cx_a - cx_b)
            dist_y = abs(cy_a - cy_b)
            
            half_w = (a["x2"] - a["x1"] + b["x2"] - b["x1"]) / 2
            half_h = (a["y2"] - a["y1"] + b["y2"] - b["y1"]) / 2

            if dist_x < half_w + 15 and dist_y < half_h + 15:
                return True
                
            return False

        detections = sorted(detections, key=lambda d: d["conf"], reverse=True)
        merged = []

        for det in detections:
            merged_with_existing = False
            for kept in merged:
                if are_close_or_overlap(det, kept):
                    kept["x1"] = min(kept["x1"], det["x1"])
                    kept["y1"] = min(kept["y1"], det["y1"])
                    kept["x2"] = max(kept["x2"], det["x2"])
                    kept["y2"] = max(kept["y2"], det["y2"])
                    kept["conf"] = max(kept["conf"], det["conf"])
                    merged_with_existing = True
                    break
            if not merged_with_existing:
                merged.append(det.copy())

        return merged