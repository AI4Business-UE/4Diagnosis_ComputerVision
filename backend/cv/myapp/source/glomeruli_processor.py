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
        output_dir: Optional[str] = None,
        conf: float = 0.5,
        iou: float = 0.45,
        imgsz: int = 1024,
        patch_size: int = 1024,
        overlap: int = 256,   
    ):
        self.path = Path(path_tiff)
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir) if output_dir else self.path.parent
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.patch_size = patch_size
        self.overlap = overlap

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
                raise ValueError(f"Nie udało się wczytać obrazu: {self.path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
                }
            )

        return detections

    def detect_glomeruli(self, save_patches: bool = False) -> List[Dict[str, Any]]:
        img = self.load_image()
        h, w = img.shape[:2]
        self.glomeruli = []

        patches_dir = self.output_dir / "patches"
        if save_patches:
            patches_dir.mkdir(parents=True, exist_ok=True)

        step = self.patch_size - self.overlap

        for y in range(0, h, step):
            for x in range(0, w, step):
                y_end = min(y + self.patch_size, h)
                x_end = min(x + self.patch_size, w)
                patch = img[y:y_end, x:x_end]

                ph, pw = patch.shape[:2]
                if ph < 50 or pw < 50:
                    continue

                detections = self.predict_on_patch(patch, x_offset=x, y_offset=y)
                self.glomeruli.extend(detections)

                if save_patches and detections:
                    patch_vis = patch.copy()

                    for det in detections:
                        x1 = det["x1"] - x
                        y1 = det["y1"] - y
                        x2 = det["x2"] - x
                        y2 = det["y2"] - y
                        conf = det["conf"]

                        cv2.rectangle(patch_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(
                            patch_vis,
                            f"{conf:.2f}",
                            (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )

                    cv2.imwrite(
                        str(patches_dir / f"patch_x{x}_y{y}_det.jpg"),
                        cv2.cvtColor(patch_vis, cv2.COLOR_RGB2BGR)
                    )

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

        def iou(a, b):
            x1 = max(a["x1"], b["x1"])
            y1 = max(a["y1"], b["y1"])
            x2 = min(a["x2"], b["x2"])
            y2 = min(a["y2"], b["y2"])

            inter = max(0, x2 - x1) * max(0, y2 - y1)
            if inter == 0:
                return 0.0

            area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
            area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])

            return inter / (area_a + area_b - inter)

        detections = sorted(detections, key=lambda d: d["conf"], reverse=True)

        merged = []

        for det in detections:
            keep = True
            for kept in merged:
                if iou(det, kept) > iou_thresh:
                    keep = False
                    break
            if keep:
                merged.append(det)

        return merged