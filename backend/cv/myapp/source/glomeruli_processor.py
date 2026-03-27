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
    ):
        self.path = Path(path_tiff)
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir) if output_dir else self.path.parent
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.patch_size = patch_size

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

    def detect_glomeruli(self) -> List[Dict[str, Any]]:
        img = self.load_image()
        h, w = img.shape[:2]
        self.glomeruli = []

        step = self.patch_size
        for y in range(0, h, step):
            for x in range(0, w, step):
                y_end = min(y + self.patch_size, h)
                x_end = min(x + self.patch_size, w)
                patch = img[y:y_end, x:x_end]

                ph, pw = patch.shape[:2]
                if ph < 50 or pw < 50:
                    continue

                self.glomeruli.extend(
                    self.predict_on_patch(patch, x_offset=x, y_offset=y)
                )

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
            label = f"glomerulus {conf:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

        out_path = out_path or str(self.output_dir / f"{self.path.stem}_glomeruli.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        return out_path
