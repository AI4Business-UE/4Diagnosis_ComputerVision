from pathlib import Path
from ultralytics import YOLO

from .tissue_length_processor import TissueLengthProcessor
from .fibrosis_processor import FibrosisProcessor
from .glomeruli_processor import GlomeruliProcessor
from .mask import generate_mask
from pathlib import Path
from django.conf import settings



class ProcessedImage():
    MODEL_PATH = Path(settings.BASE_DIR) / "myapp" / "source" / "model" / "best_100.pt"

    def __init__(self, path_tiff):
        self.path = Path(path_tiff)
        self.job_dir = self.path.parent  # slides/<job_id>

        self.glomeruli_fibrosis_classes = {} # Klasy zwłóknienia kłębuszków
        self.glomeruli = None                # Dynamiczna tablica na kłębuszki - 3 wymiary (wsp X, wsp Y, klasa)
        self.tissue_length = None            # Długość tkanki
        self.tissue_fibrosis_classe = {}     # Stopnie zwłóknienia tkanki

    def calculate_tissue_length(self):
        processor = TissueLengthProcessor(str(self.path), output_dir=self.job_dir)
        result = processor.process_image()
        self.tissue_length = result.get("length")
        return result

    def detect_glomeruli(self, conf=0.1, iou=0.45, imgsz=1024, patch_size=1024):
        processor = GlomeruliProcessor(
        path_tiff=str(self.path),
        model_path=str(self.MODEL_PATH),
        output_dir=str(self.job_dir),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        patch_size=patch_size,
    )
        self.glomeruli = processor.detect_glomeruli() or []
        processor.save_annotated_image()
        return self.glomeruli


    def count_glomeruli(self):
        if self.glomeruli is None:
            self.detect_glomeruli()
        return len(self.glomeruli or [])

    
    # Funkcja analizująca stopień zwłóknienia tkanki
    def calculate_fibrosis_degree(self):
        processor = FibrosisProcessor(str(self.path), output_dir=self.job_dir)
        result = processor.process_image()
        return result
