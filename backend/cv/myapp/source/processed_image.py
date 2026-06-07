from .tissue_length_processor import TissueLengthProcessor
from .fibrosis_processor import FibrosisProcessor
from .glomeruli_processor import GlomeruliProcessor
from .mask import generate_mask
from pathlib import Path
from django.conf import settings



class ProcessedImage():
    MODEL_PATH = Path(settings.BASE_DIR) / "myapp" / "source" / "model" / "yolov8m_Ludzie01_classification_2026-05-31_oversample2x.pt"

    def __init__(self, path_tiff):
        self.path = Path(path_tiff)
        self.job_dir = self.path.parent  # slides/<job_id>
        self.mask_path = self.job_dir / f"{self.path.stem}_mask.tiff"

        self.glomeruli_fibrosis_classes = {} # Klasy zwłóknienia kłębuszków
        self.glomeruli = None                # Dynamiczna tablica na kłębuszki - 3 wymiary (wsp X, wsp Y, klasa)
        self.tissue_length = None            # Długość tkanki
        self.tissue_fibrosis_classe = {}     # Stopnie zwłóknienia tkanki

    def calculate_tissue_length(self):
        processor = TissueLengthProcessor(str(self.path), output_dir=self.job_dir)
        result = processor.process_image()
        self.tissue_length = result.get("length")
        return result

    def detect_glomeruli(self, conf=0.15, iou=0.3, imgsz=1024, tile_size=4000):
        mrxs_files = list(self.job_dir.glob("*.mrxs"))
        if not mrxs_files:
            raise FileNotFoundError(f"Brak pliku .mrxs w {self.job_dir}")

        processor = GlomeruliProcessor(
            path_mrxs=str(mrxs_files[0]),
            model_path=str(self.MODEL_PATH),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            tile_size=tile_size,
        )
        self.glomeruli = processor.detect_glomeruli() or []
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

    # Funkcja generująca maskę tkanki #### xx
    def generate_tissue_mask(self, mode="all", **kwargs):
        return generate_mask(str(self.path), mode=mode, **kwargs)
