from pathlib import Path
from ultralytics import YOLO

from .tissue_length_processor import TissueLengthProcessor
from .fibrosis_processor import FibrosisProcessor
from pathlib import Path

class ProcessedImage():

    def __init__(self, path_tiff):
        self.path = Path(path_tiff)
        self.job_dir = self.path.parent  # slides/<job_id>

        self.glomeruli_fibrosis_classes = {} # Klasy zwłóknienia kłębuszków
        self.glomeruli = None                # Dynamiczna tablica na kłębuszki - 3 wymiary (wsp X, wsp Y, klasa)
        self.tissue_length = None            # Długość tkanki
        self.tissue_fibrosis_classe = {}     # Stopnie zwłóknienia tkanki

    # Funkcja obliczająca długość tkanki
    def calculate_tissue_length(self):
        processor = TissueLengthProcessor(str(self.path), output_dir=self.job_dir)
        result = processor.process_image()

        self.tissue_length = result.get("length")
        return result


    def detect_glomeruli(self):
        result = process_tiff(str(self.path))
        return result

    def count_glomeruli(self):
        res = process_tiff(str(self.path))
        return res.get("found_count")

    
    # Funkcja analizująca stopień zwłóknienia tkanki
    def calculate_fibrosis_degree(self):
        processor = FibrosisProcessor(str(self.path), output_dir=self.job_dir)
        result = processor.process_image()
        return result