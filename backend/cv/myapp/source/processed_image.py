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


    # Funkcja wykrywająca kłębuszki
    def detect_glomeruli(self):
        number_of_glomeruli = 10 # placeholder - tutaj funkcja będzie podstawiała ilość kłębuszków
        self.glomeruli = [[None] * number_of_glomeruli for _ in range(3)]
        pass

    # Funkcja zliczająca wszystkie kłębuszki
    def count_glomeruli(self):
        if self.glomeruli is None:
            return 0
        return len(self.glomeruli[0])
    
    # Funkcja analizująca stopień zwłóknienia tkanki
    def calculate_fibrosis_degree(self):
        processor = FibrosisProcessor(str(self.path), output_dir=self.job_dir)
        result = processor.process_image()
        return result