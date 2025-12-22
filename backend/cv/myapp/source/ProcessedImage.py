class ProcessedImage(Image):

    def __init__(self, image):
        super().__init__(image.image_id, image.file_path, image.size, image.value)

        self.glomeruli_fibrosis_classes = {} # Klasy zwłóknienia kłębuszków
        self.glomeruli = None                # Dynamiczna tablica na kłębuszki - 3 wymiary (wsp X, wsp Y, klasa)
        self.tissue_length = None            # Długość tkanki
        self.tissue_fibrosis_classe = {}     # Stopnie zwłóknienia tkanki

    # Funkcja obliczająca długość tkanki
    def calculate_tissue_length(self):
        pass

    # Funkcja wykrywająca kłębuszki
    def detect_glomeruli(self):
        number_of_glomeruli = 10 # placeholder - tutaj funkcja będzie podstawiała ilość kłębuszków
        self.glomeruli = [[None] * number_of_glomeruli for _ in range(3)]
        pass

    # Funkcja analizująca stopień zwłóknienia kłębuszka
    def classify_glomeruli(self):
        pass

    # Funkcja zliczająca wszystkie kłębuszki
    def count_glomeruli(self):
        if self.glomeruli is None:
            return 0
        return len(self.glomeruli[0])
    
    # Funkcja analizująca stopień zwłóknienia tkanki
    def calculate_fibrosis_degree(self):
        pass