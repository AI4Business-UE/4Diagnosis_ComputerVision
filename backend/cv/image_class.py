class Image:
    """Klasa reprezentuje zdjęcie histopatologiczne. 
       Obiekt `Image` jest tworzony podczas etapu uploadu i przekazywany dalej do komponentów takich jak:

       - AnalysisService — inicjacja procesu analitycznego.
       - ProcessedImage — rozszerzona wersja obrazu po przeróbce.

        Na tym etapie klasa zawiera tylko atrybuty i puste metody"""
    
    def __init__(self, id = str, file_path = str, size =int, value: dict = None ):
        """Parametry:
        - id: Unikalny identyfikator zdjęcia w systemie.
        - file_path: Ścieżka do pliku.
        - size: Rozmiar pliku.
        - value: Dodatkowe metadane.
        """
        self.id = id
        self.file_path = file_path
        self.size = size
        self.value = value or {}

    def otrzymaj(self):
        """Pobieranie zdjęcia"""
        pass

    def przerób(self):
        """Metoda odpowiedzialna za preprocessing obrazu przed analizą."""
        pass
    def wyślij(self):
        """Przekazuje obraz do kolejnego modułu przetwarzania"""
        pass