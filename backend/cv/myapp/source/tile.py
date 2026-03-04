import os
from PIL import Image
import numpy as np
import logging


logger = logging.getLogger(__name__)

class TileProcessor:
    """klasa do dzielenia duzych obrazow TIFF na mniejsze kafelki
    z overlapowaniem i pomijaniem pustych fragmentow"""

    def __init__(self, tile_size=512, overlap=64, empty_threshold=0.9):
        self.tile_size = tile_size  #rozmiar kafelka w pixelach
        self.overlap = overlap      #ile pixeli sie naklada
        self.empty_threshold = empty_threshold  #jaka czesc kafelka musi byc zapelniona zeby nie byl "pusty"

    def process(self, input_path: str, output_dir: str, prefix="tile"):
        """wczytuje obraz, dzieli go na kafelki i zapisuje do output_dir.
        :param input_path: sciezka do obrazu TIFF
        :param output_dir: folder docelowy
        :param prefix: prefix nazwy pliku kafelka"""

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            img = Image.open(input_path).convert("RGB")
        except Exception as e:
            logger.error(f"Nie można otworzyć obrazu {input_path}: {e}")
            return

        width, height = img.size
        step = self.tile_size - self.overlap #o ile przesuwamy kafelki
        tile_id = 0

        for y in range(0, height, step):
            for x in range(0, width, step):
                x_end = min(x + self.tile_size, width)
                y_end = min(y + self.tile_size, height)
                tile = img.crop((x, y, x_end, y_end)) #wycinanie kafelka

                #sprawdzenie czy jest pusty
                arr = np.array(tile.convert("L")) / 255.0 #na szare kolory
                bright_ratio = np.mean(arr > 0.9) #sprawdzenie ile bialych pixeli
                if bright_ratio > self.empty_threshold: #jesli w wiekszosci pusty
                    continue

                #zapis kafelka
                tile_path = os.path.join(output_dir, f"{prefix}_{tile_id:06d}.png")

                try:    
                    tile.save(tile_path)
                except Exception as e:
                    logger.error(f"Błąd przy zapisie kafelka {tile_path}: {e}")
                    continue

                tile_id += 1

        logger.info(f"Zapisano {tile_id} kafelkow w {output_dir}")

#uzycie
if __name__ == "__main__":
    processor = TileProcessor(tile_size=512, overlap=64)
    processor.process("skan.tiff", "tiles_output")