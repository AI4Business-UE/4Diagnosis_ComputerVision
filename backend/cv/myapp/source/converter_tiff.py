import argparse
import gc
import logging
import os
import sys
import psutil
import numpy as np
from math import ceil
from pathlib import Path
from typing import Optional, Tuple, List
from PIL import Image
import openslide

# Sekcja naprawcza dla Windowsa (częsty problem z OpenSlide)
# if os.name == 'nt':
#     openslide_path = os.environ.get('OPENSLIDE_PATH')
#     if openslide_path and os.path.isdir(openslide_path):
#         os.add_dll_directory(openslide_path)

# try:
#     import openslide
# except ImportError:
#     print("CRITICAL: OpenSlide library not found. Install it via pip and system binaries.")
#     sys.exit(1)

# def setup_logging(level: str = "INFO") -> None:
#     """Setup logging configuration."""
#     logging.basicConfig(
#         level=getattr(logging, level.upper()),
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#         handlers=[logging.StreamHandler(sys.stdout)]
#     )

class MemoryMonitor:
    """
    Context manager to monitor memory usage.
    """
    def __init__(self, operation_name: str = "operation"):
        self.operation_name = operation_name
        self.start_memory = 0
        self.max_memory = 0

    def __enter__(self):
        process = psutil.Process()
        self.start_memory = process.memory_info().rss / 1024 / 1024
        self.max_memory = self.start_memory
        logging.info(f"Starting {self.operation_name} - Memory: {self.start_memory:.1f} MB")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        process = psutil.Process()
        end_memory = process.memory_info().rss / 1024 / 1024
        memory_change = end_memory - self.start_memory
        logging.info(f"Finished {self.operation_name}. Change: {memory_change:+.1f} MB")

class SlideProcessor:
    """
    Główna klasa przetwarzająca slajdy.
    Zawiera logikę kafelkowania oraz mechanizmy zapasowe (fallback).
    """

    def __init__(self, slide_path: str, level: int, tile_size: int, threshold: int, use_associated: str):
        self.slide_path = slide_path
        self.level = level
        self.tile_size = tile_size
        self.threshold = threshold
        self.use_associated = use_associated
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_associated_image(self, slide: openslide.OpenSlide) -> Optional[Image.Image]:
        """
        Pobiera obraz towarzyszący (macro, label) jeśli główny proces zawiedzie.
        Jest to strategia ratunkowa z oryginalnego kodu.
        """
        if not slide.associated_images:
            return None

        # Priorytet pobierania, jeśli tryb auto
        priority = ["label", "macro", "thumbnail"]
        
        if self.use_associated == "auto":
            for img_type in priority:
                if img_type in slide.associated_images:
                    img = slide.associated_images[img_type]
                    # Szybkie sprawdzenie czy obraz nie jest czarny
                    if np.max(np.mean(np.array(img.convert('RGB')), axis=2)) > 10:
                        self.logger.info(f"Fallback: Using associated image '{img_type}'")
                        return img.convert('RGB')
        elif self.use_associated in slide.associated_images:
            self.logger.info(f"Fallback: Using requested associated image '{self.use_associated}'")
            return slide.associated_images[self.use_associated].convert('RGB')
            
        return None

    def _analyze_tiles(self, slide: openslide.OpenSlide, dims: Tuple[int, int]) -> Tuple[List, int, int, int, int]:
        """
        Analizuje siatkę kafelków w poszukiwaniu treści.
        """
        bounds_x = int(slide.properties.get('openslide.bounds-x', 0))
        bounds_y = int(slide.properties.get('openslide.bounds-y', 0))
        bounds_width = int(slide.properties.get('openslide.bounds-width', dims[0]))
        bounds_height = int(slide.properties.get('openslide.bounds-height', dims[1]))
        
        downsample = slide.level_downsamples[self.level]
        l_x = int(bounds_x / downsample)
        l_y = int(bounds_y / downsample)
        l_w = int(bounds_width / downsample)
        l_h = int(bounds_height / downsample)

        start_col = l_x // self.tile_size
        start_row = l_y // self.tile_size
        end_col = ceil((l_x + l_w) / self.tile_size)
        end_row = ceil((l_y + l_h) / self.tile_size)

        valid_tiles = []
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = 0, 0

        self.logger.info(f"Analyzing grid: {end_col-start_col} cols x {end_row-start_row} rows")
        
        processed = 0
        for row in range(start_row, end_row):
            for col in range(start_col, end_col):
                x = col * self.tile_size
                y = row * self.tile_size
                w = min(self.tile_size, dims[0] - x)
                h = min(self.tile_size, dims[1] - y)

                try:
                    tile = slide.read_region((x, y), self.level, (w, h)).convert("RGB")
                    # Sprawdzenie progu jasności
                    if np.max(np.mean(np.array(tile), axis=2)) > self.threshold:
                        valid_tiles.append((x, y, w, h))
                        min_x = min(min_x, x)
                        min_y = min(min_y, y)
                        max_x = max(max_x, x + w)
                        max_y = max(max_y, y + h)
                    del tile
                except Exception:
                    pass # Ignorujemy błędy pojedynczych kafelków
                
                processed += 1
                if processed % 100 == 0:
                    gc.collect()
        if valid_tiles:
            return valid_tiles, int(min_x), int(min_y), int(max_x), int(max_y)
        else:
            return valid_tiles, 0, 0, 0, 0

    def _assemble_image(self, slide: openslide.OpenSlide, tiles: list, min_x: int, min_y: int, w: int, h: int) -> Image.Image:
        """Składa obraz z kafelków."""
        img = Image.new('RGB', (w, h), (0, 0, 0))
        for i, (x, y, tw, th) in enumerate(tiles):
            tile = slide.read_region((x, y), self.level, (tw, th)).convert("RGB")
            img.paste(tile, (x - min_x, y - min_y))
            del tile
            if i % 50 == 0: gc.collect()
        return img

    def crop_content(self, pil_img: Image.Image) -> Image.Image:
        """
        Przycina czarne ramki wokół obrazu.
        """
        try:
            arr = np.array(pil_img)
            mask = np.mean(arr, axis=2) > self.threshold
            coords = np.argwhere(mask)
            if coords.size == 0: return pil_img
            
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            return pil_img.crop((x0, y0, x1, y1))
        except Exception as e:
            self.logger.warning(f"Cropping failed: {e}")
            return pil_img

    def process(self) -> Optional[Image.Image]:
        """
        Główna logika z obsługą błędów i fallbackami.
        """
        try:
            with openslide.OpenSlide(self.slide_path) as slide:
                # 1. Sprawdzenie levelu
                if self.level >= slide.level_count:
                    self.logger.warning(f"Level {self.level} unavailable. Max: {slide.level_count-1}")
                    return self._get_associated_image(slide)

                dims = slide.level_dimensions[self.level]
                
                # 2. Próba główna: Kafelkowanie (Tiling)
                try:
                    with MemoryMonitor("Tile Analysis"):
                        tiles, min_x, min_y, max_x, max_y = self._analyze_tiles(slide, dims)
                    
                    if tiles:
                        crop_w, crop_h = max_x - min_x, max_y - min_y
                        
                        # Ostrzeżenie przed ogromnymi plikami
                        if crop_w * crop_h > 800000000: # ~800MP
                             self.logger.warning("Image is extremely large. Memory issues possible.")

                        with MemoryMonitor("Assembly"):
                            return self._assemble_image(slide, tiles, min_x, min_y, crop_w, crop_h)
                    else:
                        self.logger.warning("Tile analysis found no content.")

                except MemoryError:
                    self.logger.error("Out of memory during tiling.")
                
                # 3. Strategia zapasowa 1: Czytanie całego poziomu (Full Read)
                # Czasami kafelkowanie zawodzi, a odczyt całości (dla mniejszych leveli) działa.
                self.logger.info("Attempting fallback: Full level read...")
                try:
                    with MemoryMonitor("Full Read"):
                        full_img = slide.read_region((0, 0), self.level, dims).convert("RGB")
                        if np.max(np.mean(np.array(full_img), axis=2)) > self.threshold:
                            return full_img
                except MemoryError:
                    self.logger.error("Out of memory during full read.")
                except Exception as e:
                    self.logger.error(f"Full read failed: {e}")

                # 4. Strategia zapasowa 2: Associated Images
                # Ostatnia deska ratunku - pobranie np. zdjęcia etykiety lub makro
                self.logger.info("Attempting fallback: Associated images...")
                return self._get_associated_image(slide)

        except openslide.OpenSlideError as e:
            self.logger.error(f"OpenSlide file error: {e}")
            return None
        except Exception as e:
            self.logger.critical(f"Unexpected error: {e}", exc_info=True)
            return None

def save_result(image: Image.Image, path: str) -> bool:
    try:
        with MemoryMonitor("Saving"):
            image.save(path, format="TIFF", compression="tiff_lzw")
        logging.info(f"Saved: {path}")
        return True
    except Exception as e:
        logging.error(f"Save failed: {e}")
        return False
