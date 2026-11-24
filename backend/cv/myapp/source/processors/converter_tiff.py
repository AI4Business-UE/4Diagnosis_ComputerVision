import logging
import psutil
from math import ceil
from typing import Optional, Tuple, List
from PIL import Image
import openslide


class MemoryMonitor:
    """
    Context manager to monitor memory usage.
    """
    def __init__(self, operation_name: str = "operation"):
        self.operation_name = operation_name
        self.start_memory = 0.0

    def __enter__(self):
        process = psutil.Process()
        self.start_memory = process.memory_info().rss / 1024 / 1024
        logging.info(f"Starting {self.operation_name} - Memory: {self.start_memory:.1f} MB")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        process = psutil.Process()
        end_memory = process.memory_info().rss / 1024 / 1024
        memory_change = end_memory - self.start_memory
        logging.info(f"Finished {self.operation_name}. Change: {memory_change:+.1f} MB")


def _has_bright_content(img: Image.Image, threshold: int) -> bool:
    """Fast low-memory brightness check via Pillow grayscale extrema (no float64 array)."""
    _, max_val = img.convert("L").getextrema()
    return max_val > threshold


class SlideProcessor:
    """
    Main slide processor.
    Contains tiling logic and fallback strategies.
    """

    def __init__(self, slide_path: str, level: int, tile_size: int, threshold: int, use_associated: str):
        self.slide_path = slide_path
        self.level = level
        self.tile_size = tile_size
        self.threshold = threshold
        self.use_associated = use_associated
        self.logger = logging.getLogger(self.__class__.__name__)
        self.min_x = 0
        self.min_y = 0

    def _get_associated_image(self, slide: openslide.OpenSlide) -> Optional[Image.Image]:
        """
        Retrieve an associated image (macro, label) as a fallback
        when the main tiling process fails.
        """
        if not slide.associated_images:
            return None

        priority = ["label", "macro", "thumbnail"]

        if self.use_associated == "auto":
            for img_type in priority:
                if img_type in slide.associated_images:
                    img = slide.associated_images[img_type]
                    if _has_bright_content(img, 10):
                        self.logger.info(f"Fallback: Using associated image '{img_type}'")
                        return img.convert("RGB")
        elif self.use_associated in slide.associated_images:
            self.logger.info(f"Fallback: Using requested associated image '{self.use_associated}'")
            return slide.associated_images[self.use_associated].convert("RGB")

        return None

    def _analyze_tiles(self, slide: openslide.OpenSlide, dims: Tuple[int, int]) -> Tuple[List, int, int, int, int]:
        """Analyze the tile grid looking for content."""
        level0_w, level0_h = slide.level_dimensions[0]
        bounds_x = int(slide.properties.get("openslide.bounds-x", 0))
        bounds_y = int(slide.properties.get("openslide.bounds-y", 0))
        bounds_width = int(slide.properties.get("openslide.bounds-width", level0_w))
        bounds_height = int(slide.properties.get("openslide.bounds-height", level0_h))

        downsample = slide.level_downsamples[self.level]
        l_x = int(bounds_x / downsample)
        l_y = int(bounds_y / downsample)
        l_w = int(bounds_width / downsample)
        l_h = int(bounds_height / downsample)

        start_col = max(0, l_x // self.tile_size)
        start_row = max(0, l_y // self.tile_size)
        end_col = min(ceil(dims[0] / self.tile_size), ceil((l_x + l_w) / self.tile_size))
        end_row = min(ceil(dims[1] / self.tile_size), ceil((l_y + l_h) / self.tile_size))

        valid_tiles = []
        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = 0, 0

        self.logger.info(f"Analyzing grid: {max(0, end_col - start_col)} cols x {max(0, end_row - start_row)} rows")

        for row in range(start_row, end_row):
            for col in range(start_col, end_col):
                x = col * self.tile_size
                y = row * self.tile_size
                w = min(self.tile_size, dims[0] - x)
                h = min(self.tile_size, dims[1] - y)
                if w <= 0 or h <= 0:
                    continue

                try:
                    tile = slide.read_region(
                        (int(round(x * downsample)), int(round(y * downsample))),
                        self.level,
                        (w, h),
                    )
                    try:
                        if _has_bright_content(tile, self.threshold):
                            valid_tiles.append((x, y, w, h))
                            min_x = min(min_x, x)
                            min_y = min(min_y, y)
                            max_x = max(max_x, x + w)
                            max_y = max(max_y, y + h)
                    finally:
                        tile.close()
                except Exception as e:
                    self.logger.debug(f"Skipping unreadable tile ({x}, {y}): {e}")

        if valid_tiles:
            return valid_tiles, int(min_x), int(min_y), int(max_x), int(max_y)
        return valid_tiles, 0, 0, 0, 0

    def _assemble_image(self, slide: openslide.OpenSlide, tiles: list, min_x: int, min_y: int, w: int, h: int) -> Image.Image:
        """Assemble the final image from valid tiles."""
        downsample = slide.level_downsamples[self.level]

        img = Image.new("RGB", (w, h), (0, 0, 0))
        for x, y, tw, th in tiles:
            region = slide.read_region(
                (int(round(x * downsample)), int(round(y * downsample))),
                self.level,
                (tw, th),
            )
            try:
                tile_rgb = region.convert("RGB")
                img.paste(tile_rgb, (x - min_x, y - min_y))
                tile_rgb.close()
            finally:
                region.close()
        return img

    def process(self) -> Optional[Image.Image]:
        """Main processing logic with error handling and fallback strategies."""
        try:
            with openslide.OpenSlide(self.slide_path) as slide:
                if self.level >= slide.level_count:
                    self.logger.warning(f"Level {self.level} unavailable. Max: {slide.level_count - 1}")
                    return self._get_associated_image(slide)

                dims = slide.level_dimensions[self.level]

                try:
                    with MemoryMonitor("Tile Analysis"):
                        tiles, min_x, min_y, max_x, max_y = self._analyze_tiles(slide, dims)

                    if tiles:
                        crop_w, crop_h = max_x - min_x, max_y - min_y

                        if crop_w * crop_h > 800_000_000:
                            self.logger.warning("Image is extremely large. Memory issues possible.")

                        self.min_x = min_x
                        self.min_y = min_y

                        with MemoryMonitor("Assembly"):
                            return self._assemble_image(slide, tiles, min_x, min_y, crop_w, crop_h)
                    else:
                        self.logger.warning("Tile analysis found no content.")

                except MemoryError:
                    self.logger.error("Out of memory during tiling.")

                self.logger.info("Attempting fallback: Full level read...")
                try:
                    with MemoryMonitor("Full Read"):
                        region = slide.read_region((0, 0), self.level, dims)
                        try:
                            full_img = region.convert("RGB")
                        finally:
                            region.close()
                        if _has_bright_content(full_img, self.threshold):
                            return full_img
                        full_img.close()
                except MemoryError:
                    self.logger.error("Out of memory during full read.")
                except Exception as e:
                    self.logger.error(f"Full read failed: {e}")

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
