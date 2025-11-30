import argparse
import configparser
import gc
import logging
import os
import shutil
import sys
import zipfile
from math import ceil
from pathlib import Path
from typing import Optional, Tuple

import gdown
import matplotlib.pyplot as plt
import numpy as np
import openslide
import psutil
from PIL import Image


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


class MemoryMonitor:
    """Context manager to monitor memory usage during processing."""

    def __init__(self, operation_name: str = "operation"):
        self.operation_name = operation_name
        self.start_memory = 0
        self.max_memory = 0
        self.memory_readings = []

    def __enter__(self):
        process = psutil.Process()
        self.start_memory = process.memory_info().rss / 1024 / 1024
        self.max_memory = self.start_memory
        self.memory_readings = [self.start_memory]
        logging.info(f"📊 Starting {self.operation_name} - Memory: {self.start_memory:.1f} MB")
        return self

    def update(self):
        """Update current memory reading."""
        process = psutil.Process()
        current_memory = process.memory_info().rss / 1024 / 1024
        self.max_memory = max(self.max_memory, current_memory)
        self.memory_readings.append(current_memory)

    def __exit__(self, exc_type, exc_val, exc_tb):
        process = psutil.Process()
        end_memory = process.memory_info().rss / 1024 / 1024

        self.max_memory = max(self.max_memory, end_memory)
        self.memory_readings.append(end_memory)

        avg_memory = sum(self.memory_readings) / len(self.memory_readings) if self.memory_readings else 0
        memory_increase = end_memory - self.start_memory

        logging.info(f"📈 {self.operation_name} completed:")
        logging.info(f"   🔄 Memory change: {memory_increase:+.1f} MB")
        logging.info(f"   📊 Average memory: {avg_memory:.1f} MB")
        logging.info(f"   🔺 Peak memory: {self.max_memory:.1f} MB")


def log_memory_usage(context: str = "Current"):
    """Log current memory usage."""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    logging.info(f"🧠 {context} memory usage: {memory_mb:.1f} MB")


def download_and_extract_dataset(url: str, extract_path: str, data_folder: str = "data") -> Optional[str]:
    """
    Download and extract dataset from Google Drive.

    Args:
        url: Google Drive URL or file ID
        extract_path: Path to extract the zip file
        data_folder: Name of the data folder

    Returns:
        Path to the data folder if successful, None otherwise
    """
    try:
        logging.info("Downloading dataset...")

        if 'drive.google.com' in url:
            file_id = url.split('id=')[-1].split('&')[0]
            download_url = f'https://drive.google.com/uc?id={file_id}'
        else:
            download_url = url

        zip_path = "dataset.zip"
        gdown.download(download_url, zip_path, quiet=False)

        logging.info("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        os.remove(zip_path)

        extract_full_path = Path(extract_path)
        data_path = None

        for item in extract_full_path.iterdir():
            if item.is_dir() and item.name != data_folder:
                if data_path:
                    logging.warning(f"Multiple folders found, using: {data_path.name}")
                    break
                data_path = item

        if data_path:
            new_data_path = extract_full_path / data_folder
            if new_data_path.exists():
                shutil.rmtree(new_data_path)
            data_path.rename(new_data_path)
            logging.info(f"Data extracted to: {new_data_path}")
            return str(new_data_path)
        else:
            logging.error("Could not find data folder in extracted files")
            return None

    except Exception as e:
        logging.error(f"Error downloading/extracting dataset: {e}")
        return None


def display_config_info(config_path: str, max_sections: int = 3, max_keys: int = 5) -> None:
    """Display configuration file information."""
    if not os.path.exists(config_path):
        logging.warning(f"Config file not found: {config_path}")
        return

    try:
        config = configparser.ConfigParser()
        with open(config_path, 'r', encoding='utf-8-sig') as f:
            config.read_file(f)

        logging.info("--- Config (first %d sections) ---", max_sections)
        for i, section in enumerate(list(config.sections())[:max_sections]):
            print(f"[{section}]")
            for key in list(config[section])[:max_keys]:
                print(f"  {key} = {config[section][key]}")
    except Exception as e:
        logging.error(f"Error reading config file: {e}")


def analyze_tiles_for_content(
    slide: openslide.OpenSlide,
    level: int,
    tile_size: int,
    threshold: int,
    dims: Tuple[int, int]
) -> Tuple[list, int, int, int, int]:
    """Analyze tiles to find content boundaries."""
    bounds_x = int(slide.properties.get('openslide.bounds-x', 0))
    bounds_y = int(slide.properties.get('openslide.bounds-y', 0))
    bounds_width = int(slide.properties.get('openslide.bounds-width', dims[0]))
    bounds_height = int(slide.properties.get('openslide.bounds-height', dims[1]))
    downsample = slide.level_downsamples[level]
    level_bounds_x = int(bounds_x / downsample)
    level_bounds_y = int(bounds_y / downsample)
    level_bounds_width = int(bounds_width / downsample)
    level_bounds_height = int(bounds_height / downsample)

    logging.info("Using bounds at level %d: (%d, %d) size %dx%d",
                level, level_bounds_x, level_bounds_y, level_bounds_width, level_bounds_height)

    start_col = level_bounds_x // tile_size
    start_row = level_bounds_y // tile_size
    end_col = ceil((level_bounds_x + level_bounds_width) / tile_size)
    end_row = ceil((level_bounds_y + level_bounds_height) / tile_size)

    cols = end_col - start_col
    rows = end_row - start_row
    total_tiles = cols * rows
    logging.info("Number of tiles in bounds: %d x %d = %d", cols, rows, total_tiles)

    tiles_with_content = []
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = 0, 0

    logging.info("Analyzing tiles (stage 1/3)...")
    tiles_processed = 0

    for row in range(start_row, end_row):
        for col in range(start_col, end_col):
            x = col * tile_size
            y = row * tile_size

            w = min(tile_size, dims[0] - x)
            h = min(tile_size, dims[1] - y)

            tile = slide.read_region((x, y), level, (w, h))
            tile = tile.convert("RGB")

            tile_arr = np.array(tile)
            gray = np.mean(tile_arr, axis=2)

            if np.max(gray) > threshold:
                tiles_with_content.append((x, y, w, h))
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)

            del tile, tile_arr, gray
            tiles_processed += 1

            if tiles_processed % 100 == 0:
                logging.info("Processed %d/%d tiles", tiles_processed, total_tiles)

    return tiles_with_content, min_x, min_y, max_x, max_y


def assemble_image_from_tiles(
    slide: openslide.OpenSlide,
    level: int,
    tiles_with_content: list,
    min_x: int,
    min_y: int,
    crop_width: int,
    crop_height: int
) -> Image.Image:
    """Assemble final image from tiles with content."""
    logging.info("Assembling image (stage 2/3)...")
    result_image = Image.new('RGB', (crop_width, crop_height), (0, 0, 0))

    for i, (x, y, w, h) in enumerate(tiles_with_content):
        tile = slide.read_region((x, y), level, (w, h))
        tile = tile.convert("RGB")
        result_image.paste(tile, (x - min_x, y - min_y))
        del tile

        if (i + 1) % 50 == 0:
            logging.info("Assembled %d/%d tiles", i + 1, len(tiles_with_content))

    return result_image


def crop_to_content(pil_img: Image.Image, threshold: int = 15) -> Image.Image:
    """
    Analizuje obraz, znajduje obszar z treścią (nie-czarne tło)
    i przycina go do tego obszaru.

    Args:
        pil_img: Obraz w formacie PIL.Image
        threshold: Próg jasności (0-255) do odróżnienia tła od treści

    Returns:
        Przycięty obraz w formacie PIL.Image
    """
    import numpy as np

    # Konwersja obrazu do tablicy NumPy w skali szarości
    img_arr = np.array(pil_img)
    gray = np.mean(img_arr, axis=2).astype(np.uint8)

    # Utworzenie maski dla pikseli, które nie są tłem
    mask = gray > threshold

    # Znalezienie współrzędnych obszaru z treścią
    coords = np.argwhere(mask)
    if coords.size == 0:
        return pil_img

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    cropped_pil_img = pil_img.crop((x0, y0, x1, y1))
    return cropped_pil_img


def get_best_associated_image(slide: openslide.OpenSlide, preference: str = "auto") -> Optional[Image.Image]:
    """
    Get the best available associated image from the slide.

    Args:
        slide: OpenSlide object
        preference: Preferred image type ("auto", "macro", "label", "thumbnail")

    Returns:
        PIL Image or None if no suitable image found
    """
    if not slide.associated_images:
        return None

    priority_order = ["label", "macro", "thumbnail"]

    if preference == "auto":
        for img_type in priority_order:
            if img_type in slide.associated_images:
                img = slide.associated_images[img_type]
                rgb_img = img.convert('RGB')
                arr = np.array(rgb_img)
                gray = np.mean(arr, axis=2)
                if np.max(gray) > 10:
                    logging.info(f"Using associated image: {img_type} ({img.size})")
                    return rgb_img
    else:
        if preference in slide.associated_images:
            img = slide.associated_images[preference]
            rgb_img = img.convert('RGB')
            logging.info(f"Using associated image: {preference} ({img.size})")
            return rgb_img

    return None


def process_slide_in_tiles(
    slide_path: str,
    level: int,
    tile_size: int = 2048,
    threshold: int = 15,
    use_associated: str = "auto"
) -> Optional[Image.Image]:
    """
    Process slide image in tiles to conserve memory.

    Args:
        slide_path: Path to the .mrxs file
        level: Zoom level (0=highest, higher=lower resolution)
        tile_size: Size of individual tile in pixels
        threshold: Threshold for background detection

    Returns:
        Cropped image as PIL.Image or None if no content found
    """
    try:
        with openslide.OpenSlide(slide_path) as slide:
            dims = slide.level_dimensions[level]
            logging.info("Level %d dimensions: %s", level, dims)

            # First pass: find content boundaries
            with MemoryMonitor("tile analysis"):
                tiles_with_content, min_x, min_y, max_x, max_y = analyze_tiles_for_content(
                    slide, level, tile_size, threshold, dims
                )

            if not tiles_with_content:
                # Strategy 2: Try reading entire level (like the notebook does)
                logging.warning("Tile analysis found no content, trying full level %d reading...", level)

                try:
                    with MemoryMonitor("full level reading"):
                        # Read entire level
                        full_image = slide.read_region((0, 0), level, dims)
                        full_image = full_image.convert("RGB")

                        # Check if the full image has any content
                        import numpy as np
                        arr = np.array(full_image)
                        gray = np.mean(arr, axis=2)
                        if np.max(gray) > threshold:
                            logging.info("Found content in full level, applying content cropping...")
                            cropped_image = crop_to_content(full_image, threshold)
                            logging.info("Cropped to: %s", cropped_image.size)
                            return cropped_image
                        else:
                            logging.warning("Full level image has no content above threshold")

                except Exception as e:
                    logging.warning("Full level reading failed: %s", e)

                # Strategy 3: Fallback to associated images
                logging.warning("Full level reading failed, trying associated images...")
                associated_image = get_best_associated_image(slide, use_associated)
                if associated_image:
                    logging.info("Successfully retrieved associated image")
                    return associated_image
                else:
                    logging.warning("No associated images available or they have no content!")
                    return None

            crop_width = max_x - min_x
            crop_height = max_y - min_y
            logging.info("Cropped dimensions: %d x %d", crop_width, crop_height)

            with MemoryMonitor("image assembly"):
                result_image = assemble_image_from_tiles(
                    slide, level, tiles_with_content, min_x, min_y, crop_width, crop_height
                )

            gc.collect()
            return result_image

    except Exception as e:
        logging.error("Error processing slide: %s", e)
        return None


def save_and_display_image(
    image: Image.Image,
    output_path: str,
    level: int,
    show_preview: bool = True,
    max_display_size: Tuple[int, int] = (1500, 1500)
) -> None:
    """Save the processed image and optionally display a preview."""
    try:
        logging.info("Saving image (stage 3/3)...")

        with MemoryMonitor("image saving"):
            if output_path.lower().endswith('.tiff') or output_path.lower().endswith('.tif'):
                image.save(output_path, format="TIFF", compression="tiff_lzw")
            else:
                image.save(output_path, quality=100)

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logging.info("✅ Saved: %s (%.2f MB)", output_path, file_size_mb)

        if show_preview:
            logging.info("Displaying preview...")
            display_img = image.copy()
            display_img.thumbnail(max_display_size)

            plt.figure(figsize=(12, 12))
            plt.imshow(display_img)
            plt.axis('off')
            plt.title(f'Level {level} - Cropped & Tiled')
            plt.show()

        del image
        if 'display_img' in locals():
            del display_img
        gc.collect()
        logging.info("✅ Memory freed.")

    except Exception as e:
        logging.error(f"Error saving/displaying image: {e}")


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Process medical slide images with tile-based processing"
    )
    parser.add_argument(
        "--input", "-i",
        help="Path to the .mrxs slide file"
    )
    parser.add_argument(
        "--level", "-l",
        type=int,
        default=3,
        help="Zoom level (0=highest resolution, higher=lower resolution)"
    )
    parser.add_argument(
        "--tile-size", "-t",
        type=int,
        default=2048,
        help="Tile size in pixels"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=15,
        help="Background detection threshold"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: level_{level}_cropped_tiled.tiff)"
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Don't show image preview"
    )
    parser.add_argument(
        "--download-url",
        help="Google Drive URL or file ID to download dataset"
    )
    parser.add_argument(
        "--extract-path",
        default=".",
        help="Path to extract downloaded dataset"
    )
    parser.add_argument(
        "--use-associated",
        choices=["auto", "macro", "label", "thumbnail"],
        default="auto",
        help="Use associated image instead of zoom levels (auto=choose best available)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    return parser


def handle_dataset_download(args: argparse.Namespace) -> Optional[str]:
    """Handle dataset download and return slide path."""
    if not args.download_url:
        return args.input

    data_path = download_and_extract_dataset(args.download_url, args.extract_path)
    if not data_path:
        logging.error("Failed to download/extract dataset")
        return None

    # Look for .mrxs file in data directory
    data_dir = Path(data_path)
    mrxs_files = list(data_dir.glob("*.mrxs"))
    if not mrxs_files:
        logging.error("No .mrxs file found in downloaded data")
        return None

    slide_path = str(mrxs_files[0])
    logging.info("Found slide file: %s", slide_path)

    # Check for config file
    config_path = data_dir / "Slidedat.ini"
    if config_path.exists():
        display_config_info(str(config_path))

    return slide_path


def validate_inputs(slide_path: str, args: argparse.Namespace) -> bool:
    """Validate input parameters."""
    if not slide_path or not os.path.exists(slide_path):
        logging.error("Slide file not found: %s", slide_path)
        return False

    if not args.output:
        args.output = f"level_{args.level}_cropped_tiled.tiff"

    return True



def main():
    """Main function."""
    parser = create_parser()
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run quick test with level 5 processing"
    )
    parser.add_argument(
        "--test-level",
        type=int,
        default=5,
        help="Level to use for testing (default: 5)"
    )

    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.test:
        return run_quick_test(test_level=args.test_level, show_preview=not args.no_preview)

    if not args.input and not args.download_url:
        parser.error("Either --input or --download-url must be provided")

    slide_path = handle_dataset_download(args)
    if not slide_path:
        return 1

    if not validate_inputs(slide_path, args):
        return 1

    logging.info("=" * 50)
    logging.info("PROCESSING SLIDE WITH TILES")
    logging.info("=" * 50)

    with MemoryMonitor("complete slide processing"):
        cropped_image = process_slide_in_tiles(
            slide_path,
            level=args.level,
            tile_size=args.tile_size,
            threshold=args.threshold,
            use_associated=args.use_associated
        )

    if cropped_image:
        save_and_display_image(
            cropped_image,
            args.output,
            args.level,
            show_preview=not args.no_preview
        )
        logging.info("✅ Processing completed successfully!")
        return 0
    else:
        logging.error("❌ Processing failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())