import uuid
from pathlib import Path
from .converter_tiff import SlideProcessor, save_result
from .mask import generate_mask

from PIL import Image
import cv2
import numpy as np


def crop_all_images(tiff_path, mask_path, mask_preview_path, margin=100):
    print("\n--- ROZPOCZYNAM GLOBALNE PRZYCINANIE ---")
    
    # 1. Wczytujemy plik maski (zakładamy czarne tło i jasną tkankę)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print("BLAD: Nie udalo sie wczytac maski!")
        return

    # 2. Upewniamy się, że szukamy tylko tkanki (odcinamy szumy z tła)
    _, binary_mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary_mask)
    
    if coords is None:
        print("BLAD: Maska jest zupelnie pusta!")
        return
        
    # 3. Wyliczamy wspólne współrzędne dla wszystkich plików
    x, y, w, h = cv2.boundingRect(coords)
    height, width = mask.shape
    
    x_start = max(0, x - margin)
    y_start = max(0, y - margin)
    x_end = min(width, x + w + margin)
    y_end = min(height, y + h + margin)
    
    print(f"Znalazlem tkanke! Tne do wspolrzednych: Y({y_start}:{y_end}), X({x_start}:{x_end})")

    # 4. Funkcja pomocnicza do ciecia i nadpisywania
    def crop_and_save(img_path):
        if not img_path: return
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is not None:
            cropped = img[y_start:y_end, x_start:x_end]
            cv2.imwrite(str(img_path), cropped)
            print(f"Przycieto plik: {img_path}")

    # 5. Tniemy wszystkie pliki używając tej samej 'foremki'!
    crop_and_save(tiff_path)
    crop_and_save(mask_path)
    crop_and_save(mask_preview_path)
    print("--- ZAKONCZONO PRZYCINANIE ---\n")

class SlideConverter:

    @staticmethod
    def convert_to_tiff(files, base_dir, user_lvl=5):
        job_id = str(uuid.uuid4())

        slides_root = Path(base_dir) / "slides"
        slides_root.mkdir(exist_ok=True)

        job_dir = slides_root / job_id
        job_dir.mkdir()

        mrxs_path = None

        # zapis MRXS
        for f in files:
            name = Path(f.name).name
            if name.lower().endswith(".mrxs"):

                mrxs_path = job_dir / name

                with open(mrxs_path, "wb") as out:
                    for chunk in f.chunks():
                        out.write(chunk)

        if not mrxs_path:
            raise Exception("MRXS missing")

        # folder z danymi slajdu
        data_dir = job_dir / mrxs_path.stem
        data_dir.mkdir()

        # zapis plików DATA / INI
        for f in files:

            name = Path(f.name).name

            if not name.lower().endswith(".mrxs"):

                target = data_dir / name

                with open(target, "wb") as out:
                    for chunk in f.chunks():
                        out.write(chunk)

        # naprawa Index.dat (częsty problem case-sensitive)
        for f in data_dir.iterdir():
            if f.name.lower() == "index.dat" and f.name != "Index.dat":
                f.rename(data_dir / "Index.dat")

        # walidacja MRXS
        if not (data_dir / "Index.dat").exists():
            raise Exception("Index.dat missing")

        if not list(data_dir.glob("Data*.dat")):
            raise Exception("Data*.dat missing")

        if not list(data_dir.glob("*.ini")):
            raise Exception("Slidedat.ini missing")

        # konwersja do TIFF
        processor = SlideProcessor(
            slide_path=str(mrxs_path),
            level=user_lvl,
            tile_size=1024,
            threshold=10,
            use_associated="auto"
        )

        result_img = processor.process()

        if result_img is None:
            raise Exception("TIFF conversion failed")

        tiff_path = job_dir / f"{mrxs_path.stem}.tiff"

        if not save_result(result_img, str(tiff_path)):
            raise Exception("TIFF save failed")
        
        
        # Generate tissue mask automatically
        try:
            mask_result = generate_mask(str(tiff_path), mode="all", visualize=False, save_mask=True, save_preview=True)
            mask_preview_path = mask_result.get("preview_path")
            mask_path = mask_result.get("mask_path") # Musimy wyciągnąć też mask_path do przycięcia!
            print(f"Preview path: {mask_preview_path}")
            
            # --- GLOBALNE PRZYCINANIE WSZYSTKICH ZDJĘĆ ---
            if mask_path:
                crop_all_images(tiff_path, mask_path, mask_preview_path, margin=100)
            # --------------------------------------------
            
        except Exception as e:
            # Log error but don't fail the conversion
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Mask generation failed: {str(e)}")
            mask_preview_path = None

        origin_detect_path = None
        if mask_preview_path:
            origin_detect_path = job_dir / f"{mrxs_path.stem}_origin_detect.tiff"
            try:
                with Image.open(str(mask_preview_path)) as im:
                    im.convert("RGB").save(str(origin_detect_path), format="TIFF")
            except Exception as e:
                raise Exception(f"Origin detect TIFF save failed: {e}")

        return job_id, tiff_path, mask_preview_path, origin_detect_path