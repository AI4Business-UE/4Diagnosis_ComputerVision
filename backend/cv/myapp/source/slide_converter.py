import uuid
from pathlib import Path
from .converter_tiff import SlideProcessor, save_result
from .mask import generate_mask


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
            print(mask_preview_path) # xx
        except Exception as e:
            # Log error but don't fail the conversion
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Mask generation failed: {str(e)}")
            mask_preview_path = None

        return job_id, tiff_path, mask_preview_path
