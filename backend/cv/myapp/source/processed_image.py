from .tissue_length_processor import TissueLengthProcessor
from .fibrosis_processor import FibrosisProcessor
from .glomeruli_processor import GlomeruliProcessor
from .mask import generate_mask
from pathlib import Path
from django.conf import settings



class ProcessedImage():
    MODEL_PATH = Path(settings.BASE_DIR) / "myapp" / "source" / "model" / "best_100.pt"

    def __init__(self, path_tiff):
        self.path = Path(path_tiff)
        self.job_dir = self.path.parent  # slides/<job_id>
        self.mask_path = self.job_dir / f"{self.path.stem}_mask.tiff"

        self.glomeruli_fibrosis_classes = {}  # Glomeruli fibrosis classes
        self.glomeruli = None                  # Glomeruli detections (list of dicts)
        self.tissue_length = None              # Tissue length measurement
        self.tissue_fibrosis_classe = {}       # Tissue fibrosis degrees

    def calculate_tissue_length(self):
        processor = TissueLengthProcessor(str(self.path), output_dir=self.job_dir)
        result = processor.process_image()
        self.tissue_length = result.get("length")
        return result

    def detect_glomeruli(self, conf=0.5, iou=0.45, imgsz=1280, patch_size=512):
        if not self.mask_path.exists():
             self.generate_tissue_mask()

        processor = GlomeruliProcessor(
            path_tiff=str(self.path),
            model_path=str(self.MODEL_PATH),
            mask_path=str(self.mask_path) if self.mask_path.exists() else None,
            output_dir=str(self.job_dir),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            patch_size=patch_size,
        )
        self.glomeruli = processor.detect_glomeruli(save_patches=True) or []
        processor.save_annotated_image()
        return self.glomeruli


    def count_glomeruli(self):
        if self.glomeruli is None:
            self.detect_glomeruli()
        return len(self.glomeruli or [])

    
    def calculate_fibrosis_degree(self):
        """Analyze tissue fibrosis degree."""
        processor = FibrosisProcessor(str(self.path), output_dir=self.job_dir)
        result = processor.process_image()
        return result

    def generate_tissue_mask(self, mode="all", **kwargs):
        """Generate a tissue mask for the image."""
        return generate_mask(str(self.path), mode=mode, **kwargs)
