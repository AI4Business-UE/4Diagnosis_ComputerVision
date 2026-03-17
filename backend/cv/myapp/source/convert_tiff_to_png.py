import os
import logging
from PIL import Image

class TiffToPngConverter:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        Image.MAX_IMAGE_PIXELS = None
    
    def convert_tiff_file_to_png(self, tiff_path: str, output_path: str = None) -> bool:
        if not os.path.exists(tiff_path):
            self.logger.error(f"File {tiff_path} does not exist")
            return False
        
        if output_path is None:
            base_name = os.path.splitext(tiff_path)[0]
            output_path = f"{base_name}.png"

        try:
            self.logger.info(f"conversion of file {tiff_path} ")
            
            with Image.open(tiff_path) as img:

                if img.mode not in ('RGB', 'RGBA'):
                    self.logger.info(f"Converting image mode from {img.mode} to RGB.")
                    img = img.convert('RGB')
                    
                self.logger.info("Saving PNG file")
                img.save(output_path, format="PNG")
                
            self.logger.info(f"conversion completed successfully. Saved as: {output_path}")
            return True
            
        except Exception as e:

            self.logger.error(f"an unexpected error occurred during conversion of file {tiff_path}: {e}", exc_info=True)
            return False


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    converter = TiffToPngConverter()
    
    input_file = "image.tiff"
    
    success = converter.convert(input_file)
    
    if success:
        logging.getLogger(__name__).info("The entire process completed successfully.")
    else:
        logging.getLogger(__name__).warning("The process failed.")