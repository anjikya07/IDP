import logging
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)

logger = logging.getLogger(__name__)

# Load model and processor
try:
    logger.info("Loading TrOCR model and processor...")
    processor = TrOCRProcessor.from_pretrained("trocr_invoice")
    model = VisionEncoderDecoderModel.from_pretrained("trocr_invoice")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Model loaded successfully on device: {device}")
except Exception as e:
    logger.exception("Failed to load TrOCR model or processor.")
    raise RuntimeError("TrOCR model loading failed") from e

def extract_text(filepath):
    """
    Extracts text from a document image using the TrOCR model.

    Args:
        filepath (str): Path to the invoice image.

    Returns:
        str: The extracted text.
    """
    try:
        logger.info(f"Processing file: {filepath}")
        image = Image.open(filepath).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(pixel_values)

        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        logger.info(f"OCR extraction successful for file: {filepath}")
        return text

    except Exception as e:
        logger.exception(f"OCR failed for file: {filepath}")
        raise RuntimeError("OCR processing failed") from e
