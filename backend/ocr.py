import logging
import os
import zipfile
import requests
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = "trocr_invoice/final_model"
ZIP_PATH = "trocr_invoice/final_model.zip"
DRIVE_FILE_ID = "1-mNc5xS1vb-0VLMuNcP4ehxZ2d6rezdO"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

def download_model():
    if not os.path.exists(MODEL_DIR):
        try:
            os.makedirs("trocr_invoice", exist_ok=True)
            logger.info("Downloading model from Google Drive...")
            
            with requests.get(DOWNLOAD_URL, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(ZIP_PATH, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
            logger.info("Download complete. Extracting...")
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall("trocr_invoice")
            logger.info("Extraction complete.")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download model: {e}")
            raise RuntimeError(f"Model download failed: {e}")
        except zipfile.BadZipFile as e:
            logger.error(f"Invalid zip file: {e}")
            raise RuntimeError(f"Model extraction failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during model setup: {e}")
            raise RuntimeError(f"Model setup failed: {e}")

# Download and load model
try:
    download_model()
    processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Model loaded successfully on {device}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise RuntimeError(f"Model initialization failed: {e}")

def extract_text(filepath):
    """
    Extract text from image using OCR model
    
    Args:
        filepath (str): Path to the image file
        
    Returns:
        str: Extracted text from the image
        
    Raises:
        RuntimeError: If OCR processing fails
    """
    try:
        # Validate file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Load and validate image
        try:
            image = Image.open(filepath).convert("RGB")
            logger.info(f"Image loaded successfully: {image.size}")
        except Exception as e:
            raise ValueError(f"Invalid image file: {e}")
        
        # Process image through OCR model
        try:
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(pixel_values, max_length=512)
                
            extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            if not extracted_text.strip():
                logger.warning("OCR returned empty text")
                return "No text could be extracted from the image"
                
            logger.info(f"OCR extraction successful. Text length: {len(extracted_text)}")
            return extracted_text
            
        except torch.cuda.OutOfMemoryError:
            logger.error("GPU out of memory during OCR processing")
            raise RuntimeError("Image too large for processing. Please try a smaller image.")
        except Exception as e:
            logger.error(f"OCR model processing failed: {e}")
            raise RuntimeError(f"OCR processing failed: {e}")
            
    except Exception as e:
        logger.exception("OCR processing failed with unexpected error")
        if isinstance(e, (FileNotFoundError, ValueError, RuntimeError)):
            raise e
        else:
            raise RuntimeError(f"OCR failed: {str(e)}")
