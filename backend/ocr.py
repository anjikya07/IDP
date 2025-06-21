import logging
import os
import zipfile
import requests
import subprocess
import sys
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = "trocr_invoice/final_model"
ZIP_PATH = "trocr_invoice/final_model.zip"
DRIVE_FILE_ID = "1-mNc5xS1vb-0VLMuNcP4ehxZ2d6rezdO"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

# Global variables for lazy loading
processor = None
model = None
device = None
dependencies_installed = False

def install_ml_dependencies():
    """Install ML dependencies only when needed"""
    global dependencies_installed
    
    if dependencies_installed:
        return
        
    try:
        logger.info("Installing ML dependencies...")
        
        # Install PyTorch CPU version (much smaller)
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "--index-url", "https://download.pytorch.org/whl/cpu"
        ])
        
        # Install transformers
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "transformers"
        ])
        
        dependencies_installed = True
        logger.info("ML dependencies installed successfully")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        raise RuntimeError(f"Dependency installation failed: {e}")

def download_model():
    if not os.path.exists(MODEL_DIR):
        try:
            os.makedirs("trocr_invoice", exist_ok=True)
            logger.info("Downloading model from Google Drive...")
            
            # Use session for better connection handling
            with requests.Session() as session:
                response = session.get(DOWNLOAD_URL, stream=True, timeout=300)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(ZIP_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                if downloaded % (1024 * 1024 * 100) == 0:  # Log every 100MB
                                    logger.info(f"Downloaded {percent:.1f}%")
                        
            logger.info("Download complete. Extracting...")
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall("trocr_invoice")
            logger.info("Extraction complete.")
            
            # Clean up zip file
            if os.path.exists(ZIP_PATH):
                os.remove(ZIP_PATH)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download model: {e}")
            raise RuntimeError(f"Model download failed: {e}")
        except zipfile.BadZipFile as e:
            logger.error(f"Invalid zip file: {e}")
            raise RuntimeError(f"Model extraction failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during model setup: {e}")
            raise RuntimeError(f"Model setup failed: {e}")

def load_model():
    """Load model only when needed (lazy loading)"""
    global processor, model, device
    
    if processor is None or model is None:
        try:
            # First install dependencies
            install_ml_dependencies()
            
            # Now import the libraries (after installation)
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch
            
            logger.info("Loading OCR model...")
            download_model()  # Download if not exists
            
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
        # Load model only when first needed
        load_model()
        
        # Import torch here (after installation)
        import torch
        
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
