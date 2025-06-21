import logging
import os
import zipfile
import requests
import subprocess
import sys
from PIL import Image
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = "final_model"
ZIP_PATH = "final_model.zip"
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
        logger.info("ML dependencies already installed, skipping...")
        return
        
    try:
        logger.info("Installing ML dependencies...")
        start_time = time.time()
        
        # Install PyTorch CPU version (much smaller)
        logger.info("Installing PyTorch...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "--index-url", "https://download.pytorch.org/whl/cpu"
        ])
        logger.info("PyTorch installation completed")
        
        # Install transformers
        logger.info("Installing transformers...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "transformers"
        ])
        logger.info("Transformers installation completed")
        
        dependencies_installed = True
        elapsed = time.time() - start_time
        logger.info(f"ML dependencies installed successfully in {elapsed:.2f} seconds")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        raise RuntimeError(f"Dependency installation failed: {e}")

def download_model():
    """Download model with detailed progress logging"""
    if os.path.exists(MODEL_DIR):
        logger.info(f"Model directory already exists: {MODEL_DIR}")
        return
        
    try:
        logger.info("=== STARTING MODEL DOWNLOAD ===")
        os.makedirs("trocr_invoice", exist_ok=True)
        logger.info("Downloading model from Google Drive...")
        start_time = time.time()
        
        # Use session for better connection handling
        with requests.Session() as session:
            logger.info(f"Making request to: {DOWNLOAD_URL}")
            response = session.get(DOWNLOAD_URL, stream=True, timeout=600)  # 10 min timeout
            
            if response.status_code != 200:
                logger.error(f"Download failed with status code: {response.status_code}")
                logger.error(f"Response headers: {dict(response.headers)}")
                raise requests.exceptions.RequestException(f"HTTP {response.status_code}")
            
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            logger.info(f"Download started, total size: {total_size} bytes ({total_size/1024/1024:.1f} MB)")
            
            downloaded = 0
            chunk_count = 0
            
            with open(ZIP_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        chunk_count += 1
                        
                        # Log progress every 10MB
                        if chunk_count % 1280 == 0:  # 1280 * 8192 ≈ 10MB
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                logger.info(f"Downloaded {downloaded/1024/1024:.1f}MB ({percent:.1f}%)")
                            else:
                                logger.info(f"Downloaded {downloaded/1024/1024:.1f}MB")
                        
        download_time = time.time() - start_time
        logger.info(f"Download complete in {download_time:.2f} seconds. File size: {os.path.getsize(ZIP_PATH)} bytes")
        
        # Extract with progress
        logger.info("Starting extraction...")
        extract_start = time.time()
        
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            logger.info(f"Zip contains {len(file_list)} files")
            zip_ref.extractall("trocr_invoice")
            
        extract_time = time.time() - extract_start
        logger.info(f"Extraction complete in {extract_time:.2f} seconds")
        
        # Verify extraction
        if os.path.exists(MODEL_DIR):
            model_files = os.listdir(MODEL_DIR)
            logger.info(f"Model directory created with {len(model_files)} files: {model_files}")
        else:
            logger.error(f"Model directory not found after extraction: {MODEL_DIR}")
            raise RuntimeError("Model extraction failed - directory not created")
        
        # Clean up zip file
        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)
            logger.info("Cleanup: zip file removed")
            
        total_time = time.time() - start_time
        logger.info(f"=== MODEL DOWNLOAD COMPLETED in {total_time:.2f} seconds ===")
        
    except requests.exceptions.Timeout:
        logger.error("Model download timed out")
        raise RuntimeError("Model download timed out - please try again")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download model: {e}")
        raise RuntimeError(f"Model download failed: {e}")
    except zipfile.BadZipFile as e:
        logger.error(f"Invalid zip file: {e}")
        if os.path.exists(ZIP_PATH):
            logger.info(f"Removing corrupted zip file: {ZIP_PATH}")
            os.remove(ZIP_PATH)
        raise RuntimeError(f"Model extraction failed - corrupted download: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during model setup: {e}")
        logger.exception("Full traceback:")
        raise RuntimeError(f"Model setup failed: {e}")

def load_model():
    """Load model only when needed (lazy loading) with detailed logging"""
    global processor, model, device
    
    if processor is not None and model is not None:
        logger.info("Model already loaded, skipping...")
        return
        
    try:
        logger.info("=== STARTING MODEL LOADING ===")
        start_time = time.time()
        
        # First install dependencies
        install_ml_dependencies()
        
        # Now import the libraries (after installation)
        logger.info("Importing ML libraries...")
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        import torch
        logger.info("ML libraries imported successfully")
        
        # Download model if needed
        download_model()
        
        # Load processor
        logger.info("Loading processor...")
        processor_start = time.time()
        processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
        processor_time = time.time() - processor_start
        logger.info(f"Processor loaded in {processor_time:.2f} seconds")
        
        # Load model
        logger.info("Loading model...")
        model_start = time.time()
        model = VisionEncoderDecoderModel.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        model_time = time.time() - model_start
        logger.info(f"Model loaded in {model_time:.2f} seconds")
        
        # Set device and move model
        logger.info("Setting up device...")
        device = torch.device("cpu")  # Force CPU
        model.to(device)
        model.eval()
        logger.info(f"Model moved to device: {device}")
        
        total_time = time.time() - start_time
        logger.info(f"=== MODEL LOADING COMPLETED in {total_time:.2f} seconds ===")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.exception("Full model loading traceback:")
        raise RuntimeError(f"Model initialization failed: {str(e)}")

def extract_text(filepath):
    """
    Extract text from image using OCR model with detailed logging
    
    Args:
        filepath (str): Path to the image file
        
    Returns:
        str: Extracted text from the image
        
    Raises:
        RuntimeError: If OCR processing fails
    """
    try:
        logger.info(f"=== STARTING OCR EXTRACTION for {filepath} ===")
        start_time = time.time()
        
        # Validate file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        file_size = os.path.getsize(filepath)
        logger.info(f"File exists, size: {file_size} bytes")
        
        # Load model only when first needed
        logger.info("Loading model...")
        load_model()
        logger.info("Model loaded, starting image processing...")
        
        # Import torch here (after installation)
        import torch
        
        # Load and validate image
        try:
            logger.info("Loading image...")
            image_start = time.time()
            image = Image.open(filepath).convert("RGB")
            image_time = time.time() - image_start
            logger.info(f"Image loaded successfully: {image.size} in {image_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise ValueError(f"Invalid image file: {e}")
        
        # Process image through OCR model
        try:
            logger.info("Processing image through OCR model...")
            
            # Preprocessing
            preprocess_start = time.time()
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            preprocess_time = time.time() - preprocess_start
            logger.info(f"Image preprocessed in {preprocess_time:.2f}s, tensor shape: {pixel_values.shape}")
            
            # Model inference
            logger.info("Running model inference...")
            inference_start = time.time()
            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values, 
                    max_length=512,
                    num_beams=4,
                    early_stopping=True
                )
            inference_time = time.time() - inference_start
            logger.info(f"Model inference completed in {inference_time:.2f}s")
            
            # Decode results
            logger.info("Decoding results...")
            decode_start = time.time()
            extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            decode_time = time.time() - decode_start
            logger.info(f"Text decoded in {decode_time:.2f}s")
            
            if not extracted_text.strip():
                logger.warning("OCR returned empty text")
                return "No text could be extracted from the image"
                
            total_time = time.time() - start_time
            logger.info(f"=== OCR EXTRACTION SUCCESSFUL in {total_time:.2f}s ===")
            logger.info(f"Extracted text length: {len(extracted_text)} characters")
            logger.info(f"First 100 chars: {extracted_text[:100]}...")
            
            return extracted_text
            
        except torch.cuda.OutOfMemoryError:
            logger.error("GPU out of memory during OCR processing")
            raise RuntimeError("Image too large for processing. Please try a smaller image.")
        except Exception as e:
            logger.error(f"OCR model processing failed: {e}")
            logger.exception("OCR processing traceback:")
            raise RuntimeError(f"OCR processing failed: {e}")
            
    except Exception as e:
        logger.exception("OCR processing failed with unexpected error")
        if isinstance(e, (FileNotFoundError, ValueError, RuntimeError)):
            raise e
        else:
            raise RuntimeError(f"OCR failed: {str(e)}")
