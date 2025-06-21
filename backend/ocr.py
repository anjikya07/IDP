import logging
import os
import zipfile
import requests
import subprocess
import sys
from PIL import Image
import time
import re

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
        
        # Install gdown for Google Drive downloads
        logger.info("Installing gdown...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "gdown"
        ])
        logger.info("gdown installation completed")
        
        dependencies_installed = True
        elapsed = time.time() - start_time
        logger.info(f"ML dependencies installed successfully in {elapsed:.2f} seconds")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        raise RuntimeError(f"Dependency installation failed: {e}")

def download_with_gdown():
    """Download using gdown library which handles Google Drive better"""
    try:
        logger.info("Attempting download with gdown library...")
        import gdown
        
        # gdown can handle the Google Drive virus scan warning
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, ZIP_PATH, quiet=False)
        
        # Validate the downloaded file
        if os.path.exists(ZIP_PATH):
            file_size = os.path.getsize(ZIP_PATH)
            logger.info(f"gdown download successful, file size: {file_size} bytes")
            
            if file_size < 1024 * 1024:  # Less than 1MB
                logger.error(f"Downloaded file too small ({file_size} bytes)")
                return False
                
            # Test if it's a valid zip file
            try:
                with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    logger.info(f"Zip file validated - contains {len(file_list)} files")
                return True
            except zipfile.BadZipFile:
                logger.error("Downloaded file is not a valid zip")
                return False
        else:
            logger.error("gdown failed - file not created")
            return False
            
    except ImportError:
        logger.error("gdown not available")
        return False
    except Exception as e:
        logger.error(f"gdown method failed: {e}")
        return False

def download_with_session_bypass():
    """Attempt to bypass Google Drive virus scan using session cookies"""
    try:
        session = requests.Session()
        
        # First, get the initial page to extract confirmation token
        logger.info("Getting initial page for confirmation token...")
        response = session.get(f'https://drive.google.com/uc?id={DRIVE_FILE_ID}')
        
        # Look for confirmation token in various places
        confirm_token = None
        
        # Method 1: Look for confirm parameter in forms
        confirm_match = re.search(r'name="confirm"\s+value="([^"]+)"', response.text)
        if confirm_match:
            confirm_token = confirm_match.group(1)
            logger.info(f"Found confirmation token in form: {confirm_token}")
        
        # Method 2: Look for confirm in download links
        if not confirm_token:
            confirm_match = re.search(r'confirm=([a-zA-Z0-9_-]+)', response.text)
            if confirm_match:
                confirm_token = confirm_match.group(1)
                logger.info(f"Found confirmation token in link: {confirm_token}")
        
        # Method 3: Try common confirmation tokens
        if not confirm_token:
            for token in ['t', '1', 'yes']:
                logger.info(f"Trying confirmation token: {token}")
                test_response = session.head(
                    f'https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}&confirm={token}'
                )
                content_type = test_response.headers.get('content-type', '')
                if not content_type.startswith('text/html'):
                    confirm_token = token
                    logger.info(f"Working confirmation token found: {token}")
                    break
        
        if not confirm_token:
            logger.warning("No confirmation token found, trying without one...")
            confirm_token = 't'  # Default fallback
        
        # Download with confirmation token
        download_url = f'https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}&confirm={confirm_token}'
        logger.info(f"Downloading with confirmation: {download_url}")
        
        response = session.get(download_url, stream=True, timeout=600)
        
        # Check if we're still getting HTML
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' in content_type:
            logger.error("Still receiving HTML page after confirmation attempt")
            return False
        
        # Download the file
        total_size = int(response.headers.get('content-length', 0))
        logger.info(f"Starting download, size: {total_size} bytes")
        
        with open(ZIP_PATH, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        
        logger.info(f"Session bypass download completed: {downloaded} bytes")
        return downloaded > 1024 * 1024  # Must be at least 1MB
        
    except Exception as e:
        logger.error(f"Session bypass method failed: {e}")
        return False

def download_model():
    """Download model with enhanced Google Drive bypass methods"""
    if os.path.exists(MODEL_DIR):
        logger.info(f"Model directory already exists: {MODEL_DIR}")
        return
        
    try:
        logger.info("=== STARTING MODEL DOWNLOAD ===")
        os.makedirs("trocr_invoice", exist_ok=True)
        
        # Method 1: Try gdown first (most reliable for Google Drive)
        if download_with_gdown():
            logger.info("Successfully downloaded using gdown")
        else:
            # Method 2: Try session-based bypass
            logger.info("gdown failed, trying session bypass...")
            if download_with_session_bypass():
                logger.info("Successfully downloaded using session bypass")
            else:
                # Method 3: Try original methods as fallback
                logger.info("Session bypass failed, trying original methods...")
                download_urls = [
                    f"https://drive.google.com/uc?id={DRIVE_FILE_ID}&export=download",
                    f"https://drive.google.com/uc?id={DRIVE_FILE_ID}&confirm=t",
                    f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
                ]
                
                success = False
                for i, url in enumerate(download_urls):
                    try:
                        logger.info(f"Attempting original method {i+1}: {url}")
                        if _attempt_download(url):
                            success = True
                            break
                    except Exception as e:
                        logger.warning(f"Original method {i+1} failed: {e}")
                
                if not success:
                    raise RuntimeError("All download methods failed")
        
        # Extract the model
        logger.info("Starting extraction...")
        extract_start = time.time()
        
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            logger.info(f"Extracting {len(file_list)} files...")
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
        
        logger.info("=== MODEL DOWNLOAD COMPLETED ===")
        
    except Exception as e:
        logger.error(f"Model download failed: {e}")
        # Clean up partial download
        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)
        raise RuntimeError(f"Model download failed: {e}")

def _attempt_download(url):
    """Attempt to download from a specific URL (original method)"""
    start_time = time.time()
    
    with requests.Session() as session:
        logger.info(f"Making request to: {url}")
        response = session.get(url, stream=True, timeout=600)
        
        if response.status_code != 200:
            logger.error(f"Download failed with status code: {response.status_code}")
            raise requests.exceptions.RequestException(f"HTTP {response.status_code}")
        
        # Check if we got an HTML page (common with Google Drive errors)
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' in content_type:
            logger.error("Received HTML page instead of file - likely a Google Drive error page")
            # Log first 500 chars of response for debugging
            response_preview = response.text[:500] if hasattr(response, 'text') else "Cannot preview"
            logger.error(f"Response preview: {response_preview}")
            raise requests.exceptions.RequestException("Received HTML instead of file")
        
        total_size = int(response.headers.get('content-length', 0))
        logger.info(f"Download started, total size: {total_size} bytes ({total_size/1024/1024:.1f} MB)")
        
        # Validate file size (ML models should be at least 1MB)
        if total_size < 1024 * 1024:  # Less than 1MB
            logger.warning(f"File size suspiciously small: {total_size} bytes")
        
        downloaded = 0
        chunk_count = 0
        
        with open(ZIP_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    chunk_count += 1
                    
                    # Log progress every 10MB
                    if chunk_count % 1280 == 0:
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            logger.info(f"Downloaded {downloaded/1024/1024:.1f}MB ({percent:.1f}%)")
                        else:
                            logger.info(f"Downloaded {downloaded/1024/1024:.1f}MB")
    
    download_time = time.time() - start_time
    actual_size = os.path.getsize(ZIP_PATH)
    logger.info(f"Download complete in {download_time:.2f} seconds. File size: {actual_size} bytes")
    
    # Validate the downloaded file
    if actual_size < 1024 * 1024:  # Less than 1MB
        logger.error(f"Downloaded file too small ({actual_size} bytes) - likely not the actual model")
        # Try to read first few bytes to see what we got
        with open(ZIP_PATH, 'rb') as f:
            first_bytes = f.read(100)
            logger.error(f"First 100 bytes: {first_bytes}")
        raise RuntimeError("Downloaded file appears to be corrupted or incorrect")
    
    # Test if it's actually a zip file
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            logger.info(f"Zip file validated - contains {len(file_list)} files")
    except zipfile.BadZipFile as e:
        logger.error(f"Downloaded file is not a valid zip: {e}")
        # Log file contents for debugging
        with open(ZIP_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            content_preview = f.read(500)
            logger.error(f"File content preview: {content_preview}")
        raise RuntimeError(f"Downloaded file is not a valid zip file: {e}")
    
    logger.info("=== DOWNLOAD METHOD SUCCESSFUL ===")
    return True

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
