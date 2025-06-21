import logging
import os
import zipfile
import requests
import subprocess
import sys
from PIL import Image
import time
import re
import tempfile
import shutil
import gc

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

def get_memory_usage():
    """Get current memory usage for monitoring"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb
    except ImportError:
        return None

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
        
        # Install psutil for memory monitoring
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "psutil"
            ])
            logger.info("psutil installation completed")
        except:
            logger.warning("psutil installation failed, memory monitoring disabled")
        
        dependencies_installed = True
        elapsed = time.time() - start_time
        logger.info(f"ML dependencies installed successfully in {elapsed:.2f} seconds")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        raise RuntimeError(f"Dependency installation failed: {e}")

def extract_zip_ultra_conservative(zip_path, extract_path, max_memory_mb=300):
    """
    Ultra-conservative zip extraction with external process isolation
    Uses minimal memory by processing one file at a time with cleanup
    """
    logger.info(f"Starting ultra-conservative extraction to {extract_path}")
    logger.info(f"Memory limit: {max_memory_mb}MB")
    
    os.makedirs(extract_path, exist_ok=True)
    extracted_files = 0
    
    try:
        # First, get file list with minimal memory usage
        file_list = []
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = [info for info in zip_ref.infolist() if not info.is_dir()]
        
        total_files = len(file_list)
        logger.info(f"Zip contains {total_files} files to extract")
        
        # Process files one by one with aggressive memory management
        for i, file_info in enumerate(file_list):
            try:
                # Monitor memory before each file
                current_memory = get_memory_usage()
                if current_memory and current_memory > max_memory_mb:
                    logger.warning(f"Memory usage high before file {i+1}: {current_memory:.1f}MB")
                    # Force aggressive cleanup
                    gc.collect()
                    time.sleep(0.1)  # Brief pause for memory cleanup
                
                logger.info(f"Extracting file {i+1}/{total_files}: {file_info.filename}")
                
                # Create target directory if needed
                target_path = os.path.join(extract_path, file_info.filename)
                target_dir = os.path.dirname(target_path)
                if target_dir:
                    os.makedirs(target_dir, exist_ok=True)
                
                # Open zip file fresh for each file to minimize memory
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Use very small chunks and immediate cleanup
                    with zip_ref.open(file_info) as source:
                        with open(target_path, 'wb') as target:
                            chunk_size = 4096  # Even smaller 4KB chunks
                            chunks_processed = 0
                            
                            while True:
                                try:
                                    chunk = source.read(chunk_size)
                                    if not chunk:
                                        break
                                    target.write(chunk)
                                    chunks_processed += 1
                                    
                                    # Aggressive memory monitoring during large files
                                    if chunks_processed % 100 == 0:  # Every ~400KB
                                        current_memory = get_memory_usage()
                                        if current_memory and current_memory > max_memory_mb:
                                            logger.warning(f"Memory spike during extraction: {current_memory:.1f}MB")
                                            gc.collect()
                                            
                                except MemoryError:
                                    logger.error(f"Memory error during {file_info.filename}")
                                    raise
                
                extracted_files += 1
                
                # Cleanup after each file
                if extracted_files % 5 == 0:  # More frequent cleanup
                    gc.collect()
                    logger.info(f"Extracted {extracted_files}/{total_files} files")
                    
            except Exception as e:
                logger.error(f"Failed to extract {file_info.filename}: {e}")
                # Continue with other files, but log the error
                continue
        
        logger.info(f"Ultra-conservative extraction completed: {extracted_files}/{total_files} files")
        return extracted_files > 0
        
    except Exception as e:
        logger.error(f"Ultra-conservative extraction failed: {e}")
        return False

def try_system_unzip(zip_path, extract_path):
    """
    Try to use system unzip command if available (more memory efficient)
    """
    try:
        logger.info("Attempting system unzip command...")
        result = subprocess.run([
            'unzip', '-q', zip_path, '-d', extract_path
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info("System unzip successful")
            return True
        else:
            logger.warning(f"System unzip failed: {result.stderr}")
            return False
            
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
        logger.warning(f"System unzip not available or failed: {e}")
        return False

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
        
        # Download the file with streaming to avoid memory issues
        total_size = int(response.headers.get('content-length', 0))
        logger.info(f"Starting streaming download, size: {total_size} bytes")
        
        with open(ZIP_PATH, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Monitor memory during download
                    current_memory = get_memory_usage()
                    if current_memory and current_memory > 400:  # 400MB limit
                        logger.warning(f"High memory usage during download: {current_memory:.1f}MB")
        
        logger.info(f"Session bypass download completed: {downloaded} bytes")
        return downloaded > 1024 * 1024  # Must be at least 1MB
        
    except Exception as e:
        logger.error(f"Session bypass method failed: {e}")
        return False

def download_model():
    """Download model with enhanced memory management"""
    if os.path.exists(MODEL_DIR):
        logger.info(f"Model directory already exists: {MODEL_DIR}")
        return
        
    try:
        logger.info("=== STARTING MODEL DOWNLOAD ===")
        
        # Monitor initial memory
        initial_memory = get_memory_usage()
        if initial_memory:
            logger.info(f"Initial memory usage: {initial_memory:.1f}MB")
        
        os.makedirs("trocr_invoice", exist_ok=True)
        
        # Method 1: Try gdown first (most reliable for Google Drive)
        download_success = False
        if download_with_gdown():
            logger.info("Successfully downloaded using gdown")
            download_success = True
        else:
            # Method 2: Try session-based bypass
            logger.info("gdown failed, trying session bypass...")
            if download_with_session_bypass():
                logger.info("Successfully downloaded using session bypass")
                download_success = True
            else:
                # Method 3: Try original methods as fallback
                logger.info("Session bypass failed, trying original methods...")
                download_urls = [
                    f"https://drive.google.com/uc?id={DRIVE_FILE_ID}&export=download",
                    f"https://drive.google.com/uc?id={DRIVE_FILE_ID}&confirm=t",
                    f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
                ]
                
                for i, url in enumerate(download_urls):
                    try:
                        logger.info(f"Attempting original method {i+1}: {url}")
                        if _attempt_download(url):
                            download_success = True
                            break
                    except Exception as e:
                        logger.warning(f"Original method {i+1} failed: {e}")
        
        if not download_success:
            raise RuntimeError("All download methods failed")
        
        # Check memory before extraction
        pre_extract_memory = get_memory_usage()
        if pre_extract_memory:
            logger.info(f"Memory usage before extraction: {pre_extract_memory:.1f}MB")
        
        # Try multiple extraction methods in order of preference
        logger.info("Starting memory-optimized extraction...")
        extract_start = time.time()
        
        extraction_successful = False
        
        # Method 1: Try system unzip if available (most memory efficient)
        if try_system_unzip(ZIP_PATH, "trocr_invoice"):
            extraction_successful = True
            logger.info("Extraction successful using system unzip")
        else:
            # Method 2: Ultra-conservative Python extraction
            logger.info("System unzip failed, trying ultra-conservative Python extraction...")
            if extract_zip_ultra_conservative(ZIP_PATH, "trocr_invoice"):
                extraction_successful = True
                logger.info("Extraction successful using ultra-conservative method")
            else:
                # Method 3: Last resort - try to extract just essential files
                logger.warning("Ultra-conservative extraction failed, trying selective extraction...")
                extraction_successful = extract_essential_files_only(ZIP_PATH, "trocr_invoice")
        
        if not extraction_successful:
            raise RuntimeError("All extraction methods failed - model file may be too large for available memory")
        
        extract_time = time.time() - extract_start
        logger.info(f"Extraction complete in {extract_time:.2f} seconds")
        
        # Check memory after extraction
        post_extract_memory = get_memory_usage()
        if post_extract_memory:
            logger.info(f"Memory usage after extraction: {post_extract_memory:.1f}MB")
        
        # Verify extraction
        if os.path.exists(MODEL_DIR):
            model_files = os.listdir(MODEL_DIR)
            logger.info(f"Model directory created with {len(model_files)} files: {model_files}")
        else:
            logger.error(f"Model directory not found after extraction: {MODEL_DIR}")
            raise RuntimeError("Model extraction failed - directory not created")
        
        # Clean up zip file to free memory
        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)
            logger.info("Cleanup: zip file removed")
            # Force garbage collection after cleanup
            gc.collect()
        
        # Final memory check
        final_memory = get_memory_usage()
        if final_memory:
            logger.info(f"Final memory usage: {final_memory:.1f}MB")
        
        logger.info("=== MODEL DOWNLOAD COMPLETED ===")
        
    except Exception as e:
        logger.error(f"Model download failed: {e}")
        # Clean up partial download
        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)
        raise RuntimeError(f"Model download failed: {e}")

def _attempt_download(url):
    """Attempt to download from a specific URL with memory optimization"""
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
                    
                    # Monitor memory during download
                    if chunk_count % 1000 == 0:  # Check every ~8MB
                        current_memory = get_memory_usage()
                        if current_memory and current_memory > 400:  # 400MB limit
                            logger.warning(f"High memory usage: {current_memory:.1f}MB")
                            gc.collect()  # Force garbage collection
                    
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
        logger.error(f"Downloaded file too small ({actual_size} bytes)")
        raise RuntimeError("Downloaded file appears to be corrupted or incorrect")
    
    # Test if it's actually a zip file
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            logger.info(f"Zip file validated - contains {len(file_list)} files")
    except zipfile.BadZipFile as e:
        logger.error(f"Downloaded file is not a valid zip: {e}")
        raise RuntimeError(f"Downloaded file is not a valid zip file: {e}")
    
    logger.info("=== DOWNLOAD METHOD SUCCESSFUL ===")
    return True

def load_model():
    """Load model only when needed with memory optimization"""
    global processor, model, device
    
    if processor is not None and model is not None:
        logger.info("Model already loaded, skipping...")
        return
        
    try:
        logger.info("=== STARTING MODEL LOADING ===")
        start_time = time.time()
        
        # Monitor initial memory
        initial_memory = get_memory_usage()
        if initial_memory:
            logger.info(f"Initial memory usage: {initial_memory:.1f}MB")
        
        # First install dependencies
        install_ml_dependencies()
        
        # Now import the libraries (after installation)
        logger.info("Importing ML libraries...")
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        import torch
        logger.info("ML libraries imported successfully")
        
        # Download model if needed
        download_model()
        
        # Load processor with memory monitoring
        logger.info("Loading processor...")
        processor_start = time.time()
        processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
        processor_time = time.time() - processor_start
        
        processor_memory = get_memory_usage()
        if processor_memory:
            logger.info(f"Processor loaded in {processor_time:.2f} seconds, memory: {processor_memory:.1f}MB")
        else:
            logger.info(f"Processor loaded in {processor_time:.2f} seconds")
        
        # Load model with memory optimization
        logger.info("Loading model with memory optimization...")
        model_start = time.time()
        
        # Use lower precision and memory optimization
        model = VisionEncoderDecoderModel.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map=None  # Don't auto-assign device
        )
        
        model_time = time.time() - model_start
        
        model_memory = get_memory_usage()
        if model_memory:
            logger.info(f"Model loaded in {model_time:.2f} seconds, memory: {model_memory:.1f}MB")
        else:
            logger.info(f"Model loaded in {model_time:.2f} seconds")
        
        # Set device and move model
        logger.info("Setting up device...")
        device = torch.device("cpu")  # Force CPU
        model.to(device)
        model.eval()
        
        # Force garbage collection after model loading
        gc.collect()
        
        final_memory = get_memory_usage()
        if final_memory:
            logger.info(f"Model moved to device: {device}, final memory: {final_memory:.1f}MB")
        else:
            logger.info(f"Model moved to device: {device}")
        
        total_time = time.time() - start_time
        logger.info(f"=== MODEL LOADING COMPLETED in {total_time:.2f} seconds ===")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.exception("Full model loading traceback:")
        raise RuntimeError(f"Model initialization failed: {str(e)}")

def extract_text(filepath):
    """
    Extract text from image using OCR model with memory optimization
    
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
        
        # Monitor initial memory
        initial_memory = get_memory_usage()
        if initial_memory:
            logger.info(f"Initial memory usage: {initial_memory:.1f}MB")
        
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
        
        # Load and validate image with memory optimization
        try:
            logger.info("Loading image...")
            image_start = time.time()
            image = Image.open(filepath).convert("RGB")
            
            # Resize image if too large to save memory
            max_size = 2048  # Maximum dimension
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Image resized to {new_size} for memory optimization")
            
            image_time = time.time() - image_start
            logger.info(f"Image loaded successfully: {image.size} in {image_time:.2f}s")
            
            # Check memory after image loading
            image_memory = get_memory_usage()
            if image_memory:
                logger.info(f"Memory after image loading: {image_memory:.1f}MB")
                
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise ValueError(f"Invalid image file: {e}")
        
        # Process image through OCR model with memory monitoring
        try:
            logger.info("Processing image through OCR model...")
            
            # Preprocessing
            preprocess_start = time.time()
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            preprocess_time = time.time() - preprocess_start
            
            preprocess_memory = get_memory_usage()
            if preprocess_memory:
                logger.info(f"Image preprocessed in {preprocess_time:.2f}s, memory: {preprocess_memory:.1f}MB")
            else:
                logger.info(f"Image preprocessed in {preprocess_time:.2f}s, tensor shape: {pixel_values.shape}")
            
            # Clear image from memory
            del image
            gc.collect()
            
            # Model inference with memory optimization
            logger.info("Running model inference...")
            inference_start = time.time()
            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values, 
                    max_length=512,
                    num_beams=2,  # Reduced from 4 to save memory
                    early_stopping=True
                )
            inference_time = time.time() - inference_start
            
            # Clear pixel values from memory
            del pixel_values
            gc.collect()
            
            inference_memory = get_memory_usage()
            if inference_memory:
                logger.info(f"Model inference completed in {inference_time:.2f}s, memory: {inference_memory:.1f}MB")
            else:
                logger.info(f"Model inference completed in {inference_time:.2f}s")
            
            # Decode results
            logger.info("Decoding results...")
            decode_start = time.time()
            extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            decode_time = time.time() - decode_start
            
            # Clear generated_ids from memory
            del generated_ids
            gc.collect()
            
            logger.info(f"Text decoded in {decode_time:.2f}s")
            
            if not extracted_text.strip():
                logger.warning("OCR returned empty text")
                return "No text could be extracted from the image"
                
            total_time = time.time() - start_time
            final_memory = get_memory_usage()
            
            if final_memory:
                logger.info(f"=== OCR EXTRACTION SUCCESSFUL in {total_time:.2f}s, final memory: {final_memory:.1f}MB ===")
            else:
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
