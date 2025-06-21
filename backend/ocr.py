import logging
import os
from PIL import Image
import time
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment-based configuration for Render
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    logger.error("HF_TOKEN environment variable not set!")

# Hugging Face model configuration
HF_USERNAME = "anjikya07"
HF_MODEL_NAME = "trocr_model"
HF_MODEL_ID = f"{HF_USERNAME}/{HF_MODEL_NAME}"

# Global variables for lazy loading
processor = None
model = None
device = None

def get_memory_usage():
    """Get current memory usage for monitoring"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb
    except ImportError:
        return None

def setup_huggingface_auth():
    """Setup Hugging Face authentication"""
    try:
        from huggingface_hub import login
        if HF_TOKEN:
            logger.info("Setting up Hugging Face authentication...")
            login(token=HF_TOKEN)
            logger.info("Hugging Face authentication successful")
            return True
        else:
            logger.warning("No HF_TOKEN provided")
            return False
    except Exception as e:
        logger.error(f"Hugging Face authentication failed: {e}")
        return False

def load_model():
    """Load model from Hugging Face optimized for Render deployment"""
    global processor, model, device
    
    if processor is not None and model is not None:
        logger.info("Model already loaded, skipping...")
        return
        
    try:
        logger.info("=== STARTING MODEL LOADING FROM HUGGING FACE ===")
        start_time = time.time()
        
        # Monitor initial memory
        initial_memory = get_memory_usage()
        if initial_memory:
            logger.info(f"Initial memory usage: {initial_memory:.1f}MB")
        
        # Import libraries (pre-installed from requirements.txt)
        logger.info("Importing ML libraries...")
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        import torch
        logger.info("ML libraries imported successfully")
        
        # Setup Hugging Face authentication
        setup_huggingface_auth()
        
        # Use /tmp for cache on Render
        cache_dir = "/tmp/huggingface_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load processor with memory monitoring
        logger.info(f"Loading processor from Hugging Face model: {HF_MODEL_ID}")
        processor_start = time.time()
        try:
            processor = TrOCRProcessor.from_pretrained(
                HF_MODEL_ID,
                use_auth_token=HF_TOKEN,
                cache_dir=cache_dir
            )
        except Exception as e:
            logger.warning(f"Failed to load custom processor, trying base TrOCR processor: {e}")
            # Fallback to base TrOCR processor if custom one fails
            processor = TrOCRProcessor.from_pretrained(
                "microsoft/trocr-base-printed",
                cache_dir=cache_dir
            )
        
        processor_time = time.time() - processor_start
        processor_memory = get_memory_usage()
        if processor_memory:
            logger.info(f"Processor loaded in {processor_time:.2f} seconds, memory: {processor_memory:.1f}MB")
        else:
            logger.info(f"Processor loaded in {processor_time:.2f} seconds")
        
        # Load model with memory optimization for Render
        logger.info(f"Loading model from Hugging Face: {HF_MODEL_ID}")
        model_start = time.time()
        
        # Use lower precision and memory optimization for Render's limited resources
        try:
            model = VisionEncoderDecoderModel.from_pretrained(
                HF_MODEL_ID,
                use_auth_token=HF_TOKEN,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map=None,  # Don't auto-assign device
                cache_dir=cache_dir
            )
        except Exception as e:
            logger.warning(f"Failed to load custom model, trying base TrOCR model: {e}")
            model = VisionEncoderDecoderModel.from_pretrained(
                "microsoft/trocr-base-printed",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map=None,
                cache_dir=cache_dir
            )
        
        model_time = time.time() - model_start
        model_memory = get_memory_usage()
        if model_memory:
            logger.info(f"Model loaded in {model_time:.2f} seconds, memory: {model_memory:.1f}MB")
        else:
            logger.info(f"Model loaded in {model_time:.2f} seconds")
        
        # Set device and move model (CPU only on Render free tier)
        logger.info("Setting up device...")
        device = torch.device("cpu")
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
        logger.error(f"Failed to load model from Hugging Face: {str(e)}")
        logger.exception("Full model loading traceback:")
        raise RuntimeError(f"Model initialization failed: {str(e)}")

def extract_text(filepath):
    """
    Extract text from image using OCR model optimized for Render
    
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
        
        # Import torch here
        import torch
        
        # Load and validate image with memory optimization
        try:
            logger.info("Loading image...")
            image_start = time.time()
            image = Image.open(filepath).convert("RGB")
            
            # Resize image if too large to save memory (important for Render's limited RAM)
            max_size = 1536  # Reduced from 2048 for Render
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
            
            # Model inference with memory optimization for Render
            logger.info("Running model inference...")
            inference_start = time.time()
            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values, 
                    max_length=256,  # Reduced from 512 for faster processing
                    num_beams=2,     # Reduced beam search for memory
                    early_stopping=True,
                    do_sample=False  # Deterministic output
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

# Example usage function
def main():
    """Example usage of the OCR system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='OCR Text Extraction')
    parser.add_argument('image_path', help='Path to the image file')
    args = parser.parse_args()
    
    try:
        extracted_text = extract_text(args.image_path)
        print(f"\nExtracted Text:\n{extracted_text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
