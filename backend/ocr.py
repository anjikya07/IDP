import logging
import os
from PIL import Image
import time
import gc
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use public custom model
HF_MODEL_ID = "anjikya07/trocr_model"

# Globals
processor = None
model = None
device = None

def get_memory_usage():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return None

def load_model():
    global processor, model, device

    if processor is not None and model is not None:
        logger.info("Model already loaded.")
        return

    try:
        logger.info("=== LOADING PUBLIC MODEL ===")
        start_time = time.time()

        cache_dir = "/tmp/huggingface_cache"
        os.makedirs(cache_dir, exist_ok=True)

        try:
            processor = TrOCRProcessor.from_pretrained(HF_MODEL_ID, cache_dir=cache_dir)
            model = VisionEncoderDecoderModel.from_pretrained(
                HF_MODEL_ID,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map=None,
                cache_dir=cache_dir
            )
        except Exception as e:
            logger.warning(f"Failed to load custom model: {e}")
            logger.info("Falling back to microsoft/trocr-base-printed")
            processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", cache_dir=cache_dir)
            model = VisionEncoderDecoderModel.from_pretrained(
                "microsoft/trocr-base-printed",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map=None,
                cache_dir=cache_dir
            )

        device = torch.device("cpu")
        model.to(device)
        model.eval()

        gc.collect()
        logger.info(f"Model loaded in {time.time() - start_time:.2f}s")

    except Exception as e:
        logger.exception("Model loading failed")
        raise RuntimeError(f"Model initialization failed: {e}")

def extract_text(filepath):
    try:
        logger.info(f"=== OCR START for {filepath} ===")
        start_time = time.time()

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        load_model()

        image = Image.open(filepath).convert("RGB")
        max_size = 1536
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            image = image.resize(
                (int(image.width * ratio), int(image.height * ratio)),
                Image.Resampling.LANCZOS
            )

        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        del image
        gc.collect()

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_length=256,
                num_beams=2,
                early_stopping=True,
                do_sample=False
            )

        extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        del pixel_values, generated_ids
        gc.collect()

        logger.info(f"OCR done in {time.time() - start_time:.2f}s")
        return extracted_text if extracted_text.strip() else "No text could be extracted from the image"

    except Exception as e:
        logger.exception("OCR failed")
        raise RuntimeError(f"OCR failed: {e}")
