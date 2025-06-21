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
        os.makedirs("trocr_invoice", exist_ok=True)
        logger.info("Downloading model from Google Drive...")
        with requests.get(DOWNLOAD_URL, stream=True) as r:
            r.raise_for_status()
            with open(ZIP_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info("Download complete. Extracting...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall("trocr_invoice")
        logger.info("Extraction complete.")

download_model()

# Load model
processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def extract_text(filepath):
    try:
        image = Image.open(filepath).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    except Exception as e:
        logger.exception("OCR processing failed.")
        raise RuntimeError("OCR failed") from e
