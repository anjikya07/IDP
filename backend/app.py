from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from ocr import extract_text
from extractor import extract_fields
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)  

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file siz

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route("/api/process", methods=["POST"])
def process_document():
    filepath = None  
    
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Check file extension
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
        if not ('.' in file.filename and 
                file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({"error": "Invalid file type. Please upload an image file."}), 400
        
        # Save file securely
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"Processing file: {filename}")
        
        # Extract text using OCR
        raw_text = extract_text(filepath)
        
        # Extract structured fields
        structured_data = extract_fields(raw_text)
        
        logger.info(f"Successfully processed file: {filename}")
        
        return jsonify({
            "raw_text": raw_text,
            "extracted_fields": structured_data
        })
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500
        
    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Cleaned up file: {filepath}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup file {filepath}: {str(cleanup_error)}")

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 16MB."}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
