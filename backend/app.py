from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from ocr import extract_text
from extractor import extract_fields
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route("/favicon.ico")
def favicon():
    return '', 204

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for Render"""
    return jsonify({
        "status": "healthy", 
        "service": "invoice-ocr-api",
        "version": "1.0"
    }), 200

@app.route("/", methods=["GET"])
def root():
    """Root endpoint"""
    return jsonify({
        "message": "Invoice OCR API is running",
        "endpoints": {
            "process": "/api/process",
            "health": "/health"
        }
    }), 200

@app.route("/api/process", methods=["POST"])
def process_document():
    filepath = None  # Initialize filepath variable
    
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
        
        # Extract text using OCR (model will load on first use)
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
        # 🧹 FILE CLEANUP CODE GOES HERE
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
