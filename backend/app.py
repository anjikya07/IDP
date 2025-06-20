from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from ocr import extract_text
from extractor import extract_fields
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route("/api/process", methods=["POST"])
def process_document():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filepath)

    raw_text = extract_text(filepath)
    structured_data = extract_fields(raw_text)

    return jsonify({
        "raw_text": raw_text,
        "extracted_fields": structured_data
    })

if __name__ == "__main__":
    app.run(debug=True)
