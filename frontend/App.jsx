// App.jsx
import React, { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [fields, setFields] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [darkMode, setDarkMode] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setFields(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      setFile(droppedFile);
      setPreviewUrl(URL.createObjectURL(droppedFile));
      setFields(null);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragActive(true);
  };

  const handleDragLeave = () => {
    setDragActive(false);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return alert("Please select an image file.");

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setFields(null);

const API_URL = process.env.REACT_APP_API_BASE || "http://localhost:5000";

const response = await fetch(`${API_URL}/api/process`, {
  method: "POST",
  body: formData,
});

      const data = await response.json();
      setFields(data.extracted_fields || {});
    } catch (err) {
      console.error(err);
      alert("Failed to connect to backend.");
    } finally {
      setLoading(false);
    }
  };

  const containerStyle = {
    minHeight: "100vh",
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
    padding: "2em",
    background: darkMode ? "#1a1a1a" : "#f7f9fc",
    color: darkMode ? "#f0f0f0" : "#000",
    fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
    transition: "all 0.3s ease"
  };

  const cardStyle = {
    width: "100%",
    maxWidth: "700px",
    background: darkMode ? "#2c2c2c" : "white",
    borderRadius: "16px",
    boxShadow: darkMode ? "0 20px 40px rgba(255,255,255,0.05)" : "0 20px 40px rgba(0, 0, 0, 0.1)",
    padding: "2em",
    textAlign: "center",
    animation: "fadeIn 1s ease-in-out"
  };

  const titleStyle = {
    fontSize: "2.5em",
    color: darkMode ? "#f0f0f0" : "#2c3e50",
    marginBottom: "0.5em"
  };

  const subtitleStyle = {
    fontSize: "1em",
    color: darkMode ? "#bbb" : "#555",
    marginBottom: "1.5em"
  };

  const dropzoneStyle = {
    padding: "2em",
    border: `2px dashed ${dragActive ? '#2980b9' : '#ccc'}`,
    borderRadius: "10px",
    background: dragActive ? "#e0f7fa" : darkMode ? "#333" : "#f9f9f9",
    cursor: "pointer",
    marginBottom: "1em",
    transition: "background 0.3s, border-color 0.3s"
  };

  const previewStyle = {
    maxWidth: "100%",
    maxHeight: "200px",
    borderRadius: "8px",
    marginTop: "1em",
    boxShadow: "0 4px 10px rgba(0, 0, 0, 0.1)",
    animation: "fadePreview 0.5s ease"
  };

  const buttonStyle = {
    padding: "12px 24px",
    background: "#3498db",
    border: "none",
    color: "white",
    borderRadius: "8px",
    cursor: "pointer",
    fontSize: "1em",
    marginTop: "1em",
    transition: "background 0.3s ease, transform 0.2s ease"
  };

  const resultStyle = {
    textAlign: "left",
    marginTop: "2em",
    background: darkMode ? "#444" : "#fdfdfd",
    padding: "1em",
    borderRadius: "10px",
    border: darkMode ? "1px solid #666" : "1px solid #ddd",
    boxShadow: "0 4px 10px rgba(0,0,0,0.05)",
    transition: "all 0.5s ease-in-out"
  };

  const fieldCard = {
    background: darkMode ? "#333" : "#fff",
    marginBottom: "0.8em",
    padding: "0.75em 1em",
    borderRadius: "8px",
    boxShadow: "0 2px 6px rgba(0,0,0,0.05)",
    border: "1px solid #ddd"
  };

  const spinnerStyle = {
    marginTop: "1em",
    width: "40px",
    height: "40px",
    border: "4px solid #eaf6ff",
    borderTop: "4px solid #3498db",
    borderRadius: "50%",
    animation: "spin 1s linear infinite",
    display: "inline-block"
  };

  return (
    <div style={containerStyle}>
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        @keyframes fadePreview {
          from { opacity: 0; transform: scale(0.95); }
          to { opacity: 1; transform: scale(1); }
        }
        button:hover {
          background: #2980b9;
          transform: translateY(-2px);
        }
      `}</style>

      <div style={cardStyle}>
        <h1 style={titleStyle}>📄 Invoice OCR Extractor</h1>
        <p style={subtitleStyle}>This tool is designed exclusively for extracting data from <strong>invoice images</strong>.</p>
        <button style={{ ...buttonStyle, marginBottom: "1em", background: darkMode ? "#222" : "#555" }} onClick={() => setDarkMode(!darkMode)}>
          Toggle {darkMode ? "Light" : "Dark"} Mode
        </button>

        <form onSubmit={handleSubmit}>
          <div
            style={dropzoneStyle}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => document.getElementById("file-input").click()}
          >
            {file ? <p>📄 {file.name}</p> : <p>📤 Drag & drop an invoice image here, or click to select a file</p>}
          </div>
          <input
            id="file-input"
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            style={{ display: "none" }}
          />
          {previewUrl && <img src={previewUrl} alt="Preview" style={previewStyle} />}
          <button type="submit" style={buttonStyle}>Upload & Extract</button>
        </form>

        {loading && <div style={spinnerStyle}></div>}

        {fields && (
          <div style={resultStyle}>
            <h2>✅ Extracted Fields:</h2>
            {Object.entries(fields).map(([key, value]) => (
              <div key={key} style={fieldCard}>
                <strong>{key}:</strong> {value}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
