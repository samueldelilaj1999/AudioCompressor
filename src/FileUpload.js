import React, { useState } from "react";
import { convertTo24kHz } from "./ffmpegUtils";
import { encryptAudioWithOnnx } from "./onnxAudioEncoder";
import "./FileUpload.css";

const FileUpload = () => {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);

  const handleFileChange = (event) => {
    const uploadedFile = event.target.files[0];
    if (uploadedFile) {
      setFile(uploadedFile);
      setStatus("File selected: " + uploadedFile.name);
    }
  };

  const startProcessing = async () => {
    if (!file) {
      setStatus("Please upload a file first.");
      return;
    }

    setIsProcessing(true);
    try {
      setStatus("Converting audio to 24kHz...");
      const tensorData = await convertTo24kHz(file);

      setStatus("Encoding audio with ONNX...");
      await encryptAudioWithOnnx(tensorData, file.name.split(".")[0]);

      setStatus("Audio processed and sent to the backend successfully.");
    } catch (error) {
      setStatus("Error: " + error.message);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="file-upload-container">
      <h1>Audio Processing Tool</h1>
      <div className="upload-area">
        <label className="upload-button">
          Select File
          <input
            type="file"
            onChange={handleFileChange}
            disabled={isProcessing}
          />
        </label>
        {file && <p className="file-name">{file.name}</p>}
        <button
          className="process-button"
          onClick={startProcessing}
          disabled={isProcessing}
        >
          {isProcessing ? "Processing..." : "Start"}
        </button>
      </div>
      {status && <p className="status">{status}</p>}
    </div>
  );
};

export default FileUpload;
