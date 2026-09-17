import React, { useState, useRef, useEffect } from 'react';
import { UploadCloud, Image as ImageIcon, X, ArrowRight } from 'lucide-react';

export function ImageInput({ onVerify, isLoading, disabled = false }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [caption, setCaption] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (file) {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      return () => URL.revokeObjectURL(url);
    } else {
      setPreviewUrl(null);
    }
  }, [file]);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type.startsWith('image/')) {
        setFile(droppedFile);
      }
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleClear = () => {
    setFile(null);
    setCaption('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (file && !isLoading) {
      onVerify(file, caption);
    }
  };

  return (
    <form className="input-card" onSubmit={handleSubmit}>
      <div className="input-card-header">
        <div className="card-title-group">
          <ImageIcon size={20} className="text-primary" />
          <h3 className="card-title">Image Ingestion & Verification</h3>
        </div>
        <span className="card-hint">JPEG, PNG, WebP up to 15 MB</span>
      </div>

      {!file ? (
        <div
          className={`dropzone ${dragActive ? 'dropzone-active' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".jpg,.jpeg,.png,.webp,.bmp,.tiff"
            className="hidden-file-input"
            onChange={handleFileChange}
            disabled={disabled || isLoading}
          />
          <div className="dropzone-content">
            <UploadCloud size={40} className="dropzone-icon" />
            <p className="dropzone-title">Click to upload or drag and drop image</p>
            <p className="dropzone-subtitle">Supported formats: JPG, PNG, WEBP, BMP, TIFF</p>
          </div>
        </div>
      ) : (
        <div className="preview-container">
          <div className="preview-image-wrapper">
            <img src={previewUrl} alt="Selected preview" className="preview-image" />
            <button
              type="button"
              className="preview-remove-btn"
              onClick={handleClear}
              title="Remove image"
              disabled={isLoading}
            >
              <X size={16} />
            </button>
          </div>
          <div className="preview-meta">
            <span className="preview-filename">{file.name}</span>
            <span className="preview-filesize">{(file.size / (1024 * 1024)).toFixed(2)} MB</span>
          </div>
        </div>
      )}

      <div className="form-group">
        <label htmlFor="image-caption" className="form-label">
          Accompanying Caption or Context (Optional)
        </label>
        <input
          id="image-caption"
          type="text"
          className="form-input"
          placeholder="e.g. Breaking news reported on social media..."
          value={caption}
          onChange={(e) => setCaption(e.target.value)}
          disabled={disabled || isLoading}
        />
        <span className="form-help">
          Caption will be aggregated with any text detected by optical character recognition (EasyOCR).
        </span>
      </div>

      <div className="form-actions">
        {file && (
          <button
            type="button"
            className="btn btn-secondary"
            onClick={handleClear}
            disabled={isLoading}
          >
            Clear
          </button>
        )}
        <button
          type="submit"
          className="btn btn-primary"
          disabled={!file || isLoading || disabled}
        >
          <span>{isLoading ? 'Verifying...' : 'Verify Image'}</span>
          <ArrowRight size={16} />
        </button>
      </div>
    </form>
  );
}
