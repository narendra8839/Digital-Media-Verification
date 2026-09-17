import React, { useState, useRef, useEffect } from 'react';
import { Layers, UploadCloud, Film, Image as ImageIcon, X, ArrowRight } from 'lucide-react';

export function MultimodalInput({ onVerify, isLoading, disabled = false }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [text, setText] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (file && file.type.startsWith('image/')) {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      return () => URL.revokeObjectURL(url);
    } else {
      setPreviewUrl(null);
    }
  }, [file]);

  const isVideo = file && (file.type.startsWith('video/') || /\.(mp4|avi|mov|mkv|webm)$/i.test(file.name));

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
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleClear = () => {
    setFile(null);
    setText('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if ((file || text.trim()) && !isLoading) {
      onVerify({ file, text: text.trim() });
    }
  };

  return (
    <form className="input-card" onSubmit={handleSubmit}>
      <div className="input-card-header">
        <div className="card-title-group">
          <Layers size={20} className="text-primary" />
          <h3 className="card-title">Unified Multimodal Verification</h3>
        </div>
        <span className="card-hint">Media + Accompanying Text / Social Post</span>
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
            accept=".jpg,.jpeg,.png,.webp,.mp4,.avi,.mov,.mkv,.webm"
            className="hidden-file-input"
            onChange={handleFileChange}
            disabled={disabled || isLoading}
          />
          <div className="dropzone-content">
            <UploadCloud size={40} className="dropzone-icon" />
            <p className="dropzone-title">Click to upload or drag & drop media (Image or Video)</p>
            <p className="dropzone-subtitle">Routes to facial deepfake analysis, OCR, and speech recognition</p>
          </div>
        </div>
      ) : isVideo ? (
        <div className="file-info-badge">
          <div className="file-info-icon">
            <Film size={24} />
          </div>
          <div className="file-info-details">
            <span className="file-info-name">{file.name}</span>
            <span className="file-info-size">{(file.size / (1024 * 1024)).toFixed(2)} MB</span>
          </div>
          <button
            type="button"
            className="btn-icon text-muted"
            onClick={handleClear}
            title="Remove media"
            disabled={isLoading}
          >
            <X size={18} />
          </button>
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
        <label htmlFor="multimodal-text" className="form-label">
          Accompanying Post Text, Headline, or Social Caption
        </label>
        <textarea
          id="multimodal-text"
          className="form-textarea"
          rows={4}
          placeholder="Enter claim text or social media caption that accompanies this media..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={disabled || isLoading}
        />
        <span className="form-help">
          The system will cross-reference this text alongside extracted OCR/ASR for propaganda techniques and hate speech.
        </span>
      </div>

      <div className="form-actions">
        {(file || text) && (
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
          disabled={(!file && !text.trim()) || isLoading || disabled}
        >
          <span>{isLoading ? 'Processing Multimodal...' : 'Verify Multimodal'}</span>
          <ArrowRight size={16} />
        </button>
      </div>
    </form>
  );
}
