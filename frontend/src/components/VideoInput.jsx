import React, { useState, useRef } from 'react';
import { Video, UploadCloud, X, ArrowRight, Clock, Film } from 'lucide-react';

export function VideoInput({ onVerify, isLoading, disabled = false }) {
  const [file, setFile] = useState(null);
  const [caption, setCaption] = useState('');
  const [asyncMode, setAsyncMode] = useState(false);
  const [sampleFps, setSampleFps] = useState(1.0);
  const [maxFrames, setMaxFrames] = useState(16);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

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
      if (droppedFile.type.startsWith('video/') || /\.(mp4|avi|mov|mkv|webm|m4v)$/i.test(droppedFile.name)) {
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
      onVerify(file, {
        caption,
        asyncMode,
        sampleFps: Number(sampleFps),
        maxFrames: Number(maxFrames)
      });
    }
  };

  return (
    <form className="input-card" onSubmit={handleSubmit}>
      <div className="input-card-header">
        <div className="card-title-group">
          <Video size={20} className="text-primary" />
          <h3 className="card-title">Video Ingestion & Verification</h3>
        </div>
        <span className="card-hint">MP4, AVI, MOV, WebM up to 100 MB</span>
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
            accept=".mp4,.avi,.mov,.mkv,.webm,.m4v"
            className="hidden-file-input"
            onChange={handleFileChange}
            disabled={disabled || isLoading}
          />
          <div className="dropzone-content">
            <UploadCloud size={40} className="dropzone-icon" />
            <p className="dropzone-title">Click to upload or drag and drop video</p>
            <p className="dropzone-subtitle">Supported formats: MP4, AVI, MOV, MKV, WebM</p>
          </div>
        </div>
      ) : (
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
            title="Remove video"
            disabled={isLoading}
          >
            <X size={18} />
          </button>
        </div>
      )}

      <div className="form-row">
        <div className="form-group flex-1">
          <label htmlFor="video-caption" className="form-label">
            Accompanying Context or Claim (Optional)
          </label>
          <input
            id="video-caption"
            type="text"
            className="form-input"
            placeholder="e.g. Broadcast video claiming altered footage..."
            value={caption}
            onChange={(e) => setCaption(e.target.value)}
            disabled={disabled || isLoading}
          />
        </div>

        <div className="form-group w-32">
          <label htmlFor="video-fps" className="form-label">
            Sample FPS
          </label>
          <select
            id="video-fps"
            className="form-input"
            value={sampleFps}
            onChange={(e) => setSampleFps(e.target.value)}
            disabled={disabled || isLoading}
          >
            <option value="0.5">0.5 FPS (Sparse)</option>
            <option value="1.0">1.0 FPS (Default)</option>
            <option value="2.0">2.0 FPS (Dense)</option>
          </select>
        </div>
      </div>

      <div className="checkbox-row">
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={asyncMode}
            onChange={(e) => setAsyncMode(e.target.checked)}
            disabled={disabled || isLoading}
          />
          <span>Enable Asynchronous Background Job Processing (FastAPI BackgroundTasks)</span>
        </label>
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
          <span>{isLoading ? 'Processing Video...' : 'Verify Video'}</span>
          <ArrowRight size={16} />
        </button>
      </div>
    </form>
  );
}
