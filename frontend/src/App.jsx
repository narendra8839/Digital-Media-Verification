import React, { useState } from 'react';
import { AlertCircle, RotateCcw, Shield } from 'lucide-react';
import { Header } from './components/Header';
import { InputSelector, MODES } from './components/InputSelector';
import { ImageInput } from './components/ImageInput';
import { VideoInput } from './components/VideoInput';
import { TextInput } from './components/TextInput';
import { MultimodalInput } from './components/MultimodalInput';
import { ProcessingIndicator } from './components/ProcessingIndicator';
import { ResultDashboard } from './components/ResultDashboard';
import { useBackendHealth } from './hooks/useBackendHealth';
import { useVerification } from './hooks/useVerification';

export function App() {
  const [activeMode, setActiveMode] = useState(MODES.IMAGE);
  const [inputMeta, setInputMeta] = useState({
    input_type: '',
    filename: '',
    originalImageUrl: null
  });

  const {
    isOnline,
    isChecking,
    lastChecked,
    checkHealth
  } = useBackendHealth();

  const {
    isLoading,
    loadingMessage,
    result,
    error,
    activeJobId,
    jobStatus,
    verifyText,
    verifyImage,
    verifyVideo,
    verifyMultimodal,
    reset
  } = useVerification();

  const handleReset = () => {
    reset();
    setInputMeta({
      input_type: '',
      filename: '',
      originalImageUrl: null
    });
  };

  const handleImageVerify = async (file, caption) => {
    const url = URL.createObjectURL(file);
    setInputMeta({
      input_type: 'IMAGE',
      filename: file.name,
      originalImageUrl: url
    });
    await verifyImage(file, caption);
  };

  const handleVideoVerify = async (file, options) => {
    setInputMeta({
      input_type: 'VIDEO',
      filename: file.name,
      originalImageUrl: null
    });
    await verifyVideo(file, options);
  };

  const handleTextVerify = async (text) => {
    setInputMeta({
      input_type: 'TEXT',
      filename: 'direct_input.txt',
      originalImageUrl: null
    });
    await verifyText(text);
  };

  const handleMultimodalVerify = async ({ file, text }) => {
    let url = null;
    let filename = text ? 'text_payload' : 'media_payload';
    if (file) {
      filename = file.name;
      if (file.type.startsWith('image/')) {
        url = URL.createObjectURL(file);
      }
    }
    setInputMeta({
      input_type: 'MULTIMODAL',
      filename,
      originalImageUrl: url
    });
    await verifyMultimodal({ file, text });
  };

  return (
    <div className="app-layout">
      <Header
        isOnline={isOnline}
        isChecking={isChecking}
        lastChecked={lastChecked}
        onRefresh={checkHealth}
      />

      <main className="main-container">
        {/* Error Alert Box */}
        {error && (
          <div className="error-alert" role="alert">
            <div className="error-alert-header">
              <AlertCircle size={20} className="text-danger flex-shrink-0" />
              <div className="error-alert-text">
                <strong>Verification Request Failed</strong>
                <p>{error}</p>
              </div>
            </div>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={handleReset}
            >
              <RotateCcw size={14} />
              <span>Dismiss & Try Again</span>
            </button>
          </div>
        )}

        {/* Dynamic View State */}
        {result ? (
          <ResultDashboard
            result={result}
            inputMeta={inputMeta}
            onReset={handleReset}
          />
        ) : isLoading ? (
          <ProcessingIndicator
            message={loadingMessage}
            activeJobId={activeJobId}
            jobStatus={jobStatus}
          />
        ) : (
          <div className="workspace-card">
            <InputSelector
              activeMode={activeMode}
              onSelectMode={setActiveMode}
              disabled={isLoading}
            />

            <div className="input-panel-wrapper">
              {activeMode === MODES.IMAGE && (
                <ImageInput
                  onVerify={handleImageVerify}
                  isLoading={isLoading}
                />
              )}

              {activeMode === MODES.VIDEO && (
                <VideoInput
                  onVerify={handleVideoVerify}
                  isLoading={isLoading}
                />
              )}

              {activeMode === MODES.TEXT && (
                <TextInput
                  onVerify={handleTextVerify}
                  isLoading={isLoading}
                />
              )}

              {activeMode === MODES.MULTIMODAL && (
                <MultimodalInput
                  onVerify={handleMultimodalVerify}
                  isLoading={isLoading}
                />
              )}
            </div>
          </div>
        )}
      </main>

      <footer className="app-footer">
        <div className="footer-content">
          <div className="footer-left">
            <Shield size={16} className="text-muted" />
            <span>
              <strong>Digital Media Verification System</strong> • Ethical & Responsible AI Project
            </span>
          </div>
          <div className="footer-right">
            <span>Decision Support Telemetry • Academic B.Tech Demonstration</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
