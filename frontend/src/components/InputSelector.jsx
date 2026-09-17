import React from 'react';
import { Image as ImageIcon, Video, FileText, Layers } from 'lucide-react';

export const MODES = {
  IMAGE: 'IMAGE',
  VIDEO: 'VIDEO',
  TEXT: 'TEXT',
  MULTIMODAL: 'MULTIMODAL',
};

export function InputSelector({ activeMode, onSelectMode, disabled = false }) {
  const tabs = [
    { id: MODES.IMAGE, label: 'Image Verification', icon: ImageIcon, desc: 'Facial deepfakes & image OCR' },
    { id: MODES.VIDEO, label: 'Video Verification', icon: Video, desc: 'Temporal deepfakes, OCR & Whisper ASR' },
    { id: MODES.TEXT, label: 'Text Verification', icon: FileText, desc: 'Propaganda & hate speech analysis' },
    { id: MODES.MULTIMODAL, label: 'Multimodal Combined', icon: Layers, desc: 'Media + caption unified routing' },
  ];

  return (
    <nav className="mode-tabs-container" aria-label="Input Modality Selection">
      {tabs.map((tab) => {
        const Icon = tab.icon;
        const isActive = activeMode === tab.id;
        return (
          <button
            key={tab.id}
            type="button"
            className={`mode-tab-btn ${isActive ? 'active' : ''}`}
            onClick={() => onSelectMode(tab.id)}
            disabled={disabled}
          >
            <div className="tab-icon-wrap">
              <Icon size={18} />
            </div>
            <div className="tab-text-wrap">
              <span className="tab-title">{tab.label}</span>
              <span className="tab-desc">{tab.desc}</span>
            </div>
          </button>
        );
      })}
    </nav>
  );
}
