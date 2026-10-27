# Digital Media Verification (DMV) — Research Hub

Welcome to the central research directory for the **Digital Media Verification (DMV)** project. This directory organizes all scientific documentation, system architectures, empirical evaluations, robustness benchmarks, publication-quality figures, reproduction instructions, and manuscript drafts necessary to audit, reproduce, and cite this research.

---

## 1. Directory Structure and Overview

```
research/
├── README.md                          # This master index
│
├── architecture/                      # System design and component engineering
│   ├── SYSTEM_ARCHITECTURE.md         # End-to-end multimodal architecture & Mermaid diagram
│   ├── SYSTEM_DESIGN.md               # FastAPI gateway, REST contracts, artifact lifecycle
│   └── MODEL_AND_PIPELINE_DESIGN.md   # EfficientNet-B4, RoBERTa, BERT, Whisper, Grad-CAM, MC Dropout
│
├── methodology/                       # Experimental design and verification standards
│   ├── DATASETS_AND_SPLITS.md         # FaceForensics++, SemEval-2020, HateXplain, split hygiene
│   ├── EVALUATION_PROTOCOL.md         # Metrics, video frame sampling, label orientation, MC Dropout
│   └── ROBUSTNESS_EXPERIMENTS.md      # 5 corruption operators, sweeps, V3 acceptance gates, PRNG note
│
├── results/                           # Authoritative evaluation reports and data
│   ├── MODEL_EVALUATION_SUMMARY.md    # Multi-modal benchmark synthesis across all 3 domains
│   ├── DEEPFAKE_ROBUSTNESS_V3_REPORT.md # Comprehensive final V3 robustness report & reconciliation
│   ├── DEEPFAKE_ROBUSTNESS_V2_REPORT.md # Authoritative V2 robustness report (frequency residual audit)
│   ├── V3_RESUME_HANDOFF.md           # Formal decision handoff & acceptance gate summary
│   ├── tables/                        # Publication-ready CSV and Markdown data tables
│   │   ├── research_figures_data.csv  # Machine-readable tabular data for Figures A–F
│   │   └── research_figures_data.md   # Markdown documentation and provenance guide
│   └── data/                          # Authoritative JSON evaluation outputs (per-video records)
│       ├── baseline_robustness.json   # 100-video development benchmark baseline
│       ├── candidate_v3a_robustness.json
│       ├── candidate_v3b_robustness.json
│       ├── candidate_v3c_robustness.json
│       ├── reserved_baseline_robustness.json # 100-video reserved comparison subset
│       └── severity_curve_v3.json     # 12-condition multi-severity degradation curve
│
├── figures/                           # Publication figures (Vector SVG & 300 DPI PNG)
│   ├── README.md                      # Figure catalog, captions, generator scripts
│   ├── png/                           # High-resolution raster images (Figs A through F)
│   └── svg/                           # Resolution-independent vector graphics (Figs A through F)
│
├── reproducibility/                   # Deterministic audit and reproduction guides
│   ├── REPRODUCING_RESULTS.md         # Step-by-step reproduction instructions and commands
│   ├── ENVIRONMENT_AND_DEPENDENCIES.md# OS, Python, PyTorch, Node.js, FFmpeg, and font specs
│   └── CHECKPOINT_MANIFEST.md         # Cryptographic SHA-256 hashes, sizes, paths for all weights
│
└── manuscript/                        # Academic paper submission materials
    ├── PAPER_OUTLINE.md               # Detailed academic manuscript outline and section plan
    ├── FIGURE_CAPTIONS.md             # Formal, peer-reviewed captions for Figures 1–6 and Tables 1–5
    └── REFERENCES.md                  # Genuine BibTeX records for models, datasets, and methods
```

---

## 2. Key Research Findings and Conclusions

1. **Multimodal Performance:**
   - **Visual Deepfakes (EfficientNet-B4):** 88.00% accuracy, 0.8321 Macro F1, and 96.30% Fake Precision on the 700-video FaceForensics++ C23 test split under a 4:1 class imbalance.
   - **Propaganda Techniques (RoBERTa-base):** Class-balanced focal loss improved Macro F1 from 0.4175 to **0.5132** (+0.0957) across 14 SemEval-2020 Task 11 techniques on the article-disjoint validation split.
   - **Hate Speech (BERT-base):** 67.41% accuracy and 0.6617 Macro F1 on HateXplain ($N=1,924$).
2. **Robustness Vulnerability & Decision:**
   - The production baseline is vulnerable to spatial low-pass filtering (Gaussian blur and $0.25\times$ downsampling collapse Real Recall to 12.0%).
   - Experimental Candidate V3 series (V3-A, V3-B, V3-C) successfully recovered Gaussian noise accuracy (70.0%–71.0%, real recall 86%–92%), but **all candidates catastrophically failed the JPEG $Q=30$ fake recall gate** (achieving only 16%–34% recall).
   - **Decision:** The **production baseline model (`5878ae38...`) is retained in production**. V3 candidates remain archived as experimental models.
3. **PRNG Realization Reconciliation:**
   - Candidate V3-A’s 70.0% benchmark accuracy (NumPy `RandomState` MT19937) and 72.0% severity sweep accuracy (NumPy `default_rng` PCG64) under Gaussian noise ($\sigma=15$) both evaluate the identical 100-video development subset and identical model weights. The 2-video discrepancy is due strictly to stochastic noise realizations on borderline fake samples; both pass the $\ge 65\%$ gate.
4. **Assistive Nature & Governance:**
   - The system is designed as an assistive forensic triage tool, not an infallible legal truth engine.
   - An automated human-in-the-loop gate is triggered whenever predictive variance $\sigma^2 > 0.08$ or cross-modal conflicts arise.

---

## 3. Quick Links to Historical Audits

Prior phase-by-phase implementation and training milestones are preserved in the repository’s historical documentation:
- [docs/research_history/CURRENT_MVP_AUDIT.md](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/docs/research_history/CURRENT_MVP_AUDIT.md)
- [docs/research_history/PHASE_4_EVALUATION.md](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/docs/research_history/PHASE_4_EVALUATION.md)
- [docs/research_history/PHASE_5_MODEL_OPTIMIZATION.md](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/docs/research_history/PHASE_5_MODEL_OPTIMIZATION.md)
- [docs/research_history/PHASE_6_INTEGRATION.md](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/docs/research_history/PHASE_6_INTEGRATION.md)
- [docs/PROJECT_LIMITATIONS.md](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/docs/PROJECT_LIMITATIONS.md)
- [docs/ARTIFACT_MANAGEMENT.md](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/docs/ARTIFACT_MANAGEMENT.md)
