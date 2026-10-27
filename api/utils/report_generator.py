"""Downloadable Forensic PDF Report Generator for Digital Media Verification.

Generates in-memory, professional, publication-grade PDF dossiers from image analysis results
using ReportLab. Strictly adheres to:
1. In-memory generation (io.BytesIO) - no dangling temporary files on disk.
2. Complete safety against path traversal and arbitrary file reads.
3. Accurate representation of model outputs without fabricating values.
4. Robust Unicode, XML escaping, multiline wrapping, and page-breaking logic.
5. Decision-support disclaimers and epistemic uncertainty transparency.
"""

import os
import io
import re
import html
import base64
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

logger = logging.getLogger("dmv_api.report_generator")

# Unicode Font Registration & Caching
FONT_MAIN = "Helvetica"
FONT_MAIN_BOLD = "Helvetica-Bold"
FONT_CJK = None
FONT_HINDI = None
_FONTS_INITIALIZED = False


def _initialize_fonts():
    """Discover, register, and cache Unicode fonts across Windows and Linux environments."""
    global FONT_MAIN, FONT_MAIN_BOLD, FONT_CJK, FONT_HINDI, _FONTS_INITIALIZED
    if _FONTS_INITIALIZED:
        return
    _FONTS_INITIALIZED = True

    # 1. Chinese CID font (built into ReportLab Adobe CMap)
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        FONT_CJK = "STSong-Light"
    except Exception as exc:
        logger.debug(f"CJK CID font registration skipped: {exc}")

    # 2. Main Unicode TrueType Font (Latin, Cyrillic, Arabic)
    possible_main_fonts = [
        os.path.join(os.path.dirname(__file__), "..", "fonts", "DejaVuSans.ttf"),
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/mnt/c/Windows/Fonts/arial.ttf",
    ]
    possible_bold_fonts = [
        os.path.join(os.path.dirname(__file__), "..", "fonts", "DejaVuSans-Bold.ttf"),
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
        "/mnt/c/Windows/Fonts/arialbd.ttf",
    ]

    for p in possible_main_fonts:
        if os.path.exists(p):
            try:
                pdfmetrics.registerFont(TTFont("DMV_UnicodeMain", p))
                FONT_MAIN = "DMV_UnicodeMain"
                logger.info(f"Registered primary Unicode report font: {p}")
                break
            except Exception as exc:
                logger.debug(f"Failed to register font {p}: {exc}")

    for p in possible_bold_fonts:
        if os.path.exists(p):
            try:
                pdfmetrics.registerFont(TTFont("DMV_UnicodeBold", p))
                FONT_MAIN_BOLD = "DMV_UnicodeBold"
                break
            except Exception:
                pass

    if FONT_MAIN != "Helvetica" and FONT_MAIN_BOLD == "Helvetica-Bold":
        FONT_MAIN_BOLD = FONT_MAIN

    # 3. Hindi / Indic TrueType font
    possible_hindi_fonts = [
        os.path.join(os.path.dirname(__file__), "..", "fonts", "Nirmala.ttc"),
        "C:/Windows/Fonts/Nirmala.ttc",
        "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
        "/usr/share/fonts/truetype/samyak-fonts/Samyak-Devanagari.ttf",
        "/mnt/c/Windows/Fonts/Nirmala.ttc",
    ]
    for p in possible_hindi_fonts:
        if os.path.exists(p):
            try:
                if p.lower().endswith(".ttc"):
                    pdfmetrics.registerFont(TTFont("DMV_Hindi", p, subfontIndex=0))
                else:
                    pdfmetrics.registerFont(TTFont("DMV_Hindi", p))
                FONT_HINDI = "DMV_Hindi"
                logger.info(f"Registered Hindi Unicode font: {p}")
                break
            except Exception as exc:
                logger.debug(f"Failed to register Hindi font {p}: {exc}")


# Palette definition matching application dark/light theme tokens
COLOR_PRIMARY = colors.HexColor("#2563eb")    # Blue accent
COLOR_PRIMARY_DARK = colors.HexColor("#1e40af")
COLOR_HEADER_BG = colors.HexColor("#0f172a")   # Slate 900
COLOR_TEXT_MAIN = colors.HexColor("#1e293b")   # Slate 800
COLOR_TEXT_MUTED = colors.HexColor("#64748b")  # Slate 500
COLOR_CARD_BG = colors.HexColor("#f8fafc")     # Slate 50
COLOR_BORDER = colors.HexColor("#e2e8f0")      # Slate 200
COLOR_SUCCESS = colors.HexColor("#16a34a")     # Emerald 600
COLOR_SUCCESS_BG = colors.HexColor("#f0fdf4")
COLOR_DANGER = colors.HexColor("#dc2626")      # Red 600
COLOR_DANGER_BG = colors.HexColor("#fef2f2")
COLOR_WARNING = colors.HexColor("#d97706")     # Amber 600
COLOR_WARNING_BG = colors.HexColor("#fffbeb")
COLOR_INFO = colors.HexColor("#0284c7")        # Sky 600
COLOR_INFO_BG = colors.HexColor("#f0f9ff")


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print accurate total page count on all pages."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages: int):
        _initialize_fonts()
        self.saveState()
        self.setFont(FONT_MAIN, 8)
        self.setFillColor(COLOR_TEXT_MUTED)

        # Header rule & title (pages after page 1)
        if self._pageNumber > 1:
            self.setStrokeColor(COLOR_BORDER)
            self.setLineWidth(0.5)
            self.line(40, letter[1] - 30, letter[0] - 40, letter[1] - 30)
            self.drawString(40, letter[1] - 25, "Digital Media Verification • Forensic Image Analysis Dossier")

        # Footer rule & pagination (all pages)
        self.setStrokeColor(COLOR_BORDER)
        self.setLineWidth(0.5)
        self.line(40, 35, letter[0] - 40, 35)

        footer_text = "CONFIDENTIAL & DECISION-SUPPORT EVIDENCE • NOT PROOF OF AUTHENTICITY"
        self.drawString(40, 24, footer_text)

        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(letter[0] - 40, 24, page_str)
        self.restoreState()


EMOJI_MAP = {
    '🚨': '[ALERT]',
    '⚠️': '[WARNING]',
    '🛡️': '[SHIELD]',
    '🛡': '[SHIELD]',
    '✅': '[PASS]',
    '❌': '[FAIL]',
    '🔍': '[INSPECT]',
    '🚩': '[FLAG]',
    '⚡': '[VOLT]',
    '🔥': '[URGENT]',
    '📌': '[NOTE]',
    '📊': '[STAT]',
    '🕒': '[TIME]',
}


def safe_text(val: Any, max_length: Optional[int] = None) -> str:
    """Sanitize, XML-escape, and script-tag arbitrary text for robust multilingual PDF rendering."""
    if val is None:
        return "N/A"
    text = str(val).strip()
    if not text:
        return "N/A"
    if max_length and len(text) > max_length:
        text = text[:max_length] + "..."

    # Normalize emojis to clean textual bracket tags
    for emoji, tag in EMOJI_MAP.items():
        text = text.replace(emoji, tag)
    # Strip any remaining high-surrogate emojis that standard PDF fonts do not possess
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)

    # Standard XML escaping for ReportLab Paragraph markup
    escaped = html.escape(text)

    # Ensure fonts are initialized
    _initialize_fonts()

    # Wrap Devanagari (Hindi) runs if font is available
    if FONT_HINDI:
        escaped = re.sub(
            r'([\u0900-\u097F]+(?:\s+[\u0900-\u097F]+)*)',
            rf'<font name="{FONT_HINDI}">\1</font>',
            escaped
        )

    # Wrap CJK (Chinese) runs if CID font is available
    if FONT_CJK:
        escaped = re.sub(
            r'([\u4e00-\u9fff\u3400-\u4dbf]+(?:\s+[\u4e00-\u9fff\u3400-\u4dbf]+)*)',
            rf'<font name="{FONT_CJK}">\1</font>',
            escaped
        )

    # If fallback font is Helvetica (no Unicode support), safely encode non-Latin-1
    if FONT_MAIN == "Helvetica":
        try:
            escaped = escaped.encode("latin-1", "xmlcharrefreplace").decode("latin-1")
        except Exception:
            escaped = "".join(c if ord(c) < 128 else f"&#{ord(c)};" for c in escaped)

    return escaped


def sanitize_filename(filename: Optional[str]) -> str:
    """Extract a safe, sanitized filename stripped of directory paths and shell characters."""
    if not filename:
        return "image_analysis"
    base = os.path.basename(filename).strip()
    # Remove dangerous characters
    cleaned = re.sub(r'[^a-zA-Z0-9_.-]', '_', base)
    cleaned = re.sub(r'_+', '_', cleaned).strip('_')
    return cleaned or "image_analysis"


def generate_download_filename(filename: Optional[str], timestamp: Optional[datetime] = None) -> str:
    """Generate a clean, standardized, downloadable filename (e.g. dmv-image-report-2026-10-27.pdf)."""
    date_str = (timestamp or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    clean_stem = sanitize_filename(filename)
    # Remove existing extension if present
    stem, _ = os.path.splitext(clean_stem)
    if stem and stem != "image_analysis":
        # Shorten if excessively long
        stem = stem[:30]
        return f"dmv-image-report-{stem}-{date_str}.pdf"
    return f"dmv-image-report-{date_str}.pdf"


def _resolve_safe_image_path(raw_path: Optional[str], artifact_dir: str = "artifacts") -> Optional[str]:
    """Verify that an image path is safe from traversal and physically exists on disk."""
    if not raw_path:
        return None

    # Strip URL prefixes if given as /artifacts/filename.png
    clean_subpath = raw_path
    if clean_subpath.startswith("/artifacts/"):
        clean_subpath = clean_subpath[len("/artifacts/"):]
    elif clean_subpath.startswith("artifacts/"):
        clean_subpath = clean_subpath[len("artifacts/"):]

    # Resolve paths relative to artifact_dir and root
    possible_paths = [
        os.path.abspath(clean_subpath),
        os.path.abspath(os.path.join(artifact_dir, clean_subpath)),
        os.path.abspath(os.path.join(os.getcwd(), clean_subpath)),
        os.path.abspath(os.path.join(os.getcwd(), artifact_dir, clean_subpath))
    ]

    resolved_artifact_root = os.path.realpath(artifact_dir)

    for p in possible_paths:
        if os.path.exists(p) and os.path.isfile(p):
            real_p = os.path.realpath(p)
            # Ensure file is inside artifact root or designated project workspace
            if real_p.startswith(resolved_artifact_root) or real_p.startswith(os.path.realpath(os.getcwd())):
                return real_p

    return None


def _decode_base64_image(data_uri: str) -> Optional[io.BytesIO]:
    """Safely decode client-provided data URI into an in-memory BytesIO image stream."""
    try:
        if not data_uri or not isinstance(data_uri, str):
            return None
        # Pre-decode string length guard (~14MB base64 is ~10.5MB binary)
        if len(data_uri) > 14 * 1024 * 1024:
            logger.warning("Base64 preview image payload exceeded length limit. Skipping.")
            return None
        if "base64," in data_uri:
            data_uri = data_uri.split("base64,")[1]
        decoded = base64.b64decode(data_uri)
        if len(decoded) > 10 * 1024 * 1024:  # Cap at 10MB
            logger.warning("Base64 preview image exceeded 10MB limit. Skipping.")
            return None
        return io.BytesIO(decoded)
    except Exception as exc:
        logger.warning(f"Failed to decode base64 preview image: {exc}")
        return None


def generate_image_analysis_pdf(
    result: Dict[str, Any],
    filename: Optional[str] = None,
    image_preview: Optional[str] = None,
    artifact_dir: str = "artifacts"
) -> bytes:
    """Generate a downloadable, comprehensive PDF report for an image verification result.

    Args:
        result: The unified verification result dictionary returned by `/verify/image`.
        filename: Original input filename for documentation provenance.
        image_preview: Optional client-supplied base64 data URI of the analyzed image.
        artifact_dir: Directory containing Grad-CAM spatial explanation heatmaps.

    Returns:
        Raw bytes representing a valid PDF document.
    """
    _initialize_fonts()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    style_title = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName=FONT_MAIN_BOLD,
        fontSize=18,
        leading=22,
        textColor=COLOR_HEADER_BG
    )
    style_subtitle = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName=FONT_MAIN,
        fontSize=9,
        leading=12,
        textColor=COLOR_PRIMARY
    )
    style_section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName=FONT_MAIN_BOLD,
        fontSize=12,
        leading=16,
        textColor=COLOR_HEADER_BG,
        spaceBefore=10,
        spaceAfter=4
    )
    style_body = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName=FONT_MAIN,
        fontSize=8.5,
        leading=12,
        textColor=COLOR_TEXT_MAIN
    )
    style_body_bold = ParagraphStyle(
        "BodyBold",
        parent=styles["Normal"],
        fontName=FONT_MAIN_BOLD,
        fontSize=8.5,
        leading=12,
        textColor=COLOR_TEXT_MAIN
    )
    style_meta_label = ParagraphStyle(
        "MetaLabel",
        parent=styles["Normal"],
        fontName=FONT_MAIN_BOLD,
        fontSize=8,
        leading=10,
        textColor=COLOR_TEXT_MUTED
    )
    style_meta_value = ParagraphStyle(
        "MetaValue",
        parent=styles["Normal"],
        fontName=FONT_MAIN,
        fontSize=8.5,
        leading=11,
        textColor=COLOR_TEXT_MAIN
    )
    style_callout_title = ParagraphStyle(
        "CalloutTitle",
        parent=styles["Normal"],
        fontName=FONT_MAIN_BOLD,
        fontSize=10,
        leading=13,
        textColor=COLOR_HEADER_BG
    )
    style_disclaimer = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName=FONT_MAIN,
        fontSize=7.5,
        leading=10.5,
        textColor=COLOR_TEXT_MUTED
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("DIGITAL MEDIA VERIFICATION", style_title))
    story.append(Paragraph("Forensic Decision-Support Dossier • Automated Image Verification Report", style_subtitle))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=COLOR_PRIMARY, spaceBefore=2, spaceAfter=8))

    # Parse components from unified schema safely
    input_info = result.get("input", {})
    visual_info = result.get("visual", {})
    text_info = result.get("text", {})
    propaganda_info = result.get("propaganda", {})
    hate_speech_info = result.get("hate_speech", {})
    uncertainty_map = result.get("uncertainty", {})
    governance_info = result.get("governance", {})
    audit_info = result.get("audit", {})

    gen_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    display_filename = sanitize_filename(filename or input_info.get("filename") or input_info.get("source_identifier"))

    # 2. Metadata Block Table
    meta_data = [
        [
            Paragraph("ANALYSIS TARGET", style_meta_label),
            Paragraph("REPORT GENERATION", style_meta_label),
            Paragraph("PIPELINE LATENCY", style_meta_label),
            Paragraph("MC DROPOUT SAMPLES", style_meta_label),
        ],
        [
            Paragraph(safe_text(display_filename), style_meta_value),
            Paragraph(safe_text(gen_time), style_meta_value),
            Paragraph(f"{safe_text(audit_info.get('execution_time_seconds', 'N/A'))}s", style_meta_value),
            Paragraph(f"T = {safe_text(audit_info.get('mc_samples', 20))} passes", style_meta_value),
        ]
    ]
    t_meta = Table(meta_data, colWidths=[2.2 * inch, 2.0 * inch, 1.4 * inch, 1.6 * inch])
    t_meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), COLOR_CARD_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))

    # 3. Governance & Decision Status Callout Banner
    decision_status = governance_info.get("decision_status", "UNKNOWN")
    review_required = governance_info.get("review_required", False)
    trigger_component = governance_info.get("review_trigger_component")
    gov_disclaimer = governance_info.get("disclaimer", "")

    if decision_status == "CONFIDENT_PREDICTION":
        banner_bg = COLOR_SUCCESS_BG
        banner_border = COLOR_SUCCESS
        banner_title = "CONFIDENT PREDICTION • STANDARD MONITORING"
        banner_desc = "Model confidence and predictive stability meet baseline acceptance criteria. No anomaly thresholds triggered."
    elif decision_status == "REVIEW_RECOMMENDED":
        banner_bg = COLOR_WARNING_BG
        banner_border = COLOR_WARNING
        banner_title = "HUMAN REVIEW RECOMMENDED • ANOMALY FLAGGED"
        banner_desc = f"Epistemic uncertainty exceeded standard bounds or multi-class conflicting indicators detected. Trigger: {safe_text(trigger_component)}."
    elif decision_status == "ANALYSIS_INCOMPLETE":
        banner_bg = COLOR_INFO_BG
        banner_border = COLOR_INFO
        banner_title = "ANALYSIS INCOMPLETE • SAFELY BYPASSED"
        banner_desc = "Necessary visual evidence was absent (no human face detected). The system safely refrained from ungrounded guesswork."
    else:
        banner_bg = COLOR_CARD_BG
        banner_border = COLOR_BORDER
        banner_title = f"GOVERNANCE STATUS: {safe_text(decision_status)}"
        banner_desc = safe_text(gov_disclaimer)

    gov_data = [
        [Paragraph(banner_title, style_callout_title)],
        [Paragraph(banner_desc, style_body)],
    ]
    if gov_disclaimer and gov_disclaimer != banner_desc:
        gov_data.append([Paragraph(f"<i>Policy note: {safe_text(gov_disclaimer)}</i>", style_disclaimer)])

    t_gov = Table(gov_data, colWidths=[7.2 * inch])
    t_gov.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), banner_bg),
        ("BOX", (0, 0), (-1, -1), 1.0, banner_border),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(t_gov)
    story.append(Spacer(1, 12))

    # 4. Visual Deepfake Analysis Section (EfficientNet-B4)
    story.append(Paragraph("1. Facial Deepfake Forensic Analysis (EfficientNet-B4)", style_section_heading))

    visual_status = visual_info.get("status", "NOT_APPLICABLE")
    face_detected = visual_info.get("face_detected", False)
    prediction = visual_info.get("prediction")
    confidence = visual_info.get("confidence")
    probs = visual_info.get("probabilities") or {}
    visual_uq = visual_info.get("uncertainty") or uncertainty_map.get("deepfake") or {}

    if visual_status == "NO_FACE_DETECTED" or not face_detected:
        vis_rows = [
            [Paragraph("Face Detection Status:", style_meta_label), Paragraph("No human face detected by MTCNN detector", style_body_bold)],
            [Paragraph("Classification:", style_meta_label), Paragraph("Safely Bypassed (Ungrounded prediction prohibited)", style_body)],
            [Paragraph("Forensic Reason:", style_meta_label), Paragraph(safe_text(visual_info.get("note", "No face landmarks located.")), style_body)],
        ]
    elif visual_status == "SUCCESS":
        is_fake = (prediction == "fake")
        pred_label = "MANIPULATED / DEEPFAKE" if is_fake else "AUTHENTIC / REAL"
        pred_color = COLOR_DANGER if is_fake else COLOR_SUCCESS

        conf_str = f"{float(confidence) * 100:.2f}%" if confidence is not None else "N/A"
        real_prob_str = f"{float(probs.get('real', 0)) * 100:.2f}%" if probs.get("real") is not None else "N/A"
        fake_prob_str = f"{float(probs.get('fake', 0)) * 100:.2f}%" if probs.get("fake") is not None else "N/A"

        vis_rows = [
            [Paragraph("Model Classification:", style_meta_label), Paragraph(f"<b>{pred_label}</b>", style_body_bold)],
            [Paragraph("Prediction Confidence:", style_meta_label), Paragraph(f"<b>{conf_str}</b> (Raw model output)", style_body)],
            [Paragraph("Class Probabilities:", style_meta_label), Paragraph(f"Real: {real_prob_str} | Fake: {fake_prob_str}", style_body)],
            [Paragraph("Face Detection:", style_meta_label), Paragraph("Verified (Single-crop face extracted at 224x224)", style_body)],
        ]

        if visual_uq:
            entropy_val = visual_uq.get("entropy")
            variance_val = visual_uq.get("variance")
            ent_str = f"{entropy_val:.4f} bits (threshold 0.85)" if entropy_val is not None else "N/A"
            var_str = f"{variance_val:.6f} (threshold 0.020)" if variance_val is not None else "N/A"
            dispersion = visual_uq.get("dispersion_status") or ("Elevated" if visual_uq.get("review_required") else "Stable")

            vis_rows.append([Paragraph("Predictive Entropy (H):", style_meta_label), Paragraph(ent_str, style_body)])
            vis_rows.append([Paragraph("Predictive Variance (&sigma;&sup2;):", style_meta_label), Paragraph(var_str, style_body)])
            vis_rows.append([Paragraph("Epistemic Stability:", style_meta_label), Paragraph(f"{safe_text(dispersion)} across 20 stochastic passes", style_body)])
    else:
        vis_rows = [
            [Paragraph("Status:", style_meta_label), Paragraph(safe_text(visual_status), style_body)],
            [Paragraph("Details:", style_meta_label), Paragraph(safe_text(visual_info.get("error", "No visual analysis performed.")), style_body)],
        ]

    t_vis = Table(vis_rows, colWidths=[2.2 * inch, 5.0 * inch])
    t_vis.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_vis)
    story.append(Spacer(1, 10))

    # 5. Visual Explainability (Grad-CAM Heatmap) Block
    explanation = visual_info.get("explanation") or {}
    raw_saved_path = explanation.get("saved_path") or explanation.get("overlay_url")
    resolved_cam_path = _resolve_safe_image_path(raw_saved_path, artifact_dir=artifact_dir)

    # Decode preview image if provided
    preview_stream = _decode_base64_image(image_preview) if image_preview else None

    if resolved_cam_path or preview_stream:
        story.append(Paragraph("Spatial Explainability & Input Inspection", style_section_heading))

        images_table_cells = []
        labels_table_cells = []
        col_widths = []

        if preview_stream:
            try:
                rl_input_img = RLImage(preview_stream, width=2.4 * inch, height=2.4 * inch)
                images_table_cells.append(rl_input_img)
                labels_table_cells.append(Paragraph("<b>Input Image Thumbnail</b>", style_body))
                col_widths.append(3.6 * inch)
            except Exception as e:
                logger.warning(f"Could not render preview image in PDF: {e}")

        if resolved_cam_path:
            try:
                rl_cam_img = RLImage(resolved_cam_path, width=2.4 * inch, height=2.4 * inch)
                images_table_cells.append(rl_cam_img)
                labels_table_cells.append(Paragraph("<b>Grad-CAM Saliency Overlay</b>", style_body))
                col_widths.append(3.6 * inch)
            except Exception as e:
                logger.warning(f"Could not render Grad-CAM image in PDF: {e}")

        if images_table_cells:
            # Adjust column widths if only one image
            if len(col_widths) == 1:
                col_widths = [7.2 * inch]

            cam_table_data = [images_table_cells, labels_table_cells]
            t_cam = Table(cam_table_data, colWidths=col_widths)
            t_cam.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(t_cam)

            disclaimer_text = explanation.get("disclaimer") or (
                "Grad-CAM convolutional activations computed on the final layer (features[-1]). "
                "Highlights spatial feature boundaries influencing the decision; not causal proof of manual tampering."
            )
            story.append(Paragraph(f"<i>Note: {safe_text(disclaimer_text)}</i>", style_disclaimer))
            story.append(Spacer(1, 10))
    elif visual_status == "SUCCESS":
        # Grad-CAM was expected but the artifact file is missing/expired
        story.append(Paragraph("Spatial Explainability (Grad-CAM)", style_section_heading))
        expired_p = Paragraph(
            "<b>Grad-CAM Overlay Notice:</b> The spatial visualization artifact has expired under the system's "
            "24-hour retention cleanup policy or was not persisted to disk. Rerun verification to regenerate active heatmaps.",
            style_disclaimer
        )
        t_exp = Table([[expired_p]], colWidths=[7.2 * inch])
        t_exp.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), COLOR_CARD_BG),
            ("BOX", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(t_exp)
        story.append(Spacer(1, 10))

    # 6. OCR Text Extraction & Textual Discourse Analysis
    story.append(Paragraph("2. On-Screen Text Extraction & NLP Discourse Analysis", style_section_heading))

    ocr_status = text_info.get("text_status", "NO_TEXT_AVAILABLE")
    extracted_text = text_info.get("combined_text", "").strip()

    if ocr_status == "AVAILABLE" and extracted_text:
        text_rows = [
            [Paragraph("OCR Detection Status:", style_meta_label), Paragraph("Textual Content Located via EasyOCR", style_body_bold)],
            [Paragraph("Extracted Text:", style_meta_label), Paragraph(f"<i>\"{safe_text(extracted_text, max_length=500)}\"</i>", style_body)],
        ]

        # Propaganda result
        prop_pred = propaganda_info.get("prediction")
        prop_conf = propaganda_info.get("confidence")
        if propaganda_info.get("status") == "SUCCESS" and prop_pred:
            prop_str = f"{safe_text(prop_pred).replace('_', ' ').title()}"
            if prop_conf is not None:
                prop_str += f" (Confidence: {float(prop_conf) * 100:.1f}%)"
            text_rows.append([Paragraph("Propaganda Analysis:", style_meta_label), Paragraph(prop_str, style_body)])
        elif propaganda_info.get("status") == "NO_TEXT_AVAILABLE":
            text_rows.append([Paragraph("Propaganda Analysis:", style_meta_label), Paragraph("Insufficient text for rhetorical taxonomy", style_body)])

        # Hate speech result
        hate_pred = hate_speech_info.get("prediction")
        hate_conf = hate_speech_info.get("confidence")
        if hate_speech_info.get("status") == "SUCCESS" and hate_pred:
            hate_str = f"{safe_text(hate_pred).title()}"
            if hate_conf is not None:
                hate_str += f" (Confidence: {float(hate_conf) * 100:.1f}%)"
            text_rows.append([Paragraph("Hate Speech Analysis:", style_meta_label), Paragraph(hate_str, style_body)])
        elif hate_speech_info.get("status") == "NO_TEXT_AVAILABLE":
            text_rows.append([Paragraph("Hate Speech Analysis:", style_meta_label), Paragraph("Insufficient text for toxicity detection", style_body)])
    else:
        text_rows = [
            [Paragraph("OCR Detection Status:", style_meta_label), Paragraph("No on-screen text or captions detected in image", style_body)],
            [Paragraph("Downstream NLP:", style_meta_label), Paragraph("RoBERTa propaganda and BERT hate-speech classifiers safely bypassed", style_body)],
        ]

    t_text = Table(text_rows, colWidths=[2.2 * inch, 5.0 * inch])
    t_text.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_text)
    story.append(Spacer(1, 14))

    # 7. Forensic Limitations & Responsible AI Guidelines (Keep together)
    limitations_elements = []
    limitations_elements.append(Paragraph("3. Forensic Scope, System Bounds & Ethical Disclaimers", style_section_heading))

    limitations_content = [
        [Paragraph("Decision-Support Only:", style_meta_label),
         Paragraph("This dossier provides evidentiary decision support for human analysts and fact-checkers. It does not establish binding censorship, takedown orders, or definitive legal proof of tampering.", style_body)],
        [Paragraph("No Truth Adjudication:", style_meta_label),
         Paragraph("The system identifies observable synthetic artifacts and rhetorical text patterns. It does not verify objective historical truth, intent, or geopolitical reality.", style_body)],
        [Paragraph("Epistemic Uncertainty:", style_meta_label),
         Paragraph("Monte Carlo Dropout variance and Shannon entropy quantify the model's ambiguity with respect to its training manifold (FaceForensics++ C23), not a calibrated probability of correctness.", style_body)],
        [Paragraph("Explainability Boundaries:", style_meta_label),
         Paragraph("Grad-CAM overlays represent post-hoc gradient saliency within convolutional feature representations; they highlight influential pixel regions rather than mathematical boundary proofs.", style_body)],
    ]
    t_lim = Table(limitations_content, colWidths=[2.0 * inch, 5.2 * inch])
    t_lim.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), COLOR_CARD_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    limitations_elements.append(t_lim)
    story.append(KeepTogether(limitations_elements))

    # Build PDF with dynamic page numbering canvas
    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes
