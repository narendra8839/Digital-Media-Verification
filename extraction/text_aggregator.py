"""Text aggregation module preserving source provenance across OCR, ASR, and user inputs."""

from typing import Dict, Any, List, Optional


class TextAggregator:
    """Combines and deduplicates text from multiple input streams while preserving exact provenance."""

    @staticmethod
    def aggregate(user_text: Optional[str] = None,
                  ocr_results: Optional[List[Dict[str, Any]]] = None,
                  asr_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Aggregate textual content from USER_TEXT, OCR, and ASR streams.

        Args:
            user_text: Direct user input or caption string
            ocr_results: List of OCR result dictionaries (from image or sampled video frames)
            asr_result: ASR transcript result dictionary

        Returns:
            Structured dictionary with segments, combined text, sources, and status.
        """
        segments = []
        seen_texts = set()
        active_sources = set()

        # 1. User caption / text input
        if user_text and user_text.strip():
            clean_user = user_text.strip()
            segments.append({
                "source": "USER_TEXT",
                "text": clean_user
            })
            seen_texts.add(clean_user.lower())
            active_sources.add("USER_TEXT")

        # 2. OCR text (from image or video frames)
        if ocr_results:
            for ocr_item in ocr_results:
                txt = ocr_item.get("text", "").strip()
                if txt and txt.lower() not in seen_texts:
                    seen_texts.add(txt.lower())
                    seg = {
                        "source": "OCR",
                        "text": txt,
                        "confidence": ocr_item.get("confidence", 0.0)
                    }
                    if "frame_index" in ocr_item:
                        seg["frame_index"] = ocr_item["frame_index"]
                    if "timestamp" in ocr_item:
                        seg["timestamp"] = ocr_item["timestamp"]
                    segments.append(seg)
                    active_sources.add("OCR")

        # 3. ASR transcript (from video audio)
        if asr_result:
            asr_txt = asr_result.get("text", "").strip()
            if asr_txt and asr_txt.lower() not in seen_texts:
                seen_texts.add(asr_txt.lower())
                segments.append({
                    "source": "ASR",
                    "text": asr_txt
                })
                active_sources.add("ASR")

        # Combine unique segments
        combined_text = " ".join(seg["text"] for seg in segments).strip()
        has_text = len(combined_text) > 0

        return {
            "segments": segments,
            "combined_text": combined_text,
            "sources": sorted(list(active_sources)),
            "text_status": "AVAILABLE" if has_text else "NO_TEXT_AVAILABLE",
            "segment_count": len(segments)
        }


def aggregate_text(user_text: Optional[str] = None,
                   ocr_results: Optional[List[Dict[str, Any]]] = None,
                   asr_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience functional wrapper for TextAggregator.aggregate."""
    return TextAggregator.aggregate(user_text=user_text, ocr_results=ocr_results, asr_result=asr_result)
