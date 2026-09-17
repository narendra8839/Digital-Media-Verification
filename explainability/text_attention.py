"""Token-level gradient attribution and rationale alignment for NLP models (BERT, RoBERTa)."""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union


class TokenAttribution:
    """Computes gradient-based token attribution (Grad x Input) for Transformer models."""

    def __init__(self, model: torch.nn.Module, tokenizer: Any, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

        # Identify embedding layer based on backbone type
        if hasattr(model, "bert") and hasattr(model.bert, "embeddings"):
            self.embed_layer = model.bert.embeddings.word_embeddings
            self.model_type = "bert"
        elif hasattr(model, "roberta") and hasattr(model.roberta, "embeddings"):
            self.embed_layer = model.roberta.embeddings.word_embeddings
            self.model_type = "roberta"
        elif hasattr(model, "embeddings") and hasattr(model.embeddings, "word_embeddings"):
            self.embed_layer = model.embeddings.word_embeddings
            self.model_type = "generic"
        else:
            raise ValueError("Unsupported model architecture: word_embeddings layer could not be found.")

    def attribute(self, text: str, target_class: Optional[int] = None,
                  max_length: int = 128, label_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Compute token-level attribution scores for the given text.

        Args:
            text: Input string
            target_class: Target class index to explain (default: predicted class)
            max_length: Tokenizer truncation max length
            label_mapping: Optional mapping from class index string to name

        Returns:
            Dictionary containing predicted label, confidence, tokens, and attribution scores.
        """
        inputs = self.tokenizer(text, truncation=True, padding=True, max_length=max_length, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Detach and set requires_grad on input embeddings
        input_embeds = self.embed_layer(input_ids).detach().clone().requires_grad_(True)

        self.model.zero_grad()
        if self.model_type == "roberta":
            outputs = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask)
        else:
            token_type_ids = inputs.get("token_type_ids", None)
            outputs = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask, token_type_ids=token_type_ids)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy().tolist()
        pred_idx = int(torch.argmax(logits, dim=1).item())

        if target_class is None:
            target_class = pred_idx

        target_score = logits[0, target_class]
        target_score.backward(retain_graph=True)

        # Grad x Input norm
        grad = input_embeds.grad[0] # (seq_len, hidden_dim)
        raw_saliency = torch.norm(grad * input_embeds[0], dim=-1).detach().cpu().numpy()

        # Token strings
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # Normalize saliency to [0, 1]
        s_min, s_max = float(np.min(raw_saliency)), float(np.max(raw_saliency))
        if s_max - s_min > 1e-8:
            norm_saliency = (raw_saliency - s_min) / (s_max - s_min)
        else:
            norm_saliency = np.zeros_like(raw_saliency)

        token_scores = []
        for tok, s in zip(tokens, norm_saliency):
            # Clean representation for display
            display_tok = tok.replace("Ġ", "").replace("##", "")
            if not display_tok:
                display_tok = tok
            token_scores.append({
                "token": tok,
                "clean_token": display_tok,
                "score": round(float(s), 4)
            })

        # Rank influential tokens excluding special tokens
        special_tokens = {self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token,
                          "<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "[PAD]"}
        ranked_tokens = [t for t in token_scores if t["token"] not in special_tokens]
        ranked_tokens.sort(key=lambda x: x["score"], reverse=True)

        pred_label = str(pred_idx)
        prob_dict = {str(i): round(p, 4) for i, p in enumerate(probs)}
        if label_mapping is not None:
            pred_label = label_mapping.get(str(pred_idx), str(pred_idx))
            prob_dict = {label_mapping.get(str(i), f"class_{i}"): round(p, 4) for i, p in enumerate(probs)}

        return {
            "prediction": pred_label,
            "confidence": round(float(probs[pred_idx]), 4),
            "class_index": pred_idx,
            "probabilities": prob_dict,
            "tokens": tokens,
            "token_scores": token_scores,
            "top_tokens": ranked_tokens[:5],
            "disclaimer": "Token attribution is a post-hoc explanatory signal and should not be interpreted as causal proof."
        }


def evaluate_rationale_alignment(attributed_tokens: List[Dict[str, Any]],
                                 ground_truth_rationales: List[List[int]],
                                 post_tokens: List[str],
                                 top_k: Optional[int] = None,
                                 threshold: float = 0.5) -> Dict[str, Any]:
    """Evaluate alignment between model-attributed tokens and human-annotated ground truth rationales.

    Args:
        attributed_tokens: Output from TokenAttribution.attribute()['token_scores']
        ground_truth_rationales: List of binary rationale lists from HateXplain annotators
        post_tokens: Original token list from HateXplain sample
        top_k: If set, selects the top-k highest scoring non-special tokens
        threshold: If top_k is None, selects tokens with score >= threshold

    Returns:
        Dictionary with overlap tokens, precision, recall, and F1 score.
    """
    if not ground_truth_rationales or len(ground_truth_rationales) == 0:
        return {
            "rationale_available": False,
            "overlap_count": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "note": "No ground-truth rationales available for this sample (e.g., normal post)."
        }

    # Majority vote / union of annotator rationales
    # If any annotator marked the token as rationale, or majority vote (>= 1 annotator)
    num_annotators = len(ground_truth_rationales)
    seq_len = len(ground_truth_rationales[0])
    gt_union = [0] * seq_len
    for rat in ground_truth_rationales:
        for i, val in enumerate(rat[:seq_len]):
            if val == 1:
                gt_union[i] = 1

    # Map attributed subwords back to word-level tokens
    special_tokens = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}
    filtered_attrib = [t for t in attributed_tokens if t["token"] not in special_tokens]

    # Align by text matching with post_tokens
    # Determine which post_tokens are considered "selected" by the model attribution
    model_selected_indices = set()

    if top_k is not None and top_k > 0:
        sorted_attrib = sorted(filtered_attrib, key=lambda x: x["score"], reverse=True)
        top_subwords = [t["token"].replace("##", "").replace("Ġ", "").lower() for t in sorted_attrib[:top_k]]
        for idx, pt in enumerate(post_tokens):
            pt_clean = pt.lower().strip()
            if any(sw == pt_clean or (len(sw) >= 2 and sw in pt_clean) or (len(pt_clean) >= 2 and pt_clean in sw) for sw in top_subwords):
                model_selected_indices.add(idx)
    else:
        for t in filtered_attrib:
            if t["score"] >= threshold:
                clean_tok = t["token"].replace("##", "").replace("Ġ", "").lower()
                for idx, pt in enumerate(post_tokens):
                    pt_clean = pt.lower().strip()
                    if clean_tok == pt_clean or (len(clean_tok) >= 2 and clean_tok in pt_clean) or (len(pt_clean) >= 2 and pt_clean in clean_tok):
                        model_selected_indices.add(idx)

    gt_indices = set(idx for idx, val in enumerate(gt_union) if val == 1)

    overlap = model_selected_indices.intersection(gt_indices)
    overlap_tokens = [post_tokens[i] for i in sorted(list(overlap)) if i < len(post_tokens)]

    p = len(overlap) / max(1, len(model_selected_indices))
    r = len(overlap) / max(1, len(gt_indices))
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

    return {
        "rationale_available": True,
        "ground_truth_count": len(gt_indices),
        "model_selected_count": len(model_selected_indices),
        "overlap_count": len(overlap),
        "overlap_tokens": overlap_tokens,
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "note": "Evaluated against HateXplain human rationale annotations. Alignment does not guarantee causal faithfulness."
    }
