"""BERT model definition for 3-class HateXplain classification (normal, offensive, hatespeech)."""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig


class BERTHateSpeech(nn.Module):
    def __init__(self, num_labels: int = 3, pretrained_name: str = "bert-base-uncased"):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pretrained_name, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_name, config=self.config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
