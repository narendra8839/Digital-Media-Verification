"""RoBERTa model definition for fine-grained propaganda technique classification."""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig


class RoBERTaPropaganda(nn.Module):
    def __init__(self, num_labels: int = 14, pretrained_name: str = "roberta-base"):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pretrained_name, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_name, config=self.config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
