"""
E2EMDEL/model.py

Joint learning Transformer model
"""

import logging

import torch
from torch import nn
from transformers import AutoModelForTokenClassification

import GoshiBoshi.utils as utils


logger = logging.getLogger(__name__)


class JointMDEL(nn.Module):
    def __init__(self, args, num_classes=17):
        super(JointMDEL, self).__init__()
        self.args = args
        self.bert = AutoModelForTokenClassification.from_pretrained(
            'allenai/scibert_scivocab_uncased'
        )
        self.bert.classifier = nn.Linear(self.bert.config.hidden_size,
                                         num_classes)

    def forward(self, data):
        inp, tgt, segs, masks = data
        logits = self.bert(inp.to(self.args.device),
                           attention_mask=masks.to(self.args.device),
                           token_type_ids=segs.to(self.args.device))
        return logits
