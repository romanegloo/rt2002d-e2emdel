"""
E2EMDEL/model.py

Joint learning Transformer model
"""

import logging

import torch
from torch import nn

from transformers import BertTokenizer, BertForTokenClassification

logger = logging.getLogger(__name__)


class MDEL(nn.Module):
    def __init__(self):
        super(MDEL, self).__init__()
        self.wp_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert =\
            BertForTokenClassification.from_pretrained('bert-base-uncased')

    def forward(self, data):
        pass
