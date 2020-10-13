"""
E2EMDEL/model.py

Joint learning Transformer model
"""

import logging

import torch
from torch import nn
from transformers import AutoModel, AutoModelForTokenClassification

import GoshiBoshi.utils as utils

logger = logging.getLogger(__name__)


class JointMDEL(nn.Module):
    def __init__(self, args, num_types):
        super(JointMDEL, self).__init__()
        self.args = args
        self.num_tags = len('IOB')  # Fixed
        self.bert = AutoModel.from_pretrained('allenai/scibert_scivocab_cased')
        self.tag_classifier = nn.Linear(self.bert.config.hidden_size,
                                        self.num_tags)
        self.type_classifier = nn.Linear(self.bert.config.hidden_size,
                                         num_types + 1)  # +1 for 'N'ull class
        self.ent_proj = nn.Linear(self.bert.config.hidden_size,
                                  self.bert.config.hidden_size)


    def forward(self, x, segs, masks):
        # inp, tgt, segs, masks = data
        last_hid, pooled = self.bert(x.to(self.args.device),
                                     attention_mask=masks.to(self.args.device),
                                     token_type_ids=segs.to(self.args.device))
        out0 = self.tag_classifier(last_hid)
        out1 = self.type_classifier(last_hid) 
        out2 = self.ent_proj(last_hid)

        return out0, out1, out2
