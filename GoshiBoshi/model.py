"""
E2EMDEL/model.py

Joint learning Transformer model
"""

import logging
import code

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

from GoshiBoshi.config import MM_ST, BERT_MODEL

logger = logging.getLogger(__name__)


class JointMDEL(nn.Module):
    """Joint model for mentional detection and entity linking
    Returns three outputs; 1) logits for iob tagging 2) logits for semtantic
       types 3) projected names, and
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.bert = AutoModel.from_pretrained(BERT_MODEL)
        self.bert_dim = self.bert.config.hidden_size
        self.name_proj = nn.Linear(self.bert_dim, self.bert_dim)

    def forward(self):
        """Variants of jointMDEL implements this"""
        raise NotImplementedError

class JointMDEL_OT(JointMDEL):
    """IOB tagging and the semtantic type tagging are in one kind"""
    def __init__(self, device='cpu'):
        super().__init__(device)
        logger.info("Initiating JointMDEL-OT model..")

        self.st_classifier = nn.Linear(self.bert_dim,
                                         2*len(MM_ST) + 1)
    
    def forward(self, x, segs, masks):
        last_hid, _ = self.bert(x.to(self.device),
                                     attention_mask=masks.to(self.device),
                                     token_type_ids=segs.to(self.device))
        out_st = self.st_classifier(last_hid)
        out_name = self.name_proj(last_hid)

        return None, out_st, out_name

class JointMDEL_L(JointMDEL):
    """IOB tagging classifier is located at lower level"""
    def __init__(self, device='cpu'):
        super().__init__(device)
        logger.info("Initiating JointMDEL-L model..")

        self.iob_classifier = nn.Linear(self.bert_dim, len('iob'))
        self.st_classifier = nn.Linear(self.bert_dim, len(MM_ST) + 1)

    def forward(self, x, segs, masks):
        last_hid, _ = self.bert(x.to(self.device),
                                     attention_mask=masks.to(self.device),
                                     token_type_ids=segs.to(self.device))
        out_iob = self.iob_classifier(last_hid)
        out_st = self.st_classifier(last_hid)
        out_name = self.name_proj(last_hid)

        return out_iob, out_st, out_name
  
class JointMDEL_H(JointMDEL):
    """IOB tagging classifier is located at hier level"""
    def __init__(self, device='cpu'):
        super().__init__(device)
        logger.info("Initiating JointMDEL-H model..")

        self.st_classifier = nn.Linear(self.bert_dim, len(MM_ST) + 1)

        chunk_enc_dim = 512
        enc_layer = nn.TransformerEncoderLayer(d_model=chunk_enc_dim, nhead=8)
        self.trans_iob = nn.Sequential(
            nn.Linear(len(MM_ST)+1+self.bert_dim, chunk_enc_dim),
            nn.TransformerEncoder(enc_layer, num_layers=2),
            nn.Linear(chunk_enc_dim, len('iob'))
        )

    def forward(self, x, segs, masks):
        last_hid, _ = self.bert(x.to(self.device),
                                     attention_mask=masks.to(self.device),
                                     token_type_ids=segs.to(self.device))
        out_st = self.st_classifier(last_hid)
        out_name = self.name_proj(last_hid)

        # Inputs for chunking
        logprobs_type = F.log_softmax(out_st[:,1:-1,:], dim=-1)
        x = torch.cat((logprobs_type, out_name[:,1:-1,:]), dim=-1)
        out_iob = self.trans_iob(x)

        return out_iob, out_st, out_name

    
