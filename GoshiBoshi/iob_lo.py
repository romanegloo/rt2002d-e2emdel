import code
import logging

import torch
from torch import nn

import GoshiBoshi.config as cfg
from GoshiBoshi.model import JointMDEL

logger = logging.getLogger(__name__)


class IOB_LO(JointMDEL):
    """IOB tagging classifier is located at lower level"""
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        logger.info("Initializing JointMDEL-L model..")

        st_dim = len(cfg.MM_ST) + 1
        prj_dim = cfg.NAME_DIM + len(cfg.MM_ST)
        lstm_hid_size = 512
        self.lstm_tagger = nn.LSTM(self.bert_dim, lstm_hid_size, 2,
                                   batch_first=True, bidirectional=True)
        self.iob_classifier = nn.Linear(2*lstm_hid_size, len('iob'))
        self.st_classifier = nn.Linear(self.bert_dim, st_dim)
        self.name_proj = nn.Linear(self.bert_dim, cfg.NAME_DIM)
        self.soft_logits = nn.LogSoftmax(dim=-1)
        self.matching = nn.Linear(prj_dim, prj_dim)

    def forward(self, x, segs, masks):
        x = x.to(self.device)
        masks = masks.to(self.device)
        segs = segs.to(self.device)
        last_hid, _, hids = \
            self.bert(x, attention_mask=masks, token_type_ids=segs)
        out, _ = self.lstm_tagger(last_hid)
        out_iob = self.iob_classifier(out)
        # out_iob = self.iob_classifier(last_hid)
        out_st = self.st_classifier(last_hid)
        soft_logits_st = self.soft_logits(out_st)[:,:,:-1]
        c_emb = self.name_proj(last_hid)
        c_emb = self.matching(torch.cat((c_emb, soft_logits_st), dim=-1))

        return out_iob, out_st, c_emb
