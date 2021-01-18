import code
import logging

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence

import GoshiBoshi.config as cfg
from GoshiBoshi.model import JointMDEL
from GoshiBoshi.crf import CRF

logger = logging.getLogger(__name__)


class IOB_ONE(JointMDEL):
    """IOB tagging and the semtantic type tagging are in one kind"""
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        logger.info("Initializing JointMDEL-OT model..")

        st_dim = 2 * len(cfg.MM_ST) + 1  # 2 (B/I) * st + 1 (O)
        prj_dim = cfg.NAME_DIM + len(cfg.MM_ST)
        self.st_classifier = nn.Linear(self.bert_dim, st_dim)
        self.name_proj = nn.Linear(self.bert_dim, cfg.NAME_DIM)
        self.soft_logits = nn.LogSoftmax(dim=-1)
        self.sum_st = nn.AvgPool1d(kernel_size=2)
        self.matching = nn.Linear(prj_dim, prj_dim)
        self.lstm_tagger = nn.LSTM(cfg.BERT_MODEL_DIM, cfg.LSTM_HIDDEN_DIM,
                                   num_layers=cfg.LSTM_NUM_LAYERS,
                                   batch_first=True, bidirectional=True)
        self.crf = CRF(2*cfg.LSTM_HIDDEN_DIM, st_dim)
        self.dropout = nn.Dropout(p=cfg.DROPOUT_RATIO)

    def forward(self, exs, norm_space, pred_lvl_m=False):
        x, masks, y0, y1, y2 = exs

        x = x.to(self.device)
        masks = masks.to(self.device)
        y0 = y0.to(self.device)
        y1 = y1.to(self.device)
        y2 = y2.to(self.device)

        # lm
        last_hid, _, _ = self.bert(x, attention_mask=masks)

        # semantic type
        rnn_inp = pack_padded_sequence(last_hid, masks.sum(dim=-1),
                                       batch_first=True)
        lstm_out, _ = self.lstm_tagger(rnn_inp)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        d_lstm_out = self.dropout(lstm_out)
        # crf
        iobst_loss, iobst_out = self.crf(d_lstm_out, y1, masks)

        # st_out = self.st_classifier(last_hid)
        # pull out semantic types from iob tagging
        soft_logits_st = self.sum_st(self.soft_logits(iobst_out))

        # entity
        c_emb = self.name_proj(last_hid)
        ent_out = self.matching(torch.cat((c_emb, soft_logits_st), dim=-1))

        # loss
        ent_loss = self.focal_loss(ent_out, y2, masks, norm_space=norm_space)
        joint_loss = (iobst_loss + ent_loss)

        if self.training:
            return joint_loss, None, iobst_loss, ent_loss
        # else

        null_idx = y1.size(-1) - 1
        pred_iobst = self.crf.decode(iobst_out, masks)
        # separate iob tags and semantic types apart
        pred_iob = []
        for l in pred_iobst:
            nl = np.array(l)
            nl_null = (nl == null_idx)
            nl_ = 2 * ((nl + 1) % 2)
            nl_[nl_null] = 1
            pred_iob.append(nl_)
        pred_iob = pad_sequence([torch.tensor(l) for l in pred_iob],
                                batch_first=True, padding_value=-1)\
                                    .to(self.device)
        pred_st = pad_sequence([torch.tensor(np.array(l)//2)
                                for l in pred_iobst],
                               batch_first=True, padding_value=-1)\
                                   .to(self.device)
        y1_ = torch.BoolTensor(y1.size(0), y1.size(1), 22).fill_(0).to(self.device)
        y1_[:,:,-1] = True
        for i, l in enumerate(y1):
            for j, e in enumerate(l):
                ti = e.long().argmax()
                if ti != null_idx:
                    y1_[i,j,-1] = False
                    y1_[i,j,ti//2] = True

        outcomes = self.eval_retrieval_performance(t0=(pred_iob, y0, masks),
                                                   t1=(pred_st, y1_, masks),
                                                   t2=(ent_out, y2, masks),
                                                   mention_level=pred_lvl_m,
                                                   norm_space=norm_space)
        return outcomes, None, iobst_loss, ent_loss
