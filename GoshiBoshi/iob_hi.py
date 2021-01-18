import code
import logging

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import GoshiBoshi.config as cfg
from GoshiBoshi.model import JointMDEL
from GoshiBoshi.crf import CRF

logger = logging.getLogger(__name__)


class IOB_HI(JointMDEL):
    """IOB tagging classifier is located at hier level"""
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        logger.info("Initializing JointMDEL-H model..")

        prj_dim = cfg.NAME_DIM + len(cfg.MM_ST)
        self.st_classifier = nn.Linear(self.bert_dim, len(cfg.MM_ST) + 1)
        self.name_proj = nn.Linear(self.bert_dim, cfg.NAME_DIM)
        self.soft_logits = nn.Softmax(dim=-1)
        self.matching = nn.Linear(prj_dim, prj_dim, bias=False)

        self.lstm_tagger = nn.LSTM(prj_dim + 1, cfg.LSTM_HIDDEN_DIM,
                                   num_layers=cfg.LSTM_NUM_LAYERS,
                                   batch_first=True, bidirectional=True)
        self.crf = CRF(2*cfg.LSTM_HIDDEN_DIM, len('iob'))
        self.dropout = nn.Dropout(p=cfg.DROPOUT_RATIO)

    def forward(self, exs, norm_space, pred_lvl_m=False):
        x, masks, y0, y1, y2 = exs[:5]

        x = x.to(self.device)
        masks = masks.to(self.device)
        y0 = y0.to(self.device)
        y1 = y1.to(self.device)
        y2 = y2.to(self.device)

        # lm
        last_hid, _, hids = self.bert(x, attention_mask=masks)

        # semantic type
        st_out = self.st_classifier(last_hid)
        soft_logits_st = self.soft_logits(st_out)

        # entity
        c_emb_ = self.name_proj(last_hid)
        ent_out = self.matching(torch.cat((c_emb_, soft_logits_st[:,:,:-1]), dim=-1))

        # bi-lstm
        rnn_inp = torch.cat((c_emb_, soft_logits_st), dim=-1)
        rnn_inp = pack_padded_sequence(rnn_inp, masks.sum(dim=-1),
                                       batch_first=True)
        lstm_out, _ = self.lstm_tagger(rnn_inp)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        d_lstm_out = self.dropout(lstm_out)
        # crf
        iob_loss, iob_out = self.crf(d_lstm_out, y0, masks)

        # loss
        st_loss = self.focal_loss(st_out, y1, masks)
        # st_loss = self.focal_loss(soft_logits_st, y1, masks)
        ent_loss = self.focal_loss(ent_out, y2, masks, norm_space=norm_space)
        joint_loss = (iob_loss + st_loss + ent_loss)

        if self.training:
            return joint_loss, iob_loss, st_loss, ent_loss
        # else
        pred_iob = self.crf.decode(iob_out, masks)
        outcomes = self.eval_retrieval_performance(t0=(pred_iob, y0, masks),
                                                   t1=(st_out, y1, masks),
                                                   t2=(ent_out, y2, masks),
                                                   mention_level=pred_lvl_m,
                                                   norm_space=norm_space)
        if len(exs) > 5:  # test with annotations
            return outcomes, iob_loss, st_loss, ent_loss, exs[5]
        return outcomes, iob_loss, st_loss, ent_loss
