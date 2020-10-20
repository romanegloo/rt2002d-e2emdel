"""
loss.py

focal loss for imbalanced data
"""
import code

import torch
from torch import nn
from torch.nn import functional as F

from GoshiBoshi.config import MM_ST

class FocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, y0, y1, y2, name_space):
        # Don't need starting and ending special tokens
        out_iob, out_st, out_name = (out[:,1:-1,:] if out is not None else None
                                     for out in logits)
        if logits[2].size(1) != logits[1].size(1):
            out_iob = logits[2]

        dev = out_st.device
        y0 = y0.to(dev)
        y1 = y1.to(dev)
        y2 = y2.to(dev)
        name_space = name_space.to(dev)

        # Loss0: Tagging
        if out_iob is None:
            loss0 = None
        else:
            logprobs_tag = F.log_softmax(out_iob, dim=-1)
            ce = F.cross_entropy(
                logprobs_tag.view(-1, logprobs_tag.size(-1)),
                y0.long().argmax(dim=-1).flatten(),
                reduction='none'
            )
            pt = torch.exp(-ce)
            loss0 = (self.alpha * (1-pt)**self.gamma * ce).mean()
        
        # Loss1: Types
        logprobs_st = F.log_softmax(out_st, dim=-1)
        ce = F.cross_entropy(
            logprobs_st.view(-1, logprobs_st.size(-1)),
            y1.long().argmax(dim=-1).flatten(),
            reduction='none'
        )
        pt = torch.exp(-ce)
        loss1 = (self.alpha * (1-pt)**self.gamma * ce).mean()

        # Loss2: Entity linking
        pred_st = logprobs_st[:,:,:-1]
        if out_iob is None:  # For the one-tag model
            pred_st = pred_st.reshape(*(pred_st.size()[:-1]),
                                    len(MM_ST), 2).sum(axis=-1)
        name = torch.cat((pred_st, out_name), dim=-1)
        norm_scores = torch.matmul(name, name_space.T)
        logprobs_name = F.log_softmax(norm_scores, dim=-1)
        ce = F.cross_entropy(
            logprobs_name.view(-1, logprobs_name.size(-1)),
            y2.long().argmax(dim=-1).flatten(),
            reduction='none'
        )
        pt = torch.exp(-ce)
        loss2 = (self.alpha * (1-pt)**self.gamma * ce).mean()


        if out_iob is None:
            return (loss1 + loss2), None, loss1, loss2
        return (loss0 + loss1 + loss2), loss0, loss1, loss2
