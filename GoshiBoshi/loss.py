"""
loss.py

focal loss for imbalanced data
"""
import code

import torch
from torch import nn
from torch.nn import functional as F

from GoshiBoshi.config import MM_ST
import GoshiBoshi.utils as utils

class FocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma1=5, gamma2=3, norm_space=None):
        super().__init__()
        self.alpha = alpha
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.norm_space = norm_space

    def forward(self, logits, y, masks, norm_space=None):
        bsz, lens = logits.shape[:2]
        if norm_space is not None:
            logits = torch.matmul(logits, norm_space.T)
        ce = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.long().argmax(dim=-1).flatten(),
            reduction='none'
        )
        pt = torch.exp(-ce)
        loss_focal = torch.where(pt < 0.2, (1-pt)**self.gamma1,
                                           (1-pt)**self.gamma2)
        loss_focal = self.alpha * loss_focal * ce

        return loss_focal.view(bsz, lens)[masks].sum() / bsz
        return loss
