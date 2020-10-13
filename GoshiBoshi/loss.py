"""
loss.py

focal loss for imbalanced data
"""
import code

import torch
from torch import nn
from torch.nn import functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, y0, y1, y2, name_space):
        logits_tag, logits_type, prj_name = (t[:,1:-1,:] for t in logits)
            # Don't need starting and ending special tokens
        dev = logits_tag.device
        y0 = y0.to(dev); y1 = y1.to(dev); y2 = y2.to(dev)
        name_space = name_space.to(dev)

        # Loss0: Tagging
        logprobs_tag = F.log_softmax(logits_tag, dim=-1)
        ce = F.cross_entropy(
            logprobs_tag.view(-1, logprobs_tag.size(-1)),
            y0.long().argmax(dim=-1).flatten(),
            reduction='none'
        )
        pt = torch.exp(-ce)
        loss0 = (self.alpha * (1-pt)**self.gamma * ce).mean()
        
        # Loss1: Types
        logprobs_type = F.log_softmax(logits_type, dim=-1)
        ce = F.cross_entropy(
            logprobs_type.view(-1, logprobs_type.size(-1)),
            y1.long().argmax(dim=-1).flatten(),
            reduction='none'
        )
        pt = torch.exp(-ce)
        loss1 = (self.alpha * (1-pt)**self.gamma * ce).mean()

        # Loss2: Entity linking
        prj_name = torch.cat((logprobs_type[:,:,:-1], prj_name), dim=-1)
        norm_scores = torch.matmul(prj_name, name_space.T)
        logprobs_name = F.log_softmax(norm_scores, dim=-1)
        ce = F.cross_entropy(
            logprobs_name.view(-1, logprobs_name.size(-1)),
            y2.long().argmax(dim=-1).flatten(),
            reduction='none'
        )
        pt = torch.exp(-ce)
        loss2 = (self.alpha * (1-pt)**self.gamma * ce).mean()

        return loss0, loss1, loss2