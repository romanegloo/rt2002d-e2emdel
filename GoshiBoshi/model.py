"""
E2EMDEL/model.py

Joint learning Transformer model
"""

import code
import logging
from collections import defaultdict

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel

import GoshiBoshi.config as cfg
from GoshiBoshi.loss import FocalLoss

logger = logging.getLogger(__name__)


class JointMDEL(nn.Module):
    """Joint model for mentional detection and entity linking
    Returns three outputs; 1) logits for iob tagging 2) logits for semtantic
       types 3) projected names, and
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = self.args.device
        self.bert = AutoModel.from_pretrained(cfg.BERT_MODEL,
                                              output_hidden_states=True)
        self.bert_dim = self.bert.config.hidden_size
        self.focal_loss = FocalLoss(alpha=args.focal_alpha)

    def forward(self):
        """Variants of jointMDEL implements this"""
        raise NotImplementedError

    def eval_retrieval_performance(self, t0, t1, t2, mention_level=False,
                                   norm_space=None):
        outcomes = {
            't0': {'tp': 0, 'fp': 0, 'fn': 0},
            't1': {'tp': 0, 'fp': 0, 'fn': 0},
            't2': {'tp': 0, 'fp': 0, 'fn': 0}
        }

        # Token-level prediction
        if not mention_level:
            # t0
            pred, y0, masks = t0
            gt = y0.long().argmax(dim=-1)
            if isinstance(pred, list):
                # Output of CRF decode is not padded
                pred = pad_sequence([torch.tensor(l) for l in pred],
                                    batch_first=True, padding_value=-1)\
                                        .to(self.device)
            null_idx = 1  # 'O' in IOB
            pred_mask = (pred != null_idx).logical_and(masks)
            gt_mask = (gt != null_idx).logical_and(masks)
            outcomes['t0']['tp'] = (pred[pred_mask] == gt[pred_mask]).sum().item()
            outcomes['t0']['fp'] = (pred[pred_mask] != gt[pred_mask]).sum().item()
            outcomes['t0']['fn'] = (gt[gt_mask] != pred[gt_mask]).sum().item()

            # t1
            pred, y1, masks = t1
            if pred.dim() == 3:
                pred = pred.argmax(dim=-1)
            gt = y1.long().argmax(dim=-1)  # shape: (b, l)
            null_idx = y1.size(-1) - 1
            pred_mask = (pred != null_idx).logical_and(masks)
            gt_mask = (gt != null_idx).logical_and(masks)
            outcomes['t1']['tp'] = (pred[pred_mask] == gt[pred_mask]).sum().item()
            outcomes['t1']['fp'] = (pred[pred_mask] != gt[pred_mask]).sum().item()
            outcomes['t1']['fn'] = (gt[gt_mask] != pred[gt_mask]).sum().item()

            # t2
            ent_out, y2, masks = t2
            pred = torch.matmul(ent_out, norm_space.T).argmax(dim=-1)
            gt = y2.long().argmax(dim=-1)
            null_idx = y2.size(-1) - 1
            pred_mask = (pred != null_idx).logical_and(masks)
            neg_mask = (pred == null_idx).logical_and(masks)
            gt_mask = (gt != null_idx).logical_and(masks)
            # outcomes['t2']['tp'] = (pred[pred_mask] == gt[pred_mask]).sum().item()
            outcomes['t2']['tp'] = \
                y2[pred_mask].gather(-1, pred[pred_mask].unsqueeze(-1))\
                    .long().sum().item()
            # outcomes['t2']['fp'] = (pred[pred_mask] != gt[pred_mask]).sum().item()
            outcomes['t2']['fp'] = \
                y2[pred_mask].gather(-1, pred[pred_mask].unsqueeze(-1))\
                    .logical_not().sum().item()
            # outcomes['t2']['fn'] = (gt[gt_mask] != pred[gt_mask]).sum().item()
            outcomes['t2']['fn'] = \
                (pred[neg_mask] != y2[neg_mask].long().argmax(-1))\
                    .long().sum().item()
            return outcomes
        else:  # Mention-level prediction
            pred0, y0, masks = t0
            gt0 = y0.long().argmax(dim=-1)
            if isinstance(pred0, list):
                # Output of CRF decode is not padded
                pred0 = pad_sequence([torch.tensor(l) for l in pred0],
                                    batch_first=True, padding_value=-1)\
                                        .to(self.device)
            out1, y1, _ = t1
            pred1 = out1.argmax(dim=-1)
            out2, y2, _ = t2
            pred2 = torch.matmul(out2, norm_space.T).argmax(dim=-1)
            gt2 = y2.long().argmax(dim=-1)

            # Construct predicted mentions
            # `predictions` is a dictionary of mentions which key is (i, j)
            # where i is the example idx and j is the token idx.
            # A mention is represented by
            #   (starting idx, length, [predicted type indices],
            #                          [predicted entity indices]).
            # These indices are used in voting for the best predictions.
            predictions = dict()
            a_mention = [-1, -1, [], []]
            for i, ex in enumerate(pred0):
                seq_len = masks[i].sum()
                for j, idx in enumerate(ex):
                    if j == seq_len:
                        break
                    ti = pred1[i,j].item()
                    ei = pred2[i,j].item()
                    if idx == 1:  # 'O' in IOB
                        # if mention cache is not empty, submit the entry
                        if a_mention[0] >= 0:
                            predictions[(i, a_mention[0])] = a_mention[1:]
                            a_mention = [-1, -1, [], []]
                    else:  # I or B
                        if idx == 2:  # B in IOB
                            if a_mention[0] >= 0:  # register the previous mention
                                predictions[(i, a_mention[0])] = a_mention[1:]
                                a_mention = [-1, -1, [], []]
                            a_mention = [j, 1, [ti], [ei]]
                        else:  # I in IOB, continue the current mention
                            if a_mention[0] >= 0:
                                a_mention[1] += 1
                                a_mention[2].append(ti)
                                a_mention[3].append(ei)
                if a_mention[0] >= 0:
                    predictions[(i, a_mention[0])] = a_mention[1:]
                    a_mention = [-1, -1, [], []]

            return predictions
