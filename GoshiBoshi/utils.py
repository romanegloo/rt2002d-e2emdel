"""
utils.py

Classes and methods used in common
"""
import random
import csv
import logging
import code
from collections import defaultdict
import os

import numpy as np
import torch
from torch.nn import functional as F

from transformers import AutoModel, AutoTokenizer

import GoshiBoshi.config as cfg

logger = logging.getLogger(__name__)


def set_seed(seed, n_gpu=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    if isinstance(lengths, list):
        lengths = torch.tensor(lengths)
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

def retrieval_outcomes(logits, batch, name_space):
    """
    Measure micro (token-level) retrieval performance

    """
    _, _, x_mask, y0, y1, y2 = batch
    outcomes = {
        'y0': { 'tp': 0, 'fp': 0, 'fn': 0},
        'y1': { 'tp': 0, 'fp': 0, 'fn': 0},
        'y2': { 'tp': 0, 'fp': 0, 'fn': 0}
    }
    out_iob, out_st, c_emb = \
        (out[:,1:-1,:].to('cpu') if out is not None else None for out in logits)
    x_mask = x_mask[:,1:-1]
    logprobs_st = F.log_softmax(out_st, dim=-1)

    # y0: IOB tagging
    gt = y0.long().argmax(dim=-1)
    null_idx_0 = 1  # 'O' in IOB
    null_idx_1 = y1.size(-1) - 1 # Assume that last idx is for null class
    if out_iob is None:
        pred = out_st.argmax(dim=-1)
        # Convert to IOB tensor
        pred_y0 = torch.ones(pred.size(), dtype=torch.uint8)
        pred_y0[(pred % 2 == 1) & (pred != null_idx_1)] = 0  # 'I'
        pred_y0[(pred % 2 == 0) & (pred != null_idx_1)] = 2  # 'B'
        pred_mask = (pred_y0 != null_idx_0).logical_and(x_mask)
        pred = pred_y0
        gt_mask = (gt != null_idx_0).logical_and(x_mask)
    else:
        pred = out_iob.argmax(dim=-1)
        pred_mask = (pred != null_idx_0).logical_and(x_mask)
        gt_mask = (gt != null_idx_0).logical_and(x_mask)
    outcomes['y0']['tp'] = (pred[pred_mask] == gt[pred_mask]).sum().item()
    outcomes['y0']['fp'] = (pred[pred_mask] != gt[pred_mask]).sum().item()
    outcomes['y0']['fn'] = (gt[gt_mask] != pred[gt_mask]).sum().item()

    # y1: Semantic types
    if out_iob is None:
        pred = F.avg_pool1d(out_st, kernel_size=2, ceil_mode=True).argmax(dim=-1)
        gt = F.avg_pool1d(y1.double(), kernel_size=2, ceil_mode=True).argmax(dim=-1)
        null_idx = int(y1.size(-1) / 2)
    else:
        pred = out_st.argmax(dim=-1)
        gt = y1.long().argmax(dim=-1)
        null_idx = y1.size(-1) - 1
    pred_mask = (pred != null_idx).logical_and(x_mask)
    gt_mask = (gt != null_idx).logical_and(x_mask)
    outcomes['y1']['tp'] = (pred[pred_mask] == gt[pred_mask]).sum().item()
    outcomes['y1']['fp'] = (pred[pred_mask] != gt[pred_mask]).sum().item()
    outcomes['y1']['fn'] = (gt[gt_mask] != pred[gt_mask]).sum().item()

    # y2: Entity linking
    x = logprobs_st[:,:,:-1]
    if out_iob is None:  # For the one-tag model
        x = x.reshape(*(x.size()[:-1]), len(cfg.MM_ST), 2).sum(axis=-1)
    norm_scores = torch.matmul(c_emb, name_space.T)
    pred = norm_scores.argmax(dim=-1)
    gt = y2.long().argmax(dim=-1)
    null_idx = y2.size(-1) - 1
    pred_mask = (pred != null_idx).logical_and(x_mask)
    gt_mask = (gt != null_idx).logical_and(x_mask)
    outcomes['y2']['tp'] = (pred[pred_mask] == gt[pred_mask]).sum().item()
    outcomes['y2']['fp'] = (pred[pred_mask] != gt[pred_mask]).sum().item()
    outcomes['y2']['fn'] = (gt[gt_mask] != pred[gt_mask]).sum().item()

    return outcomes

class TraningStats:
    def __init__(self):
        self.epoch = 0
        self.steps = 1
        self.n_exs = 0
        self.lr = 0
        self.loss = defaultdict(list)
        self.ret_outcomes = {
            'y0': {'tp': 0, 'fp': 0, 'fn': 0},
            'y1': {'tp': 0, 'fp': 0, 'fn': 0},
            'y2': {'tp': 0, 'fp': 0, 'fn': 0}
        }
        self.ret_scores = {
            'y0': defaultdict(list),
            'y1': defaultdict(list),
            'y2': defaultdict(list)
        }
        self.is_best = False
        self.best_score = 0

    def update_trn(self, losses):
        self.steps += 1
        for l, k in zip(losses, ['trn_tag', 'trn_st', 'trn_ent']):
            if l is not None:
                self.loss[k].append(l.item())

    def update_dev(self, losses, ret_out):
        for l, k in zip(losses, ['dev_tag', 'dev_st', 'dev_ent']):
            if l is not None:
                self.loss[k].append(l.item())
        for k1 in ret_out:
            for k2 in ret_out[k1]:
                self.ret_outcomes[k1][k2] += ret_out[k1][k2]

    def report_trn(self):
        """Print out training losses"""
        keys = ['trn_tag', 'trn_st', 'trn_ent']
        loss_avg = (sum(l) / len(l) if len(l) > 0 else 0
                    for l in (self.loss[k] for k in keys))
        msg = (
            'Epoch {} Steps {:>5}/{} -- loss ({:.3f}\t{:.3f}\t{:.3f}) lr {:.8f}'
            ''.format(self.epoch, self.steps, self.n_exs, *loss_avg, self.lr)
        )
        logger.info(msg)
        for k in keys:
            self.loss[k] = []

    def report_dev(self):
        """Print out validation losses and retrieval scores

        precision: the percentage of mentions predicted that are correct
        recall: the percentage of entities present in the corpus that are found
            by the model
        """
        keys = ['dev_tag', 'dev_st', 'dev_ent']
        loss_avg = (sum(l) / len(l) if len(l) > 0 else 0
                    for l in (self.loss[k] for k in keys))
        n_batches = len(self.loss['dev_ent'])
        msg = (
            '[DEV] #batches {} loss ({:.3f}, {:.3f}, {:.3f})'
            ''.format(n_batches, *loss_avg)
        )
        logger.info(msg)
        for k in keys:
            self.loss[k] = []
        epsilon = 1e-7
        self.is_best = False
        for k1 in self.ret_outcomes:
            s = self.ret_outcomes[k1]
            p = s['tp'] / (s['tp'] + s['fp'] + epsilon)
            r = s['tp'] / (s['tp'] + s['fn'] + epsilon)
            f = 2 * p * r / (p + r + epsilon)
            for k, v in zip(['steps', 'p', 'r', 'f'], (self.steps, p, r, f)):
                self.ret_scores[k1][k].append(v)
            msg = '[DEV] {} (p/r/f {:.3f}\t{:.3f}\t{:.3f})'.format(k1, p, r, f)
            logger.info(msg)
            # update best score
            if k1 == 'y2' and self.best_score < f:
                self.is_best = True
                self.best_score = f
        self.ret_outcomes = {
            'y0': {'tp': 0, 'fp': 0, 'fn': 0},
            'y1': {'tp': 0, 'fp': 0, 'fn': 0},
            'y2': {'tp': 0, 'fp': 0, 'fn': 0}
        }

def save_model(mdl, args, stat):
    checkpoint = {
        'model': mdl.state_dict(),
        'args': args,
        'stat': stat,
    }
    torch.save(checkpoint, cfg.BEST_MDL_FILE)
    logger.info(f'best score: {stat.best_score:.3f}, '
                f' Saving a checkpoint {cfg.BEST_MDL_FILE}')


### create n-grams
import re
def get_ngrams(string, n=3):
    string = string.encode("ascii", errors="ignore").decode()
    string = string.lower()
    chars_to_remove = [')', '(', '.', '|', '[', ']', '{', '}', "'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string) # remove the list of chars defined above
    string = string.replace('&', 'and')
    string = string.replace(',', ' ').replace('-', ' ')
    string = string.title() # Capital at start of each word
    string = re.sub(' +',' ',string).strip() # combine whitespace
    string = ' ' + string + ' ' # pad
    string = re.sub(r'[,-./]|\sBD', r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]
