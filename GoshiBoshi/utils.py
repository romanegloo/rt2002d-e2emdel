"""
utils.py

Classes and methods used in common
"""
from collections import defaultdict, Counter
import random
import csv
import logging
import code
import os

import numpy as np
import torch
from torch.nn import functional as F

from transformers import AutoModel, AutoTokenizer

from GoshiBoshi.config import *

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

def compute_retrieval_scores(logits, x_mask, y1, y2, name_space):
    """
    measure micro retrieval performance
    
    precision: the percentage of mentions predicted that are correct
    recall: the percentage of entities present in the corpus that are found
            by the model
    """
    out_iob, out_st, out_name = \
        (out[:,1:-1,:].to('cpu') if out is not None else None for out in logits)
    x_mask = x_mask[:,1:-1]
    # marker_idx = torch.tensor([i for i in range(L-1) if i%2!=0])
    # out_type = out_type.index_select(1, marker_idx)
    # out_type = out_type[:,:,3:]
    # out_ent = out_ent.index_select(1, marker_idx)
    # x_mask = x_mask.index_select(1, marker_idx)

    scores = []
    epsilon = 1e-7
    logprobs_st = F.log_softmax(out_st, dim=-1)
    for (out, y, norm) in ((out_st, y1, None),
                           (out_name, y2, name_space)):
        if norm is not None:
            x = logprobs_st[:,:,:-1]
            if out_iob is None:  # For the one-tag model
                x = x.reshape(*(x.size()[:-1]), len(MM_ST), 2).sum(axis=-1)
            out = torch.cat((x, out_name), dim=-1)
            norm_scores = torch.matmul(out, norm.T)
            pred = norm_scores.argmax(dim=-1)
        else:
            pred = out.argmax(dim=-1)
        gt = y.long().argmax(dim=-1)
        # Assume that the last idx is for the null class
        null_idx = y.size(-1) - 1
        pred_mask = (pred != null_idx).logical_and(x_mask)
        gt_mask = (gt != null_idx).logical_and(x_mask)

        tp = (pred[pred_mask] == gt[pred_mask]).sum().item()
        fp = (pred[pred_mask] != gt[pred_mask]).sum().item() 
        fn = (gt[gt_mask] != pred[gt_mask]).sum().item()

        prec = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * prec * recall / (prec + recall + epsilon) 
        scores.append((prec, recall, f1))

    return scores


def infer_ret(logits, x_mask, annotations, name_space, types, names):
    out_iob, out_st, out_name = (out[:,1:-1,:] if out is not None else None
                                    for out in logits)
    if logits[2].size(1) != logits[1].size(1):
        out_iob = logits[2]

    # if logits[0].size(1) != logits[1].size(1):
    #     logits_tag = logits[0]
    # else:
    #     logits_tag = logits[0][:,1:-1,:]

    dev = out_st.device
    name_space = name_space.to(dev)

    # out_st.shape = [N, L, (2T+1)]
    if out_iob is None:
        pred_tag = out_st.argmax(dim=-1)
        pred_tag[pred_tag == out_st.size(-1)-1] = -1
        pred_tag[torch.logical_and(pred_tag >= 0, pred_tag%2 == 0)] = 2
        pred_tag[torch.logical_and(pred_tag >= 0, pred_tag%2 == 1)] = 0
        pred_tag[pred_tag==-1] = 1
    else:
        pred_tag = out_iob.argmax(dim=-1)

    # logits_tag = logits_tag.argmax(dim=-1)

    if out_iob is None:
        probs = F.softmax(out_st, dim=-1)
        x = probs[:,:,:-1]
        probs_type = x.reshape(*(x.size()[:-1]), len(MM_ST), 2).sum(axis=-1)
        logprobs_st = torch.log(
            torch.cat((probs_type, probs[:, :, -1].unsqueeze(-1)), dim=-1)
        )
        pred_st = logprobs_st.argmax(dim=-1)
    else:
        logprobs_st = F.log_softmax(out_st, dim=-1)
        pred_st = out_st.argmax(dim=-1)

    out_name = torch.cat((logprobs_st[:,:,:-1], out_name), dim=-1)
    name_scores = torch.matmul(out_name, name_space.T)
    name_scores = name_scores.argmax(dim=-1)

    annt_dict = dict()
    for i, annts in enumerate(annotations):
        for bi, l, _, t, n in annts:
            annt_dict[(i, bi, l)] = (t, n[5:])

    pred_dict = dict()
    mention = None
    for i, ex in enumerate(pred_tag):
        for j, idx in enumerate(ex):
            if j == x_mask[i].sum():
                break
            ti = pred_st[i,j].item()
            ei = name_scores[i,j].item()
            if idx == 0:  # I
                if mention is not None:
                    mention[1] += 1
                    mention[2].append(ti)
                    mention[3].append(ei)
            elif idx == 1:  # O
                if mention is not None:
                    pred_dict[(i, *mention[:2])] = (mention[2], mention[3])
                    mention = None
            elif idx == 2:  # B
                if mention is not None:
                    pred_dict[(i, *mention[:2])] = (mention[2], mention[3])
                    mention = None
                mention = [j, 1, [ti], [ei]]
        if mention is not None:
            pred_dict[(i, *mention[:2])] = (mention[2], mention[3])
            mention = None

    tp, fp, fn = 0, 0, 0
    for pk, (tl, el) in pred_dict.items():
        if pk in annt_dict:
            t_maj = Counter(tl).most_common()[0][0]
            n_maj = Counter(el).most_common()[0][0]
            if types[t_maj] == annt_dict[pk][0] and \
                    names[n_maj] == annt_dict[pk][1]:
                tp += 1
            else:
                fp += 1
        else:
            fp += 1
    fn = len(annt_dict) - tp
    
    return tp, fp, fn

class TraningStats:
    def __init__(self):
        self.epoch = 0
        self.steps = 1
        self.n_exs = 0
        self.lr = 0
        self.cum_losses_trn = [[], [], []]
        self.cum_losses_dev = [[], [], []]
        self.best_dev_loss = 999999
        self.ret_scores_type = []
        self.ret_scores_entity = []
    
    def update(self, losses, ret_scores=None, mode='trn'):
        loss0, loss1, loss2 = losses
        if mode == 'trn':
            self.steps += 1
            for loss, cum in zip(losses, self.cum_losses_trn):
                if loss is not None:
                    cum.append(loss.item())
        else:
            for loss, cum in zip(losses, self.cum_losses_dev):
                if loss is not None:
                    cum.append(loss.item())
            if ret_scores is not None:
                s1, s2 = ret_scores
                self.ret_scores_type.append(s1)
                self.ret_scores_entity.append(s2)
    
    def report(self, mode='trn'):
        if mode == 'trn':
            loss_avg = (sum(l) / len(l) if len(l) > 0 else 0
                        for l in self.cum_losses_trn)
            self.cum_losses_trn = [[], [], []]
            msg = (
                'Epoch {} Steps {:>5}/{} -- loss ({:.3f}, {:.3f}, {:.3f}) '
                ' lr {:.8f}'.format(self.epoch, self.steps, self.n_exs,
                                    *loss_avg, self.lr))
        elif mode == 'dev':
            loss_avg = (sum(l) / len(l) if len(l) > 0 else 0
                        for l in self.cum_losses_dev)
            self.cum_losses_dev = [[], [], []]

            p1 = [s[0] for s in self.ret_scores_type]
            p1 = sum(p1) / len(p1) if len(p1) > 0 else -1
            r1 = [s[1] for s in self.ret_scores_type]
            r1 = sum(r1) / len(r1) if len(r1) > 0 else -1
            f1a = [s[2] for s in self.ret_scores_type]
            f1a = sum(f1a) / len(f1a) if len(f1a) > 0 else -1
            p2 = [s[0] for s in self.ret_scores_entity]
            p2 = sum(p2) / len(p2) if len(p2) > 0 else -1
            r2 = [s[1] for s in self.ret_scores_entity]
            r2 = sum(r2) / len(r2) if len(r2) > 0 else -1
            f1b = [s[2] for s in self.ret_scores_entity]
            f1b = sum(f1b) / len(f1b) if len(f1b) > 0 else -1
            self.ret_scores_type = []
            self.ret_scores_entity = []
            msg = (
                '[DEV] loss ({:.3f}, {:.3f}, {:.3f}), '
                'ret. (p1 {:.3f} r1 {:.3f} f1 {:.3f}) '
                '(p2 {:.6f} r2 {:.6f} f2 {:.6f})'
                ''.format(*loss_avg, p1, r1, f1a, p2, r2, f1b)
            )
        logger.info(msg)
            
    def is_best(self):
        loss_avg = (sum(l) / len(l) if len(l) > 0 else 0
                    for l in self.cum_losses_dev)
        loss = sum(loss_avg)
        if loss <= self.best_dev_loss:
            self.best_dev_loss = loss
            return True
        return False

def save_model(mdl, args, stat):
    checkpoint = {
        'model': mdl.state_dict(),
        'args': args,
        'stat': stat,
    }
    # fname = '{}_{}_{}.pt'.format(args.model_type, stat.steps, args.exp_id)
    fname = 'best.pt'
    # fname_pttn = f'{args.model_type}_*_{args.exp_id}.pt'
    # for fpath in Path(args.dir_model).glob(fname_pttn):
    #     fpath.unlink()
    torch.save(checkpoint, os.path.join(args.data_dir, fname))
    logger.info(f'Loss: {stat.best_dev_loss:.3f}, '
                f' Saving a checkpoint {fname}')
