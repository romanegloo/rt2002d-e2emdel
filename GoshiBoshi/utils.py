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

logger = logging.getLogger(__name__)


def set_seed(seed, n_gpu=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
        
def get_tokenizer(mdl_name):
    if mdl_name == 'SciBERT':
        return AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')

def get_bert_model(mdl_name):
    if mdl_name == 'SciBERT':
        return AutoModel.from_pretrained('allenai/scibert_scivocab_cased')

def csv_to_ner_split_examples(args, split_ranges=[0., .8, .9]):
    examples = []
    if args.dataset == 'KaggleNER':
        """
        Kaggle NER Example:
        Sentence # | Word | POS | Tag
        Sentence: 1 | Thousands | NNS | O
                    | of | IN | O
        """
        with open(args.ds_kaggle_path, encoding='latin1') as f:
            next(f)
            reader = csv.reader(f)
            sent_len = 0
            for l in reader:
                if l[0] != '':
                    sent_len = len(l[1]) + 1
                    examples.append(([l[1]], [l[2]], [l[3]]))
                else:
                    sent_len += len(l[1]) + 1
                    if sent_len >= args.max_sent_len:
                        continue
                    examples[-1][0].append(l[1])
                    examples[-1][1].append(l[2])
                    examples[-1][2].append(l[3])
        random.shuffle(examples)
        n_exs = len(examples)
        sr = split_ranges + [1.]

        return (
            examples[int(sr[i] * n_exs) : int(sr[i+1] * n_exs)]
            for i in range(len(sr)-1)
        )

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

# def compute_loss(logits, y0, y1, y2, norm_space):
#     out_tag, out_type, out_ent = (out.to('cpu') for out in logits)
#     N, L, C1 = out_type.size()
#     # marker_idx = torch.tensor([i for i in range(L-1) if i%2!=0])
#     # out_type = out_type.index_select(1, marker_idx)
#     # out_ent = out_ent.index_select(1, marker_idx)

#     # Log likelihood
#     log_probs_tag = F.log_softmax(out_tag, dim=-1)
#     log_probs_type = F.log_softmax(out_type, dim=-1)
#     out_ent = torch.cat((log_probs_type[:,:,:-1], out_ent), dim=-1)
#     norm_scores = torch.matmul(out_ent, norm_space.T)
#     log_probs_name = F.log_softmax(norm_scores, dim=-1)

#     rho = 0.1
#     # NLLLoss of tags
#     loss0 = -(log_probs_tag[:,1:-1,1] * y0.long()[:,:,1]).sum() * rho
#     loss0 += -(log_probs_tag[:,1:-1,[0,2]] * y1.long()[:,:,[0,2]]).sum() * (1-rho) 
#     loss0 /= y1.sum()

#     # NLLLoss of entity types
#     loss1 = -(log_probs_type[:,1:-1,-1] * y1.long()[:,:,-1]).sum() * rho
#     loss1 += -(log_probs_type[:,1:-1,:-1] * y1.long()[:,:,:-1]).sum() * (1-rho)
#     loss1 /= y1.sum()

#     # NLLLoss of entity names 
#     loss2 = -(log_probs_name[:,1:-1,-1] * y2.long()[:,:,-1]).sum() * rho
#     loss2 += -(log_probs_name[:,1:-1,:-1] * y2.long()[:,:,:-1]).sum() * (1-rho)
#     loss2 /= y1.sum()

#     return loss0, loss1, loss2

def compute_retrieval_scores(logits, x_mask, y1, y2, name_norm):
    """
    measure micro retrieval performance
    
    precision: the percentage of mentions predicted that are correct
    recall: the percentage of entities present in the corpus that are found
            by the model
    """
    out_tag, out_type, out_ent = (out[:,1:-1,:].to('cpu') for out in logits)
    x_mask = x_mask[:,1:-1]
    # marker_idx = torch.tensor([i for i in range(L-1) if i%2!=0])
    # out_type = out_type.index_select(1, marker_idx)
    # out_type = out_type[:,:,3:]
    # out_ent = out_ent.index_select(1, marker_idx)
    # x_mask = x_mask.index_select(1, marker_idx)

    scores = []
    epsilon = 1e-7
    logprobs_type = F.log_softmax(out_type, dim=-1)
    for (out, y, norm) in ((out_type, y1, None),
                           (out_ent, y2, name_norm)):
        if norm is not None:
            out = torch.cat((logprobs_type[:,:,:-1], out_ent), dim=-1)
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
    logits_tag, logits_type, prj_name = (t[:,1:-1,:] for t in logits)
    dev = logits_tag.device
    name_space = name_space.to(dev)

    logits_tag = logits_tag.argmax(dim=-1)

    logprobs_type = F.log_softmax(logits_type, dim=-1)
    logits_type = logits_type.argmax(dim=-1)

    prj_name = torch.cat((logprobs_type[:,:,:-1], prj_name), dim=-1)
    name_scores = torch.matmul(prj_name, name_space.T)
    name_scores = name_scores.argmax(dim=-1)

    annt_dict = dict()
    for i, annts in enumerate(annotations):
        for bi, l, _, t, n in annts:
            annt_dict[(i, bi, l)] = (t, n[5:])

    pred_dict = dict()
    mention = None
    for i, ex in enumerate(logits_tag):
        for j, idx in enumerate(ex):
            if j == x_mask[i].sum():
                break
            ti = logits_type[i,j].item()
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
                cum.append(loss.item())
        else:
            for loss, cum in zip(losses, self.cum_losses_dev):
                cum.append(loss.item())
            if ret_scores is not None:
                s1, s2 = ret_scores
                self.ret_scores_type.append(s1)
                self.ret_scores_entity.append(s2)
    
    def report(self, mode='trn'):
        if mode == 'trn':
            loss_avg = (sum(l) / len(l) for l in self.cum_losses_trn)
            self.cum_losses_trn = [[], [], []]
            msg = (
                'Epoch {} Steps {:>5}/{} -- loss ({:.3f}, {:.3f}, {:.3f}) '
                ' lr {:.8f}'.format(self.epoch, self.steps, self.n_exs,
                                    *loss_avg, self.lr))
        elif mode == 'dev':
            loss_avg = (sum(l) / len(l) for l in self.cum_losses_dev)
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
        loss_avg = (sum(l) / len(l) for l in self.cum_losses_dev)
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
