"""
utils.py

Classes and methods used in common
"""
from collections import defaultdict
import random
import csv
import logging
import code

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
        return AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

def get_bert_model(mdl_name):
    if mdl_name == 'SciBERT':
        return AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

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

def compute_loss(logits, y1, y2, norm_space):
    out_type, out_ent = logits
    out_type = out_type.to('cpu')
    out_ent = out_ent.to('cpu')
    N, L, C1 = out_type.size()
    marker_idx = torch.tensor([i for i in range(L-1) if i%2!=0])
    out_type = out_type.index_select(1, marker_idx)
    out_ent = out_ent.index_select(1, marker_idx)

    # NLLLoss over entity types
    log_probs_iob = F.log_softmax(out_type[:,:,:3], dim=-1)
    log_probs_type = F.log_softmax(out_type[:,:,3:], dim=-1)

    rho = 0.1
    loss1 = -(log_probs_iob[:,:,1] * y1.long()[:,:,1]).sum() * rho
    loss1 += -(log_probs_iob[:,:,[0,2]] * y1.long()[:,:,[0,2]]).sum() * (1-rho) 
    loss1 /= y1.sum()

    loss2 = -(log_probs_type[:,:,-1] * y1.long()[:,:,-1]).sum() * rho
    loss2 += -(log_probs_type[:,:,:-1] * y1.long()[:,:,3:-1]).sum() * (1-rho)
    loss2 /= y1.sum()

    # loss1 = -(log_probs[:,:,:-1] * y1.long()[:,:,:-1]).sum()
    # loss1 += -(log_probs[:,:,-1] * y1.long()[:,:,-1]).sum() * .1
    # loss1 /= y1.sum()

    # NLLLoss over entity names 
    norm_scores = torch.matmul(out_ent, norm_space.T)
    log_probs = F.log_softmax(norm_scores, dim=-1)
    loss3 = -(log_probs[:,:,-1] * y2.long()[:,:,-1]).sum() * rho
    loss3 += -(log_probs[:,:,:-1] * y2.long()[:,:,:-1]).sum() * (1-rho)
    loss3 /= y1.sum()

    return (loss1+loss2+loss3), loss1, loss2, loss3

def compute_retrieval_scores(logits, y1, y2, x_mask,
                             y1_null_idx=1, y2_null_idx=30415):
    """
    precision: the percentage of mentions predicted that are correct
    recall: the percentage of entities present in the corpus that are found
            by the model
    """
    out_type, out_ent = logits
    out_type = out_type.to('cpu')
    out_ent = out_ent.to('cpu')
    N, L, C1 = out_type.size()
    marker_idx = torch.tensor([i for i in range(L-1) if i%2!=0])
    out_type = out_type.index_select(1, marker_idx)
    out_type = out_type[:,:,3:]
    out_ent = out_ent.index_select(1, marker_idx)
    x_mask = x_mask.index_select(1, marker_idx)

    scores = []
    epsilon = 1e-7
    for (out, y, null_idx) in ((out_type, y1, y1_null_idx),
                               (out_ent, y2, y2_null_idx)):
        pred = out.argmax(dim=-1)
        gt = y.long().argmax(dim=-1)
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

class TraningStats:
    def __init__(self):
        self.epoch = 0
        self.steps = 1
        self.n_exs = 0
        self.lr = 0
        self.cum_trn_loss1 = []
        self.cum_trn_loss2 = []
        self.cum_dev_loss1 = []
        self.cum_dev_loss2 = []
        self.best_dev_loss = 999999
        self.ret_scores_type = []
        self.ret_scores_entity = []
    
    def update(self, loss1, loss2, ret_scores=None, mode='trn'):
        if mode == 'trn':
            self.steps += 1
            self.cum_trn_loss1.append(loss1.item())
            self.cum_trn_loss2.append(loss2.item())
        else:
            self.cum_dev_loss1.append(loss1.item())
            self.cum_dev_loss2.append(loss2.item())
            if ret_scores is not None:
                s1, s2 = ret_scores
                self.ret_scores_type.append(s1)
                self.ret_scores_entity.append(s2)
    
    def report(self, mode='trn'):
        if mode == 'trn':
            loss1_ = sum(self.cum_trn_loss1) / len(self.cum_trn_loss1)
            loss2_ = sum(self.cum_trn_loss2) / len(self.cum_trn_loss2)
            self.cum_trn_loss1 = []
            self.cum_trn_loss2 = []
            msg = (
                'Epoch {} Steps {:>5}/{} -- loss ({:.3f}, {:.3f}) lr {:.8f}'
                ''.format(self.epoch, self.steps, self.n_exs,
                          loss1_, loss2_, self.lr))
        elif mode == 'dev':
            loss1_ = sum(self.cum_dev_loss1) / len(self.cum_dev_loss1)
            loss2_ = sum(self.cum_dev_loss2) / len(self.cum_dev_loss2)
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

            self.cum_dev_loss1, self.cum_dev_loss2 = [], []
            self.ret_scores_type = []
            self.ret_scores_entity = []
            msg = (
                '[DEV] loss ({:.3f}, {:.3f}), '
                'ret. (p1 {:.3f} r1 {:.3f} f1 {:.3f}) '
                '(p2 {:.4f} r2 {:.4f} f2 {:.4f})'
                ''.format(loss1_, loss2_, p1, r1, f1a, p2, r2, f1b)
            )
        logger.info(msg)
            
    def is_best(self):
        if self.steps < 1000:
            return False
        loss1 = sum(self.cum_dev_loss1) / len(self.cum_dev_loss1)
        loss2 = sum(self.cum_dev_loss2) / len(self.cum_dev_loss2)
        loss = loss1 + loss2
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
    fpath = os.path.join('data/', fname)
    if not os.path.exists(fpath):
        torch.save(checkpoint, fpath)
        logger.info(f'    - Saving checkpoint {fname}')
