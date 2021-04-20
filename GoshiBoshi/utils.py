"""
utils.py

Classes and methods used in common
"""
import code
import random
import csv
import logging
from collections import defaultdict
import os

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
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


class Entity:
    def __init__(self, id, st=None, st_root=None):
        self.str_id = id
        self.names = []
        self.st = st
        self.st_root = st_root
        self.mm_count = [0, 0, 0, 0]  # Trn, Dev, Tst, augmented


class Entities:
    def __init__(self):
        self.cuis = dict()
        # Read the UMLS semantic type 'is-a' relationships
        self.st_rel = self.read_st_rel()
        # Read CUIs
        self.read_cuis()
        # Filter CUIs and read names of them
        self.read_cui_names()

    def read_st_rel(self):
        """Read the hierarchical relationships between UMLS semantic types;
        this is for merging the descendents of 21 semantic types to the
        top-level.
        """
        st_map = {t: t for t in cfg.MM_ST}  # initialize with the 21 types
        with open(cfg.UMLS_SRSTRE) as f:
            while True:
                len_before = len(st_map)
                for line in f:
                    st1, r, st2, _ = line.split('|')
                    if st2 in st_map and r == 'T186':  # T186: 'is-a' relationship
                        st_map[st1] = st_map[st2]
                if len(st_map) != len_before:  # Repeat
                    f.seek(0)
                else:
                    break
        return st_map

    def read_cuis(self):
        """Reading CUIs that belong to the 21 semantic types and descendents"""
        print('=> Reading CUIs from MRSTY...')
        with open(cfg.UMLS_MRSTY) as f:
            for line in f:
                flds = line.split('|')
                if flds[1] in self.st_rel:
                    self.cuis[flds[0]] = Entity(flds[0], st=flds[1],
                                                st_root=self.st_rel[flds[1]])
        print(f'- {len(self.cuis)} CUIs found for the st21pv semantic types')

    def read_cui_names(self):
        """Filter CUIS with certian criteria and Read cui names from MRCONSO"""
        print('=> Reading CUI names from MRCONSO...')

        pbar = tqdm(total=15479756)
        with open(cfg.UMLS_MRCONSO) as f:
            for line in f:
                pbar.update()
                flds = line.split('|')
                # 0: CUI, 1: LAT, 2: TS, 4: STT, 6: ISPREF, 11: TTY, 14: STRING
                if flds[0] in self.cuis and \
                        flds[1] == 'ENG' and \
                        flds[2] == 'P' and \
                        flds[4] == 'PF' and \
                        flds[11] in cfg.MM_ONT and \
                        flds[16] == 'N':
                    if flds[14].lower() not in [n for n in self.cuis[flds[0]].names]:
                        self.cuis[flds[0]].names.append(flds[14].lower())
        pbar.close()

    def __len__(self):
        return len(self.cuis)

    def __getitem__(self, k):
        if k in self.cuis:
            return self.cuis[k]
        return

class TraningStats:
    def __init__(self):
        self.epoch = 0
        self.steps = 1
        self.n_exs = 0
        self.lr = 0
        self.loss = defaultdict(list)
        self.ret_outcomes = {
            't0': {'tp': 0, 'fp': 0, 'fn': 0},
            't1': {'tp': 0, 'fp': 0, 'fn': 0},
            't2': {'tp': 0, 'fp': 0, 'fn': 0}
        }
        self.ret_scores = {
            't0': defaultdict(list),
            't1': defaultdict(list),
            't2': defaultdict(list)
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
            if k1 == 't2' and self.best_score < f:
                self.is_best = True
                self.best_score = f
        self.ret_outcomes = {
            't0': {'tp': 0, 'fp': 0, 'fn': 0},
            't1': {'tp': 0, 'fp': 0, 'fn': 0},
            't2': {'tp': 0, 'fp': 0, 'fn': 0}
        }

def save_model(mdl, args, stat):
    checkpoint = {
        'model': mdl.state_dict(),
        'args': args,
        'stat': stat,
    }
    torch.save(checkpoint, args.best_mdl_file)
    logger.info(f'best score: {stat.best_score:.3f}, '
                f' Saving a checkpoint {args.best_mdl_file}')


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


def init_linear(input_linear):
    """Initialize linear transformation
    """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform_(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()

def log_sum_exp(vec, m_size):
    """
    calculate log of exp sum
    args:
        vec (batch_size, vanishing_dim, hidden_dim) : input tensor
        m_size : hidden_dim
    return:
        batch_size, hidden_dim
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M

    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)  # B * M