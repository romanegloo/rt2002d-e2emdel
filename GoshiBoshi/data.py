"""data.py

DataLoader, Reads in training examples for mention detection and entity linking
"""

import logging
import code
from itertools import accumulate
import pickle
from typing import List, Tuple
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from GoshiBoshi.config import *
import GoshiBoshi.config as cfg
import GoshiBoshi.utils as utils


logger = logging.getLogger(__name__)

class MedMentionDataset(Dataset):

    def __init__(self, exs, mode, ni2cui, special_tokens=None,
                 max_sent_len=196, one_tag=False):
        """
        MedMentions dataset class

        Args:
            exs (tuple): Sentence-level annotation examples
            mode (str): Running mode {'trn', 'val', 'tst'}
            name_mapping (list): ni2cui (name index to CUI), i.e., list of CUIs
            special_tokens (list, optional): Mapping of the BERT special
                tokens. Defaults to None.
            max_sent_len (int, optional): Maximum length of input sentnece.
                Defaults to 196.
            one_tag (bool, optional): Flag indicating if the model is one-tag.
                Defaults to False.
        """
        self.mode = mode
        self.examples = [rec for rec in exs if rec['ds'] == mode]
        self.one_tag = one_tag
        self.tags = 'IOB'
        self.tag2idx = {t: i for i, t in enumerate(self.tags)}
        if not one_tag:
            self.types = MM_ST + ['N']  # N for 'N'ull class
        else:
            self.types = [MM_ST[int(i/2)]+'-B' if i%2==0 else MM_ST[int(i/2)]+'-I'
                         for i in range(2*len(MM_ST))] + ['N']
        self.type2idx = {k: i for (i, k) in enumerate(self.types)}
        self.ni2cui = ni2cui + ['N']
        self.cui2idx = defaultdict(list)
        for i, cui in enumerate(ni2cui):
            self.cui2idx[cui].append(i)
        self.cui2idx['N'] = [len(ni2cui)]
        if special_tokens is None:
            self.bert_special_tokens = cfg.BERT_SPECIAL_TOKENS
        else:
            self.bert_special_tokens = special_tokens
        self.max_sent_len = max_sent_len
        self.not_found = set()

    def __len__(self):
        return(len(self.examples))

    def __getitem__(self, idx):
        """
        Return each example in the following format:
            x: Tensor, sequence of token indices
            y1: List of entity type labels (token_ids, [label indices])
            y2: List of entity name labels (token_ids, [label indicies])
        """
        ex = self.examples[idx]
        src_len = len(ex['token_ids'][:self.max_sent_len])
        # x = torch.tensor([self.bert_special_tokens['[MASK]'], 0]).repeat(src_len)
        # for i, v in enumerate(ex['token_ids'][:self.max_sent_len]):
        #     x[2*i+1] = v
        x = torch.tensor([self.bert_special_tokens['[CLS]']] +
                         ex['token_ids'][:self.max_sent_len] +
                         [self.bert_special_tokens['[SEP]']])
        if self.mode == 'tst':
            return x, src_len, ex['annotations']

        g_i, g_o, g_b = (self.tag2idx[g] for g in 'IOB')
        t_n, e_n = self.type2idx['N'], self.cui2idx['N']

        # Tagging labels (IOB)
        y0 = torch.zeros(len(self.tags), dtype=torch.bool)
        y0[g_o] = True
        y0 = y0.repeat(src_len, 1)

        # Entity type labels (list of entity IDs over tokens)
        y1 = torch.zeros(len(self.types), dtype=torch.bool)
        y1[t_n] = True
        y1 = y1.repeat(src_len, 1)

        # Entity name labels
        y2 = torch.zeros(len(self.ni2cui), dtype=torch.bool)
        y2[e_n] = True
        y2 = y2.repeat(src_len, 1)

        for (bi, l, m, typeid, entid) in ex['annotations']:
            if bi+l-1 >= self.max_sent_len:
                continue
            if not self.one_tag:
                t = self.type2idx[typeid]
            else:
                t = self.type2idx[typeid+'-B']
            cui = entid[5:]
            if cui not in self.cui2idx:
                self.not_found.add(cui)
            e = self.cui2idx[cui]
            y0[bi, g_b], y0[bi+1:bi+l, g_i], y0[bi:bi+l, g_o] = True, True, False
            if not self.one_tag:
                y1[bi:bi+l, t], y1[bi:bi+l, t_n]= True, False
            else:
                y1[bi, t], y1[bi+1:bi+l, t+1], y1[bi:bi+l, t_n] = True, True, False
            y2[bi:bi+l, e], y2[bi:bi+l, e_n] = True, False

        return x, src_len, y0, y1, y2



def batchify(batch):
    tst = True if len(batch[0]) == 3 else False

    N = len(batch)
    x = [ex[0] for ex in batch]
    x = pad_sequence(x, batch_first=True)
    x_lens = [ex[1]+2 for ex in batch]
    y_lens = [ex[1] for ex in batch]
    segs = torch.zeros_like(x)
    x_mask = utils.sequence_mask(x_lens, max(x_lens))
    if tst:
        annt = [ex[2] for ex in batch]
        return x, segs, x_mask, annt

    C0, C1, C2 = [batch[0][i+2].size(-1) for i in range(3)]
    # todo shouldn't be filled with null class index?
    y0_mask = torch.zeros(N, max(y_lens), C0, dtype=torch.bool)
    y1_mask = torch.zeros(N, max(y_lens), C1, dtype=torch.bool)
    y2_mask = torch.zeros(N, max(y_lens), C2, dtype=torch.bool)
    for i, ex in enumerate(batch):
        y0_mask[i,:y_lens[i],:] = ex[2]
        y1_mask[i,:y_lens[i],:] = ex[3]
        y2_mask[i,:y_lens[i],:] = ex[4]

    return x, segs, x_mask, y0_mask, y1_mask, y2_mask
