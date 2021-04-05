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

import GoshiBoshi.config as cfg
import GoshiBoshi.utils as utils


logger = logging.getLogger(__name__)


class MedMentionDataset(Dataset):

    def __init__(self, exs, ds_type, mdl_type, ni2cui,
                 max_sent_len=cfg.MAX_SENT_LEN, nn='mm'):
        """ MedMentions dataset class

        Args:
            exs (tuple): Sentence-level annotation examples
            ds_type (str): Running mode {'trn', 'dev', 'tst'}
            mdl_type (str): data format depends on model type
            ni2cui (list): name index to cui mapping, i.e., list of CUIs
            nn (str): name normalization mode {'mm', 'tst_exc'}
                'mm' for the entire MedMentions CUIs
                'tst_exc' for the CUIs only appear in the test dataset
        """
        self.ds_type = ds_type
        self.mdl_type = mdl_type
        self.max_sent_len = max_sent_len

        # Examples
        self.examples = [rec for rec in exs if rec['ds'] == ds_type]
        # Semantic types
        if self.mdl_type != 'one-tag':
            self.st = cfg.MM_ST + ['N']
        else:
            self.st = [cfg.MM_ST[i//2]+'-B' if i%2==0 else cfg.MM_ST[i//2]+'-I'
                       for i in range(2*len(cfg.MM_ST))] + ['N']
        self.st2idx = {k: i for (i, k) in enumerate(self.st)}

        # Entity name normalization indices
        # - We have a different entity name norm space depends on ds_type
        self.ni2cui = []
        for n in ni2cui:
            if ds_type == 'trn' and n[2][0] > 0:
                # Anything seen in the training dataset
                self.ni2cui.append(n)
            elif ds_type == 'dev' and n[2][1] > 0:
                # Anything seen in the validation dataset
                self.ni2cui.append(n)
            elif ds_type == 'tst':
                if nn == 'tst_exc':
                    if n[2][0] == 0 and n[2][2] > 0: # CUIs in Tst exclusive
                        self.ni2cui.append(n)
                elif nn == 'mm':
                    if sum(n[2][:3]) - n[2][3] > 0:
                        self.ni2cui.append(n)
        logger.info('ds_type: {}, nn space: {} [len {}]'.format(ds_type, nn,
                                                            len(self.ni2cui)))
        self.nids = [n[0] for n in self.ni2cui]

        if self.ni2cui[-1][1] != 'N':  # Add 'N'ull class, if not exist
            self.ni2cui.append((-1, 'N', [0, 0, 0, 0]))

        self.cui2idx = dict()
        for i, n in enumerate(self.ni2cui):
            if n[1] in self.cui2idx:
                self.cui2idx[n[1]]['indices'].append(i)
            else:
                self.cui2idx[n[1]] = {
                    'tst_exc': (n[2][2] > 0 and n[2][0] == 0),
                    'indices': [i]
                }

    def __len__(self):
        return(len(self.examples))

    def __getitem__(self, idx):
        """ Return each example in the following format:
        """
        ex = self.examples[idx]
        x = torch.tensor([cfg.BERT_SP_MAP['[CLS]']] +
                         ex['token_ids'][:self.max_sent_len-2] +
                         [cfg.BERT_SP_MAP['[SEP]']])
        src_len = len(ex['token_ids'][:cfg.MAX_SENT_LEN-2]) + 2

        # Create default labels (y0, y1, y2)
        I, O, B = 0, 1, 2  # indices for each tag
        TN, EN = self.st2idx['N'], self.cui2idx['N']['indices'][0]

        y0 = torch.zeros(len('IOB'), dtype=torch.bool)
        y0[O] = True
        y0 = y0.repeat(src_len, 1)

        y1 = torch.zeros(len(self.st), dtype=torch.bool)
        y1[TN] = True
        y1 = y1.repeat(src_len, 1)

        y2 = torch.zeros(len(self.ni2cui), dtype=torch.bool)
        y2[EN] = True
        y2 = y2.repeat(src_len, 1)

        # Fill in the annotation labels
        for (bi, l, m, typeid, entid) in ex['annotations']:
            bi += 1  # for [CLS] token
            if bi + 1 + l > self.max_sent_len:
                continue

            # y0: IOB (only the first token of the subword tokens is 'B')
            y0[bi, B], y0[bi+1:bi+l, I], y0[bi:bi+l, O] = True, True, False

            # y1: Semantic types
            if self.mdl_type == 'one-tag':
                tgtB = self.st2idx[typeid+'-B']
                tgtI = self.st2idx[typeid+'-I']
                y1[bi, tgtB], y1[bi+1:bi+l, tgtI], y1[bi:bi+l, TN] = \
                    True, True, False
            else:
                tgt = self.st2idx[typeid]
                y1[bi:bi+l, tgt], y1[bi:bi+l, TN] = True, False

            # y2: Entity
            cui = entid[5:]
            if cui in self.cui2idx:
                # cui might not be indexed in case it is in both trn and tst
                # while testing zero-shot
                tgt = self.cui2idx[entid[5:]]['indices']  # without 'UMLS:'
                y2[bi:bi+l, tgt], y2[bi:bi+l, EN] = True, False

        if self.ds_type == 'tst':
            return x, src_len, y0, y1, y2, ex['annotations']
        return x, src_len, y0, y1, y2


def batchify(batch):
    bsz = len(batch)

    # sort lengths
    src_lens, sorted_idx = \
        torch.tensor([ex[1] for ex in batch]).sort(descending=True)

    # x
    x = pad_sequence([batch[i][0] for i in sorted_idx], batch_first=True)
    # attention masks
    x_mask = utils.sequence_mask(src_lens, max(src_lens))
    # targets
    y0 = pad_sequence([batch[i][2] for i in sorted_idx], batch_first=True)
    y1 = pad_sequence([batch[i][3] for i in sorted_idx], batch_first=True)
    y2 = pad_sequence([batch[i][4] for i in sorted_idx], batch_first=True)

    if len(batch[0]) == 6:  # test
        annt = [ex[5] for ex in batch]
        annt = [annt[i] for i in sorted_idx]
        return x, x_mask, y0, y1, y2, annt

    return x, x_mask, y0, y1, y2
