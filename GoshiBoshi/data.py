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

import GoshiBoshi.utils as utils


logger = logging.getLogger(__name__)

class MedMentionDataset(Dataset):

    def __init__(self, exs, mode, max_sent_len=160):
        self.examples = [rec for rec in exs['examples'] if rec['ds'] == mode]
        self.typeIDs = list('IOB') + exs['typeIDs'] + ['N']  # 'N' for no class
        self.typeID2idx = {k: i for (i, k) in enumerate(self.typeIDs)}
        self.names = exs['ni2cui'] + ['N']
        self.cui2idx = defaultdict(list)
        self.cui2idx['N'] = [len(exs['ni2cui'])]
        for k, v in exs['cui2ni'].items():
            self.cui2idx[k] = v
        self.bert_special_tokens = exs['bert_special_tokens_map']
        self.max_sent_len = max_sent_len

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
        # x = torch.tensor([self.bert_special_tokens['[ME]'],
        #                   self.bert_special_tokens['[MB]'],
        #                   0]).repeat(src_len)
        x = torch.tensor([self.bert_special_tokens['[MASK]'], 0]).repeat(src_len)
        for i, v in enumerate(ex['token_ids'][:self.max_sent_len]):
            x[2*i+1] = v
        x = torch.cat((torch.tensor([self.bert_special_tokens['[CLS]']]), x,
                       torch.tensor([self.bert_special_tokens['[SEP]']])))

        t_i, t_o, t_b = (self.typeID2idx[k] for k in 'IOB')
        t_n, e_n = self.typeID2idx['N'], self.cui2idx['N']

        # Entity type labels (list of entity IDs over tokens)
        y1 = torch.zeros(len(self.typeIDs), dtype=torch.bool)
        y1[t_o], y1[t_n] = True, True
        y1 = y1.repeat(src_len, 1)

        # Entity name labels
        y2 = torch.zeros(len(self.names), dtype=torch.bool)
        y2[e_n] = True
        y2 = y2.repeat(src_len, 1)

        for (bi, l, m, typeid, entid) in ex['annotations']:
            if bi+l-1 >= self.max_sent_len:
                continue
            t = self.typeID2idx[typeid]
            e = self.cui2idx[entid[5:]]
            y1[bi:bi+l, t], y1[bi+1:bi+l, t_i]= True, True
            y1[bi:bi+l, t_o], y1[bi:bi+l, t_n] = False, False
            y1[bi, t_b] = True
            y2[bi:bi+l, e], y2[bi:bi+l, e_n] = True, False
            # y1[b, t], y1[b, to] = True, False
            # y2[a:b+1, e], y2[a:b+1, no] = True, False
            # y2[b, e], y2[b, no] = True, False  # check this works
        
        return x, y1, y2, src_len


def batchify(batch):
    N = len(batch)
    C1 = batch[0][1].size(-1)
    C2 = batch[0][2].size(-1)
    x = [ex[0] for ex in batch]
    x = pad_sequence(x, batch_first=True)
    x_lengths = [2*ex[3]+2 for ex in batch]
    y_lengths = [ex[3] for ex in batch]
    segs = torch.zeros_like(x)
    x_mask = utils.sequence_mask(x_lengths, max(x_lengths))
    y1_mask = torch.zeros(N, max(y_lengths), C1, dtype=torch.bool)
    y2_mask = torch.zeros(N, max(y_lengths), C2, dtype=torch.bool)
    for i, ex in enumerate(batch):
        y1_mask[i,:y_lengths[i],:] = ex[1]
        y2_mask[i,:y_lengths[i],:] = ex[2]

    return x, segs, x_mask, y1_mask, y2_mask


class KaggleNERDataset(Dataset):
    lbl2idx = {k: i for i, k in enumerate(
        ['O', 'B-org', 'I-org', 'B-per', 'I-per', 'B-geo', 'I-geo',
         'B-tim', 'I-tim', 'B-art', 'I-art', 'B-gpe', 'I-gpe', 'B-eve', 'I-eve',
         'B-nat', 'I-nat']
    )}
    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return(len(self.examples))
    
    def __getitem__(self, idx):
        words, _, tags = self.examples[idx]
        sw_ids, token_start_idxs = self.subword_tokenize_to_ids(words)
        y = []
        tag_idx = 0
        for i, sw_id in enumerate(sw_ids):
            if sw_id in self.tokenizer.all_special_ids:
                y.append(self.lbl2idx['O'])
            elif i not in token_start_idxs:
                y.append(y[-1])
            else:
                y.append(self.lbl2idx[tags[tag_idx]])
                tag_idx += 1
        assert len(sw_ids) == len(y)
        
        return sw_ids, y
    
    def subword_tokenize(self, tokens):
        """Huggingface default tokenizer does not provide token boundaries which
        makes it difficult to align features on tokenized subwords. 
        `subword_tokenize` and `subword_tokenize_to_ids` provided sequence of
        subword indices along with its original token boundaries

        Parameters
            tokens: List of str
        
        Returns
            - List of subwords, flanked by the special symboles for BERT
            - List of indices of the tokenized subwords
        """
        tkn = self.tokenizer
        subwords = [tkn.tokenize(t) for t in tokens]
        subword_lengths = [len(seg) for seg in subwords]
        # Flatten subwords list and flank with BERT special tokens
        subwords = [tkn.cls_token] + \
            [subword for sublist in subwords for subword in sublist] + \
            [tkn.sep_token]
        token_start_idxs = [0] + list(accumulate(subword_lengths[:-1]))
        token_start_idxs = [1+l for l in token_start_idxs]
        return subwords, token_start_idxs

    def subword_tokenize_to_ids(self, tokens):
        """Converts subwords to ids after applying `subword_tokenize`
        
        Parameters
            Tokens: List of str
        
        Returns
            - List of subword IDs
            - A mask indiciating padding tokens
        """
        subwords, token_start_idxs = self.subword_tokenize(tokens)
        subword_ids = self.tokenizer.convert_tokens_to_ids(subwords)
        return subword_ids, token_start_idxs
