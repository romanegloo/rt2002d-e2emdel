"""data.py

DataLoader, Reads in training examples for mention detection and entity linking
"""

import logging
import code
from itertools import accumulate
import pickle

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import GoshiBoshi.utils as utils
# import utils

logger = logging.getLogger(__name__)

class MedMentionDataset(Dataset):

    def __init__(self, exs, mode):
        self.examples = [rec for rec in exs['examples'] if rec['ds'] == mode]
        self.typeIDs = ['O'] + exs['typeIDs']  # 'O' for no-type 
        self.typeID2idx = {k: i for (i, k) in enumerate(self.typeIDs)}
        self.entityIDs = ['O'] + exs['UMLS_concepts']  # 'O' for no-entity
        self.entityID2idx = {k: i for (i, k) in enumerate(self.entityIDs)}
        self.bert_special_tokens_map = exs['bert_special_tokens_map']

    def __len__(self):
        return(len(self.examples))

    def __getitem__(self, idx):
        ex = self.examples[idx]
        x = [self.bert_special_tokens_map['[CLS]']] + ex['token_ids']
        y1 = [self.typeID2idx['O']] * (len(ex['tokens']) + 1)
        y2 = [self.entityID2idx['O']] * (len(ex['tokens']) + 1)
        for (bi, l, _, typeid, entid) in ex['annotations']:
            y1[bi+1:bi+1+l] = [self.typeID2idx[typeid]] * l
            y2[bi+1:bi+1+l] = [self.entityID2idx[entid]] * l
        return x, y1, y2


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

def batchify(batch, device='cuda:0'):
    inputs = [torch.tensor(ex[0], dtype=torch.long) for ex in batch]
    inputs = pad_sequence(inputs, batch_first=True)
    labels = [torch.tensor(ex[1], dtype=torch.long) for ex in batch]
    labels = pad_sequence(labels, batch_first=True, padding_value=-1)
    src_lengths = [len(ex[0]) for ex in batch]
    segs = torch.zeros_like(inputs)
    attention_masks = utils.sequence_mask(src_lengths, inputs.size(1))

    return inputs, labels, segs, attention_masks


if __name__ == '__main__':
    tokenizer = utils.get_tokenizer('SciBERT')
    code.interact(local=locals())
