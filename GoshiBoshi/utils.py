"""
utils.py

Classes and methods used in common
"""
from collections import defaultdict
import random
import csv

import numpy as np
import torch
from torch.nn import functional as F

from transformers import AutoModel, AutoTokenizer

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
        with open(args.path_ds_kaggle, encoding='latin1') as f:
            next(f)
            reader = csv.reader(f)
            for l in reader:
                if l[0] != '':
                    examples.append(([l[1]], [l[2]], [l[3]]))
                else:
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

def compute_loss(criterion, logits, targets, pad_index=-1):
    """
    logits: N x L x C
    targets: N x L
    """
    logits = logits.view(-1, logits.size(-1))
    log_probs = F.log_softmax(logits, dim=-1)
    tgt_flat = targets.flatten()
    losses = criterion(log_probs, tgt_flat)
    mask = (tgt_flat != pad_index)
    loss = (losses * mask.long()).sum() / mask.sum()

    return loss

# class WVModel:
#     """WordVector Model"""
#     def __init__(self, vocab_size=0):
#         self.vocab_size = vocab_size
#         self.sym2idx = defaultdict(lambda: len(self.sym2idx))
#         self.idx2sym = dict()
#         self.emb = None
    
#     def __len__(self):
#         return len(self.sym2idx)
    
#     def __getitem__(self, k):
#         if isinstance(k, int):
#             return self.idx2sym[k]
#         elif isinstance(k, str):
#             return self.sym2idx[k]
#         else:
#             return

#     def close(self):
#         self.sym2idx = defaultdict(lambda: self.sym2idx['UNK'], self.sym2idx)
#         self.vocab_size = len(self)
