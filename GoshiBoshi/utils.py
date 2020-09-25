"""
utils.py

Classes and methods used in common
"""
from collections import defaultdict
import random
import csv
import logging

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

def compute_retrieval_scores(logits, targets, pad_index=-1):
    """
    logits (pred): N x L x C
    targets (truth): N x L

    Assume that 0 is for negative class.

    precision: the percentage of mentions predicted that are correct
    recall: the percentage of entities present in the corpus that are found
            by the model
    """
    pred = logits.argmax(dim=-1)
    mask = (targets != pad_index)
    mask_pred = (pred != 0).logical_and(mask)
    mask_truth = (targets != 0).logical_and(mask)

    tp = (pred * mask == targets).sum().item()
    fp = (pred[mask_pred] != targets[mask_pred]).sum().item()
    fn = (targets[mask_truth] != pred[mask_truth]).sum().item()

    epsilon = 1e-7
    prec = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * prec * recall / (prec + recall + epsilon)
    return prec, recall, f1

class TraningStats:
    def __init__(self):
        self.epoch = 0
        self.steps = 1
        self.n_exs = 0
        self.lr = 0
        # # Analysis purpose for later use
        # self.train_loss_log = []
        # self.valid_loss_log = []
        self.cum_train_loss = []
        self.cum_valid_loss = []
        self.ret_scores = [[], [], []]  # precision, recall, f1
    
    def update(self, loss, ret_scores=None, mode='trn'):
        if mode == 'trn':
            self.steps += 1
            self.cum_train_loss.append(loss)
        else:
            self.cum_valid_loss.append(loss)
            if ret_scores is not None:
                for (i, score) in enumerate(ret_scores):
                    self.ret_scores[i].append(score)
    
    def report(self, mode='trn'):
        if mode == 'trn':
            loss_ = sum(self.cum_train_loss) / len(self.cum_train_loss)
            self.cum_train_loss = []
            msg = (
                'Epoch {} Steps {:>5}/{} -- loss {:.3f}, lr {:.8f}'
                ''.format(self.epoch, self.steps, self.n_exs, loss_, self.lr)
            )
        else:
            loss_ = sum(self.cum_valid_loss) / len(self.cum_valid_loss)
            if len(self.ret_scores[0]) > 0:
                prec, recall, f1 = \
                    (sum(scores) / len(scores) for scores in self.ret_scores)
            else:
                prec, recall, f1 = -1, -1, -1
                    
            self.cum_valid_loss = []
            self.ret_scores = [[], [], []]
            msg = (
                '[DEV] loss {:.3f}, precision {:.3f}, recall {:.3f}, f1 {:.3f}'
                ''.format(loss_, prec, recall, f1)
            )
        logger.info(msg)
            
