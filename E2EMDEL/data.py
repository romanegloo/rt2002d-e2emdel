"""data.py

DataLoader, Reads in training examples for mention detection and entity linking
"""

import logging
import csv
import code

import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

class KaggleNERDataset(Dataset):
    fpath = 'data/KaggleNER/ner_dataset.csv'
    def __init__(self, vocab=None):
        """
        asdf
        """
        self.examples = []
        self.vocab = vocab

        logger.info('Reading Kaggle NER corpus')
        with open(self.fpath, encoding='latin1') as f:
            next(f)
            reader = csv.reader(f)
            for l in reader:
                if l[0] != '':
                    self.examples.append(([l[1]], [l[2]], [l[3]]))
                else:
                    self.examples[-1][0].append(l[1])
                    self.examples[-1][1].append(l[2])
                    self.examples[-1][2].append(l[3])

    def __len__(self):
        return(len(self.examples))
    
    def __getitem__(self, idx):
        return idx


if __name__ == '__main__':
    ds = KaggleNERDataset()
    print(' '.join(ds.examples[0][0]))
    code.interact(local=locals())