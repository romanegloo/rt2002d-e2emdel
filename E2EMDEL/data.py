"""data.py

DataLoader, Reads in training examples for mention detection and entity linking
"""

import logging
import csv
import code

import torch
from torch.utils.data import Dataset, DataLoader

from pytorch_fast_elmo import FastElmo, batch_to_char_ids

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
    char_ids = batch_to_char_ids(ds.examples[0][0])
    options_file = 'data/elmo/elmo_2x2048_256_2048cnn_1xhighway_options.json'
    weight_file = 'data/elmo/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5'
    elmo = FastElmo(options_file, weight_file)
    print(elmo(char_ids))
    code.interact(local=locals())