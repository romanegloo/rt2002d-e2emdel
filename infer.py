"""
infer.py

Run test with the best saved model
"""

import code
import os
import logging
import argparse
import pickle
from tqdm import tqdm

import torch
# from torch import nn
from torch.utils.data import DataLoader

from GoshiBoshi.config import MM_ST
from GoshiBoshi.data import MedMentionDataset, batchify
from GoshiBoshi.model import JointMDEL_L, JointMDEL_H, JointMDEL_OT
import GoshiBoshi.utils as utils

logger = logging.getLogger(__name__)

# Configuration -----------------------------------------------------------
parser = argparse.ArgumentParser(
    'Joint Mention Dection and Entity Linking to complete PubTator',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# Paths
parser.add_argument('--data_dir', type=str, default='data/',
                    help='Data dicretory')
parser.add_argument('--best_mdl_file', type=str, default='best.pt',
                    help='Filename of the best model')
parser.add_argument('--ds_medmention_path', type=str,
                    default='data/MedMentions/mm.pkl',
                    help='Path to the preprocessed MedMention data file')

# Runtime
parser.add_argument('--random_seed', type=int, default=12345,
                    help='Random seed')

# Model configuration
parser.add_argument('--batch_size', type=int, default=8,
                    help='Number of examples in running train/valid steps')
parser.add_argument('--max_sent_len', type=int, default=196,
                    help='Maximum sequence length')
args = parser.parse_args()

# Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: [ %(message)s ]',
    datefmt='%b%d %H:%M',
    handlers=[
            logging.FileHandler("train.log"),
            logging.StreamHandler()
    ]
)

# Set random seed
utils.set_seed(args.random_seed)

# GPU
if torch.cuda.is_available():
    args.device = torch.device('cuda:0')
    logger.info('Running on GPUs')
else:
    args.device = torch.device('cpu')
    logger.info('Running on CPUs')

mode = 'tst'
# Load test dataset
exs = pickle.load(open(args.ds_medmention_path, 'rb'))
ds = MedMentionDataset(exs, mode, max_sent_len=args.max_sent_len)

checkpoint = torch.load(os.path.join(args.data_dir, args.best_mdl_file))
args_saved = checkpoint['args']

# Load a saved model
if args_saved.model_name == 'tag-lo':
    model = JointMDEL_L(device=args.device)
elif args_saved.model_name == 'tag-hi':
    model = JointMDEL_H(device=args.device)
elif args_saved.model_name == 'one-tag':
    model = JointMDEL_OT(device=args.device)
model.to(args.device)
model.load_state_dict(checkpoint['model'])
norm_space = torch.cat((exs['entity_names'],
                        torch.zeros(exs['entity_names'].size(-1)).unsqueeze(0)))

# Evaluate
stats = utils.TraningStats()
model.eval()
it = DataLoader(ds, batch_size=args.batch_size, collate_fn=batchify)
pbar = tqdm(total=len(it))
TP, FP, FN = 0, 0, 0
with torch.no_grad():
    for i, batch in enumerate(it):
        inputs, segs, x_mask, annts = batch
        out = model(inputs, segs, x_mask)
        tp, fp, fn = utils.infer_ret(out, x_mask, annts, 
                                     norm_space, ds.types, ds.names)
        TP += tp
        FP += fp
        FN += fn
        pbar.set_description('(%s %s %s)' % (TP, FP, FN))
        pbar.update()
pbar.close()

epsilon = 1e-7
prec = TP / (TP + FP + epsilon)
recall = TP / (TP + FN + epsilon)
f1 = 2 * prec * recall / (prec + recall + epsilon) 

print(prec, recall, f1)