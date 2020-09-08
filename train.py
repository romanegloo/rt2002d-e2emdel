"""
train.py

Main script for training and saving E2E MD-EL models for PubTator annotation.
Modify the configuration file () for experiments.
"""

import sys
import logging
import argparse
import code

import torch
from torch import nn
from torch.utils.data import DataLoader

from GoshiBoshi.data import KaggleNERDataset, batchify
from GoshiBoshi.model import JointMDEL
import GoshiBoshi.utils as utils

from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


def train(args, ds, mdl, crit, optim, sch, stats):
    mdl.train()
    train_it = DataLoader(ds['tr'],
                          batch_size=args.batch_size,
                          shuffle=True,
                          collate_fn=batchify)
    stats.n_exs = len(train_it)
    for batch in train_it:
        optim.zero_grad()
        outputs = mdl(batch)

        loss = utils.compute_loss(crit, outputs[0], batch[1].to(args.device))
        loss.backward()
        optim.step()
        nn.utils.clip_grad_norm_(mdl.parameters(), 10)
        sch.step()

        stats.update(loss.item())

        # Update stats
        if stats.steps % args.log_interval == 0:
            stats.lr = optim.param_groups[0]['lr']
            stats.report()



        


if __name__ == '__main__':
    # Configuration -----------------------------------------------------------
    parser = argparse.ArgumentParser(
        'Joint Mention Dection and Entity Linking to complete PubTator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Paths
    parser.add_argument('--dir_data_dir', type=str,
                        default='data/', help='Data dicretory')
    parser.add_argument('--path_ds_kaggle', type=str,
                        default='data/KaggleNER/ner_dataset.csv',
                        help='Path to the Kaggle NER data file')

    # Runtime
    parser.add_argument('--random_seed', type=int, default=12345,
                        help='Random seed')

    # Model configuration
    parser.add_argument('--model_name', type=str, default='SciBERT',
                        choices=['SciBERT'],
                        help='Encoder model type')
    parser.add_argument('--dataset', type=str, default='KaggleNER',
                        choices=['KaggleNER'],
                        help='Dataset to which model will fit')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Default learning rate')
    parser.add_argument('--num_warmup_steps', type=int, default=0,
                        help='Number of steps for warmup in optimize schedule')
    parser.add_argument('--num_training_steps', type=int, default=0,
                        help='Number of steps for training')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of examples in running train/valid steps')
    parser.add_argument('--max_sent_len', type=int, default=256,
                        help='Maximum sequence length')
                        
    # Runtime environmnet
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log interval for training')
    parser.add_argument('--eval_interval', type=int, default=2000,
                        help='Log interval for validation')

    args = parser.parse_args()

    # Logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s: [ %(message)s ]',
        datefmt='%b%d %H:%M'
    )

    # Set random seed
    utils.set_seed(args.random_seed)
    # GPU
    if torch.cuda.is_available():
        args.device = torch.device('cuda:0')
    else:
        args.device = torch.device('cpu')
    

    # Read a tokenizer (vocab is defined in the tokenizer)
    tokenizer = utils.get_tokenizer(args.model_name)
    # Read a dataset
    ds = dict()
    ds['tr'], ds['va'], ds['ts'] = (
        KaggleNERDataset(examples=exs, tokenizer=tokenizer) for exs in
        utils.csv_to_ner_split_examples(args)
    )
    if args.num_training_steps == 0:
        args.num_training_steps = \
            int(len(ds['tr']) / args.batch_size) * args.epochs
    if args.num_warmup_steps == 0:
        args.num_warmup_steps = int(args.num_training_steps * .05)

    # Model
    model = JointMDEL(args).to(args.device)
    criterion = nn.NLLLoss(ignore_index=-1, reduction='none')
    optimizer = AdamW(model.parameters(), lr=float(args.lr))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        args.num_warmup_steps,
        args.num_training_steps
    )

    # Train -------------------------------------------------------------------
    stats = utils.TraningStats()
    model.train()
    for epoch in range(1, args.epochs + 1):
        stats.epoch = epoch
        logger.info('*** Epoch {} starts ***'.format(epoch))
        train(args, ds, model, criterion, optimizer, scheduler, stats)
        break

# inference? (check this https://www.kaggle.com/abhishek/entity-extraction-model-using-bert-pytorch)