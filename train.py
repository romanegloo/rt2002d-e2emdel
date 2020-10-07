"""
train.py

Main script for training and saving E2E MD-EL models for PubTator annotation.
Modify the configuration file () for experiments.
"""

import sys
import logging
import argparse
import pickle
import code

import torch
from torch import nn
from torch.utils.data import DataLoader

from GoshiBoshi.data import KaggleNERDataset, MedMentionDataset, batchify
from GoshiBoshi.model import JointMDEL
import GoshiBoshi.utils as utils

from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


def train(args, ds, mdl, optim, sch, name_norm, stats):
    mdl.train()
    train_it = DataLoader(ds['trn'],
                          batch_size=args.batch_size,
                          shuffle=True,
                          collate_fn=batchify)
    stats.n_exs = len(train_it)
    for batch in train_it:
        inputs, segs, x_mask, y1_mask, y2_mask = batch
        optim.zero_grad()
        out = mdl(inputs, segs, x_mask)
        losses = utils.compute_loss(out, y1_mask, y2_mask, name_norm)
        # loss1 = utils.compute_loss(out, type_labels, ent_labels, y_masks)
        losses[0].backward()
        optim.step()
        nn.utils.clip_grad_norm_(mdl.parameters(), 10)
        sch.step()
        stats.update(losses[2], losses[3])

        # Update stats
        if stats.steps % args.log_interval == 0:
            stats.lr = optim.param_groups[0]['lr']
            stats.report()
        if stats.steps % args.eval_interval == 0:
            dev_eval(args, ds, mdl, name_norm, stats)


def dev_eval(args, ds, mdl, name_norm, stats):
    mdl.eval()
    it = DataLoader(ds['dev'], batch_size=args.batch_size, shuffle=True,
                    collate_fn=batchify)
    with torch.no_grad():
        for i, batch in enumerate(it):
            inputs, segs, x_mask, y1_mask, y2_mask = batch
            out = mdl(inputs, segs, x_mask)
            losses = utils.compute_loss(out, y1_mask, y2_mask, name_norm)
            ret_scores = \
                utils.compute_retrieval_scores(out, y1_mask, y2_mask, x_mask)
            if i == 0:  # debug
                pred, _ = out
                pred = pred.to('cpu')
                N, L, C1 = pred.size()
                marker_idx = torch.tensor([i for i in range(L-1) if i%2!=0])
                pred = pred.index_select(1, marker_idx)
                print(y1_mask[:,:,:3].long().argmax(dim=-1)[0])
                print(pred[:,:,:3].argmax(dim=-1)[0])
                print(y1_mask[:,:,3:].long().argmax(dim=-1)[0])
                print(pred[:,:,3:].argmax(dim=-1)[0])
            stats.update(losses[1], losses[2], ret_scores=ret_scores, mode='dev')
            if i >= 200:
                break
    if stats.is_best():
        utils.save_model(mdl, args, stats)

    stats.report(mode='dev')
    mdl.train()
            

if __name__ == '__main__':
    # Configuration -----------------------------------------------------------
    parser = argparse.ArgumentParser(
        'Joint Mention Dection and Entity Linking to complete PubTator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Paths
    parser.add_argument('--data_dir', type=str,
                        default='data/', help='Data dicretory')
    parser.add_argument('--ds_kaggle_path', type=str,
                        default='data/KaggleNER/ner_dataset.csv',
                        help='Path to the Kaggle NER data file')
    parser.add_argument('--ds_medmention_path', type=str,
                        default='data/MedMentions/mm.pkl',
                        help='Path to the preprocessed MedMention data file')

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
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='Default learning rate')
    parser.add_argument('--num_warmup_steps', type=int, default=0,
                        help='Number of steps for warmup in optimize schedule')
    parser.add_argument('--num_training_steps', type=int, default=0,
                        help='Number of steps for training')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of examples in running train/valid steps')
    parser.add_argument('--max_sent_len', type=int, default=168,
                        help='Maximum sequence length')
                        
    # Runtime environmnet
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs to train')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log interval for training')
    parser.add_argument('--eval_interval', type=int, default=200,
                        help='Log interval for validation')

    args = parser.parse_args()

    # Logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s: [ %(message)s ]',
        datefmt='%b%d %H:%M'
    )
    fh = logging.FileHandler('train.log')
    logger.addHandler(fh)

    # Set random seed
    utils.set_seed(args.random_seed)
    # GPU
    if torch.cuda.is_available():
        args.device = torch.device('cuda:0')
        logger.info('Running on GPUs')
    else:
        args.device = torch.device('cpu')
        logger.info('Running on CPUs')
    

    # Read a tokenizer (vocab is defined in the tokenizer)
    tokenizer = utils.get_tokenizer(args.model_name)
    tokenizer.add_special_tokens({'bos_token': '[MB]', 'eos_token': '[ME]'})

    # Read a dataset
    exs = pickle.load(open(args.ds_medmention_path, 'rb'))
    ds = {mode: MedMentionDataset(exs, mode, max_sent_len=args.max_sent_len)
          for mode in ['trn', 'dev', 'tst']}

    if args.num_training_steps == 0:
        args.num_training_steps = \
            int(len(ds['trn']) / args.batch_size) * args.epochs
    if args.num_warmup_steps == 0:
        args.num_warmup_steps = int(args.num_training_steps * .05)

    # Model
    model = JointMDEL(args, emb_size=len(tokenizer), 
                      num_types=len(exs['typeIDs']))
    model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=float(args.lr))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        args.num_warmup_steps,
        args.num_training_steps
    )
    norm_space = torch.cat(
        (exs['entity_names'],
         torch.zeros(model.bert.config.hidden_size).unsqueeze(0)))

    # Train -------------------------------------------------------------------
    stats = utils.TraningStats()
    model.train()
    for epoch in range(1, args.epochs + 1):
        stats.epoch = epoch
        logger.info('*** Epoch {} starts ***'.format(epoch))
        train(args, ds, model, optimizer, scheduler, norm_space, stats)

# inference? (check this https://www.kaggle.com/abhishek/entity-extraction-model-using-bert-pytorch)
