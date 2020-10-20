# pylint: disable=redefined-outer-name

"""
train.py

Main script for training and saving E2E MD-EL models for PubTator annotation.
Modify the configuration file () for experiments.
"""

import logging
import argparse
import pickle
import code

import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer

from GoshiBoshi.config import MM_ST, BERT_MODEL
from GoshiBoshi.data import MedMentionDataset, batchify
from GoshiBoshi.model import JointMDEL_L, JointMDEL_H, JointMDEL_OT
import GoshiBoshi.utils as utils
from GoshiBoshi.loss import FocalLoss

logger = logging.getLogger(__name__)


def train(args, ds, mdl, crit, optim, sch, name_norm, stats):
    """Fit the model over an epoch"""
    mdl.train()
    train_it = DataLoader(ds['trn'],
                          batch_size=args.batch_size,
                          shuffle=True,
                          collate_fn=batchify)
    stats.n_exs = len(train_it)
    for batch in train_it:
        inputs, segs, x_mask, y0, y1, y2 = batch
        optim.zero_grad()
        out = mdl(inputs, segs, x_mask)
        l, l0, l1, l2 = crit(out, y0, y1, y2, name_norm) 
        l.backward()
        optim.step()
        nn.utils.clip_grad_norm_(mdl.parameters(), 10)
        sch.step()
        stats.update((l0, l1, l2))

        # Update stats
        if stats.steps % args.log_interval == 0:
            stats.lr = optim.param_groups[0]['lr']
            stats.report()
            # print(ds['trn'].not_found)
        if stats.steps % args.eval_interval == 0:
            dev_eval(args, ds, mdl, crit, name_norm, stats)


def dev_eval(args, ds, mdl, crit, name_norm, stats):
    """Validation loop"""
    mdl.eval()
    it = DataLoader(ds['dev'], batch_size=args.batch_size, shuffle=True,
                    collate_fn=batchify)
    with torch.no_grad():
        for i, batch in enumerate(it):
            inputs, segs, x_mask, y0, y1, y2 = batch
            out = mdl(inputs, segs, x_mask)
            l, l0, l1, l2 = crit(out, y0, y1, y2, name_norm)
            ret_scores = \
                utils.compute_retrieval_scores(out, x_mask, y1, y2, name_norm)
            if i == 0:  # debug
                _, pred, _ = out
                x_len = x_mask[0].sum()
                print(y1.long().argmax(dim=-1)[0][:x_len])
                print(pred[:,1:-1,:].argmax(dim=-1)[0][:x_len])
            stats.update((l0, l1, l2), ret_scores=ret_scores, mode='dev')
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
    parser.add_argument('--ds_medmention_path', type=str,
                        default='data/MedMentions/mm.pkl',
                        help='Path to the preprocessed MedMention data file')

    # Runtime
    parser.add_argument('--random_seed', type=int, default=12345,
                        help='Random seed')

    # Model configuration
    parser.add_argument('--model_name', type=str, default='tag-lo',
                choices=['one-tag', 'tag-lo', 'tag-hi'],
                help=('Select different model architecture. ('
                        'one-tag: IOB as in part of named entity classes, '
                        'tag-lo: seperate classifier for IOB chunking, '
                        'tag-hi: chunking classifier on top of the model)'))
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='Default learning rate')
    parser.add_argument('--num_warmup_steps', type=int, default=0,
                        help='Number of steps for warmup in optimize schedule')
    parser.add_argument('--num_training_steps', type=int, default=0,
                        help='Number of steps for training')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of examples in running train/valid steps')
    parser.add_argument('--max_sent_len', type=int, default=196,
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
        datefmt='%b%d %H:%M',
        handlers=[logging.FileHandler("train.log"),
                  logging.StreamHandler()]
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

    # Read a tokenizer (vocab is defined in the tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

    # Read a dataset
    exs = pickle.load(open(args.ds_medmention_path, 'rb'))
    ds = {mode: MedMentionDataset(exs, mode, 
                                  max_sent_len=args.max_sent_len,
                                  one_tag=(args.model_name=='one-tag'))
          for mode in ['trn', 'dev']}

    if args.num_training_steps == 0:
        args.num_training_steps = \
            int(len(ds['trn']) / args.batch_size) * args.epochs
    if args.num_warmup_steps == 0:
        args.num_warmup_steps = int(args.num_training_steps * .05)

    # Model
    if args.model_name == 'one-tag':
        model = JointMDEL_OT(device=args.device)
    if args.model_name == 'tag-lo':
        model = JointMDEL_L(device=args.device)
    elif args.model_name == 'tag-hi':
        model = JointMDEL_H(device=args.device)

    model.to(args.device)
    criterion = FocalLoss()
    optimizer = AdamW(model.parameters(), lr=float(args.lr))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        args.num_warmup_steps,
        args.num_training_steps
    )
    norm_space = torch.cat((exs['entity_names'],
                            torch.zeros(exs['entity_names'].size(-1)).unsqueeze(0)))

    # Train -------------------------------------------------------------------
    stats = utils.TraningStats()
    model.train()
    for epoch in range(1, args.epochs + 1):
        stats.epoch = epoch
        logger.info('*** Epoch %s starts ***', epoch)
        train(args, ds, model, criterion, optimizer, scheduler, norm_space, stats)
