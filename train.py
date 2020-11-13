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

from transformers import AdamW
from transformers import AutoTokenizer

import GoshiBoshi.config as cfg
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
        stats.update_trn((l0, l1, l2))

        # Update stats
        if stats.steps % args.log_interval == 0:
            stats.lr = optim.param_groups[0]['lr']
            stats.report_trn()
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
            ret_out = utils.retrieval_outcomes(out, batch, name_norm)
            stats.update_dev((l0, l1, l2), ret_out)
            if i >= 300:
                break
    stats.report_dev()
    if stats.is_best:
        utils.save_model(mdl, args, stats)


if __name__ == '__main__':
    # Configuration -----------------------------------------------------------
    parser = argparse.ArgumentParser(
        'Joint Mention Dection and Entity Linking to complete PubTator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model configuration
    parser.add_argument('--model_name', type=str, default=cfg.MDL_NAME,
                        choices=['one-tag', 'tag-lo', 'tag-hi'],
                        help='Different models with varying IOB tagging positions')
    parser.add_argument('--batch_size', type=int, default=cfg.BSZ,
                        help='Number of examples in train/valid runs')
    parser.add_argument('--max_sent_len', type=int, default=cfg.MAX_SENT_LEN,
                        help='Maximum sequence length')

    # Optimization
    parser.add_argument('--lr', type=float, default=cfg.LR,
                        help='Default learning rate')
    parser.add_argument('--scheduler', type=str, default='linear',
                        choices=['linear', 'cyclic_cosine', 'plateau'],
                        help='Learning rate scheduler')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Focal loss parameter, alpha')
    parser.add_argument('--focal_gamma', type=float, default=2.,
                        help='Focal loss parameter, gamma')

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
    utils.set_seed(cfg.RND_SEED)

    # GPU
    if torch.cuda.is_available():
        args.device = torch.device('cuda:0')
        logger.info('Running on GPUs')
    else:
        args.device = torch.device('cpu')
        logger.info('Running on CPUs')

    # Load a pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.BERT_MODEL)

    # Read a dataset
    exs = pickle.load(open(cfg.MM_DS_FILE, 'rb'))
    special_tokens = dict(zip(tokenizer.all_special_tokens,
                              tokenizer.all_special_ids))
    ds = {mode: MedMentionDataset(exs['examples'], mode, exs['ni2cui'],
                                  special_tokens=exs['bert_special_tokens_map'],
                                  max_sent_len=args.max_sent_len,
                                  one_tag=(args.model_name=='one-tag'))
          for mode in ['trn', 'dev']}

    # Default values for optimization
    args.num_training_steps = int(len(ds['trn']) / args.batch_size) * args.epochs
    args.num_warmup_steps = int(args.num_training_steps * .05)


    # Model
    if args.model_name == 'one-tag':
        model = JointMDEL_OT(args)
    if args.model_name == 'tag-lo':
        model = JointMDEL_L(args)
    elif args.model_name == 'tag-hi':
        model = JointMDEL_H(args)

    model.to(args.device)
    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    optimizer = AdamW(model.parameters(), lr=float(args.lr))
    if args.scheduler == 'linear':
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            args.num_warmup_steps,
            args.num_training_steps
        )
    elif args.scheduler == 'cyclic_cosine':
        from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            args.num_warmup_steps,
            args.num_training_steps,
            int(args.epochs / 2)
        )
    elif args.scheduler == 'plateau':
        raise NotImplementedError

    norm_space = torch.cat((exs['name_embs'],
                            torch.zeros(exs['name_embs'].size(-1)).unsqueeze(0)))

    # Train -------------------------------------------------------------------
    stats = utils.TraningStats()
    model.train()
    for epoch in range(1, args.epochs + 1):
        stats.epoch = epoch
        logger.info('*** Epoch %s starts ***', epoch)
        train(args, ds, model, criterion, optimizer, scheduler, norm_space, stats)
