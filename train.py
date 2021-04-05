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

from GoshiBoshi import IOB_HI, IOB_LO, IOB_ONE
import GoshiBoshi.config as cfg
from GoshiBoshi.data import MedMentionDataset, batchify
import GoshiBoshi.utils as utils
from GoshiBoshi.loss import FocalLoss

logger = logging.getLogger(__name__)


def train(args, ds, mdl, optim, sch, stats):
    """Fit the model over an epoch"""
    mdl.train()
    train_it = DataLoader(ds['trn'],
                          batch_size=args.batch_size,
                          shuffle=True,
                          collate_fn=batchify)
    stats.n_exs = len(train_it)
    for batch in train_it:
        optim.zero_grad()
        losses = mdl(batch, nn_trn)
        losses[0].backward()
        optim.step()
        nn.utils.clip_grad_norm_(mdl.parameters(), 10)
        sch.step()
        stats.update_trn(losses[1:])

        # Update stats
        if stats.steps % args.log_interval == 0:
            stats.lr = optim.param_groups[0]['lr']
            stats.report_trn()
        if stats.steps % args.eval_interval == 0:
            dev_eval(args, ds, mdl, stats)


def dev_eval(args, ds, mdl, stats):
    """Validation loop"""
    mdl.eval()
    it = DataLoader(ds['dev'], batch_size=args.batch_size, shuffle=True,
                    collate_fn=batchify)
    with torch.no_grad():
        for i, batch in enumerate(it):
            # inputs, segs, x_mask, y0, y1, y2 = batch
            ret_out, l0, l1, l2 = mdl(batch, nn_dev)
            # ret_out = utils.retrieval_metrics(out, batch, name_norm)
            stats.update_dev((l0, l1, l2), ret_out)
            if i >= 500:
                break
    stats.report_dev()
    if stats.is_best:
        utils.save_model(mdl, args, stats)
    mdl.train()


if __name__ == '__main__':
    # Configuration -----------------------------------------------------------
    parser = argparse.ArgumentParser(
        'Joint Mention Dection and Entity Linking to complete PubTator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Path
    parser.add_argument('--ds_path', type=str, default=cfg.MM_DS_FILE,
                        help="Path to a training dataset file")
    parser.add_argument('--best_mdl_file', type=str, default=cfg.BEST_MDL_FILE,
                        help="File path where to save the best trained model")

    # Model configuration
    parser.add_argument('--model_name', type=str, default=cfg.MDL_NAME,
                        choices=['one-tag', 'tag-lo', 'tag-hi'],
                        help='Different models with varying IOB tagging positions')
    parser.add_argument('--batch_size', type=int, default=cfg.BSZ,
                        help='Number of examples in train/valid runs')
    parser.add_argument('--x_max_sent_len', type=int, default=cfg.MAX_SENT_LEN,
                        help='Maximum sentence length of inputs to BERT '
                        '(in number of tokens)')

    # Optimization
    parser.add_argument('--lr', type=float, default=cfg.LR,
                        help='Default learning rate')
    parser.add_argument('--scheduler', type=str, default='linear',
                        choices=['linear', 'cyclic_cosine', 'plateau'],
                        help='Learning rate scheduler')
    parser.add_argument('--focal_alpha', type=float, default=1.0,
                        help='Focal loss parameter, alpha')
    parser.add_argument('--focal_gamma1', type=float, default=5.,
                        help='Focal loss parameter, gamma')
    parser.add_argument('--focal_gamma2', type=float, default=3.,
                        help='Focal loss parameter, gamma')

    # Runtime environmnet
    parser.add_argument('--epochs', type=int, default=cfg.EPOCHS,
                        help='Number of epochs to train')
    parser.add_argument('--log_interval', type=int, default=cfg.LOG_INTV,
                        help='Log interval for training')
    parser.add_argument('--eval_interval', type=int, default=cfg.EVAL_INTV,
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
    exs = pickle.load(open(args.ds_path, 'rb'))
    special_tokens = dict(zip(tokenizer.all_special_tokens,
                              tokenizer.all_special_ids))
    ds = {
        mode: MedMentionDataset(exs['examples'], mode, args.model_name,
                                exs['ni2cui'])
        for mode in ['trn', 'dev']
    }

    # Default values for optimization
    args.num_training_steps = int(len(ds['trn']) / args.batch_size) * args.epochs
    args.num_warmup_steps = int(args.num_training_steps * .05)


    # Model
    # normalization space (zeros for the null class)
    nn_trn = torch.cat(
        (torch.index_select(exs['name_embs'], dim=0,
                            index=torch.tensor(ds['trn'].nids)),
         torch.zeros(exs['name_embs'].size(-1)).unsqueeze(0))
    ).to(args.device)
    nn_dev = torch.cat(
        (torch.index_select(exs['name_embs'], dim=0,
                            index=torch.tensor(ds['dev'].nids)),
         torch.zeros(exs['name_embs'].size(-1)).unsqueeze(0))
    ).to(args.device)
    # name_embs = torch.cat(
    #     (exs['name_embs'],
    #      torch.zeros(exs['name_embs'].size(-1)).unsqueeze(0))
    # ).to(args.device)
    if args.model_name == 'one-tag':
        print('ont-tag')
        model = IOB_ONE(args)
    if args.model_name == 'tag-lo':
        model = IOB_LO(args)
    elif args.model_name == 'tag-hi':
        model = IOB_HI(args)

    model.to(args.device)
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

    # Train -------------------------------------------------------------------
    stats = utils.TraningStats()
    model.train()
    for epoch in range(1, args.epochs + 1):
        stats.epoch = epoch
        logger.info('*** Epoch %s starts ***', epoch)
        train(args, ds, model, optimizer, scheduler, stats)
