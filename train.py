"""
train.py

Main script for training and saving E2E MD-EL models for PubTator annotation.
Modify the configuration file () for experiments.
"""

import sys
import logging
import argparse
import code

from torch.utils.data import DataLoader, RandomSampler
from torch import nn

from GoshiBoshi.data import KaggleNERDataset, batchify
from GoshiBoshi.model import JointMDEL
import GoshiBoshi.utils as utils

from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


def train(args, ds, mdl, crit, optim, sch):
    mdl.train()
    steps = 0
    for batch in DataLoader(ds['tr'],
                            batch_size=args.batch_size,
                            shuffle=True,
                            collate_fn=batchify):
        optim.zero_grad()
        outputs = mdl(batch)
        loss = utils.compute_loss(crit, outputs[0], batch[1])
        loss.backward()
        optim.step()
        nn.utils.clip_grad_norm_(mdl.parameters(), 10)
        if steps % 5 == 0:
            print(loss)
        steps += 1
# todo loss something like this
# def compute_loss(logits, target, length):
#     """
#     Args:
#         logits: A Variable containing a FloatTensor of size
#             (batch, max_len, num_classes) which contains the
#             unnormalized probability for each class.
#         target: A Variable containing a LongTensor of size
#             (batch, max_len) which contains the index of the true
#             class for each corresponding step.
#         length: A Variable containing a LongTensor of size (batch,)
#             which contains the length of each data in a batch.
#     Returns:
#         loss: An average loss value masked by the length.
#     """

#     # logits_flat: (batch * max_len, num_classes)
#     logits_flat = logits.view(-1, logits.size(-1))
#     # log_probs_flat: (batch * max_len, num_classes)
#     log_probs_flat = functional.log_softmax(logits_flat)
#     # target_flat: (batch * max_len, 1)
#     target_flat = target.view(-1, 1)
#     # losses_flat: (batch * max_len, 1)
#     losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
#     # losses: (batch, max_len)
#     losses = losses_flat.view(*target.size())
#     # mask: (batch, max_len)
#     mask = _sequence_mask(sequence_length=length, max_len=target.size(1))
#     losses = losses * mask.float()
#     loss = losses.sum() / length.float().sum()
#     return loss

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
    parser.add_argument('--num_warmup_steps', type=int, default=5000,
                        help='Number of steps for warmup in optimize schedule')
    parser.add_argument('--num_training_steps', type=int, default=200000,
                        help='Number of steps for training')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of examples in running train/valid steps')
                        
    # Runtime environmnet
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train')
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='Log interval for training')
    parser.add_argument('--eval_interval', type=int, default=20000,
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

    # Read a tokenizer (vocab is defined in the tokenizer)
    tokenizer = utils.get_tokenizer(args.model_name)
    # Read a dataset
    ds = dict()
    ds['tr'], ds['va'], ds['ts'] = (
        KaggleNERDataset(examples=exs, tokenizer=tokenizer) for exs in
        utils.csv_to_ner_split_examples(args)
    )

    # Model
    model = JointMDEL(args)
    criterion = nn.NLLLoss(ignore_index=-1, reduction='none')
    optimizer = AdamW(model.parameters(), lr=float(args.lr))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        args.num_warmup_steps,
        args.num_training_steps
    )

    # Train -------------------------------------------------------------------
    model.train()
    for epoch in range(1, args.epochs + 1):
        train(args, ds, model, criterion, optimizer, scheduler)
        break
