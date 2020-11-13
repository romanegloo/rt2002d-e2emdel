"""
infer.py

Run test with the best model saved.
"""

import code
import logging
import argparse
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import GoshiBoshi.config as cfg
import GoshiBoshi.utils as utils
from GoshiBoshi.data import MedMentionDataset, batchify
from GoshiBoshi.model import JoMEDaL_Eval

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    # Configuration -----------------------------------------------------------
    parser = argparse.ArgumentParser(
        'Joint Mention Detection and Entity Linking',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Paths
    parser.add_argument('--best_mdl_file', type=str, default=cfg.BEST_MDL_FILE,
                        help='Filename of the best model')
    parser.add_argument('--norm_file', type=str, default=cfg.NS_FILE,
                        help='Data file for UMLS concept normalization')
    parser.add_argument('--tfidf_file', type=str, default=cfg.TFIDF_FILE,
                        help='n-gram tfidf vectorizer and corpus')
    # Runtime
    parser.add_argument('--use_mm_subset', action='store_true',
                       help='use the MedMentions subset of name norm space')
    parser.add_argument('--random_seed', type=int, default=cfg.RND_SEED,
                        help='Random seed')
    parser.add_argument('--batch_size', type=int, default=cfg.BSZ,
                        help='Number of examples in running train/valid steps')

    # Model
    parser.add_argument('--max_sent_len', type=int, default=cfg.MAX_SENT_LEN,
                        help='Maximum sentence length of example inputs')

    # Features
    parser.add_argument('--enable_ngram_tfidf', action='store_true',
                        help='Enable the post-process for tfidf string match')
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

    # Set defaults-------------------------------------------------------------
    # Random seed
    utils.set_seed(args.random_seed)
    # GPU
    if torch.cuda.is_available():
        args.device = torch.device('cuda:0')
        logger.info('Running on GPUs')
    else:
        args.device = torch.device('cpu')
        logger.info('Running on CPUs')

    # Load the best JoMeDaL model for evaluation and test dataset
    logger.info('Loading a pretrained model and the test dataset...')
    evaluator = JoMEDaL_Eval(args)
    mm = pickle.load(open(cfg.MM_DS_FILE, 'rb'))
    if args.use_mm_subset:
        evaluator.name_embs = mm['name_embs'].to(args.device)
        evaluator.ni2cui = mm['ni2cui'] + ['N']
    ds = MedMentionDataset(mm['examples'], 'tst',
                           evaluator.ni2cui,
                           max_sent_len=args.max_sent_len,
                           one_tag=(evaluator.model_name == 'one-tag'))
    evaluator.stypes = ds.types
    del mm


    # if args.norm_space_file != '':
    #     logger.info('Loading saved name embeddings...')
    #     norm_space = pickle.load(open(args.norm_space_file, 'rb'))
    #     name_embs = norm_space['name_embs']
    #     name_mapping = norm_space['ni2cui']
    # else:
    #     name_embs = exs['name_embs']
    #     name_mapping = exs['ni2cui']
    # name_embs = torch.cat((name_embs,
    #                     torch.zeros(exs['name_embs'].size(-1)).unsqueeze(0)))
    # # name_embs_t = name_embs[:, dim_h:]

    # evaluator.stypes = ds.types
    # evaluator.names = mm['ni2cui'] + ['N']  # debug
    it = DataLoader(ds, batch_size=args.batch_size, collate_fn=batchify,
                    shuffle=True)

    # Evaluate
    pbar = tqdm(total=len(it))
    with torch.no_grad():
        for i, batch in enumerate(it):
            evaluator.predict(batch)
            pbar.set_description(evaluator.retrieval_status)
            pbar.update()
    pbar.close()

    print(evaluator.compute_retrieval_metrics())
