"""test.py

Load pre-trained model and evaluate the model on test datasets
"""

import code
import logging
import argparse
import pickle
from tqdm import tqdm
from collections import namedtuple, Counter

import torch
from torch.utils.data import DataLoader

import GoshiBoshi.utils as utils
from GoshiBoshi.data import MedMentionDataset, batchify
from GoshiBoshi import IOB_HI, IOB_LO, IOB_ONE
import GoshiBoshi.config as cfg

logger = logging.getLogger(__name__)


class MM_Eval:
    """Evaluation class for Entity Recognition and Linking with MedMention DS"""
    def __init__(self, args):
        self.args = args
        self.st = cfg.MM_ST
        self.ds = None  # test dataset
        self.metrics = {
            't0': {'tp': 0, 'fp': 0, 'fn': 0},
            't1': {'tp': 0, 'fp': 0, 'fn': 0},
            't2': {'tp': 0, 'fp': 0, 'fn': 0}
        }
        self.checkpoint = torch.load(self.args.best_mdl_file)
        self.override_configs(self.checkpoint['args'])
        self.model = None
        self.ni2cui = None

    def override_configs(self, cfg):
        fields = ['model_name', 'focal_alpha', 'focal_gamma1', 'focal_gamma2']
        for f in fields:
            setattr(self.args, f, getattr(cfg, f))

    def load_model(self):
        logger.info('=> Loading model parameters')
        if self.args.model_name == 'one-tag':
            logger.info('- Model type: one-tag')
            self.model = IOB_ONE(self.args)
        elif self.args.model_name == 'tag-hi':
            logger.info('- Model type: tag-hi')
            self.model = IOB_HI(self.args)
        self.model.load_state_dict(self.checkpoint['model'])
        self.model.eval().to(self.args.device)

    def test_mm(self, ds):
        code.interact(local=dict(globals(), **locals()))
        # it = DataLoader(ds, batch_size=8, collate_fn=batchify)
        # pbar = tqdm(total=len(it))
        # with torch.no_grad():
        #     for i, batch in enumerate(it):
        #         predictions, _, _, _ = self.model(batch[:5], pred_lvl_m=True)
        #         self.compute_retrieval_metrics(predictions, batch[5])
        #         e = 'tp {} fp {} fn {}'.format(*list(self.metrics['t2'].values()))
        #         pbar.set_description(e)
        #         pbar.update()
        # pbar.close()
        #
        # epsilon = 1e-7
        # for t, s in self.metrics.items():
        #     precision = s['tp'] / (s['tp'] + s['fp'] + epsilon)
        #     recall = s['tp'] / (s['tp'] + s['fn'] + epsilon)
        #     f1 = 2 * precision * recall / (precision + recall)
        #     print('[{}] p {:.3} r {:.3} f1 {:.3}'.format(t, precision, recall, f1))


#     def compute_retrieval_metrics(self, predictions, annotations):
#         # Read GT annotations in dictionary
#         gt = dict()
#         for i, annts in enumerate(annotations):
#             for bi, l, _, t, e in annts:
#                 gt[(i, bi+1)] = (l, t, e[5:])
#                     # Remove 'UMLS:' from e,
#                     # prediction index counts +1 for BOS token
#
#         metrics = {
#             't0': {'tp': 0, 'fp': 0, 'fn': 0},
#             't1': {'tp': 0, 'fp': 0, 'fn': 0},
#             't2': {'tp': 0, 'fp': 0, 'fn': 0}
#         }
#         # Remove predicted mentions that contain only null classes
#         t1_nill = len(self.st)
#         t2_nill = len(self.ni2cui) - 1
#         pred_ = dict()
#         for k, (l, t, e) in predictions.items():
#             # right-trim
#             if l >= 2 and (t[-1] != t[-2] or e[-1] != e[-2]):
#                 l = l - 1
#                 t = t[:-1]
#                 e = e[:-1]
#             # remove unclassified mentions
#             t = [v for v in t if v != t1_nill]
#             e = [v for v in e if v != t2_nill]
#             if len(t) > 0 and len(e) > 0:
#                 pred_[k] = predictions[k]
#
#         for k, (l, t, e) in pred_.items():
#             # k: key, l: mention length,
#             # t: list of predicted types, e: list of predicted entities
#             t = [v for v in t if v != t1_nill]
#             e = [v for v in e if v != t2_nill]
#             if k in gt and l == gt[k][0]:
#             # if k in gt and abs(l - gt[k][0]) <= 2:
#             #     if l != gt[k][0]:
#             #         print(k)
#             #         print(pred_[k])
#             #         if (k[0], k[1]+l) in pred_:
#             #             print(pred_[k[0], k[1]+l])
#             #         if (k[0], k[1]+l) in gt:
#             #             print(gt[k[0], k[1]+l])
#             #         print(gt[k])
#             #         code.interact(local=dict(globals(), **locals()))
#                 metrics['t0']['tp'] += 1
#                 t_maj = Counter(t).most_common()[0][0] if l > 2 else t[0]
#                 if self.st[t_maj] == gt[k][1]:
#                     metrics['t1']['tp'] += 1
#                 else:
#                     metrics['t1']['fp'] += 1
#                 e_maj = Counter([self.ni2cui[i] for i in e]).most_common()[0][0]
#                 if e_maj == gt[k][2]:
#                     metrics['t2']['tp'] += 1
#                 else:
#                     metrics['t2']['fp'] += 1
#             else:
#                 for t in metrics:
#                     metrics[t]['fp'] += 1
#         for t in metrics:
#             metrics[t]['fn'] = len(gt) - metrics[t]['tp']
#
#         for t in metrics:
#             for s in metrics[t]:
#                 self.metrics[t][s] += metrics[t][s]
#         # code.interact(local=dict(globals(), **locals()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Joint Mention Detection and Entity Linking',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Paths
    parser.add_argument('--best_mdl_file', type=str,
                        default='data/best.pt',
                        help='Filename of the best model')
    parser.add_argument('--norm_file', type=str,
                        default='data/MedMentions/norm_space.pkl',
                        help='Data file for UMLS concept normalization')
    parser.add_argument('--mm_file', type=str,
                        default='data/MedMentions/mm.pkl',
                        help='MedMention file containing processed examples')
    # Runtime
    parser.add_argument('--test_exclusive', action='store_true',
                        help='name normalization space is restricted to the '
                        'test dataset exclusively')
    parser.add_argument('--random_seed', type=int, default=cfg.RND_SEED,
                        help='Random seed')
    args = parser.parse_args()

    # Logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s: [ %(message)s ]',
        datefmt='%b%d %H:%M'
    )
    logging.handlers = [logging.FileHandler("train.log"),
                        logging.StreamHandler()]

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

    # Load an evaluator
    evaluator = MM_Eval(args)

    # Load name normalization space
    mm = pickle.load(open(args.mm_file, 'rb'))
    evaluator.ds = MedMentionDataset(mm['examples'], 'tst',
                                     evaluator.args.model_name, mm['ni2cui'])

    nn_tst = torch.cat(
        (torch.index_select(mm['name_embs'], dim=0,
                            index=torch.tensor(evaluator.ds.nids)),
         torch.zeros(mm['name_embs'].size(-1)).unsqueeze(0))
    ).to(args.device)
    evaluator.load_model(nn_tst)

#     if args.use_mm_subset:  # if use_mm_subset, load from mm dataset
#         logger.info('using_subset_ns: Loading norm space from mm dataset')
#         name_embs = torch.cat(
#             (mm['name_embs'],
#              torch.zeros(mm['name_embs'].size(-1)).unsqueeze(0))
#         ).to(args.device)
#         evaluator.ni2cui = mm['ni2cui'] + ['N']
#     else:
#         logger.info('using full ns: Loading from {}'.format(args.norm_file))
#         norm_space = pickle.load(open(args.norm_file, 'rb'))
#         name_embs = norm_space['name_embs'].to(args.device)
#         evaluator.ni2cui = norm_space['ni2cui'] + ['N']


    # Run~!
    evaluator.test_mm()
