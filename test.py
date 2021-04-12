"""test.py

Load pre-trained model and evaluate the model on test datasets
"""

import code
import logging
import argparse
import pickle
from tqdm import tqdm
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import GoshiBoshi.utils as utils
from GoshiBoshi.data import MedMentionDataset, batchify
from GoshiBoshi import IOB_HI, IOB_ONE
import GoshiBoshi.config as cfg

logger = logging.getLogger(__name__)


class EvalMM:
    """Evaluation class for NER and EN with MedMention DS"""
    def __init__(self, args):
        self.args = args
        self.st = cfg.MM_ST
        self.ds = None  # test dataset
        self.norm_space = None  # Name normalization space
        self.metrics = {
            't0': {'tp': 0, 'fp': 0, 'fn': 0},
            't1': {'tp': 0, 'fp': 0, 'fn': 0},
            't2': {'tp': 0, 'fp': 0, 'fn': 0}
        }
        # ST conversion counts: [o, x, o->x, x->o]
        self.conversion_counts = [0, 0, 0, 0]
        self.conversion_matrix = np.zeros((len(self.st), len(self.st)),
                                          dtype=int)
        self.miss_matrix = np.zeros((len(self.st), len(self.st)),
                                    dtype=int)
        self.checkpoint = torch.load(self.args.best_mdl_file)
        self.override_configs(self.checkpoint['args'])
        self.model = None

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

    def test_mm(self):
        it = DataLoader(self.ds, batch_size=16, collate_fn=batchify)
        pbar = tqdm(total=len(it))
        with torch.no_grad():
            for i, batch in enumerate(it):
                predictions, _, _, _ = self.model(batch[:5],
                                                  self.norm_space,
                                                  pred_lvl_m=True)
                self.compute_retrieval_metrics(predictions, batch[5])
                self.error_analysis(predictions, batch)
                e = 'tp {} fp {} fn {}'.format(*list(self.metrics['t2'].values()))
                pbar.set_description(e)
                pbar.update()

        pbar.close()

        epsilon = 1e-7
        for t, s in self.metrics.items():
            precision = s['tp'] / (s['tp'] + s['fp'] + epsilon)
            recall = s['tp'] / (s['tp'] + s['fn'] + epsilon)
            f1 = 2 * precision * recall / (precision + recall)
            if t != 't2':
                print('{:.3} & {:.3} & {:.3} &'.format(precision, recall, f1))
            else:
                print('{:.3} & {:.3} & {:.3} \\\\'.format(precision, recall, f1))

    def error_analysis(self, predictions, batch):
        annotations = batch[5]
        gt, pred = self.get_gt_pred(predictions, annotations)
        # for k in gt:
        #     if k not in predictions:
        #         code.interact(local=dict(globals(), **locals()))

        for k, (l, t, e) in pred.items():
            if k in gt and l == gt[k][0]:
                i, j = self.st.index(t), self.st.index(cuis[e].st_root)
                self.conversion_matrix[i][j] += 1
                if t == gt[k][1]:
                    self.conversion_counts[0] += 1
                    if cuis[e].st_root != gt[k][1]:
                        self.conversion_counts[2] += 1
                else:
                    self.conversion_counts[1] += 1
                    if cuis[e].st_root == gt[k][1]:
                        self.conversion_counts[3] += 1
                if e != gt[k][2]:
                    i = self.st.index(gt[k][1])
                    j = self.st.index(cuis[e].st_root)
                    self.miss_matrix[i][j] += 1

    def compute_retrieval_metrics(self, predictions, annotations):
        gt, pred = self.get_gt_pred(predictions, annotations)
        metrics = {
            't0': {'tp': 0, 'fp': 0, 'fn': 0},
            't1': {'tp': 0, 'fp': 0, 'fn': 0},
            't2': {'tp': 0, 'fp': 0, 'fn': 0}
        }
        for k, (l, t, e) in pred.items():
            if k in gt and l == gt[k][0]:
                if self.args.zero_shot and\
                        not (gt[k][2] in self.ds.cui2idx and
                             self.ds.cui2idx[gt[k][2]]['tst_exc']):
                    continue
                metrics['t0']['tp'] += 1
                if t == gt[k][1]:
                    metrics['t1']['tp'] += 1
                else:
                    metrics['t1']['fp'] += 1
                if e == gt[k][2]:
                    metrics['t2']['tp'] += 1
                else:
                    metrics['t2']['fp'] += 1
            else:  # fp^o
                if not (e in self.ds.cui2idx and self.ds.cui2idx[e]['tst_exc']):
                    continue
                for ck in metrics:
                    metrics[ck]['fp'] += 1
        for ck in metrics:
            if self.args.zero_shot:
                n = sum([1 for k, annt in gt.items()
                         if (annt[2] in self.ds.cui2idx and
                             self.ds.cui2idx[annt[2]]['tst_exc'])])
            else:
                n = len(gt)
            metrics[ck]['fn'] = n - metrics[ck]['tp']

        for ck in metrics:
            for s in metrics[ck]:
                self.metrics[ck][s] += metrics[ck][s]

    def get_gt_pred(self, predictions, annotations):
        gt = dict()
        pred = dict()
        for i, annts in enumerate(annotations):
            for bi, l, _, t, e in annts:
                # Remove 'UMLS:' from e,
                # prediction index counts +1 for BOS token
                gt[(i, bi+1)] = (l, t, e[5:])
        # Remove predictions that contain only null classes
        t1_nill = len(self.st)
        t2_nill = len(self.ds.nids)

        # k: key, l: length of a mention,
        # t: list of token-level type predictions
        # e: list of token-level entity predictions
        seen_mention_spans = None
        if self.args.zero_shot:
            seen_mention_spans =\
                [(k[0], k[1], k[1] + m[0]) for k, m in gt.items()
                 if not (m[2] in self.ds.cui2idx and
                         self.ds.cui2idx[m[2]]['tst_exc'])]
        for k, (l, t, e) in predictions.items():
            # # right trim
            # if l >= 2 and (t[-1] != t[-2] or e[-1] != e[-2]):
            #     t = t[:-1]
            #     e = e[:-1]
            # remove unclassified mentions
            t = [v for v in t if v != t1_nill]
            e = [v for v in e if v != t2_nill]
            if len(t) == 0 or len(e) == 0:
                continue
            # Skip any predictions that overlap a mention of seen entity
            if self.args.zero_shot:
                if any(k[0] == m[0] and
                       ((m[1] <= k[1] < m[2]) or (m[1] <= k[1] + l - 1 < m[2]))
                       for m in seen_mention_spans):
                    continue
                # if not (gt[k][2] in self.ds.cui2idx and
                #         self.ds.cui2idx[gt[k][2]]['tst_exc']):
                #     continue
            t_maj = Counter(t).most_common()[0][0] if l > 2 else t[0]
            e_maj = Counter(e).most_common()[0][0] if l > 2 else e[0]
            pred[k] = (predictions[k][0], self.st[t_maj],
                       self.ds.ni2cui[e_maj][1])
        return gt, pred


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
    parser.add_argument('--zero_shot', action='store_true',
                        help='Compute the performance only with the annotations '
                        'that appear in the test dataset')
    parser.add_argument('--nn', type=str, default='mm',
                        help='Name normalization space (mm: full mm CUIs, '
                        'tst_ex: test exclusive CUIs)')
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
    evaluator = EvalMM(args)

    # Load name normalization space
    mm = pickle.load(open(args.mm_file, 'rb'))
    cuis = mm['cuis']
    evaluator.ds = MedMentionDataset(mm['examples'], 'tst',
                                     evaluator.args.model_name, mm['ni2cui'],
                                     nn=args.nn)
    tokenizer = AutoTokenizer.from_pretrained(cfg.BERT_MODEL)

    nn_tst = torch.cat(
        (torch.index_select(mm['name_embs'], dim=0,
                            index=torch.tensor(evaluator.ds.nids)),
         torch.zeros(mm['name_embs'].size(-1)).unsqueeze(0))
    ).to(args.device)
    evaluator.norm_space = nn_tst
    evaluator.load_model()

    # Run~!
    evaluator.test_mm()