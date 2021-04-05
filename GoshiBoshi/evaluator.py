import logging
from collections import Counter, defaultdict
import pickle

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import linear_kernel  # dot product

from GoshiBoshi import IOB_HI, IOB_LO, IOB_ONE
import GoshiBoshi.config as cfg

logger = logging.getLogger(__name__)

class JoMEDaL_Eval:
    """ Evaluation model for Entity detection and Linking performance"""
    def __init__(self, args):
        self.args = args
        self.r = defaultdict(int)

        checkpoint = torch.load(args.best_mdl_file)
        self.args_saved = checkpoint['args']
        self.cfg = checkpoint['configs']
        self.model_name = self.cfg['model_name']

        logger.info('Loading a tokenizer(%s)', cfg.BERT_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.BERT_MODEL)

        logger.info('Loading best JoMEDaL model saved...')
        if self.model_name == 'tag-lo':
            self.model = IOB_LO(self.args_saved)
        elif self.model_name == 'tag-hi':
            self.model = IOB_HI(self.args_saved)
        elif self.model_name == 'one-tag':
            self.model = IOB_ONE(self.args_saved)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval().to(self.args.device)

        logger.info('Loading UMLS concept normalization space...')
        norm_space = pickle.load(open(args.norm_file, 'rb'))
        if args.use_mm_subset:
            # These will be loaded from the saved best model
            self.name_embs = None
            self.ni2cui = None
        else:
            self.name_embs = norm_space['name_embs'].to(args.device)
            self.ni2cui = norm_space['ni2cui'] + ['N']

        logger.info('Loading TF-IDF vectorizer and transformed names...')
        tfidf = pickle.load(open(args.tfidf_file, 'rb'))
        self.vectorizer = tfidf['vectorizer']
        self.tfidf = tfidf['names_ngram']
        self.tfidf_s = tfidf['names_ngram_s']

        self.stypes = []

    @property
    def retrieval_status(self):
        return '({} {} {})'.format(self.r['TP'], self.r['FP'],
                                   self.r['N']-self.r['TP'])

    def predict(self, batch):
        inputs, segs, x_mask, annotations = batch
        out = self.model(inputs, segs, x_mask)
        out_iob, out_st, c_emb = (o[:,1:-1,:] if o is not None else None
                                  for o in out)

        if self.model_name == 'one-tag':  # out_iob is None
            # IOB
            pred = out_st.argmax(dim=-1)
            null_idx_1 = out_st.size(-1) - 1
            pred_tag = torch.ones(pred.size(), dtype=torch.uint8)
            pred_tag[(pred % 2 == 1) & (pred != null_idx_1)] = 0
            pred_tag[(pred % 2 == 0) & (pred != null_idx_1)] = 2
            # Semantic type
            pred = F.avg_pool1d(out_st, kernel_size=2, ceil_mode=True)
            pred_st = pred.argmax(dim=-1)
        else:  # Either 'tag-lo' or 'tag-hi'
            # IOB
            pred_tag = out_iob.argmax(dim=-1)
            # Semantic type
            # logprobs_st = F.log_softmax(out_st, dim=-1)
            pred_st = out_st.argmax(dim=-1)

        # Compute name similarities
        name_scores = torch.matmul(c_emb, self.name_embs.T)
        if self.args.enable_ngram_tfidf:
            top_ids = torch.topk(name_scores, 30, dim=-1)
        else:
            top_ids = name_scores.argmax(-1)

        # Construct predicted mentions
        predictions = dict()
        a_mention = [-1, -1, [], [], defaultdict(int)]
        # `predictions` is a dictionary of mentions which key is (i, j) where i
        # is the example idx and j is the token idx. A mention is represented by
        # (starting idx, length, [predicted type indices], [predicted entity indices]).
        # These indices are used in voting for the best predictions.
        for i, ex in enumerate(pred_tag):
            seq_len = x_mask[i].sum()
            for j, idx in enumerate(ex):
                if j == seq_len:
                    break
                if idx == 1:  # O in IOB
                    if a_mention[0] >= 0:
                        predictions[(i, a_mention[0])] = a_mention[1:]
                        a_mention = [-1, -1, [], [], defaultdict(int)]
                else:  # I or B
                    ti = pred_st[i,j].item()
                    if self.args.enable_ngram_tfidf:
                        ei = top_ids[1][i,j].tolist()
                        ei_scores = top_ids[0][i,j].tolist()
                    else:
                        ei = top_ids[i,j].item()
                    if idx == 2:  # B in IOB
                        if a_mention[0] >= 0:  # register the previous mention
                            predictions[(i, a_mention[0])] = a_mention[1:]
                            a_mention = [-1, -1, [], [], defaultdict(int)]
                        a_mention[:3] = (j, 1, [ti])
                        if self.args.enable_ngram_tfidf:
                            a_mention[3] = ei
                            for s, e in zip(ei_scores, ei):
                                a_mention[4][e] += s
                        else:
                            a_mention[3] = [ei]
                            a_mention[4][ei] = 1
                    else:  # I in IOB
                        if a_mention[0] >= 0:
                            a_mention[1] += 1
                            a_mention[2].append(ti)
                            if self.args.enable_ngram_tfidf:
                                a_mention[3].extend(ei)
                                for s, e in zip(ei_scores, ei):
                                    a_mention[4][e] += s
                            else:
                                a_mention[3].append(ei)
                                a_mention[4][ei] = 1
            if a_mention[0] >= 0:
                predictions[(i, a_mention[0])] = a_mention[1:]
                a_mention = [-1, -1, [], [], defaultdict(int)]

        # Compute and update retrieval results
        self.__update_retrieval_metrics(predictions, annotations,
                                        inputs, x_mask)

    def __update_retrieval_metrics(self, pred, gt, x, x_mask):
        # Read gt annotations
        annt_dict = dict()
        for i, annts in enumerate(gt):
            for bi, l, _, t, e in annts:
                annt_dict[(i, bi)] = (l, t, e[5:])  # remove 'UMLS:' from e

        # print('GT:', annt_dict)
        # print('\nPRED:', {k: [v[0], [self.ni2cui[i] for i in v[2][:10]]]
        #                   for k, v in pred.items()})
        for k, (l, t, e, s) in pred.items():
            # k: key, l: mention length,
            # t: list of predicted types, e: list of predicted entities
            if k in annt_dict:
                x_i = x[k[0]][x_mask[k[0]]][1:-1]
                phrase = self.tokenizer.decode(x_i[k[1]:k[1]+l])
                t_maj = Counter(t).most_common()[0][0]
                if self.args.enable_ngram_tfidf:
                    n_maj = self.__vote_best_entity(e, s, phrase)
                else:
                    n_maj = Counter(e).most_common()[0][0]
                if self.ni2cui[n_maj] == annt_dict[k][2]:
                    # print('HIT', annt_dict[k][2], self.ni2cui[n_maj], k, l)
                    self.r['TP'] += 1
                else:
                    self.r['FP'] += 1
            else:
                self.r['FP'] += 1
        self.r['N'] += len(annt_dict)


    def __vote_best_entity(self, ent_inds, scores, phrase):
        """
        Choose the entity from the `ent_inds` which exactly matches the
        entity name in terms of n-gram tfidf vector
        """
        sorted_scores = [(k, v) for k, v in
                         sorted(scores.items(), key=lambda s: s[1], reverse=True)]

        phrase_vec = self.vectorizer.transform([phrase])
        if self.args.use_mm_subset:
            tfidf_cossim = linear_kernel(phrase_vec, self.tfidf_s[ent_inds])
        else:
            tfidf_cossim = linear_kernel(phrase_vec, self.tfidf[ent_inds])
        tfidf_ranked = [i[0] for i in sorted(enumerate(tfidf_cossim[0]),
                                             key=lambda x: x[1], reverse=True)]

        if abs(tfidf_cossim[0][tfidf_ranked[0]] - 1) < 1e-7:
            return ent_inds[tfidf_ranked[0]]
        else:
            return sorted_scores[0][0]

    def compute_retrieval_metrics(self):
        epsilon = 1e-7
        self.r['FN'] = self.r['N'] - self.r['TP']
        precision = self.r['TP'] / (self.r['TP'] + self.r['FP'] + epsilon)
        recall = self.r['TP'] / (self.r['TP'] + self.r['FN'] + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)

        return precision, recall, f1