"""
prep_mm.py

Convert PubTator format of MedMention annotation data for model training 
consumption. Examples are in sentences.

output format:
examples = {
    'data': [
        {
            'pmid': str(pubmed document id),
            'sent_no': int(sentence ordinal number)
            'raw': str(original sentence),
            'tokens': list of tokens(str), tokenized by a pre-trained BERT tokenizer,
            'annotations': [
                (t_k, l, type, entity code), ...
            ]
        },
        ...
    ],
    'typeIDs': [T001, T04, ..., Txx]
}
"""
from typing import List
import gzip
import re
import logging
from itertools import accumulate
from operator import add
import code
from tqdm import tqdm
import pickle
from collections import Counter

from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer


MM_FILE = 'data/MedMentions/corpus_pubtator.txt.gz'
MM_pmids_trn = 'data/MedMentions/corpus_pubtator_pmids_trng.txt'
MM_pmids_dev = 'data/MedMentions/corpus_pubtator_pmids_dev.txt'
MM_pmids_tst = 'data/MedMentions/corpus_pubtator_pmids_test.txt'
MM_OUT_FILE = 'data/MedMentions/mm.pkl'
logger = logging.getLogger(__name__)

def proc_doc(tokenizer, doc):
    r_title = re.compile(r'^((\d+)\|t\|)(.*)$')
    r_body = re.compile(r'^((\d+)\|a\|)(.*)$')
    r_annt = re.compile(r'^(\S+)\t(\d+)\t(\d+)\t(.*)\t(.*)\t(.*)$')

    did = None
    title = ''
    body = ''
    annotations = []
    for l in doc:
        m = r_title.match(l)
        if m:
            if did is None:
                did = m.group(2)
            title = m.group(3)
        m = r_body.match(l)
        if m:
            body = m.group(3)
        m = r_annt.match(l)
        if m:
            annotations.append(m.group(0).split('\t'))

    def align_annts(ex, annts, tokenizer, offsets):
        offsets.append(offsets[-1] + len(ex['raw']) + 1)
        aligned = []

        # We need to keep track of the character positions between the origial
        # and sub-word tokenized tokens (including [UNK])
        
        # Tokenization should be done per word
        tok_to_orig_index = []
        orig_to_tok_index = []
        word_split_tokens = ex['raw'].split()
        word_split_offsets = [0]
        for w in word_split_tokens:
            word_split_offsets.append(word_split_offsets[-1] + len(w) + 1)
        all_tokens = []
        for (i, token) in enumerate(ex['raw'].split()):
            orig_to_tok_index.append(len(all_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_tokens.append(sub_token)
        ex['tokens'] = all_tokens
        ex['token_ids'] = tokenizer.convert_tokens_to_ids(all_tokens)
            
        for annt in annts:
            if int(annt[2]) < offsets[-2]:
                continue
            if int(annt[1]) >= offsets[-1]:
                break
            b, e = int(annt[1]) - offsets[-2], int(annt[2]) - offsets[-2]
            if b < 0:
                raise IndexError
            span = [0, word_split_offsets[-1]]
            for (i, v) in enumerate(word_split_offsets):
                if v <= b:
                    span[0] = i
                if v < e:
                    span[1] = i
            span_in_subtokens = \
                [i for (i, v) in enumerate(tok_to_orig_index)
                 if span[0] <= v <= span[1]]
            # refine span_in_subtokens to exact subword mapping
            t_s, t_e = -1, -1
            phrase = annt[3].lower()
            for t in span_in_subtokens:
                if t_s == -1 and all_tokens[t].startswith(phrase[:2]):
                    t_s = t
                if t_s > 0 and all_tokens[t].endswith(phrase[:2]):
                    t_e = t
            if t_s >= 0 and t_e >= 0:
                span_in_subtokens = list(range(t_s, t_e+1))
            
            aligned.append((span_in_subtokens[0], len(span_in_subtokens),
                            *annt[3:]))

        return aligned
        
    examples = []
    sent_no = 0
    offsets = [0]
    # Tokenize title
    ex = {'pmid': did, 'sent_no': sent_no, 'raw': title}
    ex['annotations'] = align_annts(ex, annotations, tokenizer, offsets)
    examples.append(ex)

    # Tokenize body
    sents = sent_tokenize(body)
    for si in range(len(sents)):
        success = False
        while not success:
            if si >= len(sents):
                break
            ex = {'pmid': did, 'sent_no': si + 1, 'raw': sents[si]}
            try:
                ex['annotations'] = align_annts(ex, annotations, tokenizer, offsets)
            except IndexError as e:
                if si != 0 and si <= len(sents):
                    sents = sents[:si-1] + [sents[si-1] + ' ' + sents[si]] \
                            + sents[si+1:]
                    del examples[-1]
                    del offsets[-2:]
                    si -= 1
                else:
                    success = True
                    continue
            else:
                success = True
                examples.append(ex)
        
    return examples, [annt[-1] for annt in annotations]
    
if __name__ == '__main__':
    # Logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s: [ %(message)s ]',
        datefmt='%b%d %H:%M'
    )

    # Read from the MedMention datafile
    # Load BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

    data = dict.fromkeys(['examples', 'typeIDs', 'bert_special_tokens_map'])
    data['examples'] = []
    data['bert_special_tokens_map'] = dict(zip(tokenizer.all_special_tokens,
                                               tokenizer.all_special_ids))
    data['typeIDs'] = [f'T{t:03}' for t in 
                       [5, 7, 17, 22, 31, 33, 37, 38, 58, 62, 74, 82, 91, 92,
                        97, 98, 103, 168, 170, 201, 204]]
        # Types are defined [here](https://tinyurl.com/y49oovcw)

    # data['UMLS_concepts'] = defaultdict(lambda: len(data['UMLS_concepts']))

    a_doc: List[str] = []
    concept_counter = Counter()
    pbar = tqdm(total=216458)
    with gzip.open(MM_FILE, 'rt', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:  # End of file
                break
            if line == '\n':  # End of a document
                pbar.update(len(a_doc) + 1)
                exs, annts = proc_doc(tokenizer, a_doc)
                data['examples'].extend(exs)
                concept_counter.update(annts)
                a_doc = []
            else:
                a_doc.append(line)
    pbar.close()

    data['UMLS_concepts'] = list(concept_counter.keys())

    # For each example, assign dataset mode [trn/dev/tst] according to the
    # original split (https://tinyurl.com/y2yujq2o)
    pmids_trn = [pmid.rstrip() for pmid in open(MM_pmids_trn).readlines()]
    pmids_dev = [pmid.rstrip() for pmid in open(MM_pmids_dev).readlines()]
    pmids_tst = [pmid.rstrip() for pmid in open(MM_pmids_tst).readlines()]
    for ex in data['examples']:
        if ex['pmid'] in pmids_trn:
            ex['ds'] = 'trn'
        elif ex['pmid'] in pmids_dev:
            ex['ds'] = 'dev'
        elif ex['pmid'] in pmids_tst:
            ex['ds'] = 'tst'
        else:
            logger.warning('pmid {} does not exist in any of the datsets')

                
    # Saving processed examples
    logger.info('Saving {} examples into {}'
                ''.format(len(data['examples']), MM_OUT_FILE))
    pickle.dump(data, open(MM_OUT_FILE, 'wb'))
