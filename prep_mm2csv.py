"""
prep_mm.py

Convert PubTator format of MedMention annotation data for model training 
consumption.

Example annotation:
25763772        0       5       DCTN4   T103    UMLS:C4308010

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

UMLS MRSTY example:
C0000005|T116|A1.4.1.2.1.7|Amino Acid, Peptide, or Protein|AT17648347|256|

UMLS MRCONSO example:
(CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|SAUI|SCUI|SAB|TTY|CODE|STRING|SRL|SUPPRESS|CVF)
C0000005|ENG|P|L0000005|PF|S0007492|Y|A26634265||M0019694|D012711|MSH|PEP|D012711|(131)I-Macroaggregated Albumin|0|N|256|
"""
from typing import List, DefaultDict, Dict, Set, Tuple
import gzip
import re
import logging
from itertools import accumulate
from operator import add
import code
from tqdm import tqdm
import pickle
from collections import Counter, defaultdict

from nltk.tokenize import sent_tokenize

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel

import GoshiBoshi.utils as utils


MM_FILE = 'data/MedMentions/corpus_pubtator.txt.gz'
MM_pmids_trn = 'data/MedMentions/corpus_pubtator_pmids_trng.txt'
MM_pmids_dev = 'data/MedMentions/corpus_pubtator_pmids_dev.txt'
MM_pmids_tst = 'data/MedMentions/corpus_pubtator_pmids_test.txt'
MRCONSO_FILE = 'data/UMLS/MRCONSO.RRF'
MRSTY_FILE = 'data/UMLS/MRSTY.RRF'
SRSTRE_FILE = 'data/UMLS/SRSTRE1'
MM_OUT_FILE = 'data/MedMentions/mm.pkl'
BERT_MDL_DIM = 768

TYPE_ROOTS = [f'T{t:03}' for t in 
              [5, 7, 17, 22, 31, 33, 37, 38, 58, 62, 74, 82, 91, 92,
               97, 98, 103, 168, 170, 201, 204]]
ST_MAP = {t: t for t in TYPE_ROOTS}
# Read hierarchical relationships between UMLS semantic types
with open(SRSTRE_FILE) as f:
    while True:
        len_all = len(ST_MAP)
        for line in f:
            st1, r, st2, _ = line.split('|')
            if st2 in ST_MAP and r == 'T186':
                ST_MAP[st1] = ST_MAP[st2]
        if len(ST_MAP) != len_all:
            f.seek(0)
        else:
            break

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
    
def get_name_embeddings(hs, entity_types, typeIDs, lengths):
    """Given the hidden states and the lengths in batch, compute the mean of
    the vectors. Assume that `hs` is in (N x L x D) and l in (N)

    Output is in (N x D)
    """
    device = hs.device
    if isinstance(lengths, list):
        lengths = torch.tensor(lengths)
    # compute the mean vector from the outputs of the pre-trained BERT model
    n, l, d = hs.size()
    ll = lengths + torch.arange(n) * l
    b = torch.arange(n * l).view(n, l)
    b = (b < ll.view(n, 1) - 1).to(device)
    lengths = lengths.unsqueeze(1).to(device)
    mean = (hs * b.view(n, l, 1).float())[:,1:].sum(1) / lengths

    # Build one-hot vectors of the entity types
    t_embs = torch.tensor([typeIDs.index(t) for t in entity_types]).unsqueeze(-1)
    t_embs = torch.zeros(hs.size(0), len(typeIDs)).scatter_(1, t_embs, 1)

    name_embs = torch.cat((t_embs.to(device), mean), dim=1)
    
    return name_embs.to('cpu')


if __name__ == '__main__':
    # Logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s: [ %(message)s ]',
        datefmt='%b%d %H:%M'
    )

    # Read from the MedMention datafile
    # Load BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')

    data = dict.fromkeys(['examples', 'typeIDs', 'bert_special_tokens_map'])
    data['examples'] = []
    data['bert_special_tokens_map'] = dict(zip(tokenizer.all_special_tokens,
                                               tokenizer.all_special_ids))
    data['typeIDs'] = TYPE_ROOTS
    data['ontology_abbr'] = ['CPT', 'FMA', 'GO', 'HGNC', 'HPO', 'ICD10',
                            'ICD10CM', 'ICD9CM', 'MDR', 'MSH', 'MTH', 'NCBI',
                            'NCI', 'NDDF', 'NDFRT', 'OMIM', 'RXNORM',
                            'SNOMEDCT_US']
        # Types are defined [here](https://tinyurl.com/y49oovcw)


    # Read MedMention datafile; get concepts with examples
    a_doc: List[str] = []
    concept_counter: Dict[str, int] = Counter()
    logger.info('Reading MedMention dataset...')
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
    logger.info(f'{len(concept_counter)} unique concepts found in the examples' )

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

    # Read the UMLS concepts that belong to st21pv semantic types
    st21pv_cuis = dict()
    with open(MRSTY_FILE) as f:
        for line in f:
            flds = line.split('|')
            if flds[1] in ST_MAP:
                st21pv_cuis[flds[0]] = ST_MAP[flds[1]]
    logger.info('{} CUIs found for the given semantic types'
                ''.format(len(st21pv_cuis)))
    
    logger.info('Loading SciBERT model...')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    mdl = AutoModel.from_pretrained('allenai/scibert_scivocab_cased', 
                                    output_hidden_states=True)
    mdl.to(device)
    mdl.eval()

    # Read the UMLS concepts and build the name normalization vectors.
    # MM-ST21pv contains 3,170,502 concepts. 
    norm_dim = BERT_MDL_DIM + len(data['typeIDs'])
    # norm_space = torch.empty((3500000, norm_dim))
    norm_space_s = torch.empty((5*len(concept_counter), norm_dim))
    ni2cui: List[str] = []
    ni2cui_s: List[str] = []
    
    bsz = 512
    batch: List[torch.Tensor] = []  # batch of cui names
    cnt, cnt_s = 0, 0
    pbar = tqdm(total=15479756)
    with torch.no_grad(), open(MRCONSO_FILE) as f:
        for (ln, line) in enumerate(f):
            pbar.update()
            flds = line.split('|')
            if f'UMLS:{flds[0]}' not in concept_counter:
                continue
            # in st21pv AND eng AND preferred LUI
            if not (flds[0] in st21pv_cuis and flds[1] == 'ENG' and \
                flds[2] == 'P' and flds[11] in data['ontology_abbr'] and \
                flds[4] == 'PF' and flds[6] == 'Y'):
                # for dev purpose, make this stricter
                continue
            inp = tokenizer.encode(flds[14][:80])  # name
            ni2cui.append(flds[0])
            cnt += 1
            batch.append((torch.tensor(inp), st21pv_cuis[flds[0]]))
            if len(batch) == bsz:
                inputs = [x[0] for x in batch]
                entity_types = [x[1] for x in batch]
                inp_lengths = [len(x[0]) for x in batch]
                inputs = pad_sequence(inputs, batch_first=True).to(device)
                masks = \
                    utils.sequence_mask(inp_lengths, inputs.size(1)).to(device)
                _, _, output = mdl(inputs, attention_mask=masks)
                name_embs = get_name_embeddings(output[-2], entity_types,
                                                data['typeIDs'], inp_lengths)

                # norm_space[cnt-bsz:cnt] = name_embs
                for (i, cui) in enumerate(ni2cui[-bsz:]):
                    if f'UMLS:{cui}' in concept_counter:
                        norm_space_s[cnt_s] = name_embs[i]
                        ni2cui_s.append(cui)
                        cnt_s += 1
                batch = []
                        
        # Remainders in batch
        if len(batch) > 0:
            inputs = [x[0] for x in batch]
            entity_types = [x[1] for x in batch]
            inp_lengths = [len(x) for x in batch]
            inputs = pad_sequence(inputs, batch_first=True).to(device)
            masks = utils.sequence_mask(inp_lengths, inputs.size(1)).to(device)
            _, _, output = mdl(inputs, attention_mask=masks)
            name_embs = get_name_embeddings(output[-2], entity_types,
                                            data['typeIDs'], inp_lengths)
            # norm_space[cnt-len(batch):cnt] = name_embs
            for (i, cui) in enumerate(ni2cui[-len(batch):]):
                if f'UMLS:{cui}' in concept_counter:
                    norm_space_s[cnt_s] = name_embs[i]
                    ni2cui_s.append(cui)
                    # cui2ni_s[cui].append(cnt_s)
                    cnt_s += 1
            batch = []
    pbar.close()
    logger.info(f'UMLS concept definitions (total: {len(ni2cui)}'
                f' sub: {len(ni2cui_s)}) found')
    # norm_space = norm_space[:cnt]
    data['entity_names'] = norm_space_s[:cnt_s]
    data['ni2cui'] = ni2cui_s

    # Saving processed examples
    logger.info('Saving {} examples into {}'
                ''.format(len(data['examples']), MM_OUT_FILE))
    pickle.dump(data, open(MM_OUT_FILE, 'wb'))
            
