"""
filename: prep_mm.py

This script
(1) reads in the MedMention datasets
(2) transforms annotation records to model training examples
(3) stores examples in a file for training
(4) converts the concept names into name embeddings using the pretrained BERT
    encoder
(5) and stores the name normalization spaces for the inference process

Followings are the formats of different data records

[Annotations in MedMentions]
25763772        0       5       DCTN4   T103    UMLS:C4308010

[UMLS MRSTY]
C0000005|T116|A1.4.1.2.1.7|Amino Acid, Peptide, or Protein|AT17648347|256

[UMLS MRCONSO]
C0000005|ENG|P|L0000005|PF|S0007492|Y|A26634265||M0019694|D012711|MSH|PEP|D012711|(131)I-Macroaggregated Albumin|0|N|256
* columns in the above format are:
(CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STRING|SRL|SUPPRESS|CVF)
"""

import code
from typing import List
from collections import Counter, defaultdict
from tqdm import tqdm
import gzip
import re
import pickle
import random

from nltk.tokenize import sent_tokenize
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer

import GoshiBoshi.config as cfg
import GoshiBoshi.utils as utils


def load_pretrained_bert(model_name):
    """
    Load pretrained BERT model (default SciBERT_uncased) and return both
    the tokenizer and encoder of it
    """
    print(f'Loading BERT tokenizer ({model_name})')
    tok = AutoTokenizer.from_pretrained(model_name)
    print(f'Loading BERT encoder ({model_name})')
    enc = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    return tok, enc


def align_annts(ex, annts, offsets):
    """Bert tokenizer chunks a sentence into subword tokens,
    while MedMentions dataset annotates tokens in word-level. So, an additional
    alignment process is needed. `proc_doc` converts offsets of annotations
    in consideration of subword tokens.

    Args:
        ex: example placeholder
        annts: global (doc) list of annotations
        offsets: offsets of sentences in character positions
    """

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
        if int(annt[2]) < offsets[-2]:  # annotations of previous sentences
            continue
        if int(annt[1]) >= offsets[-1]: # annotations of following sentences
            break
        b, e = int(annt[1]) - offsets[-2], int(annt[2]) - offsets[-2]
        if b < 0:
            raise IndexError  # merging attempt will be done in proc_doc
        # compute span boundaries
        span = [0, word_split_offsets[-1]]
        for (i, v) in enumerate(word_split_offsets):
            if v <= b:
                span[0] = i
            if v < e:
                span[1] = i
        span_in_subtokens = \
            [i for (i, v) in enumerate(tok_to_orig_index)
                if span[0] <= v <= span[1]]

        aligned.append((span_in_subtokens[0], len(span_in_subtokens), *annt[3:]))
    ex['annotations'] = aligned


def proc_doc(doc, st2cuis=None):
    """
    Process a PubMed document, returns an example for training along with the
    annotations.

    Args:
        doc: List of document strings in PubTator format

    Returns:
        - List of training examples
        - List of UMLS concepts
    """
    pmid = ''  # document id
    title = ''
    body = ''
    exs = []
    annotations = []
    offsets = [0]

    # Regex for detecting title/abstract/annotations of a document
    r_title = re.compile(r'^((\d+)\|t\|)(.*)$')
    r_body = re.compile(r'^((\d+)\|a\|)(.*)$')
    r_annt = re.compile(r'^(\S+)\t(\d+)\t(\d+)\t(.*)\t(.*)\t(.*)$')

    for l in doc:
        m = r_title.match(l)
        if m:
            if pmid == '':
                pmid = m.group(2)
            title = m.group(3)
        m = r_body.match(l)
        if m:
            body = m.group(3)
        m = r_annt.match(l)
        if m:
            annotations.append(m.group(0).split('\t'))

    # JoMDEL takes annotated sentences
    sents = [title]
    sents.extend(sent_tokenize(body))
    for si, sent in enumerate(sents):
        # sent_tokenize might split in the middle of an annotated mention. We
        # tried to merge the two sentences when that happens.
        success = False
        while not success:
            if si >= len(sents):
                break
            ex = {'pmid': pmid, 'sent_no': si, 'raw': sent, 'annotations': [],
                  'generated': False}
            if pmid in split_trn:
                ds_grp = 'trn'
            elif pmid in split_dev:
                ds_grp = 'dev'
            elif pmid in split_tst:
                ds_grp = 'tst'
            ex['ds'] = ds_grp

            # Try to fix the broken sentences.
            try:
                align_annts(ex, annotations, offsets)
            except IndexError:
                if si != 0 and si <= len(sents):
                    sents = sents[:si-1] + [sents[si-1] + ' ' + sents[si]] \
                            + sents[si+1:]
                    del examples[-1]
                    del offsets[-2:]
                else:
                    success = True
                    continue
            else:
                success = True
                exs.append(ex)

        if st2cuis is not None:
            for annt in ex['annotations']:
                st2cuis[annt[3]][annt[4]] = annt[2]  # dict[type][cui] = mention

    return exs, [annt[-1] for annt in annotations]


def get_name_embeddings(cemb, entity_types, typeIDs, lens):
    """Given the hidden states and the lengths in batch, compute the mean of
    the vectors. Assume that `cemb` is in (N x L x D) and `lens` in (N)

    Output is in (N x D)
    """
    cemb = cemb.to('cpu')
    if isinstance(lens, list):
        lens = torch.tensor(lens)
    # Compute the mean vectors of the encoder outputs over only the varying
    # lengths of the source inputs
    n, l, d = cemb.size()
    ll = lens + torch.arange(n) * l - 1
    b = torch.arange(n * l).view(n, l)
    b = (b < ll.unsqueeze(-1)).unsqueeze(-1).long()
    lens = lens.unsqueeze(-1) - 1
    mean_c = (cemb * b)[:, 1:].sum(dim=1) / lens

    # Build one-hot vectors of the entity types
    t_embs = torch.tensor([typeIDs.index(t) for t in entity_types]).unsqueeze(-1)
    t_embs = torch.zeros(cemb.size(0), len(typeIDs)).scatter_(1, t_embs, 1)

    name_embs = torch.cat((mean_c[:,:cfg.NAME_DIM], t_embs), dim=1)

    return name_embs


def encode_exs(batch, device):
    """
    Use Bert encoder to encode concept name input and create name embeddings

    Args:
        batch: Batch of (concept name, semantic type) pairs

    Returns:
        concept name embeddings
    """
    inputs = [x[0] for x in batch]
    stypes = [x[1] for x in batch]
    inp_lens = [len(x) for x in inputs]
    inputs = pad_sequence(inputs, batch_first=True).to(device)
    masks = utils.sequence_mask(inp_lens, inputs.size(1)).to(device)
    _, _, output = encoder(inputs, attention_mask=masks)
    c_emb = output[-2]  # Take the second layer from the last
    name_embs = get_name_embeddings(c_emb, stypes, cfg.MM_ST, inp_lens)
    return name_embs


if __name__ == '__main__':
    # Set defaults ------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # Generate augmented examples (n=AUG_FACTOR additional examples per original example)
    AUG_FACTOR = 1

    # Read MedMentions split information
    split_trn = [p.rstrip() for p in open(cfg.MM_PMIDS_TRN).readlines()]
    split_dev = [p.rstrip() for p in open(cfg.MM_PMIDS_DEV).readlines()]
    split_tst = [p.rstrip() for p in open(cfg.MM_PMIDS_TST).readlines()]

    # Read the hierarchical relationships between UMLS semantic types; this is
    # for merging the descendents of the 21 semantic types to top-level.
    print('Reading ST relationships from SRSTRE...')
    st_map = {t: t for t in cfg.MM_ST}
    with open(cfg.UMLS_SRSTRE) as f:
        while True:
            len_all = len(st_map)
            for line in f:
                st1, r, st2, _ = line.split('|')
                if st2 in st_map and r == 'T186':  # T186: 'is-a' relationship
                    st_map[st1] = st_map[st2]
            if len(st_map) != len_all:  # Repeat
                f.seek(0)
            else:
                break

    # Build the mapping of UMLS CUIs to semantic types
    print('Building the CUI to Semantic type mapping...')
    cui2st = dict()
    with open(cfg.UMLS_MRSTY) as f:
        for line in f:
            flds = line.split('|')
            if flds[1] in st_map:
                cui2st[flds[0]] = st_map[flds[1]]
    print(f'- {len(cui2st)} CUIs found for the st21pv semantic types')

    # Load BERT model ---------------------------------------------------------
    tokenizer, encoder = load_pretrained_bert(cfg.BERT_MODEL)
    encoder.to(device)
    encoder.eval()

    # Process the MedMentions annotations -------------------------------------
    examples = []               # List of processed examples
    a_doc = []                  # Container for a document
    mm_concepts = Counter()     # Dictionary for the UMLS concepts
    st2cuis_mm_trn = defaultdict(dict)  # For substituting entities to augment examples

    print('Reading MedMentions examples...')
    pbar = tqdm(total=216458)
    with gzip.open(cfg.MM_RAW_FILE, 'rt', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:  # End of file
                break
            if line == '\n':  # End of a document
                pbar.update(len(a_doc) + 1)
                exs, annts = proc_doc(a_doc, st2cuis_mm_trn)
                examples.extend(exs)
                mm_concepts.update(annts)
                a_doc = []
            else:
                a_doc.append(line)
    pbar.close()
    print(f'{len(mm_concepts)} unique concepts found in the examples' )

    # Generate augmented examples
    new_examples = []
    if AUG_FACTOR > 0:
        print('Generating augmented examples...')
        for ex in tqdm(examples):
            if ex['ds'] == 'trn':
                for i in range(AUG_FACTOR):
                    ex_new = ex.copy()
                    ex_new['annotations'] = []
                    ex_new['tokens'] = []
                    ex_new['generated'] = True
                    tkn_offset = 0
                    pointer = 0
                    for annt in ex['annotations']:
                        k = random.sample(st2cuis_mm_trn[annt[3]].keys(), k=1)[0]
                        m = st2cuis_mm_trn[annt[3]][k]
                        subtokens = tokenizer.tokenize(m)
                        ex_new['tokens'] += ex['tokens'][pointer:annt[0]] + subtokens
                        pointer = annt[0] + annt[1]
                        ex_new['annotations'].append(
                            (annt[0]+tkn_offset, len(subtokens), m, annt[3], k)
                        )
                        tkn_offset += len(subtokens) - annt[1]


                    ex_new['tokens'] += ex['tokens'][pointer:]
                    ex_new['token_ids'] = \
                        tokenizer.convert_tokens_to_ids(ex_new['tokens'])
                    new_examples.append(ex_new)
    print(f'{len(new_examples)} new examples generated')

    # Read in UMLS concepts and build the name normalization spacea -----------
    print('Building the UMLS name normalization space...')
    norm_dim = len(cfg.MM_ST) + cfg.NAME_DIM
    bsz = 256
    # Reserve space
    norm = torch.empty((3500000, norm_dim))
    ni2cui = [] # ni: index in the norm space
    norm_s = torch.empty((100000, norm_dim))
    ni2cui_s = []
    name_corpus = []
    name_corpus_s = []

    batch: List[torch.Tensor] = []  # Batch of input tensors (i.e., cui names)
    pbar = tqdm(total=15479756)
    with torch.no_grad(), open(cfg.UMLS_MRCONSO) as f:
        for line in f:
            pbar.update()
            flds = line.split('|')
            cui = f'UMLS:{flds[0]}'
            # Filter
            # if cui not in mm_concepts:
            #     continue
            if not cui in mm_concepts and flds[16] != 'N':
                continue
            # 0: CUI, 1: LAT, 2: TS, 4: STT, 6: ISPREF, 11: TTY
            if not (flds[0] in cui2st and \
                    flds[1] == 'ENG' and \
                    flds[2] == 'P' and \
                    flds[4] == 'PF' and \
                    flds[6] == 'Y' and \
                    flds[11] in cfg.MM_ONT):
                continue
            name_corpus.append(flds[14])
            inp = tokenizer.encode(flds[14][:cfg.MAX_NAME_LEN])
            ni2cui.append(flds[0])
            batch.append((torch.tensor(inp), cui2st[flds[0]]))
            if len(batch) == bsz:
                name_embs = encode_exs(batch, device)
                cnt = len(ni2cui)
                norm[cnt-bsz:cnt] = name_embs
                for (i, c) in enumerate(ni2cui[-bsz:]):
                    if f'UMLS:{c}' in mm_concepts:
                        cnt_s = len(ni2cui_s)
                        norm_s[cnt_s] = name_embs[i]
                        ni2cui_s.append(c)
                        name_corpus_s.append(flds[14])
                batch = []
        if len(batch) > 0:
            name_embs = encode_exs(batch, device)
            cnt = len(ni2cui)
            norm[cnt-len(batch):cnt] = name_embs
            for (i, cui) in enumerate(ni2cui[-len(batch):]):
                if f'UMLS:{cui}' in mm_concepts:
                    cnt_s = len(ni2cui_s)
                    norm_s[cnt_s] = name_embs[i]
                    ni2cui_s.append(cui)
                    name_corpus_s.append(flds[14])
            batch = []
    pbar.close()
    print(f'UMLS concept definitions (total: {len(ni2cui)}'
          f' sub: {len(ni2cui_s)}) found')
    print('Fitting the Tfidf vectorizer...')

    # Tfidf vectorizer using n-gram features
    vectorizer = TfidfVectorizer(analyzer=utils.get_ngrams, lowercase=False)
    names_ngram = vectorizer.fit_transform(name_corpus)
    names_ngram_s = vectorizer.transform(name_corpus_s)


    # Save processed files ----------------------------------------------------
    # We save three different data files; (1) examples for training (2) full-size
    # name normalization space for evaluation (3) tfidf vectorizer
    out = {
        'examples': examples + new_examples,
        'name_embs': norm_s[:len(ni2cui_s)],
        'ni2cui': ni2cui_s,
        'bert_special_tokens_map': dict(zip(tokenizer.all_special_tokens,
                                            tokenizer.all_special_ids)),
    }

    print('Saving training examples: {}'.format(cfg.MM_DS_FILE))
    pickle.dump(out, open(cfg.MM_DS_FILE, 'wb'))

    print('Saving full-size norm space: {}'.format(cfg.NS_FILE))
    out = {
        'name_embs': norm[:len(ni2cui)],
        'ni2cui': ni2cui
    }
    pickle.dump(out, open(cfg.NS_FILE, 'wb'))

    print('Saving ngram tfidf vectorizer: {}'.format(cfg.TFIDF_FILE))
    out = {
        'vectorizer': vectorizer,
        'names_ngram': names_ngram,
        'names_ngram_s': names_ngram_s
    }
    pickle.dump(out, open(cfg.TFIDF_FILE, 'wb'))
