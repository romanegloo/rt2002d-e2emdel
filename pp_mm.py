"""
filename: pp_mm.py

Preprocessing MedMentions dataset

(1) Converts MedMentions records to model training examples
(2) Prepare concept name normalization space using the pre-trained BERT encoder
    (the BERT model should be consistent with the one used in the actual model)

Followings are the example records of different sources:

[Annotations in MedMentions]
25763772\t0\t5\tDCTN4\tT103\tUMLS:C4308010
(format: pmid\tstart_pos\tend_pos\tsemantic_type\tCUI)

[UMLS MRSTY]
C0000005|T116|A1.4.1.2.1.7|Amino Acid, Peptide, or Protein|AT17648347|256

[UMLS MRCONSO]
C0000005|ENG|P|L0000005|PF|S0007492|Y|A26634265||M0019694|D012711|MSH|PEP|D012711|(131)I-Macroaggregated Albumin|0|N|256
* columns in the above format are:
(CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STRING|SRL|SUPPRESS|CVF)
"""

import code
import argparse
import gzip
import re
import random
from collections import defaultdict
import pickle
from itertools import accumulate
from copy import deepcopy

from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel

import GoshiBoshi.config as cfg
import GoshiBoshi.utils as utils


class Entity:
    def __init__(self, id, st=None, st_root=None):
        self.str_id = id
        self.names = []
        self.st = st
        self.st_root = st_root
        self.mm_count = [0, 0, 0]  # Trn, Val, Tst


class Entities:
    def __init__(self):
        self.cuis = dict()
        # Read the UMLS semantic type 'is-a' relationships
        self.st_rel = self.read_st_rel()
        # Read CUIs
        self.read_cuis()
        # Filter CUIs and read names of them
        self.read_cui_names()

    def read_st_rel(self):
        """Read the hierarchical relationships between UMLS semantic types;
        this is for merging the descendents of 21 semantic types to the
        top-level.
        """
        st_map = {t: t for t in cfg.MM_ST}  # initialize with the 21 types
        with open(cfg.UMLS_SRSTRE) as f:
            while True:
                len_before = len(st_map)
                for line in f:
                    st1, r, st2, _ = line.split('|')
                    if st2 in st_map and r == 'T186':  # T186: 'is-a' relationship
                        st_map[st1] = st_map[st2]
                if len(st_map) != len_before:  # Repeat
                    f.seek(0)
                else:
                    break
        return st_map

    def read_cuis(self):
        """Reading CUIs that belong to the 21 semantic types and descendents"""
        print('=> Reading CUIs from MRSTY...')
        with open(cfg.UMLS_MRSTY) as f:
            for line in f:
                flds = line.split('|')
                if flds[1] in self.st_rel:
                    self.cuis[flds[0]] = Entity(flds[0], st=flds[1],
                                                st_root=self.st_rel[flds[1]])
        print(f'- {len(self.cuis)} CUIs found for the st21pv semantic types')

    def read_cui_names(self):
        """Filter CUIS with certian criteria and Read cui names from MRCONSO"""
        print('=> Reading CUI names from MRCONSO...')

        pbar = tqdm(total=15479756)
        with open(cfg.UMLS_MRCONSO) as f:
            for line in f:
                pbar.update()
                flds = line.split('|')
                # 0: CUI, 1: LAT, 2: TS, 4: STT, 6: ISPREF, 11: TTY, 14: STRING
                if flds[0] in self.cuis and\
                        flds[1] == 'ENG' and\
                        flds[2] == 'P' and\
                        flds[4] == 'PF' and\
                        flds[11] in cfg.MM_ONT and\
                        flds[16] == 'N':
                    if flds[14].lower() not in [n for n in self.cuis[flds[0]].names]:
                        self.cuis[flds[0]].names.append(flds[14].lower())
        pbar.close()

    def __len__(self):
        return len(self.cuis)

    def __getitem__(self, k):
        if k in self.cuis:
            return self.cuis[k]
        return


def read_mm_examples():
    print('=> Reading MedMentions examples...')
    mm_examples = []
    a_doc = []

    pbar = tqdm(total=216458)
    with gzip.open(cfg.MM_RAW_FILE, 'rt', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:  # End of file
                break
            if line == '\n':  # End of a document
                pbar.update(len(a_doc) + 1)
                exs = proc_doc(a_doc)
                mm_examples.extend(exs)
                a_doc = []
            else:
                a_doc.append(line)
    pbar.close()
    return mm_examples


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
    # and the sub-word tokenized tokens (including [UNK])

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


def proc_doc(doc):
    """Process a PubMed document, returns annotation examples.

    args:
        doc: List of document strings in PubTator format

    returns:
        - List of training examples
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

    # JoMoDEL takes annotated sentences
    sents = [title]
    sents.extend(sent_tokenize(body))
    sents = fix_sentence_splits(sents, annotations)

    for si, sent in enumerate(sents):
        ex = {
            'pmid': pmid,
            'sent_no': si,
            'raw': sent,
            'annotations': [],
            'generated': False,
            'ds': 'trn'
        }
        if pmid in split_dev:
            ex['ds'] = 'dev'
        elif pmid in split_tst:
            ex['ds'] = 'tst'
        try:
            align_annts(ex, annotations, offsets)
        except IndexError:  # ignore this doc
            return []
        else:
            exs.append(ex)

        for annt in ex['annotations']:
            cui = annt[4][5:]
            if not annt[2].lower() in [n for n in UMLS.cuis[cui].names]:
                UMLS.cuis[cui].names.append(annt[2].lower())
            UMLS.cuis[cui].mm_count[['trn', 'dev', 'tst'].index(ex['ds'])] += 1

    return exs


def fix_sentence_splits(sents, annts):
    """Sentence tokenizer might split a sentence in the middle of an annotation.
    Concatenate the two sentences such that all annotations can be contained in
    a sentence"""
    out = []
    text = ''
    for si, sent in enumerate(sents):
        text = (text + ' ' + sent).strip()
        offset = sum([len(s)+1 for s in out]) + len(text)
        success = True
        for annt in annts:
            a, b = int(annt[1]), int(annt[2])
            if a > offset:
                continue
            if a < offset < b:
                success = False
                break
        else:
            success = True
        if success:
            out.append(text)
            text = ''
            success = False
    return out


def gen_counterfactual_examples(exs):
    """Augment training data with counterfactual examples
    """
    # Annotations will be replaced with a concept with the same semantic type,
    # so we need to group CUIs by semantic types.
    st_cuis = defaultdict(list)
    for cui, c in UMLS.cuis.items():
        st_cuis[c.st].append(cui)

    print('=> Generating counterfactual examples...')
    new_examples = []
    for ex in tqdm(exs):
        # Only training examples can be used
        if ex['ds'] != 'trn' or len(ex['annotations']) == 0:
            continue
        for _ in range(args.aug):
            ex_ = deepcopy(ex)
            ex_['annotations'] = []
            ex_['generated'] = True
            # Change half of the annotations
            offset = 0
            altered = False
            for i, a in enumerate(ex['annotations']):
                if random.random() > .7:  # replace
                    altered = True
                    cui = a[4][5:]
                    st_org = UMLS.cuis[cui].st
                    new_cui = random.choice(st_cuis[st_org])
                    name = random.choice(UMLS.cuis[new_cui].names)
                    tokens = tokenizer.tokenize(name)
                    ex_['tokens'][a[0]+offset:a[0]+offset+a[1]] = tokens
                    ex_['annotations'].append((a[0]+offset, len(tokens),
                                               name, a[3], f'UMLS:{new_cui}'))
                    offset += len(tokens) - a[1]
                else:
                    ex_['annotations'].append((a[0]+offset, *a[1:]))
            ex_['token_ids'] = tokenizer.convert_tokens_to_ids(ex_['tokens'])
            if altered:
                new_examples.append(ex_)
        for annt in ex_['annotations']:
            cui = annt[4][5:]
            if not annt[2].lower() in [n for n in UMLS.cuis[cui].names]:
                UMLS.cuis[cui].names.append(annt[2].lower())
            UMLS.cuis[cui].mm_count[0] += 1  # +1 occurrence in training

    return new_examples


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
    lens = lens.unsqueeze(-1) - 2
    mean_c = (cemb * b)[:, 1:].sum(dim=1) / lens

    # Build one-hot vectors of the entity types
    t_embs = torch.tensor([typeIDs.index(t) for t in entity_types]).unsqueeze(-1)
    t_embs = torch.zeros(cemb.size(0), len(typeIDs)).scatter_(1, t_embs, 1)

    name_embs = torch.cat((mean_c[:,:cfg.NAME_DIM], t_embs), dim=1)

    return name_embs


def encode_exs(batch):
    """ Use BERT encoder to encode concept name input and create name embeddings

    Args:
        batch: Batch of (concept name, semantic type) pairs

    Returns:
        concept name embeddings
    """
    inputs = [x[0] for x in batch]
    stypes = [x[1] for x in batch]
    inp_lens = [len(x) for x in inputs]
    inputs = pad_sequence(inputs, batch_first=True).to(args.device)
    masks = utils.sequence_mask(inp_lens, inputs.size(1)).to(args.device)
    _, _, output = encoder(inputs, attention_mask=masks)
    c_emb = output[-1]  # Take the last layer outputs
    name_embs = get_name_embeddings(c_emb, stypes, cfg.MM_ST, inp_lens)
    return name_embs


def build_name_space():
    """Name normalization space with mapping indices to CUIs"""
    print('=> Building the UMLS concept name normalization space...')
    emb_dim = cfg.NAME_DIM + len(cfg.MM_ST)
    bsz = 512
    ni2cui = []

    # Reserve a space (we take up to three names per each cui)
    embeddings = torch.empty(
        (3 * sum(1 for k, c in UMLS.cuis.items() if sum(c.mm_count) > 0)),
        emb_dim
    )

    batch = []  # Batch of input tensors (i.e., cui names)
    pbar = tqdm(total=len(UMLS.cuis))
    with torch.no_grad():
        for k, c in UMLS.cuis.items():
            pbar.update()
            if sum(c.mm_count) == 0:
                continue
            for name in c.names[-3:]:
                inp = tokenizer.encode(name[:args.max_name_len])
                ni2cui.append((len(ni2cui), k, c.mm_count))
                batch.append((torch.tensor(inp), c.st_root))
                if len(batch) == bsz:
                    name_embs = encode_exs(batch)
                    cnt = len(ni2cui)
                    embeddings[cnt-bsz:cnt] = name_embs
                    batch = []
        if len(batch) > 0:
            name_embs = encode_exs(batch)
            cnt = len(ni2cui)
            embeddings[cnt-len(batch):cnt] = name_embs
            batch = []
    pbar.close()
    print(f'UMLS concept name embeddings (total: {len(ni2cui)}) created')

    return embeddings, ni2cui


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Pre-processing MedMentions',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--max_name_len', type=int, default=60,
                        help='Maximum length of concept name to be encoded')
    parser.add_argument('--aug', type=int, metavar='N', default=0,
                        help='Augment training data with counterfactual '
                        'examples (generate N additional examples per each '
                        'original example)')
    parser.add_argument('--file_mm_out', type=str, default=cfg.MM_DS_FILE,
                        help='Path to a file where mm datasets be saved')
    parser.add_argument('--save_full_ns', action='store_true',
                        help='Save the full concept name normalization space '
                        'in a separate file')
    args = parser.parse_args()

    # Set defaults ------------------------------------------------------------
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    # Read MedMentions trn/dev/tst splits
    split_trn = [p.rstrip() for p in open(cfg.MM_PMIDS_TRN).readlines()]
    split_dev = [p.rstrip() for p in open(cfg.MM_PMIDS_DEV).readlines()]
    split_tst = [p.rstrip() for p in open(cfg.MM_PMIDS_TST).readlines()]

    # Load pretrained BERT model
    tokenizer = AutoTokenizer.from_pretrained(cfg.BERT_MODEL)
    encoder = AutoModel.from_pretrained(cfg.BERT_MODEL,
                                        output_hidden_states=True)
    encoder.to(args.device)
    encoder.eval()

    # Read CUIs
    UMLS = Entities()

    # Read and convert MedMentions annotation examples
    examples = read_mm_examples()


    # Exclude the CUIs that do not have any name associated with
    print('=> Deleting all the CUIs without a name')
    to_delete = []
    total = len(UMLS.cuis)
    for cui, e in UMLS.cuis.items():
        if len(e.names) == 0:
            to_delete.append(cui)
    print('{} deleted from {} cuis'.format(len(to_delete), total))
    for cui in to_delete:
        del UMLS.cuis[cui]

    # Data Augmentation: Generate counterfactual examples for training
    if args.aug > 0:
        examples += gen_counterfactual_examples(examples)

    # Build the concept name normalization space
    ns, ns_ids = build_name_space()

    out = {
        'examples': examples,
        'name_embs': ns[:len(ns_ids)],
        'ni2cui': ns_ids,
    }
    print('=> Saving training examples: {}'.format(args.file_mm_out))
    pickle.dump(out, open(args.file_mm_out, 'wb'))

    # if args.save_full_ns:
    #     ns, ns_ids = build_name_space(full=True)
    #     print('=> Saving full-size norm space: {}'.format(cfg.NS_FILE))
    #     out = {
    #         'name_embs': ns[:len(ns_ids)],
    #         'ni2cui': ns_ids
    #     }
    #     pickle.dump(out, open(cfg.NS_FILE, 'wb'))
