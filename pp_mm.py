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
from copy import deepcopy

from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel
from tokenizers import BertWordPieceTokenizer

import GoshiBoshi
import GoshiBoshi.config as cfg
import GoshiBoshi.utils as utils


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


def convert_sent_chars_to_token_ids(s):
    s = normalizer_fn(s)  # Use the same normalization function used in tokenizer
    tokens = tokenizer.tokenize(s)
    idx_str = []
    idx = 0
    pos = 0
    for t in tokens:
        # skip spaces
        while True:
            if s[pos] == ' ' and pos < len(s):
                pos += 1
                idx_str.append(idx-1)
            else:
                break
        if pos >= len(s):
            break
        if t == '[UNK]':  # discard (backward tracing is too difficult)
            return []
        elif t.startswith('##'):
            if s[pos:].startswith(t[2:]):
                idx_str.extend([idx] * (len(t) - 2))
                pos += len(t) - 2
                idx += 1
        else:
            if s[pos:].startswith(t):
                idx_str.extend([idx] * len(t))
                pos += len(t)
                idx += 1

    return idx_str


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
    aligned = []  # container for annotations on subword-level tokens

    # Tokenization should be done per word in order to keep track of
    # subtoken-token mapping for labeling
    tok_to_word_index = []
    word_split_tokens = ex['raw'].split()
    word_split_offsets = [0]
    for w in word_split_tokens:
        word_split_offsets.append(word_split_offsets[-1] + len(w) + 1)
    all_tokens = []
    for (i, token) in enumerate(word_split_tokens):
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_word_index.append(i)
            all_tokens.append(sub_token)

    ex['tokens'] = all_tokens
    ex['token_ids'] = tokenizer.convert_tokens_to_ids(all_tokens)
    _mapping = convert_sent_chars_to_token_ids(ex['raw'])
    assert len(_mapping) != 0

    for annt in annts:
        if int(annt[2]) < offsets[-2]:   # annt belongs to the previous sent.
            continue
        if int(annt[2]) > offsets[-1]:  # ann belongs to the following sent
            break
        b, e = int(annt[1]) - offsets[-2], int(annt[2]) - offsets[-2]
        assert 0 <= b < e

        try:
            aligned.append((_mapping[b], _mapping[e-1]-_mapping[b]+1, *annt[3:]))
        except IndexError:
            raise AssertionError
    ex['annotations'] = aligned
    if ex['pmid'] == '27352045' and ex['sent_no'] == 12:
        code.interact(local=dict(globals(), **locals()))


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

    for line in doc:
        m = r_title.match(line)
        if m:
            if pmid == '':
                pmid = m.group(2)
            title = m.group(3)
        m = r_body.match(line)
        if m:
            body = m.group(3)
        m = r_annt.match(line)
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
        except AssertionError:  # ignore this doc
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
            offset = 0
            altered = False
            for i, a in enumerate(ex['annotations']):
                # Change half of the annotations
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
                    UMLS.cuis[new_cui].mm_count[3] += 1
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
    normalizer_fn = BertWordPieceTokenizer().normalize
    encoder = AutoModel.from_pretrained(cfg.BERT_MODEL,
                                        output_hidden_states=True)
    encoder.to(args.device)
    encoder.eval()

    # Read CUIs
    UMLS = GoshiBoshi.Entities()

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
        'cuis': UMLS.cuis
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