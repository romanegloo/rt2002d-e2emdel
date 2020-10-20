BERT_MODEL = 'allenai/scibert_scivocab_cased'
BERT_MODEL_DIM = 768

# Paths
MM_FILE = 'data/MedMentions/corpus_pubtator.txt.gz'
MM_pmids_trn = 'data/MedMentions/corpus_pubtator_pmids_trng.txt'
MM_pmids_dev = 'data/MedMentions/corpus_pubtator_pmids_dev.txt'
MM_pmids_tst = 'data/MedMentions/corpus_pubtator_pmids_test.txt'
MRCONSO_FILE = 'data/UMLS/MRCONSO.RRF'
MRSTY_FILE = 'data/UMLS/MRSTY.RRF'
SRSTRE_FILE = 'data/UMLS/SRSTRE1'

# MedMentions
MM_ST = [f'T{t:03}' for t in [5, 7, 17, 22, 31, 33, 37, 38, 58, 62, 74, 82, 91, 
                              92, 97, 98, 103, 168, 170, 201, 204]]
