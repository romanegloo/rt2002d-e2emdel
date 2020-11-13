# Pretrained BERT
BERT_MODEL = 'allenai/scibert_scivocab_uncased'
BERT_MODEL_DIM = 768
BERT_SPECIAL_TOKENS = {'[UNK]': 101, '[SEP]': 103, '[PAD]': 0, '[CLS]': 102,
                       '[MASK]': 104}  # Default setup, if not specified

# File Paths
MM_RAW_FILE = 'data/MedMentions/corpus_pubtator.txt.gz'  # Medmentions dataset
MM_PMIDS_TRN = 'data/MedMentions/corpus_pubtator_pmids_trng.txt' # training
MM_PMIDS_DEV = 'data/MedMentions/corpus_pubtator_pmids_dev.txt' # validation
MM_PMIDS_TST = 'data/MedMentions/corpus_pubtator_pmids_test.txt' # testing
MM_DS_FILE = 'data/MedMentions/mm.pkl'  # Processed MedMentions for training
NS_FILE = 'data/MedMentions/norm_space.pkl'  # UMLS concept normalization space
TFIDF_FILE = 'data/MedMentions/tfidf.pkl'  # UMLS concept normalization space
BEST_MDL_FILE = 'data/best.pt'  # Best model state
UMLS_MRCONSO = 'data/UMLS/MRCONSO.RRF'  # UMLS CUIs
UMLS_MRSTY = 'data/UMLS/MRSTY.RRF'  # UMLS semantic types
UMLS_SRSTRE = 'data/UMLS/SRSTRE1'  # UMLS relationships between semantic types

# Data
MAX_NAME_LEN = 120

# Run
RND_SEED = 12345
BSZ = 8          # Training batch size
LR = 3e-5  # Starting learning rate

# Model configuration
MDL_NAME = 'tag-hi'  # Architectures with varying locations of IOB tagging
MAX_SENT_LEN = 256   # Maximum sentence length of example inputs
NAME_DIM = 384       # Dimension of concept name embeddings


# MedMentions
MM_ONT = ['CPT', 'FMA', 'GO', 'HGNC', 'HPO', 'ICD10', 'ICD10CM', 'ICD9CM',
          'MDR', 'MSH', 'MTH', 'NCBI', 'NCI', 'NDDF', 'NDFRT', 'OMIM',
          'RXNORM', 'SNOMEDCT_US']
MM_ST = [f'T{t:03}' for t in [5, 7, 17, 22, 31, 33, 37, 38, 58, 62, 74, 82, 91,
                              92, 97, 98, 103, 168, 170, 201, 204]]

# 0 5: Virus,
# 1 7: Bacterium,
# 2 17: Anatomical Structure,
# 3 22: Body System
# 4 31: Body Substance
# 5 33: Finding
# 6 37: Injury or Poisoning
# 7 38: Biologic Function
# 8 58: Health Care Activity
# 9 62: Research Activity
# 10 74: Medical Device
# 11 82: Spatial Concept
# 12 91: Biomedical Occupation or Discipline
# 13 92: Organization
# 14 97: Professional or Occupational Group
# 15 98: Population Group
# 16 103: Chemical
# 17 168: Food
# 18 170: Intellectual Product
# 19 201: Clinical Attribute
# 20 204: Eukaryote