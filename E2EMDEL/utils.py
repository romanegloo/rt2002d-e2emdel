"""
utils.py

Classes and methods used in common
"""
from collections import defaultdict

# class WVModel:
#     """WordVector Model"""
#     def __init__(self, vocab_size=0):
#         self.vocab_size = vocab_size
#         self.sym2idx = defaultdict(lambda: len(self.sym2idx))
#         self.idx2sym = dict()
#         self.emb = None
    
#     def __len__(self):
#         return len(self.sym2idx)
    
#     def __getitem__(self, k):
#         if isinstance(k, int):
#             return self.idx2sym[k]
#         elif isinstance(k, str):
#             return self.sym2idx[k]
#         else:
#             return

#     def close(self):
#         self.sym2idx = defaultdict(lambda: self.sym2idx['UNK'], self.sym2idx)
#         self.vocab_size = len(self)