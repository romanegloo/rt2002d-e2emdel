import pickle
import code

import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib as plt

# Load stats
onetag = pickle.load(open('data/onetag.pkl', 'rb'))
taglo = pickle.load(open('data/taglo.pkl', 'rb'))
taghi = pickle.load(open('data/taghi.pkl', 'rb'))

d = dict()
for k, v in zip(('onetag', 'taglo', 'taghi'), (onetag, taglo, taghi)):
    v = v.ret_scores['y0']
    d[k] = pd.Series(v['f'], index=v['steps'])

df = pd.DataFrame(d).interpolate()
g = sns.relplot(data=df, kind='line')


code.interact(local=locals())