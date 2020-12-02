#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 15:14:31 2018

@author: Zhaoya Gong
"""
import numpy as np
import pandas as pd
from math import log
import scipy.sparse as spsp
from scipy.special import comb
import itertools as it
import networkx as nx
import re
import twokenize
from keras.preprocessing.text import Tokenizer



# multinomial coefficient implementation 1
def multinomial(params):
    if len(params) == 1:
        return 1
    return comb(sum(params), params[-1], exact=True) * multinomial(params[:-1])

# multinomial coefficient implementation 2
def multinomial2(lst):
    res, i = 1, 1
    for a in lst:
        for j in range(1,a+1):
            res *= i
            res //= j
            i += 1
    return res

# p_value is the implementation of equation A6 in Martinez-Romo et al(2011): DOI: 10.1103/PhysRevE.84.046108
# lst is a list with length 4: k1 + k2 + k3 + k4 = N, k1 = k, k2 = n1 - k, k3 = n2 - k
def p_value_A6(lst):
    N = lst[0]+lst[1]+lst[2]+lst[3]
    c1, c2 = 1, 1
    for i in range(lst[2]):
        c1 *= (1-(lst[1]+lst[0])/(N-i))

    for j in range(lst[0]):
        c2 *= ((lst[1]+lst[0]-j)*(lst[2]+lst[0]-j)/(N-lst[2]-j)/(lst[0]-j))

    return c1*c2


def extract_hash_tags(s):
    return set([re.sub(r"#+", "#", k) for k in set([k for j in set(
                       [i for i in s.split() if i.startswith("#")]) for k in re.findall(r"\W[\w']+", j) if k.startswith("#")])])

# tweet = "There are a #few #hashtags in #this text but #only a #few: http://example.org/#comments"
# print(extract_hash_tags(tweet))

# tweets = pd.read_csv('clean_USelection.txt', delimiter='\t', header=None, skipinitialspace=True, quoting=3)


################################################################################
### Read in tweet file
tweets = pd.read_csv('USelect_sample.txt', delimiter='\t', header=None, skipinitialspace=True, quoting=3)
# mcol = dict(zip(range(3,14), range(2,13)))
# mcol[2]=14
# tw = tw.rename(columns=mcol)
# idx9_11 = tw[14].apply(lambda x: (x.split()[1]) in set(['Oct', 'Nov', 'Sep']))
# tweets = tw[idx9_11]

### Extract hashtags from tweets and calculate some statistics
ht = {}
for s in tweets[12]:
    for t in extract_hash_tags(s.lower()):
        if t in ht.keys():
            ht[t] += 1
        else:
            ht[t] = 1

rank_ht = sorted(ht.items(), key=lambda kv: kv[1], reverse=True)
print(rank_ht[0:20])


################################################################################
### Construct a hashtag adjacency matrix based on coocurrance in the same tweet
ht_size = len(ht.keys())
ht_id = dict(zip(ht.keys(), range(ht_size)))
mt_hashtag = spsp.lil_matrix((ht_size, ht_size), dtype = float)

for s in tweets[12]:
    for c in it.combinations(extract_hash_tags(s.lower()), 2):
        mt_hashtag[ht_id[c[0]], ht_id[c[1]]] += 1
        mt_hashtag[ht_id[c[1]], ht_id[c[0]]] = mt_hashtag[ht_id[c[0]], ht_id[c[1]]]

# G = nx.from_scipy_sparse_matrix(mt_hashtag)
# nx.write_gexf(G, 'hashtag_mt.gexf')

### Build a similarity network of hashtags
inv_ht_id = {v: k for k, v in ht_id.items()}
N = tweets.shape[0]
p0 = 1e-6
mt_htwt = spsp.tril(mt_hashtag, k=-1, format='coo')

# =============================================================================
## using lil_matrix to iterate is not efficient
# for t, i in ht_id.items():
#     for j in range(i):
#         k = mt_hashtag[i, j]
#         n1 = ht[t]
#         n2 = ht[inv_ht_id[j]]
#         pvalue = multinomial([k, n1-k, n2-k, N-n1-n2+k])/comb(N, n1)/comb(N, n2)
#         if pvalue < p0:
#             mt_hashtag[i, j] = log(p0/pvalue)
#         else:
#             mt_hashtag[i, j] = 0.0
# =============================================================================

for idx, (i,j,w) in enumerate(zip(mt_htwt.row, mt_htwt.col, mt_htwt.data)):
    k = int(w)
    n1 = ht[inv_ht_id[i]]
    n2 = ht[inv_ht_id[j]]
    pvalue = p_value_A6([k, n1-k, n2-k, N-n1-n2+k])
    # pvalue = multinomial([k, n1-k, n2-k, N-n1-n2+k])/(comb(N, n1, exact=True)*comb(N, n2, exact=True))
    if pvalue < p0 and pvalue > 0:
        mt_htwt.data[idx] = log(p0/pvalue)
    elif pvalue == 0:
        mt_htwt.data[idx] = np.inf
    else:
        mt_htwt.data[idx] = 0.0

mt2 = (mt_htwt.tocsr().transpose()+mt_htwt.tocsc()).tolil()


################################################################################
### Setup an enlarged seed set of hashtags with labels
ref_antic = {'billclintonisrapist','clintoncollapse','clintoncorruption','clintoncrimefamily',
             'clintoncrimefoundation','corruptclintons','corrupthillary','criminalhillary',
             'crookedclinton','crookedhilary','crookedhillary','democratliesmatter','dropouthillary',
             'hillary2jail','hillary4prison','hillaryforprison','hillaryforprison2016','hillarylies',
             'hillaryliesmatter','imnotwithher','indicthillary','killary','lockherup',
             'lyingcrookedhillary','lyinghillary','neverhilary','neverhillary','ohhillno',
             'queenofcorruption','queenofcorrupton','sickhillary'}

ref_prot = {'alwaystrump','america1st','americafirst','blacks4trump','blacksfortrump',
            'buildthewall','deplorablesfortrump','draintheswamp','gaysfortrump','imwithhim','imwithyou',
            'latinosfortrump','latinoswithtrump','maga', 'maga3x','magax3','makeamericagreatagain',
            'makeamericasafeagain','onlytrump','rednationrising','securetheborder','trump2016',
            'trumpforpresident','trumppence16','trumppence2016','trumpstrong','trumptrain','trumpwins',
            'trumpwon','veteransfortrump','votegop','votetrump','votetrump2016',
            'votetrumppence','votetrumppence16','votetrumpusa','women4trump','womenfortrump'}

ref_antit = {'chickentrump','clowntrain','crookeddonald','crookedtrump','defeattrump','dirtydonald',
             'donthecon','dumbdonald','dumpthetrump','loserdonald','losertrump','lovetrumpshate',
             'lyingtrump','lyintrump','nastywoman','nastywomen','nastywomenvote','nevergop',
             'nevertrump','orangehitler','racisttrump','ripgop','stoptrump',
             'trump20never','trumpleaks','trumplies','whinylittlebitch'}

ref_proc = {'bluewave','bluewave2016','clintonkaine2016','connecttheleft','estoyconella','hereiamwithher',
            'herstory','heswithher','hillary2016','hillaryaprovenleader','hillaryforamerica',
            'hillaryforpresident','hillyes','iamwithher','imwithher','imwithher2016',
            'madamepresident','madampresident','ohhillyes','republicansforhillary','sheswithus',
            'strongertogether','turnncblue','uniteblue','unitedagainsthate','voteblue',
            'voteblue2016','votehillary','wearewithher','werewithher','whyimwithher'}

ref_prot = set(['#'+i for i in ref_prot])
ref_antit = set(['#'+i for i in ref_antit])
ref_proc = set(['#'+i for i in ref_proc])
ref_antic = set(['#'+i for i in ref_antic])

cls_ht = {k: v for k, v in ht_id.items() if mt2.rows[v]!=[]}

for t in ref_proc:
    if t in cls_ht:
        cls_ht[t]=cls_ht['#imwithher']

for t in ref_antit:
    if t in cls_ht:
        cls_ht[t]=cls_ht['#nevertrump']

for t in ref_prot:
    if t in cls_ht:
        cls_ht[t]=cls_ht['#maga']

for t in ref_antic:
    if t in cls_ht:
        cls_ht[t]=cls_ht['#neverhillary']

miss = 1
seedtag = ['#imwithher', '#nevertrump', '#maga', '#neverhillary']
seedcls = {cls_ht[i]: i for i in seedtag}


################################################################################
### Use label propagation method to classify more hashtags in the similarity network
while miss > 0:
    miss = 0
    rnd = np.random.permutation(ht_size)
    for i in rnd:
        cols = mt2.rows[i]
        if cols == [] or cls_ht[inv_ht_id[i]] in seedcls.keys():
            continue

        w = [mt2[i,j] for j in cols]
        c = [cls_ht[inv_ht_id[j]] for j in cols]
        cw = pd.DataFrame({'c': c, 'w': w})
        reduce = cw.groupby(['c'])['w'].sum()
        winner = reduce[reduce == reduce.max()].keys().tolist()

        if cls_ht[inv_ht_id[i]] in winner:
            continue
        else:
            if len(set(winner) & seedcls.keys()) != 0:
                winner = list(set(winner) & seedcls.keys())

            cls_ht[inv_ht_id[i]] = np.random.choice(winner, 1)[0]
            miss+=1


################################################################################
### Pruning the classfied hashtags in the network to only keep the significant ones
pd_cls_ht = pd.DataFrame({'id':[ht_id[i] for i in cls_ht.keys()], 'ht':list(cls_ht.keys()), 'cls':list(cls_ht.values()), 'fr': [ht[i] for i in cls_ht.keys()]})

prot_threshold = 0.001 * pd_cls_ht[(pd_cls_ht['cls']==cls_ht['#maga'])]['fr'].max()
prot = pd_cls_ht[(pd_cls_ht['cls']==cls_ht['#maga']) & (pd_cls_ht['fr']>prot_threshold)]

proc_threshold = 0.001 * pd_cls_ht[(pd_cls_ht['cls']==cls_ht['#imwithher'])]['fr'].max()
proc = pd_cls_ht[(pd_cls_ht['cls']==cls_ht['#imwithher']) & (pd_cls_ht['fr']>proc_threshold)]

antit_threshold = 0.001 * pd_cls_ht[(pd_cls_ht['cls']==cls_ht['#nevertrump'])]['fr'].max()
antit = pd_cls_ht[(pd_cls_ht['cls']==cls_ht['#nevertrump']) & (pd_cls_ht['fr']>antit_threshold)]

antic_threshold = 0.001 * pd_cls_ht[(pd_cls_ht['cls']==cls_ht['#neverhillary'])]['fr'].max()
antic = pd_cls_ht[(pd_cls_ht['cls']==cls_ht['#neverhillary']) & (pd_cls_ht['fr']>antic_threshold)]


################################################################################
### Export to networkx graph for gephi for visualization
cls_ids = prot['id'].tolist() + proc['id'].tolist() + antit['id'].tolist() + antic['id'].tolist()
mt_cls = mt2[cls_ids,:][:, cls_ids]
dmt_cls = mt_cls.toarray()
dmt_cls[np.isinf(dmt_cls)] = 999.
mt_cls = spsp.lil_matrix(dmt_cls)

g_cls = nx.from_scipy_sparse_matrix(mt_cls)

sub_id = dict(zip(range(len(cls_ids)), cls_ids))
for i in g_cls.nodes:
    g_cls.nodes[i]['ht'] = inv_ht_id[sub_id[i]]
    g_cls.nodes[i]['cls'] = int(cls_ht[inv_ht_id[sub_id[i]]])
    g_cls.nodes[i]['fr'] = int(ht[inv_ht_id[sub_id[i]]])

# nx.write_gexf(g_cls, "g_cls.gexf")


################################################################################
### Classify tweets according to labeled hashtags to generate the training data

#cls_ids_ht = {inv_ht_id[t]: cls_ht[inv_ht_id[t]] for t in cls_ids}
ht_valid = pd.read_csv('use_ht_valid.csv')
lb_id = {'Anti-Clinton':cls_ht['#neverhillary'], 'Anti-Trump':cls_ht['#nevertrump'], 'Pro-Trump':cls_ht['#maga'], 'Pro-Clinton':cls_ht['#imwithher']}
cls_ids_ht = {h: lb_id[l] for h, l in zip(ht_valid['ht'], ht_valid['lb'])}

tweet_label = np.zeros(len(tweets))
supC = set([cls_ht['#imwithher'], cls_ht['#nevertrump']])
supT = set([cls_ht['#maga'], cls_ht['#neverhillary']])

for i, s in enumerate(tweets[12]):
    l = [cls_ids_ht[t] for t in extract_hash_tags(s.lower()) if t in cls_ids_ht]
    if l == []: # unlabeled
        tweet_label[i] = -1
        continue
    u, idx = np.unique(l, return_inverse=True)
    mx = np.bincount(idx)
    winner = u[mx == mx.max()]
    if len(winner) > 1:
        if set(winner) == supC: # support Clinton
            tweet_label[i] = 99
        if set(winner) == supT: # support Trump
            tweet_label[i] = 1000
    elif len(winner) == 1:
        tweet_label[i] = winner[0]
    else:
        tweet_label[i] = np.nan


################################################################################
### Output training data to the formats used for word embedding learning
l_tw = pd.DataFrame({'tt': tweets[12].str.lower(), 'lb': tweet_label})
train_l_tw = l_tw[l_tw['lb']>0]

train_C = train_l_tw[(train_l_tw['lb'] == float(cls_ht['#imwithher'])) | (train_l_tw['lb'] == float(cls_ht['#nevertrump'])) | (train_l_tw['lb'] == 99.0)]
train_T = train_l_tw[(train_l_tw['lb'] == float(cls_ht['#maga'])) | (train_l_tw['lb'] == float(cls_ht['#neverhillary'])) | (train_l_tw['lb'] == 1000.0)]

with open('train_C.txt', 'w') as out:
    for tw in train_C['tt']:
        out.write(tw+'\n')

with open('train_T.txt', 'w') as out:
    for tw in train_T['tt']:
        out.write(tw+'\n')

tokenized_corpus = []
for t in train_l_tw['tt']:
    tokenized_corpus.append([tk for tk in twokenize.tokenizeRawTweetText(t) if not (tk.startswith('@') or re.match(twokenize.url, tk) or re.match(twokenize.punctSeq, tk))])
    #tokenized_corpus.append([tk for tk in twokenize.tokenizeRawTweetText(t) if not (tk.startswith('@') or re.match(twokenize.url, tk) or re.match(twokenize.punctSeq, tk) or (tk.startswith('#') and (tk in cls_ids_ht)))])

tok = Tokenizer(char_level=False)
tok.fit_on_texts(tokenized_corpus)
#print(tok.word_counts)
#print(tok.document_count)
#print(tok.word_index)
#print(tok.word_docs)

with open('vocab.csv', 'w') as out:
    out.write('<unk> 0\n')
    for k, v in tok.word_index.items():
        out.write(' '.join([k, str(v)]))
        out.write('\n')
