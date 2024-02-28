#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import scipy as sp
import pandas as pd

import random
import itertools
import timeit
import time as ti

import networkx as nx
import netlsd
import distanceclosure as dc
from portrait_divergence import portrait_divergence


# In[19]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[8]:


def weighten(G):
    for (u, v, w) in G.edges(data=True):
        w['weight'] = np.random.uniform()
    return G


# In[9]:


V = 1000
D = 0.01
E = round(D*V*(V-1)/2)

G = nx.barabasi_albert_graph(V, round(E/V))
G = weighten(G)


# In[11]:


def removal(H):
    H.remove_edge(*random.choice(list(H.edges)))

def addition(H):
    u, v = random.choice(list(H.nodes)), random.choice(list(H.nodes))
    while (u, v) in H.edges:
        u, v = random.choice(list(H.nodes)), random.choice(list(H.nodes))
    H.add_edge(u, v, weight = np.random.uniform())
    
def random_switching(H):
    removal(H)
    addition(H)

def degree_preserving_switching(H):
    a, b = random.choice(list(H.edges))
    c, d = random.choice(list(H.edges))
    while (a, c) in H.edges or (b, d) in H.edges:
        a, b = random.choice(list(H.edges))
        c, d = random.choice(list(H.edges))

    H.remove_edge(a, b)
    H.remove_edge(c, d)
    H.add_edge(a, c, weight = np.random.uniform())
    H.add_edge(b, d, weight = np.random.uniform())


# In[12]:


def euc_distance(m, H, G, mG = None):
    #if mG is None:
    #    mG = m(G)
    return np.linalg.norm(m(H) - mG)

def lap_spec_d(H, G, mG = None):
    return euc_distance(nx.laplacian_spectrum, H, G, mG)

def adj_spec_d(H, G, mG = None):
    return euc_distance(nx.adjacency_spectrum, H, G, mG)

def nlap_spec_d(H, G, mG = None):
    return euc_distance(nx.normalized_laplacian_spectrum, H, G, mG)

def netlsd_heat_d(H, G, mG = None):
    return euc_distance(netlsd.heat, H, G, mG)

def portrait_div_d(H, G):
    return portrait_divergence(H, G)


# In[13]:


def test(G, perturbation, metrics, K, N):
    time = []
    apsp_G = dc.metric_backbone(G, weight='weight')
    
    Distances_full = {m_id : [ [] for _ in range(K) ] for m_id, _, _ in metrics}
    Distances_apsp = {m_id : [ [] for _ in range(K) ] for m_id, _, _ in metrics}
    
    prec_full = {}
    prec_apsp = {}
    
    for m_id, md, m in metrics:
        if m is not None:
            prec_full[m_id] = m(G)
            prec_apsp[m_id] = m(apsp_G)
            md_f = md(G, G, prec_full[m_id])
            md_a = md(apsp_G, apsp_G, prec_apsp[m_id])
            for i in range(K):
                Distances_full[m_id][i].append(md_f)
                Distances_apsp[m_id][i].append(md_a)
        else:
            md_f = md(G, G)
            md_a = md(apsp_G, apsp_G)
            for i in range(K):
                Distances_full[m_id][i].append(md_f)
                Distances_apsp[m_id][i].append(md_a)
    for i in range(K):
        H = G.copy()
        for j in range(N):
            start = timeit.default_timer()
            perturbation(H)
            apsp_H = dc.metric_backbone(H, weight='weight')
            
            for m_id, md, m in metrics:
                if m is not None:
                    Distances_full[m_id][i].append(md(H, G, prec_full[m_id]))
                    Distances_apsp[m_id][i].append(md(apsp_H, apsp_G, prec_apsp[m_id]))
                else:
                    Distances_full[m_id][i].append(md(H, G))
                    Distances_apsp[m_id][i].append(md(apsp_H, apsp_G))

            time.append(timeit.default_timer() - start)
            print(f'Perturbation nb {j + i*N}')
            print(f'Time spent               = ' + ti.strftime('%H:%M:%S', ti.gmtime(int(np.sum(time)))))
            print(f'Estimated time remaining = ' + ti.strftime('%H:%M:%S', ti.gmtime(int(np.mean(time) * (N*K - 1 - j + i*N)))))
            print()
    
    return Distances_full, Distances_apsp


# In[14]:


metrics = [
    ("lap", lap_spec_d, nx.laplacian_spectrum), 
    ("adj", adj_spec_d, nx.adjacency_spectrum),
    ("nlap", nlap_spec_d, nx.normalized_laplacian_spectrum),
    ("netlsd", netlsd_heat_d, netlsd.heat),
    ("portrait", portrait_div_d, None)
]


# In[17]:


N = 1000
K = 5

Distances_full, Distances_apsp = test(G, removal, metrics, K, N)


# In[76]:


df_full = pd.concat({k : pd.DataFrame(a).T.agg(['mean', 'std'], axis=1) for k, a in Distances_full.items()}, axis=1)
df_apsp = pd.concat({k : pd.DataFrame(a).T.agg(['mean', 'std'], axis=1) for k, a in Distances_apsp.items()}, axis=1)


# In[77]:


df_full.to_csv("results_full.csv")
df_apsp.to_csv("results_apsp.csv")

