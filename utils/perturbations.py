from random import choice
import numpy as np

def edge_removal(H, w):
    H.remove_edge(*choice(list(H.edges)))

def edge_addition(H, w):
    while True:
        u, v = choice(list(H.nodes)), choice(list(H.nodes))
        if u != v and (u, v) not in list(H.edges):
            break
    H.add_edge(u, v, weight = w())
    
def random_edge_switching(H, w):
    edge_removal(H, w)
    edge_addition(H, w)

def degree_preserving_edge_switching(H, w):
    while True:
        a, b = choice(list(H.edges))
        c, d = choice(list(H.edges))
        if len(set([a, b, c, d])) == 4 and (a, c) not in list(H.edges) and (b, d) not in list(H.edges):
            break

    ab_weight = H[a][b]['weight']
    cd_weight = H[c][d]['weight']
    H.remove_edge(a, b)
    H.remove_edge(c, d)
    H.add_edge(a, c, weight = ab_weight)
    H.add_edge(b, d, weight = cd_weight)