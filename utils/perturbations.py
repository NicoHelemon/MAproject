from random import choice
import numpy as np

def edge_removal(H):
    H.remove_edge(*choice(list(H.edges)))

def edge_addition(H):
    while True:
        u, v = choice(list(H.nodes)), choice(list(H.nodes))
        if u != v and (u, v) not in list(H.edges):
            break
    H.add_edge(u, v, weight = np.random.uniform())
    
def random_edge_switching(H):
    edge_removal(H)
    edge_addition(H)

def degree_preserving_edge_switching(H):
    while True:
        a, b = choice(list(H.edges))
        c, d = choice(list(H.edges))
        if len(set([a, b, c, d])) == 4 and (a, c) not in list(H.edges) and (b, d) not in list(H.edges):
            break

    H.remove_edge(a, b)
    H.remove_edge(c, d)
    H.add_edge(a, c, weight = np.random.uniform())
    H.add_edge(b, d, weight = np.random.uniform())