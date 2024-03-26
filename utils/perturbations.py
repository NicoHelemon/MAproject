from random import choice

class EdgeRemoval:
    def __init__(self):
        self.name = 'Edge removal'
        self.id = 'Rem'
    
    def __call__(self, H, w):
        H.remove_edge(*choice(list(H.edges)))

class EdgeAddition:
    def __init__(self):
        self.name = 'Edge addition'
        self.id = 'Add'
    
    def __call__(self, H, w):
        while True:
            u, v = choice(list(H.nodes)), choice(list(H.nodes))
            if u != v and (u, v) not in list(H.edges):
                break
        H.add_edge(u, v, weight = w)

class RandomEdgeSwitching:
    def __init__(self):
        self.name = 'Random edge switching'
        self.id = 'Switch'
        self.edge_removal  = EdgeRemoval()
        self.edge_addition = EdgeAddition()
    
    def __call__(self, H, w):
        self.edge_removal(H, w)
        self.edge_addition(H, w)
    
class DegreePreservingEdgeSwitching:
    def __init__(self):
        self.name = 'Deg preserving edge switching'
        self.id = 'Deg'
    
    def __call__(self, H, w):
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