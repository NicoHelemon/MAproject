from random import choice

class EdgeRemoval:
    def __init__(self):
        self.name = 'Edge removal'
        self.id = 'Rem'
    
    def __call__(self, H, w):
        u, v = choice(list(H.edges))
        H.remove_edge(u, v)
        return u, v

class EdgeAddition:
    def __init__(self):
        self.name = 'Edge addition'
        self.id = 'Add'
    
    def __call__(self, H, w):
        while True:
            u, v = choice(list(H.nodes)), choice(list(H.nodes))
            if u != v and not H.has_edge(u, v):
                break
        H.add_edge(u, v, weight = w)
        return u, v

class RandomEdgeSwitching:
    def __init__(self):
        self.name = 'Random edge switching'
        self.id = 'Switch'
        self.edge_removal  = EdgeRemoval()
        self.edge_addition = EdgeAddition()
    
    def __call__(self, H, w):
        a, b = self.edge_removal(H, w)
        c, d = self.edge_addition(H, w)
        return a, b, c, d
    
class DegreePreservingEdgeSwitching:
    def __init__(self):
        self.name = 'Deg preserving edge switching'
        self.id = 'Deg'
    
    def __call__(self, H, w):
        while True:
            a, b = choice(list(H.edges))
            c, d = choice(list(H.edges))
            if len(set([a, b, c, d])) == 4 and not H.has_edge(a, c) and not H.has_edge(b, d):
                break

        ab_weight = H[a][b]['weight']
        cd_weight = H[c][d]['weight']
        H.remove_edge(a, b)
        H.remove_edge(c, d)
        H.add_edge(a, c, weight = ab_weight)
        H.add_edge(b, d, weight = cd_weight)
        return a, b, c, d