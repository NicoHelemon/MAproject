"""
def adj_spec_d(H, G, mG = None):
    return euc_distance(nx.adjacency_spectrum, H, G, mG)
    """

"""metrics = list(zip(
    ["lap", "adj", "nlap", "netlsd", "portrait"],
    [lap_spec_d, adj_spec_d, nlap_spec_d, netlsd_heat_d, portrait_div_d],
    [nx.laplacian_spectrum, nx.adjacency_spectrum, nx.normalized_laplacian_spectrum, netlsd.heat, None]
))"""

"""
def weighted_LFR(n = 1000, d = 0.01, s = 10):
    m = round(d*n*(n-1)/2)

    tau1 = 3
    tau2 = 1.1
    mu = 0.2
    while True:
        try:
            G = nx.LFR_benchmark_graph(
                n, tau1, tau2, mu, average_degree=10, max_degree=100, min_community = 5, seed = s)
            break
        except:
            continue
    
    while G.number_of_edges() > m:
        edge_removal(G)

    return weighten(G)
"""

"""
def weighted_CM(n = 1000, d = 0.01, t = 1.7):
    m = round(d*n*(n-1)/2)

    while True:
        while True:
            seq = sorted([math.ceil(d) for d in nx.utils.powerlaw_sequence(n, t)], reverse=True)
            if sum(seq) % 2 == 0:
                break
            
        G = nx.Graph(nx.configuration_model(seq))
        G.remove_edges_from(nx.selfloop_edges(G))

        if m < G.number_of_edges():
            break
    
    while G.number_of_edges() > m:
        edge_removal(G)

    return weighten(G)
"""