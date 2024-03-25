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

"""
def inverse_fade(x):
    return -1/(2*x + 1) + 1

def arc_fade(x):
    return 2*np.sqrt(0.25 - (x - 0.5)**2)

rejection sampling

def add_gaussian_noise(G, σ, min = 0, max = None, absolute = False):
    for (_, _, w) in G.edges(data=True):
        noise = np.random.normal(0, σ)
        if absolute:
            noise = abs(noise)
        w['weight'] += noise
        w['weight'] = clamp(w['weight'], min, max)
    return G
"""

"""
def clamp(x, a = None, b = None):
    if a is not None:
        x = max(a, x)
    if b is not None:
        x = min(b, x)
    return x
    """

"""
def add_gaussian_noise(G, σ, weight):
    # resampling
    if weight in [exp, log]:
        for (_, _, w) in G.edges(data=True):
            new_w = w['weight'] + np.random.normal(0, σ)
            while new_w < 0:
                new_w = w['weight'] + np.random.normal(0, σ)
            w['weight'] = new_w

    # reflecting (how about circular shift?)
    elif weight in [uni]:
        for (_, _, w) in G.edges(data=True):
            new_w = w['weight'] + np.random.normal(0, σ)
            while new_w < 0 or new_w > 1:
                if new_w < 0:
                    new_w = -new_w
                else:    
                    new_w = 2 - new_w
            w['weight'] = new_w

    return G
"""

"""
for G, G_name in graphs:
    for w in list(zip(weights, weights_e))[0:1]:
        for p in perturbations[0:1]:
            distance_vs_perturbation_test_execution(
                G, G_name, w, p, metrics[0:3], K = 10, N = 500, step = 5, time_printing = True)

for G, G_name in graphs:
    for w in list(zip(weights, weights_p, weights_m))[0:1]:
        gaussian_noise_test_execution(
             G, G_name, w, metrics[0:3], sigmas = np.linspace(0, 0.1, 10+1).tolist(), 
             K = 20, time_printing = True)

for G, G_name in graphs[0:1]:
     clustering_gaussian_noise_test_execution(
          G, G_name, zip(weights, weights_p, weights_m), metrics[0:3], 0.1, time_printing = True)
          """