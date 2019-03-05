from scipy.optimize import linprog
import networkx as nx
import pprint
import json
from networkx import utils
# from networkx.utils import powerlaw_sequence
from networkx.algorithms.bipartite.generators import configuration_model
from networkx.algorithms.shortest_paths.unweighted import all_pairs_shortest_path_length
from networkx.algorithms.components import is_connected

# IMPORTANT: By default bounds on a variable are 0 <= v < +inf

# c = [-1, 4]
# A = [[-3, 1], [1, 2], [0, -1]]
# b = [6, 4, 3]
# res = linprog(c, A_ub=A, b_ub=b, options={"disp": True})
# print(res)

sequence = [5, 5, 4, 4, 3, 3, 2, 2]
# sequence = nx.random_powerlaw_tree_sequence(100, tries=5000)
# z=nx.utils.create_degree_sequence(100,powerlaw_sequence)
G=nx.configuration_model(sequence)
G=nx.Graph(G)
G.remove_edges_from(G.selfloop_edges())
SP = dict(all_pairs_shortest_path_length(G))
pprint.pprint(SP)
print(is_connected(G))

for i in range(1,10):
    good = False
    while not good:
        # Generate first G
        using_sequence = False
        # sequence = [5, 5, 4, 4, 3, 3, 2, 2]  # Set sequence
        # G=nx.configuration_model(sequence)
        G = nx.
        G=nx.Graph(G)
        G.remove_edges_from(G.selfloop_edges())
        G_sequence = list(d for n, d in G.degree())
        G_sequence.sort()
        if not using_sequence:
            sequence = G_sequence
        
        G_prime = nx.configuration_model(sequence)
        G_prime=nx.Graph(G_prime)
        G_prime.remove_edges_from(G_prime.selfloop_edges())
        G_prime_sequence = list(d for n, d in G_prime.degree())
        G_prime_sequence.sort()
        # print(G_sequence)
        # print(G_prime_sequence)
        first = False
        for i in range(0, len(G)):
            if G_prime_sequence[i] != G_sequence[i]:
                print("Bad Graphs (degree dist)")
                good = False
                break
        if not is_connected(G) or not is_connected(G_prime):
            print("Bad Graphs (disconnected)")
    print("Good graphs")
    SP = dict(all_pairs_shortest_path_length(G))
