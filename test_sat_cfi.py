import networkx as nx
import numpy as np
from automorph_method_v2 import *
from dimacs_to_edge_list import *

def permute_labels_only(G):
    nodes = list(G.nodes())
    N = len(nodes)
    permutation = np.random.permutation([i for i in range(0, N)])
    G_prime = nx.Graph()
    node_to_idx = {}
    for i in range(0, N):
        node_to_idx[nodes[i]] = i
        G_prime.add_node(i)
    for edge in G.edges():
        G_prime.add_edge(permutation[node_to_idx[edge[0]]], permutation[node_to_idx[edge[1]]])
    return G_prime

for base_or_mult in ["base", "mult"]:
    for idx in range(8, 33):
        file_num = idx * 250
        graphs = []

        print("\n-------------------------")

        for char in ["a", "b", "c", "d", "e"]:
            dim_file = 'sat_cfi_dim/sat_cfi_%s_%s_%s.dmc' % (base_or_mult, file_num, char)
            edge_file = 'sat_cfi_dim/sat_cfi_%s_%s_%s.edge_list' % (base_or_mult, file_num, char)
            # convert(dim_file, edge_file)
            G = nx.read_adjlist(edge_file, create_using=nx.Graph, nodetype=int)
            G = nx.Graph(G)
            graphs.append((G, edge_file))

        for i in range(0, len(graphs)):
            for j in range(i, len(graphs)):
                print("\n %s vs %s" % (graphs[i][1], graphs[j][1]))
                G1 = CanonicalGraph(graphs[i][0])
                G2 = CanonicalGraph(permute_labels_only(graphs[j][0]))
                equal = G1.graph_comparison(G1, G2) == 0
                if (i == j) == equal:
                    print("Correct!")
                else:
                    print("Incorrect!")
                    exit(1)
