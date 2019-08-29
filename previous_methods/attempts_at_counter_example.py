import networkx as nx
import numpy as np
from automorph_method_v3 import *
from Gods_way_is_the_best_way import *
from faster_Gway import *

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

G1 = nx.Graph()
for i in range(0, 13):
    G1.add_node(i)

G1.add_edge(0,1)
G1.add_edge(1,2)
G1.add_edge(2,3)
G1.add_edge(3,4)
G1.add_edge(4,5)

G1.add_edge(6,7)
G1.add_edge(7,8)
G1.add_edge(8,9)
G1.add_edge(9,10)
G1.add_edge(10,11)

for i in range(0, 6):
    G1.add_edge(12, 2*i)

G2 = nx.Graph(G1)

G1.add_edge(5,0)
G1.add_edge(11,6)
G2.add_edge(5,6)
G2.add_edge(11,0)

GG1 = FasterGGraph(G1, first_layer=True)
print(GG1.internal_labels)
GG2 = FasterGGraph(G2, first_layer=True)
print(GG2.internal_labels)

print(GG1.graph_comparison(GG1,GG2))
