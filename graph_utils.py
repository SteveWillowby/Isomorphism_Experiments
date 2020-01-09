import networkx as nx
import numpy as np

def zero_indexed_graph(G):
    nodes = list(G.nodes())
    nodes.sort()
    nodes_dict = {nodes[i]: i for i in range(0, len(nodes))}
    new_G = nx.Graph()
    for i in range(0, len(nodes)):
        new_G.add_node(i)
    for (a, b) in G.edges():
        new_G.add_edge(nodes_dict[a], nodes_dict[b])
    return new_G

def zero_indexed_graph_and_coloring_list(G, C):
    nodes = list(G.nodes())
    nodes.sort()
    nodes_dict = {nodes[i]: i for i in range(0, len(nodes))}
    new_G = nx.Graph()
    for i in range(0, len(nodes)):
        new_G.add_node(i)
    for (a, b) in G.edges():
        new_G.add_edge(nodes_dict[a], nodes_dict[b])
    new_C = [C[n] for n in nodes]
    return new_G, new_C

def permute_node_labels(G):
    nodes = list(G.nodes())
    N = len(nodes)
    permutation = np.random.permutation([i for i in range(0, N)])
    # print(permutation)
    G_prime = nx.Graph()
    node_to_idx = {}
    for i in range(0, N):
        node_to_idx[nodes[i]] = i
        G_prime.add_node(i)
    for edge in G.edges():
        G_prime.add_edge(permutation[node_to_idx[edge[0]]], permutation[node_to_idx[edge[1]]])
    return G_prime
    
