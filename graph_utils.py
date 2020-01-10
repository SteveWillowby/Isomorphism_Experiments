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
    
def make_graph_with_same_degree_dist(G):
    G_sequence = list(d for n, d in G.degree())
    G_sequence.sort()
    sorted_G_sequence = list((d, n) for n, d in G.degree())
    sorted_G_sequence.sort(key=lambda tup: tup[0])
    done = False
    while not done:
        G_prime = nx.configuration_model(G_sequence)
        G_prime = nx.Graph(G_prime)
        G_prime.remove_edges_from(G_prime.selfloop_edges())
        tries = 10
        while tries > 0 and (len(G.edges()) != len(G_prime.edges())):
            sorted_G_prime_sequence = list((d, n) for n, d in G_prime.degree())
            sorted_G_prime_sequence.sort(key=lambda tup: tup[0])
            #print("Sorted G_sequence:")
            #print(sorted_G_sequence)
            #print("Sorted G_prime_sequence:")
            #print(sorted_G_prime_sequence)
            missing = []
            for i in range(0, len(G.nodes())):
                while sorted_G_sequence[i][0] > sorted_G_prime_sequence[i][0]:
                    missing.append(sorted_G_prime_sequence[i][1])
                    sorted_G_prime_sequence[i] = (sorted_G_prime_sequence[i][0] + 1, sorted_G_prime_sequence[i][1])
            missing = np.random.permutation(missing)
            if len(missing) % 2 != 0:
                print("Sanity issue! Alert!")
            #print("Edges before:")
            #print(G_prime.edges())
            #print("Missing:")
            #print(missing)
            for i in range(0, int(len(missing) / 2)):
                G_prime.add_edge(missing[2*i], missing[2*i + 1])
            G_prime = nx.Graph(G_prime)
            G_prime.remove_edges_from(G_prime.selfloop_edges())
            #print("Edges after:")
            #print(G_prime.edges())
            #if not is_connected(G_prime):
                #print("Bad: G_prime disconnected")
            tries -= 1
        if not is_connected(G_prime):
            pass
        elif len(G.edges()) == len(G_prime.edges()):
            #print("Graph creation successful")
            done = True
    return G_prime
