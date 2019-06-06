from scipy.optimize import linprog
import networkx as nx
from networkx import utils
from networkx.algorithms.bipartite.generators import configuration_model
from networkx.algorithms import isomorphism
from networkx.algorithms.shortest_paths.unweighted import all_pairs_shortest_path_length
from networkx.algorithms.components import is_connected
import numpy as np

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
            if not is_connected(G_prime):
                print("Bad: G_prime disconnected")
            tries -= 1
        if not is_connected(G_prime):
            pass
        elif len(G.edges()) == len(G_prime.edges()):
            #print("Graph creation successful")
            done = True
    return G_prime

def old_counter_example():
    G = nx.Graph()
    G_prime = nx.Graph()
    for i in range(0, 8):
        G.add_node(i)
        G_prime.add_node(i)
    for G_edge in [(0, 2), (1, 3), (1, 4), (1, 7), (2, 3), (2, 5), (2, 6), (3, 5), (3, 7), (4, 6), (5, 6)]:
        G.add_edge(G_edge[0], G_edge[1])
    for G_prime_edge in [(0, 3), (1, 5), (1, 7), (2, 4), (2, 6), (3, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]:
        G_prime.add_edge(G_prime_edge[0], G_prime_edge[1])
    return (G, G_prime)

def permute_labels_only(G):
    N = len(G.nodes())
    permutation = np.random.permutation([i for i in range(0, N)])
    G_prime = nx.Graph()
    for i in range(0, N):
        G_prime.add_node(i)
    for edge in G.edges():
        G_prime.add_edge(permutation[edge[0]], permutation[edge[1]])
    return G_prime

for i in range(1,15):
    #print("Creating Pairs of Graphs")
    good = False
    while not good:
        # Generate first G
        using_sequence = False
        #sequence = [2, 2, 2, 2, 6, 4, 4, 4, 4]  # Set sequence
        #G=nx.configuration_model(sequence)

        G=nx.erdos_renyi_graph(8,0.4)
        #G=nx.watts_strogatz_graph(10,3,0.3)
        #G=nx.barabasi_albert_graph(10,2)

        G=nx.Graph(G)
        G.remove_edges_from(G.selfloop_edges())
        if not is_connected(G):
            print("Bad: G disconnected")
            continue
        good = True
        G_prime = make_graph_with_same_degree_dist(G)
        # G_prime = permute_labels_only(G)

    (G, G_prime) = old_counter_example()
    #print(G.edges())
    #print(G_prime.edges())
    #G_prime = permute_labels_only(G_prime)
    predict_iso = lp_iso_check(G, G_prime)

    # Get actual result
    GM = isomorphism.GraphMatcher(G, G_prime)
    actual_iso = GM.is_isomorphic()

    if predict_iso == actual_iso:
        print("\nCorrect!")
        print(actual_iso)
        """print("G_with_G.fun")
        print(G_with_G.fun)
        print("G_with_G.status")
        print(G_with_G.status)
        print("G_prime_with_G_prime.fun")
        print(G_prime_with_G_prime.fun)
        print("G_prime_with_G_prime.status")
        print(G_prime_with_G_prime.status)
        print("G_with_G_prime.status")
        print(G_with_G_prime.status)
        print("G_with_G_prime.fun")
        print(G_with_G_prime.fun)
        print("")"""
    else:
        print("Incorrect!")
        print("Actual:")
        print(actual_iso)
        """print("\nG_with_G:")
        print(G_with_G)
        print("\nG_prime_with_G_prime:")
        print(G_prime_with_G_prime)
        print("\nG_with_G_prime:")
        print(G_with_G_prime)"""
        print("G's edges:")
        print(G.edges())
        print("G_prime's edges:")
        print(G_prime.edges())
        break
        #G's edges:
        #[(0, 2), (1, 3), (1, 4), (1, 7), (2, 3), (2, 5), (2, 6), (3, 5), (3, 7), (4, 6), (5, 6)]
        #G_prime's edges:
        #[(0, 3), (1, 5), (1, 7), (2, 4), (2, 6), (3, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]

