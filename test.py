from scipy.optimize import linprog
import networkx as nx
import pprint
import json
from networkx import utils
# from networkx.utils import powerlaw_sequence
from networkx.algorithms.bipartite.generators import configuration_model
from networkx.algorithms.shortest_paths.unweighted import all_pairs_shortest_path_length
from networkx.algorithms.components import is_connected
import numpy as np
from numpy import random

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
            print("Graph creation successful")
            done = True
    return G_prime

# Returns all the data from running the LP
def run_LP_with_maxs(G, G_prime, G_maxs=None, G_prime_maxs=None, goal_node=None):
    N = len(G.nodes())

    # Assume weights are indexed by G and then by G_prime, so w_10 is weight from G[1] to G_prime[0]
    c = [1 for n in range(0, N*N)]
    if goal_node is not None: # If there's one node (in G, not G_prime) we really want to minimize
        for i in range(0, N):
            c[N*goal_node + i] += N*N

    SP = dict(all_pairs_shortest_path_length(G))
    SP_prime = dict(all_pairs_shortest_path_length(G_prime))
    A = []
    b = []
    for i in range(0, N):
        for j in range(i, N):
            for k in range(0, N):
                for l in range(k, N):
                    new_constraint = [0 for n in range(0, N*N)] # w_ik and w_jl
                    new_constraint[N*i + k] = -1
                    new_constraint[N*j + l] = -1
                    A.append(new_constraint)
                    b.append(-abs(SP[i][j] - SP_prime[k][l]))
                    new_constraint = [0 for n in range(0, N*N)] # w_il and w_jk
                    new_constraint[N * i + l] = -1
                    new_constraint[N * j + k] = -1
                    A.append(new_constraint)
                    b.append(-abs(SP[i][j] - SP_prime[k][l]))

    if G_maxs is not None:
        if(len(G_maxs) != N):
            print("Error! len(G_maxs) != N")
        for i in range(0, N):
            new_constraint = [0 if j < N*i or j >= N * (i+1) else 1 for j in range(0, N*N)]
            A.append(new_constraint)
            b.append(G_maxs[i])
    if G_prime_maxs is not None:
        if (len(G_prime_maxs) != N):
            print("Error! len(G_prime_maxs) != N")
        for i in range(0, N):
            new_constraint = [1 if j % N == i else 0 for j in range(0, N * N)]
            A.append(new_constraint)
            b.append(G_maxs[i])
    return linprog(c, A_ub=A, b_ub=b, method="interior-point", options={"disp":True, "maxiter":N*N*N*N*100})


for i in range(1,2):
    print("Creating Pairs of Graphs")
    good = False
    while not good:
        # Generate first G
        using_sequence = False
        sequence = [5, 5, 4, 4, 3, 3, 2, 2]  # Set sequence
        G=nx.configuration_model(sequence)
        G=nx.Graph(G)
        G.remove_edges_from(G.selfloop_edges())
        if not is_connected(G):
            print("Bad: G disconnected")
            continue
        good = True
        G_prime = make_graph_with_same_degree_dist(G)
    G_with_G = run_LP_with_maxs(G, G)
    G_prime_with_G_prime = run_LP_with_maxs(G_prime, G_prime)
    print("\n\nG with G:")
    print(G_with_G)
    print("\n\nG_prime with G_prime:")
    print(G_prime_with_G_prime)