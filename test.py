from scipy.optimize import linprog
import networkx as nx
import pprint
from networkx import utils
from networkx.algorithms.bipartite.generators import configuration_model
from networkx.algorithms import isomorphism
from networkx.algorithms.shortest_paths.unweighted import all_pairs_shortest_path_length
from networkx.algorithms.components import is_connected
import numpy as np

# IMPORTANT: By default bounds on a variable are 0 <= v < +inf

# c = [-1, 4]
# A = [[-3, 1], [1, 2], [0, -1]]
# b = [6, 4, 3]
# res = linprog(c, A_ub=A, b_ub=b, options={"disp": True})
# print(res)

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
            new_constraint = [1 if N*i <= j and j < N * (i+1) else 0 for j in range(0, N*N)]
            #print(new_constraint)
            A.append(new_constraint)
            b.append(G_maxs[i])
    if G_prime_maxs is not None:
        if (len(G_prime_maxs) != N):
            print("Error! len(G_prime_maxs) != N")
        for i in range(0, N):
            new_constraint = [1 if j % N == i else 0 for j in range(0, N * N)]
            #print(new_constraint)
            A.append(new_constraint)
            b.append(G_prime_maxs[i])
    return linprog(c, A_ub=A, b_ub=b, method="interior-point", options={"disp":False, "maxiter":N*N*N*N*100})

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

        G=nx.erdos_renyi_graph(10,0.4)
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

    # Make prediction
    G_with_G = run_LP_with_maxs(G, G)
    G_prime_with_G_prime = run_LP_with_maxs(G_prime, G_prime)
    N = len(G.nodes())
    G_maxs = [0.01 for n in range(0, N)]  # IMPORTANT: The 0.01 is for slack. It should probably be a function of N.
    G_prime_maxs = [0.01 for n in range(0, N)]  # IMPORTANT: The 0.01 is for slack. It should probably be a function of N.
    for i in range(0, N*N):
        G_maxs[int(i / N)] += G_with_G.x[i]
        G_prime_maxs[int(i / N)] += G_prime_with_G_prime.x[i]
    G_with_G_prime = run_LP_with_maxs(G, G_prime, G_maxs=G_maxs, G_prime_maxs=G_prime_maxs)
    predict_iso = G_with_G_prime.status == 0 and abs(G_with_G_prime.fun - G_with_G.fun) < 0.00001

    # Get actual result
    GM = isomorphism.GraphMatcher(G, G_prime)
    actual_iso = GM.is_isomorphic()

    if predict_iso == actual_iso:
        print("\nCorrect!")
        print(actual_iso)
        print("G_with_G.fun")
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
        print("")
    else:
        print("Incorrect!")
        print("Actual:")
        print(actual_iso)
        print("\nG_with_G:")
        print(G_with_G)
        print("\nG_prime_with_G_prime:")
        print(G_prime_with_G_prime)
        print("\nG_with_G_prime:")
        print(G_with_G_prime)

