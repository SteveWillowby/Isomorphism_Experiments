import networkx as nx
from k_tuple_test import *
import graph_utils
import alg_utils
from weisfeiler_lehman import WL
from corneil_thesis import QuotientGraph
import sys

N = int(sys.argv[1])

def enumerate_n_choose_k_graph_pairings(n, k, N):
    output_bank = []

    basic_graph = nx.Graph()
    for i in range(0, n):
        basic_graph.add_node(i)
    next_node = n
    for neighbors in alg_utils.get_all_k_sets(n, k):
        basic_graph.add_node(next_node)
        for neighbor in neighbors:
            basic_graph.add_edge(next_node, neighbor)
        next_node += 1

    print(sorted([(b, a) for (a, b) in basic_graph.edges()]))

    disconnected_graph = graph_utils.graph_union(basic_graph, basic_graph)
    unpaired_A_nodes = [i for i in range(n, len(basic_graph.nodes()))]
    unpaired_B_nodes = [i + len(basic_graph.nodes()) for i in unpaired_A_nodes]
    bank_of_already_seen_graph_types = set()
    finished_graphs = []
    # print("Running with N=%d" % N)
    fancy_enumerate_n_choose_k_graph_pairings_helper(disconnected_graph, unpaired_A_nodes, unpaired_B_nodes,\
        bank_of_already_seen_graph_types, finished_graphs)
    return finished_graphs

def fancy_enumerate_n_choose_k_graph_pairings_helper(current_graph, unpaired_A_nodes, unpaired_B_nodes, bank_of_already_seen_graph_types, finished_graphs):
    canon_A = True
    canon_B = False
    best_canon = None
    trusted = 0
    highest_k = -1
    while trusted < 5:  # Go until it uses some k or lower k's x times in a row.
        for k in range(0, len(current_graph.nodes()) - 1):
            canon_A = KTupleTest(current_graph, k, mode="Master")
            canon_B = KTupleTest(graph_utils.permute_node_labels(current_graph), k, mode="Master")
            if canon_A.matrix == canon_B.matrix:
                if k > highest_k:
                    trusted = 0
                    highest_k = k
                    best_canon = canon_A
                else:
                    trusted += 1
                break
    if highest_k >= 2:
        print("highest_k = %d" % highest_k)

    # Trust that we now have a TRUE canon and not just a lucky canon.
    canon_matrix = tuple([tuple(row) for row in best_canon.matrix])
    if canon_matrix in bank_of_already_seen_graph_types:
        return
    bank_of_already_seen_graph_types.add(canon_matrix)

    if len(unpaired_A_nodes) == 0:
        finished_graphs.append(current_graph)
        if len(finished_graphs) % 200 == 0:
            print(len(finished_graphs))
        return

    canon_colors = best_canon.internal_labels
    i_colors = set()
    for i in range(0, len(unpaired_A_nodes)):
        # Only use one of each orbit from the first half of the graph, but DONT use that restriction for the second half.
        if canon_colors[i] in i_colors:
            continue
        i_colors.add(canon_colors[i])

        for j in range(0, len(unpaired_B_nodes)):
            new_graph = nx.Graph(current_graph)
            new_graph.add_edge(unpaired_A_nodes[i], unpaired_B_nodes[j])
            new_A = unpaired_A_nodes[0:i] + unpaired_A_nodes[i+1:]
            new_B = unpaired_B_nodes[0:j] + unpaired_B_nodes[j+1:]
            fancy_enumerate_n_choose_k_graph_pairings_helper(new_graph, new_A, new_B,\
                bank_of_already_seen_graph_types, finished_graphs)

def enumerate_n_choose_k_graph_pairings_helper(N, current_graph, unpaired_A_nodes,\
        unpaired_B_nodes, bank_of_already_seen_graph_types, finished_graphs):

    if len(unpaired_A_nodes) == 0:  # Base case: Graph Fully Connected. Assume new and add it.
        finished_graphs.append(current_graph)
        if len(finished_graphs) % 100 == 0:
            print(len(finished_graphs))
            sys.stdout.flush()
        return

    coloring = KTupleTest(current_graph, k=N, mode="Servant").internal_labels
    if type(coloring) == list:
        coloring = {i: coloring[i] for i in range(0, len(coloring))}
    key = QuotientGraph(current_graph, coloring).value
    if key in bank_of_already_seen_graph_types:  # Base-ish case: Already seen this graph from an N-node centric perspective.
        return
    bank_of_already_seen_graph_types.add(key)

    for i in range(0, len(unpaired_A_nodes)):
        for j in range(0, len(unpaired_B_nodes)):
            new_graph = nx.Graph(current_graph)
            new_graph.add_edge(unpaired_A_nodes[i], unpaired_B_nodes[j])
            new_A = unpaired_A_nodes[0:i] + unpaired_A_nodes[i+1:]
            new_B = unpaired_B_nodes[0:j] + unpaired_B_nodes[j+1:]
            enumerate_n_choose_k_graph_pairings_helper(N, new_graph, new_A, new_B,\
                bank_of_already_seen_graph_types, finished_graphs)

graphs_yo = enumerate_n_choose_k_graph_pairings(6, 3, N)
print(len(graphs_yo))
"""
for graph in graphs_yo:
    edges = []
    for (a, b) in sorted(list(graph.edges())):
        if a < (len(graph.nodes()) / 2) and b >= (len(graph.nodes()) / 2):
            edges.append((a, b))
    print(edges)
"""

for i in range(0, len(graphs_yo)):
    for j in range(i, len(graphs_yo)):
        print("%d vs %d:" % (i, j)) 
        sys.stdout.flush()
        for k in range(0, 5):
            equal = k_tuple_check(graphs_yo[i], graph_utils.permute_node_labels(graphs_yo[j]))
        print("Predicted: %s" % equal)
        print("----")
        sys.stdout.flush()
