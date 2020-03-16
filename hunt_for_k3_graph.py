import networkx as nx
from k_tuple_test import *
import graph_utils
import alg_utils
from weisfeiler_lehman import WL
from corneil_thesis import QuotientGraph

def enumerate_n_choose_k_graph_pairings(n, k):
    output_bank = []

    basic_graph = nx.Graph()
    for i in range(0, n):
        basic_graph.add_node(i)
    next_node = n
    for neighbors in alg_utils.get_all_k_tuples(n, k):
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
    N = 0
    print("Running with N=%d" % N)
    enumerate_n_choose_k_graph_pairings_helper(N, disconnected_graph, unpaired_A_nodes, unpaired_B_nodes,\
        bank_of_already_seen_graph_types, finished_graphs)
    return finished_graphs

def enumerate_n_choose_k_graph_pairings_helper(N, current_graph, unpaired_A_nodes,\
        unpaired_B_nodes, bank_of_already_seen_graph_types, finished_graphs):

    if len(unpaired_A_nodes) == 0:  # Base case: Graph Fully Connected. Assume new and add it.
        finished_graphs.append(current_graph)
        if len(finished_graphs) % 100 == 0:
            print(len(finished_graphs))
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
            if len(new_A) + 1 != len(unpaired_A_nodes):
                print(new_A)
                print(unpaired_A_nodes)
                print(i)
            if len(new_B) + 1 != len(unpaired_B_nodes):
                print(new_B)
                print(unpaired_B_nodes)
                print(j)
            enumerate_n_choose_k_graph_pairings_helper(N, new_graph, new_A, new_B,\
                bank_of_already_seen_graph_types, finished_graphs)

graphs_yo = enumerate_n_choose_k_graph_pairings(5, 2)
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
        print("%d vs %d. Predicted: %s" % (i, j, k_tuple_check(graphs_yo[i], graph_utils.permute_node_labels(graphs_yo[j]))))
