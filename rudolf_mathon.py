import networkx as nx
import graph_utils 
from k_tuple_test import *
from corneil_thesis import *

def graph_from_adj_dict(adj_dict):
    G = nx.Graph()
    nodes = set([key for key, neighbors in adj_dict.items()])
    for node in nodes:
        G.add_node(node)
    for key, neighbors in adj_dict.items():
        for neighbor in neighbors:
            if neighbor not in nodes:
                nodes.add(neighbor)
                G.add_node(neighbor)
            G.add_edge(key, neighbor)
    return G

def Rudolf_Mathon_A25():
    edges = {1: [11, 12, 13, 14, 15, 16], \
             2: [11, 12, 17, 18, 19, 20], \
             3: [11, 13, 17, 21, 22, 23], \
             4: [11, 14, 18, 21, 24, 25], \
             5: [12, 13, 19, 22, 24, 25], \
             6: [12, 15, 20, 21, 23, 24], \
             7: [13, 16, 18, 20, 23, 25], \
             8: [14, 15, 17, 19, 23, 25], \
             9: [14, 16, 17, 20, 22, 24], \
            10: [15, 16, 18, 19, 21, 22]}
    return graph_from_adj_dict(edges)

def Rudolf_Mathon_B25():
    edges = {1: [11, 12, 13, 14, 15, 16], \
             2: [11, 12, 17, 18, 19, 20], \
             3: [11, 13, 17, 21, 22, 23], \
             4: [11, 14, 18, 21, 24, 25], \
             5: [12, 13, 19, 22, 24, 25], \
             6: [12, 15, 20, 21, 23, 24], \
             7: [13, 16, 18, 20, 23, 25], \
             8: [14, 15, 17, 19, 23, 25], \
             9: [14, 16, 19, 20, 21, 22], \
            10: [15, 16, 17, 18, 22, 24]}
    return graph_from_adj_dict(edges)

def Rudolf_Mathon_A35():
    edges = {
        1: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35], \
        2: [10, 11, 12, 13, 16, 17, 18, 19, 28, 29, 30, 31, 32, 33, 34, 35], \
        3: [10, 11, 12, 13, 14, 15, 18, 19, 21, 22, 23, 25, 26, 27, 34, 35], \
        4: [ 8,  9, 12, 13, 15, 16, 17, 19, 20, 22, 23, 24, 27, 30, 33, 35], \
        5: [ 8,  9, 12, 13, 14, 16, 17, 18, 20, 21, 25, 26, 27, 28, 29, 31], \
        6: [ 8,  9, 10, 11, 14, 15, 17, 19, 21, 23, 24, 25, 29, 30, 31, 32], \
        7: [ 8,  9, 10, 11, 14, 15, 16, 18, 20, 22, 24, 26, 28, 32, 33, 34], \
        8: [15, 17, 18, 19, 25, 26, 27, 31, 32, 33, 34, 35], \
        9: [14, 16, 18, 19, 21, 22, 23, 28, 29, 30, 34, 35], \
       10: [15, 16, 17, 18, 20, 21, 23, 26, 27, 29, 30, 33], \
       11: [14, 16, 17, 19, 20, 22, 24, 25, 27, 28, 31, 35], \
       12: [14, 15, 17, 18, 21, 22, 24, 25, 28, 30, 32, 33], \
       13: [14, 15, 16, 19, 20, 23, 24, 26, 29, 31, 32, 34], \
       14: [24, 26, 27, 29, 30, 32, 33, 35], \
       15: [20, 21, 22, 28, 29, 31, 32, 35], \
       16: [21, 22, 23, 25, 27, 31, 32, 35], \
       17: [22, 23, 24, 26, 27, 28, 29, 34], \
       18: [20, 23, 24, 25, 30, 31, 34, 35], \
       19: [20, 21, 25, 26, 28, 30, 33, 34], \
       20: [24, 25, 27, 28, 29, 30], \
       21: [24, 27, 28, 31, 33, 34], \
       22: [25, 26, 29, 30, 31, 34], \
       23: [24, 25, 26, 28, 32, 35], \
       24: [31, 33, 34], \
       25: [29, 32, 33], \
       26: [28, 30, 31, 33], \
       27: [30, 32, 34, 35], \
       28: [32, 35], \
       29: [33, 34, 35], \
       30: [31, 32], \
       31: [35], \
       32: [34], \
       33: [35]}
    return graph_from_adj_dict(edges)

def Rudolf_Mathon_B35():
    edges = {
         1: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35], \
         2: [10, 11, 12, 13, 16, 17, 18, 19, 28, 29, 30, 31, 32, 33, 34, 35], \
         3: [10, 11, 12, 13, 14, 15, 18, 19, 21, 22, 23, 25, 26, 27, 34, 35], \
         4: [ 8,  9, 12, 13, 15, 16, 17, 19, 20, 22, 23, 26, 27, 29, 30, 33], \
         5: [ 8,  9, 12, 13, 14, 16, 17, 18, 20, 21, 24, 25, 27, 28, 31, 35], \
         6: [ 8,  9, 10, 11, 14, 15, 17, 18, 22, 23, 24, 25, 29, 30, 31, 32], \
         7: [ 8,  9, 10, 11, 14, 15, 16, 19, 20, 21, 24, 26, 28, 32, 33, 34], \
         8: [15, 17, 18, 19, 25, 26, 27, 31, 32, 33, 34, 35], \
         9: [14, 16, 18, 19, 21, 22, 23, 28, 29, 30, 34, 35], \
        10: [15, 16, 17, 18, 20, 21, 23, 24, 27, 30, 33, 35], \
        11: [14, 16, 17, 19, 20, 22, 25, 26, 27, 28, 29, 31], \
        12: [14, 15, 17, 19, 21, 22, 24, 25, 28, 30, 32, 33], \
        13: [14, 15, 16, 18, 20, 23, 24, 26, 29, 31, 32, 34], \
        14: [24, 26, 27, 29, 30, 32, 33, 35], \
        15: [20, 21, 22, 28, 29, 31, 32, 35], \
        16: [21, 22, 23, 25, 27, 31, 32, 33], \
        17: [21, 23, 24, 26, 27, 28, 29, 34], \
        18: [20, 22, 25, 26, 28, 30, 33, 34], \
        19: [20, 23, 24, 25, 30, 31, 34, 35], \
        20: [24, 25, 27, 28, 29, 30], \
        21: [25, 26, 29, 30, 31, 34], \
        22: [24, 27, 28, 21, 22, 24], \
        23: [24, 25, 26, 28, 32, 35], \
        24: [31, 33, 34], \
        25: [29, 32, 33], \
        26: [28, 30, 31, 33], \
        27: [30, 32, 34, 35], \
        28: [32, 35], \
        29: [33, 34, 35], \
        30: [31, 32], \
        31: [35], \
        32: [34], \
        33: [35]}
    return graph_from_adj_dict(edges)

def Rudolf_Mathons_random_graph_extension(G):
    C = graph_utils.matrix_from_graph(G)
    C_size = len(C)
    C.append([1 for i in range(0, C_size)])
    for i in range(0, C_size):
        C[i].append(1)
    C[-1].append(0)
    CComp = graph_utils.complement_of_graph_matrix(C)
    M = []
    for i in range(0, len(C)):
        M.append(C[i] + CComp[i])
    for i in range(0, len(C)):
        M.append(CComp[i] + C[i])
    return graph_utils.graph_from_matrix(M)

def Rudolf_Mathon_E72_A35():
    return Rudolf_Mathons_random_graph_extension(Rudolf_Mathon_A35())

def Rudolf_Mathon_E72_B35():
    return Rudolf_Mathons_random_graph_extension(Rudolf_Mathon_B35())

if __name__ == "__main__":
    G1 = graph_utils.zero_indexed_graph(Rudolf_Mathon_A25())
    G2 = graph_utils.zero_indexed_graph(Rudolf_Mathon_B25())
    # G2 = graph_utils.permute_node_labels(G2)

    G1_labels = KTupleTest(G1, k=0, external_labels=[1 if i == 21 else 0 for i in range(0, 25)], mode="Servant").internal_labels
    G2_labels = KTupleTest(G2, k=0, external_labels=[1 if i == 21 else 0 for i in range(0, 25)], mode="Servant").internal_labels

    if type(G1_labels) is list:
        G1_labels = {i: G1_labels[i] for i in range(0, len(G1_labels))}
    if type(G2_labels) is list:
        G2_labels = {i: G2_labels[i] for i in range(0, len(G2_labels))}

    # print(G1_labels)
    # print(G2_labels)

    G1_labels_QG = QuotientGraph(G1, G1_labels)
    G2_labels_QG = QuotientGraph(G2, G2_labels)

    print(G1_labels_QG == G2_labels_QG)

    G1_sub1 = graph_utils.induced_subgraph(G1, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
    G1_sub2 = graph_utils.induced_subgraph(G1, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    G2_sub1 = graph_utils.induced_subgraph(G2, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
    G2_sub2 = graph_utils.induced_subgraph(G2, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    groups_1 = [[0, 3, 5, 6], [10, 11, 12, 13, 14, 15, 17, 19, 20, 22, 23, 24], [1, 2, 4, 7, 8, 9], [16, 18, 21]]
    groups_2 = [[12, 15, 16, 18, 20, 23], [0, 1, 3, 5, 6, 7], [10, 11, 13, 14, 17, 19, 22, 24], [2, 4, 8, 9], [21]]
    def get_positions_from_groups_row(groups):
        positions = {}
        for group_id in range(0, len(groups)):
            group = groups[group_id]
            x = group_id * 130
            y = (len(group) / 2) * 100
            for element in group:
                positions[element] = (x, y)
                y -= 100
        return positions
    def get_positions_from_groups_axial(groups):
        radial_dir = [1, 0]
        positions = {}
        for group_id in range(0, len(groups)):
            group = groups[group_id]
            d = 100
            for element in group:
                positions[element] = (d * radial_dir[0], d * radial_dir[1])
                d += 100
            radial_dir = [radial_dir[1] * -1, radial_dir[0]]
        return positions
    positions_1 = get_positions_from_groups_row(groups_1)
    positions_2 = get_positions_from_groups_axial(groups_2)
    positions_2[21] = (-300, -500)
    
    graph_utils.display_graph(G1, title="The counterexample to Corneil (1st half).", positions=positions_1,\
        colors=[x[1] for x in sorted([(n, c) for n, c in G1_labels.items()])])
    graph_utils.display_graph(G2, title="The counterexample to Corneil (2nd half).", positions=positions_2,\
        colors=[x[1] for x in sorted([(n, c) for n, c in G2_labels.items()])])
