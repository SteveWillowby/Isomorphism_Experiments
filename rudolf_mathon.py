import networkx as nx
import graph_utils 

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
    return M
