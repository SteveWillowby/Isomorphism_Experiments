import networkx as nx

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
