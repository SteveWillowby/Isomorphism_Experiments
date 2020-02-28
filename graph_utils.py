import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# def display_graph(G):

# def zero_indexed_graph(G):

# def zero_indexed_graph_and_coloring_list(G, C):

#    Assumes zero-indexed input
# def graph_union(G1, G2):

# def induced_subgraph(G, nodes):

# def permute_node_labels(G):

# Returns a new graph where the node ids are changed according to the relabeling dict.
# def relabeled_graph(G, relabeling):

# def make_graph_with_same_degree_dist(G):

# def is_3_SR(G):

# def matrix_from_graph(G):

# Note: if allow_self_loops is false, input can be just the "right half" of the adjacency matrix as follows:
#   e.g. [[1, 1, 0], [1, 0], [1]] would denote (0, 1), (0, 2), (1, 2), (2, 3).
# def graph_from_matrix(M, allow_self_loops=False):

# def complement_of_graph_matrix(M, allow_self_loops=False):

# Takes a graph a forms a meta-graph where each node is a copy of G and each edge is a single node which all
#   nodes in two copies of G connect to.
# def Justus_square_1(G):

# def peterson_graph():

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
    color_replacements = {}
    new_C = [C[n] for n in nodes]
    new_C.sort()
    next_color = 0
    color_replacements = {}
    for c in new_C:
        if c not in color_replacements:
            color_replacements[c] = next_color
            next_color += 1
    new_C = [color_replacements[C[n]] for n in nodes]
    return new_G, new_C

# Assumes zero-indexed input
def graph_union(G1, G2):
    G3 = nx.Graph(G1)
    node_start = len(G3.nodes())
    for node in G2.nodes():
        G3.add_node(node + node_start)
    for (a, b) in G2.edges():
        G3.add_edge(a + node_start, b + node_start)
    return G3

def induced_subgraph(G, nodes):
    new_G = nx.Graph()
    for n in nodes:
        new_G.add_node(n)
    for i in range(0, len(nodes)):
        n1 = nodes[i]
        for j in range(i + 1, len(nodes)):
            n2 = nodes[j]
            if G.has_edge(n1, n2):
                new_G.add_edge(n1, n2)
    return new_G

def permute_node_labels(G):
    nodes = list(G.nodes())
    permutation = np.random.permutation([i for i in range(0, len(nodes))])
    reordering = {nodes[i]: nodes[permutation[i]] for i in range(0, len(nodes))}
    return relabeled_graph(G, reordering)

def relabeled_graph(G, relabeling):
    G_prime = nx.Graph()
    for node in G.nodes():
        G_prime.add_node(relabeling[node])
    for (a, b) in G.edges():
        G_prime.add_edge(relabeling[a], relabeling[b])
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
        G_prime.remove_edges_from([(n, n) for n in G_prime.nodes()])
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
            G_prime.remove_edges_from([(n, n) for n in G_prime.nodes()])
            #print("Edges after:")
            #print(G_prime.edges())
            #if not is_connected(G_prime):
                #print("Bad: G_prime disconnected")
            tries -= 1
        if not nx.is_connected(G_prime):
            pass
        elif len(G.edges()) == len(G_prime.edges()):
            #print("Graph creation successful")
            done = True
    return G_prime

def is_2_SR(G):
    nodes = list(G.nodes())
    nodes.sort()
    dicts = [{}, {}]
    for i in range(0, len(nodes)):
        for j in range(i + 1, len(nodes)):
            t = int(G.has_edge(i, j))
            the_dict_formed_here = {}
            for l in range(0, len(nodes)):
                if l == i or l == j:
                    continue
                cnt = int(G.has_edge(i, l)) + int(G.has_edge(j, l))
                if cnt not in the_dict_formed_here:
                    the_dict_formed_here[cnt] = 0
                the_dict_formed_here[cnt] += 1

            the_dict_to_compare_to = dicts[t]
            if len(the_dict_to_compare_to) == 0:
                dicts[t] = the_dict_formed_here
            else:
                if len(the_dict_to_compare_to) != len(the_dict_formed_here):
                    return False
                for identifier, count in the_dict_formed_here.items():
                    if identifier not in the_dict_to_compare_to or the_dict_to_compare_to[identifier] != count:
                        return False
    return True

def is_3_SR(G):
    nodes = list(G.nodes())
    nodes.sort()
    dicts = [{}, {}, {}, {}]
    for i in range(0, len(nodes)):
        for j in range(i + 1, len(nodes)):
            for k in range(j + 1, len(nodes)):
                ij = int(G.has_edge(i, j))
                ik = int(G.has_edge(i, k))
                jk = int(G.has_edge(j, k))
                num_edges = ij + ik + jk
                the_dict_formed_here = {}
                for l in range(0, len(nodes)):
                    if l == i or l == j or l == k:
                        continue
                    li = int(G.has_edge(min(l, i), max(l, i)))
                    lj = int(G.has_edge(min(l, j), max(l, j)))
                    lk = int(G.has_edge(min(l, k), max(l, k)))
                    li_count = li * (ij + ik)
                    lj_count = lj * (ij + jk)
                    lk_count = lk * (ik + jk)
                    edge_count = li + lj + lk
                    identifier = (edge_count, tuple(sorted([li_count, lj_count, lk_count])))
                    if identifier not in the_dict_formed_here:
                        the_dict_formed_here[identifier] = 0
                    the_dict_formed_here[identifier] += 1

                the_dict_to_compare_to = dicts[num_edges]
                if len(the_dict_to_compare_to) == 0:
                    dicts[num_edges] = the_dict_formed_here
                else:
                    if len(the_dict_to_compare_to) != len(the_dict_formed_here):
                        return False
                    for identifier, count in the_dict_formed_here.items():
                        if identifier not in the_dict_to_compare_to or the_dict_to_compare_to[identifier] != count:
                            return False
    return True

def matrix_from_graph(G):
    nodes = sorted(list(G.nodes()))
    matrix = [[int(G.has_edge(node_a, node_b)) for node_b in nodes] for node_a in nodes]
    return matrix

def graph_from_matrix(M, allow_self_loops=False):
    G = nx.Graph()
    for i in range(0, len(M)):
        G.add_node(i)
    for i in range(0, len(M)):
        for j in range(i + (1-int(allow_self_loops)), len(M)):
            if M[i][j]:
                G.add_edge(i, j)
    return G

def complement_of_graph_matrix(M, allow_self_loops=False):
    C = [[1 - elt for elt in row] for row in M]
    if not allow_self_loops:
        for i in range(0, len(C)):
            C[i][i] = 0
    return C

def Justus_square_1(G):
    G = zero_indexed_graph(G)
    num_V = len(G.nodes())
    G_new = graph_union(G, G)
    for i in range(2, num_V):
        G_new = graph_union(G_new, G)
    for (a, b) in G.edges():
        edge_label = num_V * num_V + min(a, b) * num_V + max(a, b)
        G_new.add_node(edge_label)
        for i in range(0, num_V):
            G_new.add_edge(a * num_V + i, edge_label)
            G_new.add_edge(b * num_V + i, edge_label)
    return G_new

def gen_graph_1():
    G = nx.Graph()
    for i in range(0, 9):
        G.add_node(i)
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(3, 4)
    G.add_edge(4, 5)
    G.add_edge(5, 0)

    G.add_edge(6, 0)
    G.add_edge(6, 1)
    G.add_edge(6, 3)
    G.add_edge(6, 4)

    G.add_edge(7, 1)
    G.add_edge(7, 2)
    G.add_edge(7, 4)
    G.add_edge(7, 5)

    G.add_edge(8, 2)
    G.add_edge(8, 3)
    G.add_edge(8, 5)
    G.add_edge(8, 0)
    return G

def gen_graph_1_cycles():
    G1 = gen_graph_1()
    G2 = graph_union(G1, G1)
    G3 = graph_union(G1, G2)
    for i in range(0, 9):
        G3.add_edge(i, 9 + i)
        G3.add_edge(9 + i, 18 + i)
        G3.add_edge(i, 18 + i)
    #for i in range(0, 5):
    #    G3.add_edge(i, 18 + (i + 1) % 5)
    #for i in range(5, 9):
    #    G3.add_edge(i, 18 + 5 + ((i - 5) + 1) % 4)
    return G3

def peterson_graph():
    G = nx.Graph()
    for i in range(0, 10):
        G.add_node(i)
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(3, 4)
    G.add_edge(4, 0)

    G.add_edge(0, 5)
    G.add_edge(1, 6)
    G.add_edge(2, 7)
    G.add_edge(3, 8)
    G.add_edge(4, 9)

    G.add_edge(5, 7)
    G.add_edge(7, 9)
    G.add_edge(9, 6)
    G.add_edge(6, 8)
    G.add_edge(8, 5)

    return G

def display_graph(G, title="A graph", colors='yellow', positions=None):
    # nodelist=
    # edgelist=
    # edge_color= (can be a list of colors? a dict?)
    # pos= (a dict?)
    # labels=
    nx.draw_networkx(G, node_color=colors, pos=positions, node_size=100)
    plt.title(title)
    plt.draw()
    plt.show()
