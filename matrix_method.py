import networkx as nx
from networkx import utils
from networkx.algorithms.bipartite.generators import configuration_model
from networkx.algorithms import isomorphism
from networkx.algorithms.shortest_paths.unweighted import all_pairs_shortest_path_length
from networkx.algorithms.components import is_connected
import numpy as np

def permute_labels_only(G):
    N = len(G.nodes())
    permutation = np.random.permutation([i for i in range(0, N)])
    G_prime = nx.Graph()
    for i in range(0, N):
        G_prime.add_node(i)
    for edge in G.edges():
        G_prime.add_edge(permutation[edge[0]], permutation[edge[1]])
    return G_prime

def slow_get_adjacency_matrix(G):
    n = len(G.nodes())
    M = [[0 for c in range(0, n)] for r in range(0,n)]
    for r in range(0, n):
        for c in range(0, n):
            if G.has_edge(r, c):
                M[r][c] = 1
    return M

def swap_rows(M, i, j):
    temp_row = M[i]
    M[i] = M[j]
    M[j] = temp_row

def swap_cols(M, i, j):
    for row in range(0, len(M)):
        temp_val = M[row][i]
        M[row][i] = M[row][j]
        M[row][j] = temp_val

def get_copy_of_col(M, c):
    return [M[r][c] for r in range(0, len(M))]

def find_swap_lists(M, k):
    n = len(M)
    if k == 0:
        return [[i for i in range(0,n)]]
    main_dict = {}
    main_lists = []
    for row in range(0,n):
        curr_dict = main_dict
        for col in range(0, k):
            value = M[row][col]
            if value in curr_dict:
                if col == k-1:
                    curr_dict[value].append(row)
                else:
                    curr_dict = curr_dict[value]
            else:
                if col == k-1:
                    curr_dict[value] = [row]
                    main_lists.append(curr_dict[value])
                else:
                    curr_dict[value] = {}
                    curr_dict = curr_dict[value]
    return main_lists

def is_col_viable(M, swap_lists, target_col, actual_col_idx):
    source_col = get_copy_of_col(M, actual_col_idx)
    for sl in swap_lists:
        wrong_counts = [0,0]
        for row in range(0, len(sl)):
            if target_col[row] != source_col[row]:
                wrong_counts[source_col[row]] += 1
        if wrong_counts[0] != wrong_counts[1]:
            return False
    return True

def update_M_with_viable_column(M, swap_lists, target_col, target_col_idx, actual_col_idx):
    swap_cols(M, target_col_idx, actual_col_idx)
    source_col = get_copy_of_col(M, target_col_idx)
    for sl in swap_lists:
        wrong_locs = [[],[]]
        for row in range(0, len(sl)):
            if target_col[row] != source_col[row]:
                wrong_locs[source_col[row]].append(row)
        for wrong_idx in range(0, len(wrong_locs[0])):
            swap_rows(M, wrong_locs[0][wrong_idx], wrong_locs[1][wrong_idx])

def find_out_if_matrices_are_isomorphic(M1, M2):
    n = len(M1)
    for target_col_idx in range(0, n):
        swap_lists = find_swap_lists(M2, target_col_idx)
        target_col = get_copy_of_col(M1, target_col_idx)
        success = False
        for actual_col_idx in range(target_col_idx, n):
            if is_col_viable(M2, swap_lists, target_col, actual_col_idx):
                update_M_with_viable_column(M2, swap_lists, target_col, target_col_idx, actual_col_idx)
                success = True
                break
        if not success:
            return False
    return True

M1 = [[0,1,0,0],[1,0,1,1],[0,1,0,1],[0,1,1,0]]
M2 = [[0,1,0,0],[0,1,0,1],[0,1,1,0],[1,0,1,1]]

G1 = nx.erdos_renyi_graph(8,0.4)
#G1=nx.watts_strogatz_graph(10,3,0.3)
#G1=nx.barabasi_albert_graph(10,2)

G2 = permute_labels_only(G1)

M1 = slow_get_adjacency_matrix(G1)
M2 = slow_get_adjacency_matrix(G2)
print(M1)
print(M2)

print(find_out_if_matrices_are_isomorphic(M1, M2))
