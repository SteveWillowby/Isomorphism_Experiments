import networkx as nx
import alg_utils
import corneil_thesis

# Assumes the input graph is zero-indexed.
# Also assumes the coloring is sorted by node index.

# The edge_types is a dict mapping (source, target) to type label
# Note that the types can be different when pointing in the opposite direction.

# Updates coloring_list in place.

# BEWARE that if init_active_set is specified, the changes that would be implied by the
# regular WL algorithm might not spread through the entire graph (i.e. if the boundary
# at some iteration does not change).
# For example, if node x has a unique color, and init_active_set is set([x]), nothing
# will change.
# I recommend just making a color unique OR being sure you already ran WL without an init_active_set.
def WL(G, coloring_list, edge_types=None, init_active_set=None, return_comparable_output=False):

    nodes = [i for i in range(0, len(G.nodes()))]
    active = set(nodes)
    active_hashes = {}
    if init_active_set is not None:
        active = init_active_set
        active_hashes = {tuple(sorted(list(active))): 1}

    # non_active = set(nodes) - active
    neighbor_lists = [list(G.neighbors(n)) for n in nodes]

    node_to_color = coloring_list
    color_to_nodes = [set([]) for i in range(0, len(nodes))] # There will at most be |V| colors.
    next_color = 1
    for node in range(0, len(node_to_color)):
        color_to_nodes[node_to_color[node]].add(node)
        if node_to_color[node] >= next_color:
            next_color = node_to_color[node] + 1

    for i in range(0, next_color):
        if len(color_to_nodes[i]) == 0:
            print("ERRONEOUS INPUT! MISSING COLOR %d in coloring_list passed to WL()" % i)
    

    color_counts = [0 for i in range(0, len(nodes))]
    used_full_color = [False for i in range(0, len(nodes))]

    comparable_output = []
    the_round = 0

    while len(active) > 0:
        if return_comparable_output:
            the_round += 1

        previous_partition_sizes = {node: len(color_to_nodes[node_to_color[node]]) for node in active}

        new_colors = []
        for node in active:
            if edge_types is not None:
                new_colors.append(((node_to_color[node], sorted([(node_to_color[n], edge_types[(n, node)]) for n in neighbor_lists[node]])), node))
            else:
                new_colors.append(((node_to_color[node], sorted([node_to_color[n] for n in neighbor_lists[node]])), node))
            color_counts[node_to_color[node]] += 1
            if color_counts[node_to_color[node]] == len(color_to_nodes[node_to_color[node]]):
                used_full_color[node_to_color[node]] = True

        new_colors.sort()

        had_a_change = False
        last_was_a_pass = True

        old_color = new_colors[0][0][0]
        on_first_in_partition = True
        full_color = used_full_color[old_color]

        if not used_full_color[old_color]:
            node_to_color[new_colors[0][1]] = next_color
            color_to_nodes[old_color].remove(new_colors[0][1])
            color_to_nodes[next_color].add(new_colors[0][1])
            had_a_change = True
            last_was_a_pass = False
            if return_comparable_output:
                comparable_output.append([the_round, next_color, 1, new_colors[0][0][1]])

        used_full_color[old_color] = False
        color_counts[old_color] = 0

        for i in range(1, len(new_colors)):
            new_color_or_partition = False
            prev_old_color = old_color
            old_color = new_colors[i][0][0]
            node = new_colors[i][1]
            if prev_old_color != old_color:
                if not on_first_in_partition: # If the previous partition actually used up a next_color
                    next_color += 1
                new_color_or_partition = True
                full_color = used_full_color[old_color]
                used_full_color[old_color] = False
                color_counts[old_color] = 0
                on_first_in_partition = True
            elif new_colors[i][0][1] != new_colors[i - 1][0][1]:
                if not on_first_in_partition or not full_color: # If we've used up a next_color in this partition.
                    next_color += 1
                new_color_or_partition = True
                on_first_in_partition = False

            if new_color_or_partition and return_comparable_output:
                if on_first_in_partition and full_color:
                    comparable_output.append([the_round, old_color, 1, new_colors[i][0][1]])
                else:
                    comparable_output.append([the_round, next_color, 1, new_colors[i][0][1]])
            elif return_comparable_output and len(comparable_output) > 0:
                comparable_output[-1][2] += 1

            if on_first_in_partition and full_color:
                last_was_a_pass = True
            else:
                node_to_color[node] = next_color 
                color_to_nodes[old_color].remove(node)
                color_to_nodes[next_color].add(node)
                had_a_change = True
                last_was_a_pass = False

        if not had_a_change: # No changes! We're done!
            break

        if not last_was_a_pass:
            next_color += 1

        if next_color == len(color_to_nodes):
            break

        new_active = set([])
        for node in active:
            if len(color_to_nodes[node_to_color[node]]) != previous_partition_sizes[node]:
                for neighbor in neighbor_lists[node]:
                    if len(color_to_nodes[node_to_color[neighbor]]) > 1: # Don't add singletons.
                        new_active.add(neighbor)
        active = new_active

    if not return_comparable_output:
        return None

    comparable_output = []
    for node_set in color_to_nodes:
        if len(node_set) == 0:
            break
        a_node = node_set.pop()
        comparable_output.append((len(node_set) + 1, tuple(sorted([node_to_color[n] for n in neighbor_lists[a_node]]))))
    return tuple(comparable_output)

def __tuple_substitute(t, idx, element):
    l = list(t)
    l[idx] = element
    return tuple(l)

# If k <= 1, assumes G is zero-indexed.
def k_dim_WL_coloring(G, k, init_coloring=None):
    if init_coloring is None:
        init_coloring = {n: 0 for n in G.nodes()}
    node_coloring = dict(init_coloring)
    if k <= 1:
        WL(G, node_coloring)
        if type(node_coloring) is list:
            node_coloring = {i: node_coloring[i] for i in range(0, len(node_coloring))}
        return node_coloring

    nodes = list(G.nodes())

    tuples = alg_utils.get_all_k_tuples(len(nodes), k)
    tuples = [tuple([nodes[idx] for idx in k_tup]) for k_tup in tuples]
    tuple_bank = {tup: tup for tup in tuples}  # Allows referencing a 'canonical' copy of the tuple so that memory is not wasted.

    # BEGIN assigning initial colors:
    tuple_coloring = []
    for tup in tuples:
        structure_list = []  # Contains edge info AND notes whether ith node is the same as the jth node.
        for i in range(0, k):
            for j in range(i + 1, k):
                structure_list.append((G.has_edge(tup[i], tup[j]), tup[i] == tup[j]))
        tuple_coloring.append(((tuple([node_coloring[n] for n in tup]), tuple(structure_list)), tup))
    tuple_coloring.sort()

    tuple_coloring = alg_utils.list_of_sorted_pairs_to_id_dict(tuple_coloring)
    # END assigning initial colors.

    # Enter main computation.
    done = False
    while not done:
        prev_tuple_coloring = dict(tuple_coloring)
        assert (0, 24, 24) in tuple_coloring
        tuple_coloring = [((tuple_coloring[tup],\
            tuple(sorted([tuple([tuple_coloring[__tuple_substitute(tup, i, n)] for i in range(0, k)]) for n in nodes]))), tup) for tup in tuples]
            # tuple(sorted([ tuple(sorted([tuple_coloring[n_tup] for n_tup in neighbors[tup][i]])) for i in range(0, k) ]))), tup) for tup in tuples]
        tuple_coloring.sort()
        next_color = -1
        prev_value = ()
        new_tuple_coloring = {}
        done = True
        for i in range(0, len(tuple_coloring)):
            (value, tup) = tuple_coloring[i]
            if value != prev_value:
                next_color += 1
                prev_value = value
            new_tuple_coloring[tup] = next_color
            if prev_tuple_coloring[tup] != next_color:  # If anything changed, carry on.
                done = False
        tuple_coloring = new_tuple_coloring

    # Convert tuple colors into node colors.
    new_node_coloring = {n: [[] for i in range(0, k)] for n in nodes}
    for tup in tuples:
        for i in range(0, k):
            new_node_coloring[tup[i]][i].append(tuple_coloring[tup])
    for n in nodes:
        for i in range(0, k):
            new_node_coloring[n][i].sort()
    new_node_coloring = [(new_node_coloring[n], n) for n in nodes]
    new_node_coloring.sort()

    return alg_utils.list_of_sorted_pairs_to_id_dict(new_node_coloring)

# If k <= 1, assumes G is zero-indexed.
def l_nodes_k_dim_WL_coloring(G, l, k, init_coloring=None):
    if init_coloring is None:
        init_coloring = {n: 0 for n in G.nodes()}
    node_coloring = dict(init_coloring)

    nodes = list(G.nodes())

    l_tuples = alg_utils.get_all_k_tuples(len(nodes), l)
    l_tuples = [tuple([nodes[n] for n in tup]) for tup in l_tuples]

    l_tuple_results = {}
    for l_tup in l_tuples:
        modified_coloring = [((n not in l_tup, node_coloring[n]), n) for n in nodes]
        modified_coloring.sort()
        modified_coloring = alg_utils.list_of_sorted_pairs_to_id_dict(modified_coloring)
        new_coloring = k_dim_WL_coloring(G, k, init_coloring=modified_coloring)
        the_quotient_graph = corneil_thesis.QuotientGraph(G, new_coloring)
        l_tuple_results[l_tup] = the_quotient_graph

    node_results = {n: [[] for i in range(0, l)] for n in nodes}
    for l_tup, result in l_tuple_results.items():
        for i in range(0, l):
            node_results[l_tup[i]][i].append(result)
    node_results = [([sorted(result_list) for result_list in results], n) for n, results in node_results.items()]
    node_results.sort()
    return alg_utils.list_of_sorted_pairs_to_id_dict(node_results)

"""
G = nx.Graph()
G.add_node(0)
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_node(4)
G.add_node(5)
G.add_node(6)
G.add_node(7)
G2 = nx.Graph(G)

for i in range(0, 2):
    offset = i * 4
    G.add_edge(offset+0,1+offset)
    G.add_edge(offset+0,2+offset)
    G.add_edge(offset+1,2+offset)
    G.add_edge(offset+1,3+offset)
    G.add_edge(offset+2,3+offset)
G.add_edge(0,4)
G.add_edge(3,7)
# print(G.edges())

coloring = [0, 0, 0, 0, 0, 0, 0, 0]
edge_types = {}
for (a, b) in G.edges():
    edge_types[(a, b)] = 0
    edge_types[(b, a)] = 0
edge_types[(1, 0)] = 1
edge_types[(5, 4)] = 1
print(WL(G, coloring, edge_types=edge_types, init_active_set=None, return_comparable_output=True))
print(coloring)
"""
