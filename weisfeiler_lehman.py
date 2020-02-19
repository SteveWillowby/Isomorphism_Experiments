import networkx as nx
import alg_utils

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
        comparable_output.append((len(node_set) + 1, sorted([node_to_color[n] for n in neighbor_lists[a_node]])))
    return comparable_output

def k_dim_WL_test(G1, G2, k):
    print("Beginning %d-dim WL test" % k)
    G1_max = max(G1.nodes()) + 1  # Connect all nodes to a single node.
    G2_max = max(G2.nodes()) + 1  # Connect all nodes to a single node.
    G1_nodes = [(1, n) for n in G1.nodes()] + [(1, G1_max)]
    G2_nodes = [(2, n) for n in G2.nodes()] + [(2, G2_max)]
    if len(G1_nodes) != len(G2_nodes):
        return False
    G1_edges = [((1, min(a, b)), (1, max(a, b))) for (a, b) in G1.edges()] + [((1, n), (1, G1_max)) for n in G1.nodes()]
    G2_edges = [((2, min(a, b)), (2, max(a, b))) for (a, b) in G2.edges()] + [((2, n), (2, G2_max)) for n in G2.nodes()]
    if len(G1_edges) != len(G2_edges):
        return False

    edges = set(G1_edges + G2_edges)
    nodes = G1_nodes + G2_nodes

    k_tups = alg_utils.get_all_k_permutations(len(G1_nodes), k)
    k_tups = [tuple([G1_nodes[idx] for idx in k_tup]) for k_tup in k_tups] + [tuple([G2_nodes[idx] for idx in k_tup]) for k_tup in k_tups]
    print("Got tups")

    k_tup_labels = []
    for k_tup in k_tups:
        label = []
        for var_a in k_tup:
            for var_b in k_tup:
                var_min = min(var_a, var_b)
                var_max = max(var_a, var_b)
                label.append(int((var_min, var_max) in edges))
        k_tup_labels.append((label, k_tup))
    k_tup_labels.sort()

    next_integer_label = 0
    new_k_tup_labels = {k_tup_labels[0][1]: 0}
    prev_label = k_tup_labels[0][0]
    for i in range(1, len(k_tup_labels)):
        if k_tup_labels[i][0] != prev_label:
            next_integer_label += 1
            prev_label = k_tup_labels[i][0]
        new_k_tup_labels[k_tup_labels[i][1]] = next_integer_label
    k_tup_labels = new_k_tup_labels

    total_num_integer_labels = next_integer_label + 1
    print("Got initial labels")

    # Now that the initial labels have been assigned...
    # Assign neighbors.
    neighbors = {k_tup: [] for k_tup in k_tups}
    for k_tup in k_tups:
        for i in range(0, k):
            for j in range(0, len(nodes)):
                if nodes[j] != k_tup[i] and nodes[j][0] == k_tup[i][0]:  # If different but from same graph...
                    new_tup = [elt for elt in k_tup]
                    new_tup[i] = nodes[j]
                    neighbors[k_tup].append(tuple(new_tup))
    print("Got neighbors")

    # Main algorithm loop:
    prev_num_integer_labels = 0
    while total_num_integer_labels != prev_num_integer_labels:
        # Acquire new labels.
        new_k_tup_labels = [((k_tup_labels[k_tup], sorted([k_tup_labels[neighbor] for neighbor in neighbors[k_tup]])), k_tup) for k_tup in k_tups]
        new_k_tup_labels.sort()
        next_integer_label = 0
        new_new_k_tup_labels = {new_k_tup_labels[0][1]: 0}
        prev_label = new_k_tup_labels[0][0]
        for i in range(1, len(new_k_tup_labels)):
            if new_k_tup_labels[i][0] != prev_label:
                next_integer_label += 1
                prev_label = new_k_tup_labels[i][0]
            new_new_k_tup_labels[new_k_tup_labels[i][1]] = next_integer_label
        k_tup_labels = new_new_k_tup_labels

        prev_num_integer_labels = total_num_integer_labels
        total_num_integer_labels = next_integer_label + 1

    # Now check to see if there is an overlap of labels...
    first_tup = k_tups[0]
    for i in range(1, len(k_tups)):
        other_tup = k_tups[i]
        if other_tup[0][0] != first_tup[0][0] and k_tup_labels[first_tup] == k_tup_labels[other_tup]: # Different graphs but same labels.
            return True
    return False

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
