import networkx as nx
import alg_utils
import graph_utils
from weisfeiler_lehman import *

def shortest_paths_comparison(G1, G2):
    if len(G1.nodes()) != len(G2.nodes()):
        return False
    if len(G1.edges()) != len(G2.edges()):
        return False

    # Zero-index the graphs so we can use arrays rather than dicts.
    G1 = graph_utils.zero_indexed_graph(G1)
    G1_nodes = [i for i in range(0, len(G1.nodes()))]
    G2 = graph_utils.zero_indexed_graph(G2)
    G2_nodes = [i for i in range(0, len(G2.nodes()))]

    # Compute paths
    G1_neighbors_lists = [list(G1.neighbors(n)) for n in G1_nodes]
    G2_neighbors_lists = [list(G2.neighbors(n)) for n in G2_nodes]
    G1_edge_labels = {}
    for (a, b) in G1.edges():
        G1_edge_labels[(a, b)] = []
        G1_edge_labels[(b, a)] = []
    G2_edge_labels = {}
    for (a, b) in G2.edges():
        G2_edge_labels[(a, b)] = []
        G2_edge_labels[(b, a)] = []
    G1_visited_sets = [set([n]) for n in G1_nodes]
    G2_visited_sets = [set([n]) for n in G2_nodes]
    G1_unfinished = set(G1_nodes)
    G2_unfinished = set(G2_nodes)
    path_length = 1
    while len(G1_unfinished) > 0:
        prev_G1_visited_sets = [set(v) for v in G1_visited_sets]
        prev_unfinished = set(G1_unfinished)
        for unfinished_node in prev_unfinished:
            transmitted = False
            for neighbor in G1_neighbors_lists[unfinished_node]:
                if neighbor not in prev_unfinished:
                    continue
                old_neighbor_size = len(prev_G1_visited_sets[neighbor])
                neighbor_size_with_transmission = len(prev_G1_visited_sets[neighbor] | prev_G1_visited_sets[unfinished_node])
                transmission_size = neighbor_size_with_transmission - old_neighbor_size
                if transmission_size > 0:
                    edge = (unfinished_node, neighbor)
                    if len(G1_edge_labels[edge]) == 0 or G1_edge_labels[edge][-1][0] < path_length:
                        G1_edge_labels[edge].append([path_length, transmission_size])
                    else:
                        print("I THINK THIS SHOULDN'T HAPPEN!")
                        G1_edge_labels[edge][-1][1] += transmission_size
                    transmitted = True
                    G1_unfinished.add(neighbor)

                G1_visited_sets[neighbor] |= prev_G1_visited_sets[unfinished_node]
                if len(G1_visited_sets[neighbor]) == len(G1_nodes):
                    G1_unfinished.discard(neighbor)

            if not transmitted:
                G1_unfinished.discard(unfinished_node)
        path_length += 1
    path_length = 1
    while len(G2_unfinished) > 0:
        prev_G2_visited_sets = [set(v) for v in G2_visited_sets]
        prev_unfinished = list(G2_unfinished)
        for unfinished_node in prev_unfinished:
            transmitted = False
            for neighbor in G2_neighbors_lists[unfinished_node]:
                if neighbor not in prev_unfinished:
                    continue
                old_neighbor_size = len(prev_G2_visited_sets[neighbor])
                neighbor_size_with_transmission = len(prev_G2_visited_sets[neighbor] | prev_G2_visited_sets[unfinished_node])
                transmission_size = neighbor_size_with_transmission - old_neighbor_size
                if transmission_size > 0:
                    edge = (unfinished_node, neighbor)
                    if len(G2_edge_labels[edge]) == 0 or G2_edge_labels[edge][-1][0] < path_length:
                        G2_edge_labels[edge].append([path_length, transmission_size])
                    else:
                        print("I THINK THIS SHOULDNT HAPPEN!")
                        G2_edge_labels[edge][-1][1] += transmission_size

                    transmitted = True
                    G2_unfinished.add(neighbor)

                G2_visited_sets[neighbor] |= prev_G2_visited_sets[unfinished_node]
                if len(G2_visited_sets[neighbor]) == len(G2_nodes):
                    G2_unfinished.discard(neighbor)

            if not transmitted:
                G2_unfinished.discard(unfinished_node)
        path_length += 1

    G1_edge_label_list = [(0, e) for e, l in G1_edge_labels.items()]
    G2_edge_label_list = [(0, e) for e, l in G2_edge_labels.items()]
    if not alg_utils.jointly_further_sort_by_and_compare(G1_edge_label_list, G1_edge_labels, G2_edge_label_list, G2_edge_labels):
        return False

    for (label, edge) in G1_edge_label_list:
        G1_edge_labels[edge] = label
    for (label, edge) in G2_edge_label_list:
        G2_edge_labels[edge] = label
    
    G1_canon = FlimsyCanonicalizer(G1, [0 for n in G1_nodes], G1_edge_labels)
    G2_canon = FlimsyCanonicalizer(G2, [0 for n in G2_nodes], G2_edge_labels)
    return G1_canon.matrix == G2_canon.matrix

def paths_comparison(G1, G2):
    if len(G1.nodes()) != len(G2.nodes()):
        return False
    if len(G1.edges()) != len(G2.edges()):
        return False
    G1_nodes = set(G1.nodes())
    G2_nodes = set(G2.nodes())
    G1_paths = PathSteps(G1)
    G2_paths = PathSteps(G2)
    G1_done = False
    G2_done = False
    completed_iterations = 0
    G1_edge_types = [(0, (s, t)) for (s, t) in G1.edges()] + [(0, (t, s)) for (s, t) in G1.edges()]
    G2_edge_types = [(0, (s, t)) for (s, t) in G2.edges()] + [(0, (t, s)) for (s, t) in G2.edges()]
    G1_coloring = [(0, n) for n in G1_nodes]
    G2_coloring = [(0, n) for n in G2_nodes]
    prev_max_color = 0
    while (not G1_done) and (not G2_done):
        G1_done = G1_paths.compute_next_iteration()
        G2_done = G2_paths.compute_next_iteration()
        completed_iterations += 1
        G1_latest_paths = {n: G1_paths.access(completed_iterations, n, []) for n in G1_nodes}
        G2_latest_paths = {n: G2_paths.access(completed_iterations, n, []) for n in G2_nodes}
        if not alg_utils.jointly_further_sort_by_and_compare(G1_coloring, G1_latest_paths, G2_coloring, G2_latest_paths):
            print("Color Check A Failed!")
            return False

        G1_latest_steps = {}
        for i in range(0, len(G1_edge_types)):
            (s, t) = G1_edge_types[i][1]
            if (s, t) in G1_paths.steps[completed_iterations]:
                G1_latest_steps[(s, t)] = G1_paths.steps[completed_iterations][(s, t)]
            else:
                G1_latest_steps[(s, t)] = 0
        G2_latest_steps = {}
        for i in range(0, len(G2_edge_types)):
            (s, t) = G2_edge_types[i][1]
            if (s, t) in G2_paths.steps[completed_iterations]:
                G2_latest_steps[(s, t)] = G2_paths.steps[completed_iterations][(s, t)]
            else:
                G2_latest_steps[(s, t)] = 0
        if not alg_utils.jointly_further_sort_by_and_compare(G1_edge_types, G1_latest_steps, G2_edge_types, G2_latest_steps):
            print("Paths Check A Failed!")
            return False

        G1_WL_colors = [(0, n) for n in G1_nodes]
        alg_utils.further_sort_by(G1_WL_colors, G1_coloring)
        G1_WL_colors = [(n, c) for (c, n) in G1_WL_colors]
        G1_WL_colors.sort()
        G1_WL_colors = [x[0] for x in G1_WL_colors]
        WL(G1, G1_WL_colors, {(s, t): c for (c, (s, t)) in G1_edge_types})
        G2_WL_colors = [(0, n) for n in G2_nodes]
        alg_utils.further_sort_by(G2_WL_colors, G2_coloring)
        G2_WL_colors = [(n, c) for (c, n) in G2_WL_colors]
        G2_WL_colors.sort()
        G2_WL_colors = [x[0] for x in G2_WL_colors]
        WL(G2, G2_WL_colors, {(s, t): c for (c, (s, t)) in G2_edge_types})
        #G1_WL_colors = WLColoringWithEdgeTypes(G1, {n: c for (c, n) in G1_coloring}, {(s, t): c for (c, (s, t)) in G1_edge_types})
        #G2_WL_colors = WLColoringWithEdgeTypes(G2, {n: c for (c, n) in G2_coloring}, {(s, t): c for (c, (s, t)) in G2_edge_types})
        if not alg_utils.jointly_further_sort_by_and_compare(G1_coloring, G1_WL_colors, G2_coloring, G2_WL_colors):
            print("Color Check B Failed!")
            return False
        if completed_iterations == 1 or prev_max_color < G1_coloring[-1][0]:
            G1_partial_canon = FlimsyCanonicalizer(G1, G1_WL_colors, {(s, t): c for (c, (s, t)) in G1_edge_types})
            G2_partial_canon = FlimsyCanonicalizer(G2, G2_WL_colors, {(s, t): c for (c, (s, t)) in G2_edge_types})
            if G1_partial_canon.matrix == G2_partial_canon.matrix:
                print("A total of %d orbits found" % (G1_coloring[-1][0] + 1))
                return True
        prev_max_color = G1_coloring[-1][0]

    print("Finished without Canonicalizing As Same!")
    return False


class FlimsyCanonicalizer:

    def __init__(self, G, init_coloring, edge_types):
        nodes = list(G.nodes())
        sorted_color_node_pairs = [(0, n) for n in nodes]
        alg_utils.further_sort_by(sorted_color_node_pairs, init_coloring)
        final_node_order = []
        coloring = [0 for n in nodes]
        for (c, n) in sorted_color_node_pairs:
            coloring[n] = c

        while len(sorted_color_node_pairs) > 0:
            final_node_order.append(sorted_color_node_pairs[-1][1])
            sorted_color_node_pairs.pop()
            if len(sorted_color_node_pairs) > 0:
                WL(G, coloring, edge_types, init_active_set=set([final_node_order[-1]]))
                alg_utils.further_sort_by(sorted_color_node_pairs, coloring)
        self.final_node_order = final_node_order
        self.matrix = self.node_order_to_matrix(G, final_node_order)

    def node_order_to_matrix(self, G, final_node_order):
        matrix = []
        for i in range(0, len(final_node_order)):
            next_row = []
            for j in range(i + 1, len(final_node_order)):
                if G.has_edge(final_node_order[i], final_node_order[j]):
                    next_row.append(1)
                else:
                    next_row.append(0)
            matrix.append(next_row)

        #self.ordered_labels = [self.external_labels[n] for n in final_node_order]
        return matrix

class WLColoringWithEdgeTypes:

    # Assumes the input colors are >= 0.
    # The edge_types is a dict mapping (source, target) to type label
    # Note that the types can be different when pointing in the opposite direction.
    def __init__(self, G, init_coloring_dict, edge_types, init_active_set=None):
        nodes = set(G.nodes())
        active = set(nodes)
        if init_active_set is not None:
            active = init_active_set
        non_active = nodes - active
        neighbor_sets = {n: set(G.neighbors(n)) for n in nodes}
        coloring = init_coloring_dict
        partitions = {}
        for n, c in coloring.items():
            if c not in partitions:
                partitions[c] = 1
            else:
                partitions[c] += 1
        partition_sizes = {n: partitions[coloring[n]] for n in nodes}

        while len(active) > 0:
            nc_pairs = []
            for node in active:
                neighbor_colors = [(coloring[n], edge_types[(n, node)]) for n in neighbor_sets[node]]
                neighbor_colors.sort()
                nc_pairs.append(([coloring[node], neighbor_colors], node))
            nc_pairs.sort()
            curr_partition_start = 0
            next_color = 0
            new_coloring = {nc_pairs[0][1]: 0}
            for i in range(1, len(nc_pairs)):
                if nc_pairs[i][0] != nc_pairs[i-1][0]:
                    next_color += 1
                new_coloring[nc_pairs[i][1]] = next_color
            for non_active_node in non_active:
                new_coloring[non_active_node] = coloring[non_active_node] + len(nodes)
            coloring = new_coloring

            partitions = {}
            for n, c in coloring.items():
                if c not in partitions:
                    partitions[c] = 1
                else:
                    partitions[c] += 1
            new_partition_sizes = {n: partitions[coloring[n]] for n in nodes}
            new_active = set([])
            any_change = False
            for node in active:
                if new_partition_sizes[node] != partition_sizes[node]:
                    for neighbor in neighbor_sets[node]:
                        if new_partition_sizes[neighbor] > 1:
                            new_active.add(neighbor)
                    any_change = True
            active = new_active
            non_active = nodes - active

            if not any_change:
                break
            partition_sizes = new_partition_sizes
        self.coloring = coloring

class PathSteps:

    def __init__(self, G):
        self.G = G
        self.neighbor_sets = {node: set(self.G.neighbors(node)) for node in self.G.nodes()}
        self.nodes = list(self.G.nodes())
        self.nodes.sort()
        # The final result. For each round, a set of directed edges with num of steps across.
        self.steps = {}

        self.ALT_RECORD = False # Seems to save space and produce equivalent runtime.
        if not self.ALT_RECORD:
            # First layer of dict is path length.
            # Within this, a dict for paths to a node. # is indicated by ["num"]
            # Within this, IN SORTED ORDER, num paths involving all nodes indicated.
            # Negation of a node means that the node is _not_ included in the path.
            self.record_of_paths = {0: {}}
            for i in range(0, len(self.nodes)):
                n1 = self.nodes[i]
                self.record_of_paths[0][n1] = {"num": 1}
                for j in range(0, len(self.nodes)):
                    n2 = self.nodes[j]
                    self.record_of_paths[0][n1][n2] = {"num": 0}
        else:
            self.record_of_paths = {}
            for n1 in self.nodes:
                self.record_of_paths[(0, n1, ())] = 1
                for n2 in self.nodes:
                    self.record_of_paths[(0, n1, tuple([n2]))] = 0

        self.completed_iterations = 0
        #while not self.compute_next_iteration():
        #    pass

    # Returns True if finished. False otherwise.
    def compute_next_iteration(self):
        self.completed_iterations += 1
        had_any_paths = False
        for node in self.nodes:
            num_paths = self.find_value(self.completed_iterations, node, [])
            if num_paths > 0:
                had_any_paths = True
        return not had_any_paths

    # Assumes other_incl_nodes is sorted
    def access(self, iteration, dest_node, other_incl_nodes=[]):
        if not self.ALT_RECORD:
            order = other_incl_nodes
            if iteration not in self.record_of_paths:
                return None
            d = self.record_of_paths[iteration][dest_node]
            for sub_node in order:
                if sub_node not in d:
                    return None
                d = d[sub_node]
            if "num" not in d:
                return None
            return d["num"]
        else:
            key = (iteration, dest_node, tuple(other_incl_nodes))
            if key in self.record_of_paths:
                return self.record_of_paths[key]
            #print("key (%s, %s, %s) not in record" % (key[0], key[1], key[2]))
            return None

    # Assumes other_incl_nodes is sorted
    def assign(self, value, iteration, dest_node, other_incl_nodes=[]):
        if not self.ALT_RECORD:
            order = other_incl_nodes
            if iteration not in self.record_of_paths:
                self.record_of_paths[iteration] = {n: {} for n in self.nodes}
            d = self.record_of_paths[iteration][dest_node]
            for sub_node in order:
                if sub_node not in d:
                    d[sub_node] = {}
                d = d[sub_node]
            d["num"] = value
        else:
            key = (iteration, dest_node, tuple(other_incl_nodes))
            self.record_of_paths[key] = value

    def add_steps(self, value, iteration, dest_node, neighbor):
        if iteration not in self.steps:
            self.steps[iteration] = {n: {} for n in self.nodes}
        self.steps[iteration][dest_node][neighbor] = value

    # Assumes other_incl_nodes is sorted
    def find_value(self, iteration, dest_node, other_incl_nodes=[]):
        if iteration < 0:
            raise ValueError("ERROR! THIS SHOULD NEVER OCCUR! iteration < 0 in find_value()")
        this_val = self.access(iteration, dest_node, other_incl_nodes)
        if this_val is not None:
            return this_val
        # NECESSARY CHECKS (???) BEGIN HERE.
        dest_node_all = self.access(iteration, dest_node, [])
        if dest_node_all is not None and dest_node_all == 0:
            self.assign(0, iteration, dest_node, other_incl_nodes)
            return 0
        if len(other_incl_nodes) > 1:
            for incl_node in other_incl_nodes:
                if self.find_value(iteration, dest_node, [incl_node]) == 0:
                    self.assign(0, iteration, dest_node, other_incl_nodes)
                    return 0
        # NECESSARY CHECKS (???) END HERE.
        # SPEEDUP (HOPEFULLY) HEURISTICS BEGIN HERE.
        if False and not self.ALT_RECORD:
            if self.search_existing_subset_results_for_zero(iteration, dest_node, other_incl_nodes):
                self.assign(0, iteration, dest_node, other_incl_nodes)
                return 0
        else:
            subsets_of_incl_nodes = self.subsets(other_incl_nodes, min_size=2, max_size=len(other_incl_nodes)-1)
            for subset in subsets_of_incl_nodes:
                full_subset_value = self.access(iteration, dest_node, subset)
                if full_subset_value == 0:
                    self.assign(0, iteration, dest_node, other_incl_nodes)
                    return 0
            # THE FOLLOWING SLOWS THINGS DOWN.
            #subsets_of_subset = self.subsets(subset, min_size=1, max_size=len(subset)-1)
            #for subset_of_subset in subsets_of_subset:
            #    sub_subset_value = self.find_value(iteration, dest_node, subset_of_subset)
            #    if sub_subset_value == full_subset_value:
            #        # We can refine our search to be smaller!
            #        remainder = list(set(other_incl_nodes) - (set(subset) - set(subset_of_subset)))
            #        remainder.sort()
            #        full_value = self.find_value(iteration, dest_node, remainder)
            #        self.assign(full_value, iteration, dest_node, other_incl_nodes)
            #        return full_value
        # HEURISTICS END HERE.
        neighbors = self.neighbor_sets[dest_node]
        sum_of_values = 0
        for neighbor in neighbors:
            incl_set = set(other_incl_nodes)
            incl_set.discard(neighbor)
            incl_list = list(incl_set)
            incl_list.sort()
            base_neighbor_value = self.find_value(iteration - 1, neighbor, incl_list)
            if base_neighbor_value > 0:
                incl_set.add(dest_node)
                incl_list = list(incl_set)
                incl_list.sort()
                steps_value = base_neighbor_value - self.find_value(iteration - 1, neighbor, incl_list)
                self.add_steps(steps_value, iteration, dest_node, neighbor)
                sum_of_values += steps_value
        self.assign(sum_of_values, iteration, dest_node, other_incl_nodes)
        return sum_of_values

    def subsets(self, a_sorted_list, min_size, max_size):
        subsets = []
        for size in range(min_size, max_size+1):
            indices = [i for i in range(0, size)]
            done = False
            while not done:
                subset = [a_sorted_list[i] for i in indices]
                subsets.append(subset)
                if indices[-1] == len(a_sorted_list) - 1:
                    idx_offset = 0
                    while indices[(size - 1) - idx_offset] == (len(a_sorted_list) - 1) - idx_offset:
                        idx_offset += 1
                        if idx_offset == size:
                            done = True
                            break
                    if not done:
                        target = indices[(size - 1) - idx_offset] + 1
                        while idx_offset >= 0:
                            indices[(size - 1) - idx_offset] = target
                            target += 1
                            idx_offset -= 1
                else:
                    indices[-1] += 1
        return subsets

    def search_existing_subset_results_for_zero(self, iteration, dest_node, other_incl_nodes):
        if iteration not in self.record_of_paths:
            return False
        start_dict = self.record_of_paths[iteration][dest_node]
        incl_nodes_set = set(other_incl_nodes)
        stack = [[(node, sub_dict) for node, sub_dict in start_dict.items()]]
        while len(stack) > 0:
            while len(stack[-1]) > 0:
                if stack[-1][-1][0] == "num":
                    if stack[-1][-1][1] == 0:
                        return True
                    stack[-1].pop()
                elif stack[-1][-1][0] in incl_nodes_set:
                    (investigate_node, corresponding_dict) = stack[-1].pop()
                    stack.append([(node, sub_dict) for node, sub_dict in corresponding_dict.items()])
                else:
                    stack[-1].pop()
            stack.pop()

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
print(G.edges())
for i in range(0, 2):
    offset = i * 4
    G2.add_edge((offset+1)%8,(2+offset)%8)
    G2.add_edge((offset+1)%8,(3+offset)%8)
    G2.add_edge((offset+2)%8,(3+offset)%8)
    G2.add_edge((offset+2)%8,(4+offset)%8)
    G2.add_edge((offset+3)%8,(4+offset)%8)
G2.add_edge(1,5)
G2.add_edge(4,0)
print(G2.edges())

test = PathSteps(G)
print(test.subsets([1, 3, 5, 6], 1, 4))
init_coloring = {n: 0 for n in range(0, 8)}
init_edge_types = {}
for (s, t) in G.edges():
    init_edge_types[(s, t)] = 0
    init_edge_types[(t, s)] = 0
init_coloring[0] = 0
init_edge_types[(0,1)] = 1
test2 = WLColoringWithEdgeTypes(G, init_coloring, init_edge_types, init_active_set=set([0]))
print(test2.coloring)

print(paths_comparison(G, G2))
"""
