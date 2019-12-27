import networkx as nx

class PathsMethod:

    def __init__(self, G1, coloring, paths=None):
        self.paths = paths
        if self.paths is None:
            self.paths = PathSteps(G)

class FlimsyCanonicalizer:

    def __init__(self, G, init_coloring):
        pass

class WL_Coloring:

    # Assumes the input colors are >= 0.
    def __init__(self, G, init_coloring_dict):
        nodes = set(G.nodes)
        active = set(nodes)
        non_active = set([])
        neighbor_sets = {n: set(G.neighbors(n)) for n in nodes}
        coloring = init_coloring_dict
        partitions = {}
        for n, c in coloring.items():
            if c not in partitions:
                partitions[c] = 1
            else:
                partitions[c] += 1
        partition_sizes = {n: partitions[coloring[n]] for n in nodes}

        while True:
            print(active)
            nc_pairs = []
            for node in active:
                neighbor_colors = [coloring[n] for n in neighbor_sets[node]]
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
        self.active_sets = [set(self.nodes)]
        # The final result. For each round, a set of directed edges with num of steps across.
        self.steps = {}

        self.ALT_RECORD = True # Seems to save space and produce equivalent runtime.
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
        while not self.compute_next_iteration():
            pass

    # Returns True if finished. False otherwise.
    def compute_next_iteration(self):
        self.completed_iterations += 1
        had_any_paths = False
        for node in self.nodes:
            num_paths = self.find_value(self.completed_iterations, node, [])
            print("Num to node %d: %d" % (node, num_paths))
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
        subsets_of_incl_nodes = self.subsets(other_incl_nodes, min_size=2, max_size=len(other_incl_nodes)-1)
        for subset in subsets_of_incl_nodes:
            full_subset_value = self.find_value(iteration, dest_node, subset)
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

G = nx.Graph()
G.add_node(0)
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_node(4)
G.add_node(5)
G.add_node(6)
G.add_node(7)

for i in range(0, 2):
    offset = i * 4
    G.add_edge(offset+0,1+offset)
    G.add_edge(offset+0,2+offset)
    G.add_edge(offset+1,2+offset)
    G.add_edge(offset+1,3+offset)
    G.add_edge(offset+2,3+offset)
G.add_edge(0,4)
G.add_edge(3,7)

test = PathSteps(G)
print(test.subsets([1, 3, 5, 6], 1, 4))
init_coloring = {n: 0 for n in range(0, 8)}
init_coloring[0] = 1
test2 = WL_Coloring(G, init_coloring)
print(test2.coloring)
