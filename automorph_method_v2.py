import networkx as nx

class CanonicalGraph:

    # TODO: Add support for multiple connected components.
    # use nx.connected_components(G) to get the sets of nodes
    # TODO: Add this to the graph_comparison function
    def __init__(self, G):
        self.G = G
        self.size = len(self.G.nodes())
        if self.size <= 1:
            return

        # TODO: Add complement to graph_comparison function
        self.complement = len(self.G.edges()) > (self.size * (self.size - 1)) / 4
        if self.complement:
            self.G = nx.complement(self.G)

        self.sets_to_label_info = {}
        self.label_counts = {0: 0} # I believe the initial value here does not matter.

        # one_node_graph = nx.Graph()
        # one_node_graph.add_node(0)
        self.label_properties = {} # {Label id: CG} where label_graph includes external mappings

        self.next_label = 0
        self.nodes = list(self.G.nodes())
        self.internal_mapping_to_labels = {n: 0 for n in self.nodes}
        self.external_mapping_to_labels = {n: 0 for n in self.nodes}

        self.neighborhoods = {n: CanonicalGraph(self.G.subgraph(self.G.neighbors(n)))}

        counter = 0
        while True:
            sorted_multisets = self.get_new_id_multisets_in_order()
            new_labels = self.assign_new_labels_for_sorted_multisets(sorted_multisets)
            if self.are_new_labels_effectively_the_same(new_labels):
                break
            self.mapping_to_labels = new_labels
            counter += 1

    # TODO: Confirm this ordering cannot generate cycles of "greater-ness"
    # -1: This graph "smaller"
    # 0:  Isomorphic
    # 1:  This graph "bigger"
    def graph_comparison(self, graph_b):
        # Number of nodes
        if self.size < graph_b.size:
            return -1
        if self.size > graph_b.size:
            return 1
        if self.size <= 1: # Same size and identical graphs
            return 0

        # Number of labels
        if self.next_label < graph_b.next_label:
            return -1
        if self.next_label > graph_b.next_label:
            return 1

        # Counts of labels
        for i in range(0, len(self.label_counts)):
            if self.label_counts[i] > graph_b.label_counts[i]:
                return -1
            if self.label_counts[i] < graph_b.label_counts[i]:
                return 1

        # Internal-external assignment matches
        pairings_lists = [[(self.internal_mapping_to_labels[i], self.external_mapping_to_labels[i]) for i in range(0, self.size)], \
                          [(graph_b.internal_mapping_to_labels[i], graph_b.external_mapping_to_labels[i]) for i in range(0, graph_b.size)]]
        pairings_dicts = [{}, {}]
        for i in [0, 1]:
            pairings = pairings_dicts[i]
            for pairing in pairings_lists[i]:
                if pairing in pairings:
                    pairings[pairing] += 1
                else:
                    pairings[pairing] = 1

        unique_pairings = set(pairings_lists[0]) | set(pairings_lists[1])
        unique_pairings = list(unique_pairings)
        unique_pairings.sort()
        for pairing in unique_pairings:
            if pairing not in pairings_dicts[0]:
                return -1
            if pairing not in pairings_dicts[1]:
                return 1
            if pairings_dicts[0][pairing] < pairings_dicts[1][pairing]:
                return -1
            if pairings_dicts[0][pairing] > pairings_dicts[1][pairing]:
                return 1

        # Graphs of labels
        for i in range(0, len(self.label_properties)):
            comp = graph_comparison[self.label_properties[i], graph_b.label_properties[i]]
            if comp != 0:
                return comp

        return 0

    def lexicographic_comparison(self, a, b):
        for i in range(0, min(len(a), len(b))):
            if a[i] < b[i]:
                return -1
            if b[i] < a[i]:
                return 1
        if len(a) < len(b):
            return -1
        if len(b) < len(a):
            return 1
        return 0

    # O(|E|log|E| + |E|*|V|log|V|) = O(|E||V|log|V|)
    # TODO: Implement a bucket or a radix sort.
    def get_new_id_multisets_in_order(self):
        arrays = []
        for node in self.nodes:
            s = []
            for neighbor in self.mapping_to_neighbors[node]:
                s.append(self.mapping_to_labels[neighbor])
            s.sort()
            arrays.append((node, s))
        arrays.sort(key=(lambda x: x[1]), cmp=self.lexicographic_comparison) # O(cmp * |V|log|V|) = O(|E|*|V|log|V|)
        return arrays

    # O(|E|)
    def assign_new_labels_for_sorted_multisets(self, sorted_multisets):
        new_labels = {}
        prev = []
        for i in range(0, len(sorted_multisets)):
            current = sorted_multisets[i]
            if self.lexicographic_comparison(prev, current[1]) != 0:
                self.next_label += 1
                self.label_counts[self.next_label] = 0

                sets_to_labels = self.sets_to_labels
                for neighbors_old_label in current[1]:
                    if neighbors_old_label not in sets_to_labels:
                        sets_to_labels[neighbors_old_label] = {}
                    sets_to_labels = sets_to_labels[neighbors_old_label]
                sets_to_labels["label"] = self.next_label

            new_labels[current[0]] = self.next_label
            self.label_counts[self.next_label] += 1
            prev = current[1]
        return new_labels

    # O(|V|)
    def are_new_labels_effectively_the_same(self, new_labels):
        old_group_identifiers = {}
        new_group_identifiers = {}
        for node in self.nodes:
            old_label = self.mapping_to_labels[node]
            new_label = new_labels[node]
            if old_label not in old_group_identifiers:
                old_group_identifiers[old_label] = node
            if new_label not in new_group_identifiers:
                new_group_identifiers[new_label] = node
            if old_group_identifiers[old_label] != new_group_identifiers[new_label]:
                return False
        return True

    # TODO: Make this non-recursive
    def are_sets_to_labels_equal(self, a_set, b_set):
        if len(a_set) != len(b_set):
            return False
        for set_member, value in a_set.items():
            if set_member not in b_set:
                return False
            if set_member == "label":
                if b_set["label"] != value:
                    return False
            elif not self.are_sets_to_labels_equal(a_set[set_member], b_set[set_member]):
                return False
        return True

    def is_equal(self, other_canonical_description):
        other_counts = other_canonical_description.label_counts
        for label, count in self.label_counts.items():
            if label not in other_counts or other_counts[label] != count:
                return False
        return self.are_sets_to_labels_equal(self.sets_to_labels, other_canonical_description.sets_to_labels)
