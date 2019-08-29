import networkx as nx

class FasterNeighborsRevisited:

    def __init__(self, G, external_labels=None, nodewise=True):
        self.nodewise = nodewise
        if external_labels is None:
            external_labels = {n: 0 for n in G.nodes()}

        if self.nodewise:
            self.initial_nodes = list(G.nodes())
            self.initial_G = G
            (self.G, external_labels) = self.expand_graph(G, external_labels)
        else:
            self.G = G
        self.nodes = list(self.G.nodes())
        self.mapping_to_neighbors = {n: set(self.G.neighbors(n)) for n in self.nodes}
        self.internal_labels = {n: external_labels[n] for n in self.nodes}
        self.external_labels = {n: external_labels[n] for n in self.nodes}
        self.higher_than_any_internal_label = max(len(self.nodes), max([l for n, l in self.internal_labels.items()]) + 1)
        self.label_definitions = []

        # Give things a head-start with shortest-path values.
        if self.nodewise:
            basic_overlay = FasterNeighborsRevisited(self.G, nodewise=False)
            basic_overlay = basic_overlay.internal_labels
            self.nodewise_overlays = {}
            for node in self.nodes:
                path_labels = nx.single_source_shortest_path_length(self.G, node) # Do shortest paths computation to speed things up.
                max_dist = max([d for n, d in path_labels.items()]) + 1
                for n in self.nodes:
                    if n not in path_labels:
                        path_labels[n] = max_dist
                self.nodewise_overlays[node] = {n: path_labels[n] + basic_overlay[n] * len(self.nodes) for n in self.nodes}
                max_value = max([l for n, l in self.nodewise_overlays[node].items()])
                self.nodewise_overlays[node][node] = max_value + 1

        counter = 0
        while True:
            sorted_ids = self.get_new_ids_in_order()
            new_labels = self.assign_new_labels_for_sorted_ids(sorted_ids)
            if self.are_new_labels_effectively_the_same(new_labels):
                break
            self.internal_labels = new_labels
            counter += 1

        if self.nodewise:
            self.set_canonical_form()
        else:
            self.label_pairings = [(self.internal_labels[n], external_labels[n]) for n in self.nodes]
            self.label_pairings.sort()

    def get_new_ids_in_order(self):
        ids = []
        for node in self.nodes:
            if self.nodewise:
                new_labels = {n: self.nodewise_overlays[node][n] * self.higher_than_any_internal_label + self.internal_labels[n] for n in self.nodes}
                i = (self.internal_labels[node], 1, FasterNeighborsRevisited(self.G, external_labels=new_labels, nodewise=False)) # Referencing oneself appears to be necessary!
                self.nodewise_overlays[node] = i[2].internal_labels
            else:
                neighbors = [self.internal_labels[n] for n in self.mapping_to_neighbors[node]]
                neighbors.sort()
                i = (self.internal_labels[node], neighbors) # TODO: Consider whether referencing oneself is necessary.
            ids.append((node, i))
        ids.sort(key=(lambda x: x[1]))
        return ids

    # O(|V'| * cmp_for_sub_labels)
    def assign_new_labels_for_sorted_ids(self, sorted_ids):
        new_labels = {}
        prev = None
        next_numeric_label = -1
        self.label_definitions = []
        for current in sorted_ids:
            if prev is None or prev != current[1]:
                next_numeric_label += 1
                self.label_definitions.append(current[1])

            new_labels[current[0]] = next_numeric_label
            prev = current[1]
        return new_labels

    # O(|V'|)
    def are_new_labels_effectively_the_same(self, new_labels):
        old_group_identifiers = {}
        new_group_identifiers = {}
        for node in self.nodes:
            old_label = self.internal_labels[node]
            new_label = new_labels[node]
            if old_label not in old_group_identifiers:
                old_group_identifiers[old_label] = node
            if new_label not in new_group_identifiers:
                new_group_identifiers[new_label] = node
            if old_group_identifiers[old_label] != new_group_identifiers[new_label]:
                return False
        return True

    def full_comparison(self, other):
        if not self.nodewise and other.nodewise:
            print("This comparison should never occur!")
            return -1
        if self.nodewise and not other.nodewise:
            print("This comparison should never occur!")
            return 1

        if self.nodewise:
            if self.ordered_labels < other.ordered_labels:
                return -1
            if self.ordered_labels > other.ordered_labels:
                return 1
            if self.matrix < other.matrix:
                return -1
            if self.matrix > other.matrix:
                return 1
            return 0

        # Internal-external label matching.
        if self.label_pairings < other.label_pairings:
            return -1
        if self.label_pairings > other.label_pairings:
            return 1

        # Actual label definitions.
        for i in range(0, len(self.label_definitions)):
            if self.label_definitions[i][0] < other.label_definitions[i][0]:
                return -1
            if self.label_definitions[i][0] > other.label_definitions[i][0]:
                return 1
            if self.label_definitions[i][1] < other.label_definitions[i][1]:
                return -1
            if self.label_definitions[i][1] > other.label_definitions[i][1]:
                return 1

        return 0

    def expand_graph(self, G, external_labels):
        edge_label = max([l for n, l in external_labels.items()]) + 1
        new_G = nx.Graph()
        new_labels = {}
        node_list = list(G.nodes())
        node_list.sort()
        next_node_label = node_list[-1]
        for node in node_list:
            new_G.add_node(node)
            new_labels[node] = external_labels[node]
        for edge in G.edges():
            next_node_label += 1
            new_G.add_node(next_node_label)
            new_G.add_edge(edge[0], next_node_label)
            new_G.add_edge(edge[1], next_node_label)
            new_labels[next_node_label] = edge_label

        return (new_G, new_labels)

    def __eq__(self, other):
        return self.full_comparison(other) == 0

    def __lt__(self, other):
        return self.full_comparison(other) == -1

    def __gt__(self, other):
        return self.full_comparison(other) == 1

    def __le__(self, other):
        return self.full_comparison(other) < 1

    def __ge__(self, other):
        return self.full_comparison(other) > -1

    def set_canonical_form(self):
        ordering = [[n, 0] for n in self.initial_nodes]
        self.further_sort(ordering, self.internal_labels)

        final_node_order = [ordering[0][0]]
        ordering = ordering[1:]
        for i in range(1, len(self.initial_nodes)):
            self.further_sort(ordering, self.nodewise_overlays[final_node_order[-1]])
            final_node_order.append(ordering[0][0])
            if len(ordering) > 1:
                ordering = ordering[1:]

        matrix = []
        for i in range(0, len(final_node_order)):
            next_row = []
            for j in range(i + 1, len(final_node_order)):
                if self.initial_G.has_edge(final_node_order[i], final_node_order[j]):
                    next_row.append(1)
                else:
                    next_row.append(0)
            matrix.append(next_row)

        self.final_node_order = final_node_order
        self.ordered_labels = [self.external_labels[n] for n in final_node_order]
        self.matrix = matrix

    def further_sort(self, initial_list, new_labels):
        for i in range(0, len(initial_list)):
            initial_list[i][1] = (initial_list[i][1], new_labels[initial_list[i][0]])
        initial_list.sort(key=(lambda x: x[1]))
        next_label = 0
        prev_value = initial_list[0][1]
        initial_list[0][1] = next_label
        for i in range(1, len(initial_list)):
            current_value = initial_list[i][1]
            if current_value != prev_value:
                next_label += 1
                prev_value = current_value
            initial_list[i][1] = next_label
