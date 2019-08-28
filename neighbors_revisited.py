import networkx as nx

class NeighborsRevisited:

    def __init__(self, G, external_labels=None, nodewise=True):
        self.nodewise = nodewise
        if external_labels is None:
            external_labels = {n: 0 for n in G.nodes()}

        if self.nodewise:
            (self.G, external_labels) = self.expand_graph(G, external_labels)
            print(len(self.G.nodes()))
        else:
            self.G = G
        self.nodes = list(self.G.nodes())
        self.mapping_to_neighbors = {n: set(self.G.neighbors(n)) for n in self.nodes}
        self.internal_labels = {n: external_labels[n] for n in self.nodes}
        self.next_numeric_label = max([l for n, l in external_labels.items()])
        self.label_definitions = []
        self.label_counts = []

        counter = 0
        while True:
            sorted_ids = self.get_new_ids_in_order()
            new_labels = self.assign_new_labels_for_sorted_ids(sorted_ids)
            if self.are_new_labels_effectively_the_same(new_labels):
                break
            self.internal_labels = new_labels
            if self.nodewise:
                print(counter)
            counter += 1

    def get_new_ids_in_order(self):
        ids = []
        if self.nodewise:
            self.next_numeric_label += 1
        for node in self.nodes:
            if self.nodewise:
                new_labels = {n: l for n, l in self.internal_labels.items()}
                new_labels[node] = self.next_numeric_label
                i = (self.internal_labels[node], NeighborsRevisited(self.G, external_labels=new_labels, nodewise=False)) # Referencing oneself appears to be necessary!
            else:
                neighbors = [self.internal_labels[n] for n in self.mapping_to_neighbors[node]]
                neighbors.sort()
                i = (self.internal_labels[node], neighbors) # TODO: Consider whether referencing oneself is necessary.
            ids.append((node, i))
        ids.sort(key=(lambda x: x[1]))
        return ids

    def assign_new_labels_for_sorted_ids(self, sorted_ids):
        new_labels = {}
        prev = None
        for current in sorted_ids:
            if prev is None or prev != current[1]:
                self.next_numeric_label += 1
                self.label_definitions.append((self.next_numeric_label, current[1]))
                self.label_counts.append(0)

            new_labels[current[0]] = self.next_numeric_label
            self.label_counts[-1] += 1
            prev = current[1]
        return new_labels

    # O(|V|)
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

        if len(self.nodes) < len(other.nodes): # TODO: Make sure this isn't strictly needed and is just here for speed.
            return -1
        if len(self.nodes) > len(other.nodes):
            return 1

        if len(self.label_definitions) < len(other.label_definitions):
            return -1
        if len(self.label_definitions) > len(other.label_definitions):
            return 1

        if self.label_counts < other.label_counts:
            return -1
        if self.label_counts > other.label_counts:
            return 1

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
