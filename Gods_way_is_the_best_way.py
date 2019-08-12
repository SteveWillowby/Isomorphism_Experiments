import networkx as nx

# This provides a canonical description of a graph with labeled nodes.
class GGraph:

    def __init__(self, G, external_labels=None, was_from_complement=False, nodewise=True, first_layer=False):
        self.complement = was_from_complement
        self.G = G
        self.size = len(self.G.nodes())
        self.nodewise = nodewise
        if self.size == 0:
            return
        if self.size == 1:
            if external_labels is not None:
                if len(external_labels) > 1:
                    print("EXTREME ERROR -- external_labels should just have 1 item.")
                for n, l in external_labels.items():
                    self.single_label = l
            else:
                self.single_label = None
            return

        self.G_comp = nx.complement(self.G) # Do I need this at all?

        self.nodes = list(self.G.nodes())
        if external_labels is not None:
            self.external_labels = external_labels
            self.next_label = 0
            for n, l in external_labels.items():
                if l > self.next_label:
                    self.next_label = l  # next item will ultimately be assigned next_label + 1
        else:
            self.external_labels = {n: 0 for n in self.nodes}
            self.next_label = 0

        self.first_new_label = self.next_label + 1

        self.internal_labels = {n: l for (n, l) in self.external_labels.items()}

        # If there are multiple connected components, just get the subgraphs for those.
        components_generator = nx.connected_components(self.G)
        components = []
        for component in components_generator:
            components.append(component)

        self.num_components = len(components)
        if self.num_components > 1:
            # Subgraphs 
            self.components_list = []
            for component in components:
                node_labels = {n: self.internal_labels[n] for n in component}
                component_graph = GGraph(self.G.subgraph(component), node_labels, was_from_complement=self.complement, nodewise=self.nodewise) # TODO: Be sure I don't need to pass on self.complement
                self.components_list.append(component_graph)
            self.components_list.sort(cmp=self.graph_comparison)
            return

        # Otherwise, there's a single component.

        if self.nodewise:
            self.nodewise_graphs = []
            for node in self.nodes:
                old_label = self.external_labels[node] # Save node's label
                self.external_labels[node] = self.first_new_label # Mark node as special
                if first_layer:
                    print("Diagnostic: Starting node %s of %s" % (node + 1, len(self.nodes)))
                self.nodewise_graphs.append(GGraph(G, external_labels, was_from_complement=self.complement, nodewise=False))
                self.external_labels[node] = old_label # Restore node's label
            self.nodewise_graphs.sort(cmp=self.graph_comparison)
            return

        self.label_counts = {}
        self.label_id_nums = []
        for node, label in self.internal_labels.items():
            if label in self.label_counts:
                self.label_counts[label] += 1
            else:
                self.label_counts[label] = 1
                self.label_id_nums.append(label)
        self.label_id_nums.sort()

        nodes_set = set(self.nodes)
        self.neighborhood_complements = {}
        self.neighborhood_nodes = {}
        self.neighborhood_subgraphs = {}
        for node in self.nodes:
            neighbors_set = set(self.G.neighbors(node))
            if len(neighbors_set) >= len(self.nodes) / 2:
                self.neighborhood_complements[node] = True
                self.neighborhood_nodes[node] = nodes_set - neighbors_set
                self.neighborhood_subgraphs[node] = self.G_comp.subgraph(self.neighborhood_nodes[node])
            else:
                self.neighborhood_complements[node] = False
                self.neighborhood_nodes[node] = neighbors_set
                self.neighborhood_subgraphs[node] = self.G.subgraph(self.neighborhood_nodes[node])

        self.label_graphs = {}

        counter = 0
        while True:
            sorted_neighborhoods = self.get_new_sorted_neighborhoods()
            new_labels = self.assign_new_labels_for_sorted_neighborhoods(sorted_neighborhoods)
            if counter > 0 and self.are_new_labels_effectively_the_same(new_labels): # Make sure to get label data at least once.
                break
            self.internal_labels = new_labels
            counter += 1

    # TODO: Confirm this ordering cannot generate cycles of "greater-ness"
    # -1: This graph "smaller"
    # 0:  Isomorphic
    # 1:  This graph "bigger"
    def graph_comparison(self, graph_a, graph_b):
        # Complement flag
        if (not graph_a.complement) and graph_b.complement:
            return -1
        if graph_a.complement and not graph_b.complement:
            return 1

        # Nodewise flag
        if (not graph_a.nodewise) and graph_b.nodewise:
            return -1
        if graph_a.nodewise and not graph_b.nodewise:
            return 1

        # Number of nodes
        if graph_a.size < graph_b.size:
            return -1
        if graph_a.size > graph_b.size:
            return 1
        if graph_a.size == 0: # Both empty
            return 0
        if graph_a.size == 1: # Both single nodes
            if graph_a.single_label is None and graph_b.single_label is not None:
                return -1
            if graph_a.single_label is not None and graph_b.single_label is None:
                return 1
            if graph_a.single_label < graph_b.single_label:
                return -1
            if graph_a.single_label > graph_b.single_label:
                return 1
            return 0

        # Components
        if graph_a.num_components < graph_b.num_components:
            return -1
        if graph_a.num_components > graph_b.num_components:
            return 1
        # If there are multiple components, that's all we need to compare.
        if graph_a.num_components > 1:
            for i in range(0, graph_a.num_components):
                comp = self.graph_comparison(graph_a.components_list[i], graph_b.components_list[i])
                if comp != 0:
                    return comp
            return 0

        # If nodewise is set, compare the individual graphs.
        if graph_a.nodewise:
            for i in range(0, len(graph_a.nodewise_graphs)):
                comp = self.graph_comparison(graph_a.nodewise_graphs[i], graph_b.nodewise_graphs[i])
                if comp != 0:
                    return comp
            return 0

        # Otherwise...

        # Label id nums
        if graph_a.first_new_label < graph_b.first_new_label:
            return -1
        if graph_a.first_new_label > graph_b.first_new_label:
            return 1
        if len(graph_a.label_id_nums) < len(graph_b.label_id_nums):
            return -1
        if len(graph_a.label_id_nums) > len(graph_b.label_id_nums):
            return 1
        for i in range(0, len(graph_a.label_id_nums)):
            if graph_a.label_id_nums[i] < graph_b.label_id_nums[i]:
                return -1
            if graph_a.label_id_nums[i] > graph_b.label_id_nums[i]:
                return 1

        # Counts of labels
        for label in graph_a.label_id_nums:
            if graph_a.label_counts[label] > graph_b.label_counts[label]:
                return -1
            if graph_a.label_counts[label] < graph_b.label_counts[label]:
                return 1

        # Graphs of (new) labels
        for label in graph_a.label_id_nums:
            if label >= graph_a.first_new_label:
                comp = self.graph_comparison(graph_a.label_graphs[label], graph_b.label_graphs[label])
                if comp != 0:
                    return comp

        return 0

        """ I don't think this part is necessary.
        # Internal-external assignment matches
        pairings_lists = [[(graph_a.internal_labels[i], graph_a.external_labels[i]) for i in range(0, graph_a.size)], \
                          [(graph_b.internal_labels[i], graph_b.external_labels[i]) for i in range(0, graph_b.size)]]
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
        """

    def get_new_sorted_neighborhoods(self):
        labels = []
        for node in self.nodes:
            starting_neighbor_labels = {n: self.internal_labels[n] for n in self.neighborhood_nodes[node]}
            graph = GGraph(self.neighborhood_subgraphs[node], starting_neighbor_labels, was_from_complement=self.neighborhood_complements[node], nodewise=True)
            labels.append((node, graph))
        labels.sort(key=(lambda x: x[1]), cmp=self.graph_comparison) # O(cmp * |V|log|V|) = O(|E|*|V|log|V|)
        return labels

    def assign_new_labels_for_sorted_neighborhoods(self, sorted_neighborhoods):
        new_labels = {}
        single_node_graph = nx.Graph()
        single_node_graph.add_node(0)
        prev = GGraph(single_node_graph, {0: -1})
        for i in range(0, len(sorted_neighborhoods)):
            current = sorted_neighborhoods[i]
            if self.graph_comparison(prev, current[1]) != 0:
                self.next_label += 1
                self.label_counts[self.next_label] = 0
                self.label_graphs[self.next_label] = current[1]
                self.label_id_nums.append(self.next_label)

            new_labels[current[0]] = self.next_label
            self.label_counts[self.next_label] += 1
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
