import networkx as nx
from networkx.algorithms.shortest_paths import *
import copy

# This provides a canonical description of a graph with labeled nodes.
class FasterGGraph:

    # Nodewise may be zero, one or two.
    # 0 --> don't collect automorphisms in THIS LAYER
    # 1 --> collect automorphisms and form canonical representation
    # 2 --> don't collect automorphisms in ANY LAYER
    def __init__(self, G, external_labels=None, was_from_complement=False, nodewise=1, first_layer=False, results_dict=None, layer=0):
        print(layer)
        self.layer = layer
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

        self.results_dict = results_dict
        if self.results_dict is None:
            self.results_dict = {}

        self.G_comp = nx.complement(self.G) # Do I need this at all?

        self.nodes = list(self.G.nodes())
        self.nodes.sort()

        if external_labels is not None:
            self.external_labels = external_labels

            # Unfortunately, the ORDERING of the original labels matters, so we must preserve this ordering unless we do something fancier.
            # Regularize the external labels to be in increasing order starting with the NODE with the smallest id.
            # This is useful for hashing whether this graph problem has effectively been computed before.
            # node_identifiers_for_labels = {}
            #for node in self.nodes:
            #    if self.external_labels[node] not in node_identifiers_for_labels:
            #        node_identifiers_for_labels[self.external_labels[node]] = node
            #    elif node < node_identifiers_for_labels[self.external_labels[node]]:
            #        node_identifiers_for_labels[self.external_labels[node]] = node

            # external_labels_list = [[l, n] for l, n in node_identifiers_for_labels.items()]
            external_labels_list = list(set([(l, l) for n, l in self.external_labels.items()]))
            external_labels_list.sort(key=(lambda x: x[1]))

            external_labels_replacements = {}
            for i in range(0, len(external_labels_list)):
                external_labels_replacements[external_labels_list[i][0]] = i

            self.regularized_external_labels = {n: external_labels_replacements[l] for n, l in self.external_labels.items()}
            self.next_label = len(external_labels_list) - 1

        else:
            self.external_labels = {n: 0 for n in self.nodes}
            self.regularized_external_labels = {n: 0 for n in self.nodes}
            self.next_label = 0

        # Before going any further, check to see if this graph has already been computed.
        hash_key = self.computation_id()
        if hash_key in self.results_dict:
            # print("HIT!!!!!!!!!!!!!!")
            self.__dict__ = self.results_dict[hash_key].__dict__
            self.external_labels = external_labels
            return
        # print("Miss out of %s" % (len(self.results_dict)))

        # The graph has not been computed before.

        self.first_new_label = self.next_label + 1

        self.internal_labels = {n: l for (n, l) in self.regularized_external_labels.items()}

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
                component_graph = FasterGGraph(self.G.subgraph(component), node_labels, was_from_complement=self.complement, \
                    nodewise=self.nodewise, results_dict=self.results_dict, layer=self.layer+1) # TODO: Be sure I don't need to pass on self.complement
                self.components_list.append(component_graph)
            self.components_list.sort(cmp=self.graph_comparison)

            self.results_dict[hash_key] = copy.copy(self)
            return

        # Otherwise, there's a single component.

        if self.nodewise == 1:

            # First, get some starting breakdown:
            if first_layer:
                print("Diagnostic: Starting quick prep.")
                print(self.regularized_external_labels)
            initial_labels = FasterGGraph(self.G, self.regularized_external_labels, was_from_complement=self.complement, \
                nodewise=2, results_dict=self.results_dict, layer=self.layer+1)
            initial_labels = initial_labels.internal_labels
            if first_layer:
                print("Diagnostic: Starting full prep.")
                exit()
            initial_labels = FasterGGraph(self.G, initial_labels, was_from_complement=self.complement, nodewise=0, \
                results_dict=self.results_dict, layer=self.layer+1)
            initial_labels = initial_labels.internal_labels

            nodewise_graphs = {}
            for node in self.nodes:
                if first_layer:
                    print("Diagnostic: Starting node %s of %s" % (node + 1, len(self.nodes)))

                path_labels = nx.single_source_shortest_path_length(G, node) # Do shortest paths computation to speed things up.
                combined_labels = [[n, (path_labels[n], initial_labels[n])] for n in self.nodes] # NOTE: Having path before initial is necessary.
                combined_labels.sort(key=(lambda x: x[1]))

                combined_label = 0
                node_centric_labels = {combined_labels[0][0]: combined_label}
                for i in range(1, len(self.nodes)):
                    if combined_labels[i - 1][1] != combined_labels[i][1]:
                        combined_label += 1
                    node_centric_labels[combined_labels[i][0]] = combined_label
                nodewise_graphs[node] = FasterGGraph(G, node_centric_labels, was_from_complement=self.complement, nodewise=0, \
                    results_dict=self.results_dict, layer=self.layer+1)
                for n, l in node_centric_labels.items():
                    if n != node and l == node_centric_labels[node]:
                        print("MAJOR ERROR!")

            sorted_nodewise_graphs = [[node, nodewise_graphs[node]] for node in self.nodes]
            sorted_nodewise_graphs.sort(key=(lambda x: x[1]), cmp=self.graph_comparison)

            self.automorphism_orbits = [[sorted_nodewise_graphs[0][1], 1]]

            for i in range(1, len(self.nodes)):
                if self.graph_comparison(sorted_nodewise_graphs[i - 1][1], sorted_nodewise_graphs[i][1]) == 0:
                    self.automorphism_orbits[-1][1] += 1
                else:
                    self.automorphism_orbits.append([sorted_nodewise_graphs[i][1], 1])
            if first_layer:
                print("Number of orbits: %s. Occurrences of first orbit: %s." % (len(self.automorphism_orbits), self.automorphism_orbits[0][1]))

            self.results_dict[hash_key] = copy.copy(self)
            return

            #TODO: Try to make sense of this canonical stuff which I'm not using right now.

            # Sort nodes by automorphism group and external labels

            if first_layer:
                print(self.regularized_external_labels)

            automorphism_group = 0
            node_ordering = [[sorted_nodewise_graphs[0][0], (automorphism_group, self.regularized_external_labels[sorted_nodewise_graphs[0][0]])]]
            for i in range(1, len(self.nodes)):
                if self.graph_comparison(sorted_nodewise_graphs[i - 1][1], sorted_nodewise_graphs[i][1]) != 0:
                    automorphism_group += 1
                node_ordering.append([sorted_nodewise_graphs[i][0], (automorphism_group, self.regularized_external_labels[sorted_nodewise_graphs[i][0]])])
            node_ordering.sort(key=(lambda x: x[1]))

            self.automorphism_and_group_labels = {node_ordering[i][0]: node_ordering[i][1][0] for i in range(0, len(self.nodes))}

            if first_layer:
                print(self.automorphism_and_group_labels)

            final_node_order = [node_ordering[0][0]]
            for selection_round in range(1, len(self.nodes)):
                node_ordering = node_ordering[1:] # Remove first element
                group_label = 0
                new_node_ordering = [[node_ordering[0][0], (group_label, nodewise_graphs[final_node_order[-1]].internal_labels[node_ordering[0][0]])]]
                for i in range(1, len(node_ordering)):
                    if node_ordering[i - 1][1] != node_ordering[i][1]:
                        group_label += 1
                    new_node_ordering.append([node_ordering[i][0], (group_label, nodewise_graphs[final_node_order[-1]].internal_labels[node_ordering[i][0]])])
                new_node_ordering.sort(key=(lambda x: x[1]))
                node_ordering = new_node_ordering
                final_node_order.append(node_ordering[0][0])

            # Now that we have an automorphism and label determined node ordering, we can create a canonical adjacency matrix.
            self.canonical_matrix = []
            for i in range(0, len(self.nodes)):
                first_node = final_node_order[i]
                self.canonical_matrix.append([])
                for j in range(i + 1, len(self.nodes)):
                    second_node = final_node_order[j]
                    if G.has_edge(first_node, second_node):
                        self.canonical_matrix[-1].append(1)
                    else:
                        self.canonical_matrix[-1].append(0)

            # Also using this ordering, we provide automorphism and external node labels
            self.automorphism_and_group_labels = [self.automorphism_and_group_labels[final_node_order[i]] for i in range(0, len(self.nodes))]

            self.results_dict[hash_key] = copy.copy(self)
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

        current_labels = set([l for n, l in self.internal_labels.items()])
        old_labels = set(self.label_id_nums) - current_labels
        for label in old_labels:
            if label in self.label_graphs:
                del self.label_graphs[label]
            del self.label_counts[label]
        self.label_id_nums = list(current_labels)
        self.label_id_nums.sort()

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
        if graph_a.nodewise < graph_b.nodewise:
            return -1
        if graph_a.nodewise > graph_b.nodewise:
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
        if graph_a.nodewise == 1:
            if len(graph_a.automorphism_orbits) < len(graph_b.automorphism_orbits): # Total number of nodes distinct by automorphism
                return -1
            if len(graph_a.automorphism_orbits) > len(graph_b.automorphism_orbits):
                return 1
            for i in range(0, len(graph_a.automorphism_orbits)):
                if graph_a.automorphism_orbits[i][1] < graph_b.automorphism_orbits[i][1]: # Counts of distinct node types by automorphism
                    return -1
                if graph_a.automorphism_orbits[i][1] > graph_b.automorphism_orbits[i][1]:
                    return 1
            for i in range(0, len(graph_a.automorphism_orbits)): # Actual definitions of the types of automorphisms
                comp = self.graph_comparison(graph_a.automorphism_orbits[i][0], graph_b.automorphism_orbits[i][0])
                if comp != 0:
                    return comp
            return 0

            # TODO: Make sense of why the following doesn't work.

            if graph_a.automorphism_and_group_labels < graph_b.automorphism_and_group_labels:
                return -1
            if graph_a.automorphism_and_group_labels > graph_b.automorphism_and_group_labels: # TODO: Can make faster by combining these two.
                return 1

            if graph_a.canonical_matrix < graph_b.canonical_matrix:
                return -1
            if graph_a.canonical_matrix > graph_b.canonical_matrix: # TODO: Can make faster by combining these two.
                return 1

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

        # return 0

        # Internal-external assignment matches
        pairings_lists = [[(graph_a.internal_labels[graph_a.nodes[i]], graph_a.external_labels[graph_a.nodes[i]]) for i in range(0, graph_a.size)], \
                          [(graph_b.internal_labels[graph_b.nodes[i]], graph_b.external_labels[graph_b.nodes[i]]) for i in range(0, graph_b.size)]]
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
        return 0

    def get_new_sorted_neighborhoods(self):
        labels = []
        use_nodewise = 1
        if self.nodewise == 2:
            use_nodewise = 2
        for node in self.nodes:
            starting_neighbor_labels = {n: self.internal_labels[n] for n in self.neighborhood_nodes[node]}
            graph = FasterGGraph(self.neighborhood_subgraphs[node], starting_neighbor_labels, was_from_complement=self.neighborhood_complements[node], \
                nodewise=use_nodewise, results_dict=self.results_dict,layer=self.layer+1)
            labels.append((node, graph))
        labels.sort(key=(lambda x: x[1]), cmp=self.graph_comparison) # O(cmp * |V|log|V|) = O(|E|*|V|log|V|)
        return labels

    def assign_new_labels_for_sorted_neighborhoods(self, sorted_neighborhoods):
        new_labels = {}
        single_node_graph = nx.Graph()
        single_node_graph.add_node(0)
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

    def computation_id(self):
        regularized_external_labels_list_sorted = [self.regularized_external_labels[n] for n in self.nodes]
        return (self.complement, self.nodewise, tuple(self.nodes), tuple(regularized_external_labels_list_sorted))
