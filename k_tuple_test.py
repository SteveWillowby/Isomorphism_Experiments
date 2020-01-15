import networkx as nx
import graph_utils
import alg_utils
from weisfeiler_lehman import *

class KTupleTest:

    def __init__(self, G, external_labels=None, k=2, nodewise="Master"):
        self.nodewise = nodewise

        if external_labels is None:
            external_labels = [0 for n in G.nodes()]

        # 0-index the nodes AND the external labels
        if self.nodewise == "Master":
            G, external_labels = graph_utils.zero_indexed_graph_and_coloring_list(G, external_labels)

        self.G = G
        self.nodes = list(self.G.nodes())
        self.nodes.sort()
        self.mapping_to_neighbors = [list(self.G.neighbors(n)) for n in self.nodes]
        self.K = k
        self.tuples = []
        #self.susbsets_of_tuples = [[]]
        self.internal_labels = {}

        for i in range(1, self.K + 1):
            tuple_candidates = alg_utils.get_all_k_tuples(len(self.nodes), i)
            self.tuples.append([])
            #if i > 1:
            #    self.subsets_of_tuples.append(alg_utils.get_all_k_tuples(i, i - 1)
            for candidate in tuple_candidates:
                if True or nx.connected.is_connected(graph_utils.induced_subgraph(self.G, candidate)):
                    self.tuples[-1].append(candidate)
                    self.internal_labels[candidate] = 0

        counter = 0
        while True:
            sorted_ids = self.get_new_ids_in_order()
            new_labels = self.assign_new_labels_for_sorted_ids(sorted_ids)
            if self.are_new_labels_effectively_the_same(new_labels):
                if self.nodewise == "Master":
                    print("Took a total of %s rounds to first get the correct labels." % (counter))
                    print("There were a total of %d labels" % (len(set([new_labels[n] for n in self.initial_G.nodes()]))))
                break
            WL(self.G, new_labels)
            self.internal_labels = new_labels
            counter += 1

        if self.nodewise == "Master":
            self.set_canonical_form()
        else:
            self.label_pairings = [(self.internal_labels[n], external_labels[n]) for n in self.nodes]
            self.label_pairings.sort()

    def get_new_ids_in_order(self):
        ids = []
        for node in self.nodes:
            if self.nodewise:
                #new_labels = [self.nodewise_overlays[node][n] * self.higher_than_any_internal_label + self.internal_labels[n] for n in self.nodes]
                #i = (self.internal_labels[node], FasterNeighborsRevisited(self.G, external_labels=new_labels, nodewise=False)) # Referencing oneself appears to be necessary!
                new_overlay = [(self.nodewise_overlays[node][n], n) for n in self.nodes]
                alg_utils.further_sort_by(new_overlay, self.internal_labels)
                for (c, n) in new_overlay:
                    self.nodewise_overlays[node][n] = c
                i = (self.internal_labels[node], WL(self.G, self.nodewise_overlays[node], return_comparable_output=True)) 
                # self.nodewise_overlays[node] = i[1].internal_labels
            ids.append((node, i))
        ids.sort(key=(lambda x: x[1]))
        return ids

    # O(|V'| * cmp_for_sub_labels)
    def assign_new_labels_for_sorted_ids(self, sorted_ids):
        new_labels = {}
        prev = None
        next_numeric_label = -1
        self.label_definitions.append([])
        for current in sorted_ids:
            if prev is None or prev != current[1]:
                next_numeric_label += 1
                self.label_definitions[-1].append(current[1])

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
                for i in range(0, len(self.matrix)):
                    if self.matrix[i] != other.matrix[i]:
                        print("%d:" % (i+1))
                        print(self.matrix[i])
                        print(other.matrix[i])
                        print("---")
                return -1
            if self.matrix > other.matrix:
                for i in range(0, len(self.matrix)):
                    if self.matrix[i] != other.matrix[i]:
                        print("%d:" % (i+1))
                        print(self.matrix[i])
                        print(other.matrix[i])
                        print("---")
                return 1
            return 0

        # Internal-external label matching.
        if self.label_pairings < other.label_pairings:
            return -1
        if self.label_pairings > other.label_pairings:
            return 1

        if len(self.label_definitions) < len(other.label_definitions):
            return -1
        if len(self.label_definitions) > len(other.label_definitions):
            return 1

        # Actual label definitions.
        for r in range(0, len(self.label_definitions)):
            if len(self.label_definitions[r]) < len(other.label_definitions[r]):
                return -1
            if len(self.label_definitions[r]) > len(other.label_definitions[r]):
                return 1
            for i in range(0, len(self.label_definitions[r])):
                if self.label_definitions[r][i][0] < other.label_definitions[r][i][0]:
                    return -1
                if self.label_definitions[r][i][0] > other.label_definitions[r][i][0]:
                    return 1
                if self.label_definitions[r][i][1] < other.label_definitions[r][i][1]:
                    return -1
                if self.label_definitions[r][i][1] > other.label_definitions[r][i][1]:
                    return 1

        return 0

    def expand_graph(self, G, external_labels):
        edge_label = max(external_labels) + 1
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

    def fancy_expand_graph(self, G, external_labels):
        expanded_edge_label = max([l for n, l in external_labels.items()]) + 1
        new_G = nx.Graph()
        new_labels = {}
        node_list = list(G.nodes())
        node_list.sort()
        next_node_label = node_list[-1]
        for node in node_list:
            new_G.add_node(node)
            new_labels[node] = external_labels[node]
        a_triangle = False
        a_non_triangle = False
        for edge in G.edges():
            triangle = False
            for n1 in G.neighbors(edge[0]):
                if n1 in G.neighbors(edge[1]):
                    triangle = True
                    break
            if triangle:
                a_triangle = True
                next_node_label += 1
                new_G.add_node(next_node_label)
                new_G.add_edge(edge[0], next_node_label)
                new_G.add_edge(edge[1], next_node_label)
                new_labels[next_node_label] = expanded_edge_label
            else:
                a_non_triangle = True
                new_G.add_edge(edge[0], edge[1])
        if self.nodewise:
            if a_non_triangle:
                if a_triangle:
                    print("Both kinds!")
                else:
                    print("All non-triangles")
            else:
                print("All triangles")

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
        l = sorted([self.internal_labels[n] for n in self.initial_G.nodes()])
        print(len(l))
        prev_i = 0
        for i in range(1, len(l)):
            if l[i] != l[i-1]:
                print(i - prev_i)
                prev_i = i
        print(len(l) - prev_i)
        ordering = [[n, 0] for n in self.initial_nodes]
        self.further_sort(ordering, self.internal_labels)
        # print("Initial Ordering:")
        # print(ordering)

        final_node_order = [ordering[0][0]]
        ordering = ordering[1:]
        for i in range(1, len(self.initial_nodes)):
            selected_index = 0

            # FROM HERE[A]....
            if ordering[0][1] != 0:
                for x in ordering:
                    x[1] -= ordering[0][1]
                #print("NOT READY!")
            #print(ordering)
            # alg_utils.further_sort_by(ordering, {x[0]: x[1] for x in  ordering})
            self.further_sort(ordering, self.nodewise_overlays[final_node_order[-1]])

            while selected_index < len(ordering):
                if selected_index + 1 < len(ordering):
                    if ordering[selected_index][1] == ordering[selected_index + 1][1]:
                        next_idx = selected_index + 1
                        while next_idx < len(ordering) and ordering[selected_index][1] == ordering[next_idx][1]:
                            next_idx += 1
                        selected_index = next_idx
                    else:
                        break
                else:
                    break
            # print("Selected index is %d" % selected_index)
            # ....TO HERE[A] is all code to make things faster. It's not strictly necessary.

            #if len(ordering) > 1 and ordering[0][1] == ordering[1][1]:
            if selected_index >= len(ordering) or (selected_index + 1 < len(ordering) and ordering[selected_index][1] == ordering[selected_index + 1][1]):
                new_labels = {n[0]: n[1] + len(final_node_order) for n in ordering}
                for j in range(0, len(final_node_order)):
                    new_labels[final_node_order[j]] = j
                new_labels = [new_labels[i] for i in range(0, len(new_labels))]
                more_refined = FasterNeighborsRevisited(self.initial_G, external_labels=new_labels, nodewise="Servant")
                self.further_sort(ordering, more_refined.internal_labels)
                selected_index = 0

            final_node_order.append(ordering[selected_index][0])
            if len(ordering) > 1:
                if selected_index + 1 < len(ordering) and ordering[selected_index][1] == ordering[selected_index + 1][1]:
                    print("Chose the %dth node with a tie (1-indexed)." % (i+1))
                ordering.pop(selected_index)
        # print("The very final node ordering is:")
        # print(final_node_order)

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
