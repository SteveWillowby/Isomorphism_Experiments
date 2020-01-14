import networkx as nx
import graph_utils
import alg_utils
from neighbors_revisited import *
from weisfeiler_lehman import *

class FasterNeighborsRevisited:

    def __init__(self, G, external_labels=None, mode="Master"):
        self.mode = mode

        if external_labels is None:
            external_labels = [0 for n in G.nodes()]

        # 0-index the nodes AND the external labels
        if self.mode == "Master":
            G, external_labels = graph_utils.zero_indexed_graph_and_coloring_list(G, external_labels)

        if self.mode:
            self.initial_nodes = list(G.nodes())
            self.initial_G = G
            (self.G, external_labels) = self.expand_graph(G, external_labels)
        else:
            self.G = G

        self.nodes = list(self.G.nodes())
        self.nodes.sort()
        self.mapping_to_neighbors = [set(self.G.neighbors(n)) for n in self.nodes]
        self.internal_labels = [external_labels[n] for n in self.nodes]
        self.external_labels = [external_labels[n] for n in self.nodes]

        # Give things a head-start.
        if self.mode:
            overall_WL_coloring = list(self.internal_labels)
            WL(self.G, overall_WL_coloring)
            self.nodewise_overlays = []
            for node in self.nodes:
                self.nodewise_overlays.append(list(overall_WL_coloring))
        #        WL(self.G, self.nodewise_overlays[node], init_active_set=set([node]))

        self.counter = 0
        while True:
            sorted_ids = self.get_new_ids_in_order()
            new_labels = self.assign_new_labels_for_sorted_ids(sorted_ids)
            if self.are_new_labels_effectively_the_same(new_labels):
                if self.mode == "Master":
                    print("Took a total of %s rounds to first get the correct labels." % (self.counter))
                    print("There were a total of %d labels" % (len(set([new_labels[n] for n in self.initial_G.nodes()]))))
                break
            # WL(self.G, new_labels)
            self.internal_labels = new_labels
            self.counter += 1

        if self.mode == "Master":
            self.set_canonical_form()

    def get_new_ids_in_order(self):
        ids = []
        for node in self.nodes:
            #new_labels = [self.nodewise_overlays[node][n] * self.higher_than_any_internal_label + self.internal_labels[n] for n in self.nodes]
            #i = (self.internal_labels[node], FasterNeighborsRevisited(self.G, external_labels=new_labels, nodewise=False)) # Referencing oneself appears to be necessary!

            original_labels = {i: (self.internal_labels[i], self.nodewise_overlays[node][i]) for i in self.nodes}

            new_overlay = [(self.nodewise_overlays[node][n], n) for n in self.nodes]
            if self.counter >= 80:
                new_overlay = [(0, n) for n in self.nodes]
            max_c = max([c for (c, n) in new_overlay])
            for i in range(0, len(new_overlay)):
                if new_overlay[i][1] == node:
                    new_overlay[i] = (max_c + 1, node)

            alg_utils.further_sort_by(new_overlay, self.internal_labels)
            for (c, n) in new_overlay:
                self.nodewise_overlays[node][n] = c

            """
            # This segment gives the node a unique color if it does not already have one.
            l = self.nodewise_overlays[node]
            node_is_singleton = False
            for i in range(0, len(l)):
                if i != node and l[i] == l[node]:
                    use_max = True
                    break
            if use_max:
                l[node] = max(l) + 1
            else:
                max_l
            """

            exp = NeighborsRevisited(self.G, {i: self.nodewise_overlays[node][i] for i in self.nodes}, nodewise=False)

            new_labels = exp.internal_labels
            # comp_output = WL(self.G, self.nodewise_overlays[node], return_comparable_output=True)

            before_after_comparison = BeforeAfterLabels(original_labels, self.nodewise_overlays[node])

            # i = (self.internal_labels[node], comp_output)
            i = (self.internal_labels[node], exp, before_after_comparison)
            self.nodewise_overlays[node] = exp.internal_labels
            """
            for j in self.nodes:
                for k in range(0, len(self.nodes)):
                    exp_v = exp.internal_labels[j] == exp.internal_labels[k]
                    i_v = self.nodewise_overlays[node][j] == self.nodewise_overlays[node][k]
                    if exp_v != i_v:
                        print("LERLKJELKRJEKLRJE")
            """
            # self.nodewise_overlays[node] = i[1].internal_labels
            ids.append((node, i))
        ids.sort(key=(lambda x: x[1]))
        return ids

    # O(|V'| * cmp_for_sub_labels)
    def assign_new_labels_for_sorted_ids(self, sorted_ids):
        new_labels = {}
        prev = None
        next_numeric_label = -1
        # self.label_definitions.append([])

        prev_idx_start = 0
        idx = -1
        if self.mode == "Master":
            print("Partition sizes")
        nodes = [sorted_ids[0][0]]

        for current in sorted_ids:
            idx += 1
            if prev is None or prev != current[1]:
                next_numeric_label += 1
                # self.label_definitions[-1].append(current[1])

                if self.mode == "Master" and idx > 0:
                    # print(idx - prev_idx_start)
                    if idx - prev_idx_start == 16:
                        print(nodes)
                    prev_idx_start = idx
                    nodes = []
            nodes.append(current[0])

            new_labels[current[0]] = next_numeric_label
            prev = current[1]
        if self.mode == "Master":
            print(idx + 1 - prev_idx_start)
            print("Num of labels found: %d" % (next_numeric_label + 1))
        return [new_labels[i] for i in range(0, len(self.nodes))]

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
        if self.mode != other.mode:
            print("This comparison should never occur! -- Different mode values!")
            exit()

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
        """
        l = sorted([self.internal_labels[n] for n in self.initial_G.nodes()])
        print(len(l))
        prev_i = 0
        for i in range(1, len(l)):
            if l[i] != l[i-1]:
                print(i - prev_i)
                prev_i = i
        print(len(l) - prev_i)
        """
        ordering = [[0, n] for n in self.initial_nodes]
        alg_utils.further_sort_by(ordering, self.internal_labels)

        final_node_order = [ordering[0][1]]
        ordering = ordering[1:]
        for i in range(1, len(self.initial_nodes)):
            selected_index = 0

            # FROM HERE[A]....
            alg_utils.further_sort_by(ordering, self.nodewise_overlays[final_node_order[-1]])

            while selected_index < len(ordering):
                if selected_index + 1 < len(ordering):
                    if ordering[selected_index][0] == ordering[selected_index + 1][0]:
                        next_idx = selected_index + 1
                        while next_idx < len(ordering) and ordering[selected_index][0] == ordering[next_idx][0]:
                            next_idx += 1
                        selected_index = next_idx
                    else:
                        break
                else:
                    break
            # ....TO HERE[A] is all code to make things faster. It's not strictly necessary.

            #if len(ordering) > 1 and ordering[0][1] == ordering[1][1]:
            if selected_index >= len(ordering) or (selected_index + 1 < len(ordering) and ordering[selected_index][0] == ordering[selected_index + 1][0]):
                new_labels = {n[1]: n[0] + len(final_node_order) for n in ordering}
                for j in range(0, len(final_node_order)):
                    new_labels[final_node_order[j]] = j
                new_labels = [new_labels[i] for i in range(0, len(new_labels))]
                more_refined = FasterNeighborsRevisited(self.initial_G, external_labels=new_labels, mode="Servant")
                alg_utils.further_sort_by(ordering, more_refined.internal_labels)
                selected_index = 0

            final_node_order.append(ordering[selected_index][1])
            if len(ordering) > 1:
                if selected_index + 1 < len(ordering) and ordering[selected_index][0] == ordering[selected_index + 1][0]:
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

class BeforeAfterLabels:

    def __init__(self, before, after):
        before_to_after = {}
        for item, before_label in before.items():
            after_label = after[item]
            key = (before_label, after_label)
            if key not in before_to_after:
                before_to_after[key] = 0
            before_to_after[key] += 1
        before_to_after = [(label, count) for label, count in before_to_after.items()]
        before_to_after.sort()
        self.value = before_to_after

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        return self.value < other.value

    def __gt__(self, other):
        return self.value > other.value

    def __le__(self, other):
        return self.value <= other.value

    def __ge__(self, other):
        return self.value >= other.value
