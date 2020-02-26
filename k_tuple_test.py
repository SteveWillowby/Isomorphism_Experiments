import networkx as nx
import graph_utils
import alg_utils
from weisfeiler_lehman import *

def k_tuple_check(G1, G2, exact_k=None):
    G1P = graph_utils.zero_indexed_graph(G1)
    G2P = graph_utils.zero_indexed_graph(G2)
    G1_nodes = list(G1P.nodes())
    G2_nodes = list(G2P.nodes())
    G1_max = max(G1_nodes) + 1
    G2_max = max(G2_nodes) + 1
    G1P.add_node(G1_max)
    G2P.add_node(G2_max)
    for node in G1_nodes:
        G1P.add_edge(node, G1_max)
    for node in G2_nodes:
        G2P.add_edge(node, G2_max)
    G3 = graph_utils.graph_union(G1P, G2P)
    if exact_k is not None:
        labels = KTupleTest(G3, k=exact_k, mode="Servant").internal_labels
        return labels[G1_max] == labels[G1_max + G2_max + 1]
    for k in range(0, len(G1_nodes) - 1):
        labels = KTupleTest(G3, k=k, mode="Servant").internal_labels
        if labels[G1_max] != labels[G1_max + G2_max + 1]:
            return False
        G1_Canon = KTupleTest(G1P, k=k, mode="Master")
        G2_Canon = KTupleTest(G2P, k=k, mode="Master")
        if G1_Canon == G2_Canon:
            return True
        print("k = %d inconclusive. Moving on to k = %d" % (k, k + 1))
        
    print("No solution found! Algorithm incomplete!")
    return None

class KTupleTest:

    def __init__(self, G, k=2, external_labels=None, mode="Master"):
        self.mode = mode

        if external_labels is None:
            external_labels = [0 for n in G.nodes()]

        # 0-index the nodes AND the external labels
        if self.mode == "Master":
            G, external_labels = graph_utils.zero_indexed_graph_and_coloring_list(G, external_labels)

        self.external_labels = external_labels

        self.G = G
        self.K = k

        self.nodes = list(self.G.nodes())
        self.nodes.sort()

        self.mapping_to_neighbors = [list(self.G.neighbors(n)) for n in self.nodes]

        connected_components = list(comp for comp in nx.connected.connected_components(self.G))
        node_to_component_mapping = {}
        for i in range(0, len(connected_components)):
            for node in connected_components[i]:
                node_to_component_mapping[node] = i

        self.tuples = []
        max_tuple_size = max(1, self.K)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!! TODO: ASSESS WHETHER OR NOT i SHOULD START AT 1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for i in range(max_tuple_size, max_tuple_size + 1):
            tuple_candidates = alg_utils.get_all_k_tuples(len(self.nodes), i)
            for candidate in tuple_candidates:
                comp = node_to_component_mapping[candidate[0]]
                all_same_comp = True
                for j in range(1, len(candidate)):
                    if node_to_component_mapping[candidate[j]] != comp:
                        all_same_comp = False
                        break
                if all_same_comp:
                    self.tuples.append(candidate)

        self.internal_labels = [self.external_labels[n] for n in self.nodes]
        self.tuple_labels = {tup: 0 for tup in self.tuples}
        for node in self.nodes:
            self.tuple_labels[tuple([node])] = self.internal_labels[node]

        WL(self.G, self.internal_labels)
        l2 = len(self.nodes) - 1
        l1 = (len(self.nodes) - 1) - int((len(self.nodes) / 2))
        counter = 0
        while True:
            self.update_tuple_ids()
            new_labels = self.acquire_new_labels()
            if self.are_new_labels_effectively_the_same(new_labels):
                if True or self.mode == "Master":
                    # print("When k = %d, it took a total of %s rounds to converge on labels." % (self.K, counter))
                    # print("There were a total of %d labels" % (len(set([new_labels[n] for n in self.G.nodes()]))))
                    # print(sorted([(new_labels[n], n) for n in self.nodes]))
                    pass
                break
            WL(self.G, new_labels)
            self.internal_labels = new_labels
            counter += 1
            # print(counter)

        if self.mode == "Master":
            self.set_canonical_form()

    def update_tuple_ids(self):
        ids = []
        for i in range(0, len(self.tuples)):
            # print(float(i) / len(self.tuples))
            tup = self.tuples[i]
            if self.K == 0:
                l = (self.tuple_labels[tup], sorted([self.internal_labels[n] for n in self.mapping_to_neighbors[tup[0]]]))
            else:
                old_labels = {n: (self.internal_labels[n], n in tup) for n in self.nodes}
                new_labels = [(0, n) for n in self.nodes]
                alg_utils.further_sort_by(new_labels, old_labels)
                new_labels = {n: l for (l, n) in new_labels}
                comp_result = WL(self.G, new_labels, return_comparable_output=True)
                # label_matching = BeforeAfterLabels(old_labels, new_labels)
                l = (self.tuple_labels[tup], comp_result)
            ids.append((l, tup))
        ids.sort()

        new_tuple_id = len(self.nodes)
        prev_identifier = ids[0][0]
        for (identifier, tup) in ids:
            if identifier != prev_identifier:
                new_tuple_id += 1
                prev_identifier = identifier
            self.tuple_labels[tup] = new_tuple_id

    def acquire_new_labels(self):
        node_ids = {node: [self.internal_labels[node]] for node in self.nodes}
        for tup in self.tuples:
            for node in tup:
                node_ids[node].append(self.tuple_labels[tup])
        for node, l in node_ids.items():
            l.sort()

        node_ids = [(node_ids[node], node) for node in self.nodes]
        node_ids.sort()

        new_labels = {}
        prev_identifier = node_ids[0][0]
        next_numeric_label = 0
        for (identifier, node) in node_ids:
            if identifier != prev_identifier:
                next_numeric_label += 1
                prev_identifier = identifier

            new_labels[node] = next_numeric_label
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

    def full_comparison(self, other, show_diff=False):
        if self.ordered_labels < other.ordered_labels:
            return -1
        if self.ordered_labels > other.ordered_labels:
            return 1
        if self.matrix < other.matrix:
            if show_diff:
                for i in range(0, len(self.matrix)):
                    if self.matrix[i] != other.matrix[i]:
                        print("%d:" % (i+1))
                        print(self.matrix[i])
                        print(other.matrix[i])
                        print("---")
            return -1
        if self.matrix > other.matrix:
            if show_diff:
                for i in range(0, len(self.matrix)):
                    if self.matrix[i] != other.matrix[i]:
                        print("%d:" % (i+1))
                        print(self.matrix[i])
                        print(other.matrix[i])
                        print("---")
            return 1
        return 0

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
        ordering = [(0, n) for n in self.nodes]
        alg_utils.further_sort_by(ordering, self.internal_labels)
        # print("Initial Ordering:")
        # print(ordering)

        final_node_order = [ordering[0][1]]
        ordering = ordering[1:]
        for i in range(1, len(self.nodes)):
            selected_index = 0

            if ordering[0][0] != 0:
                sub_value = ordering[0][0]
                for i in range(0, len(ordering)):
                    ordering[i] = (ordering[i][0] - sub_value, ordering[i][1])
                #print("NOT READY!")
            # FROM HERE[A]....
            #print(ordering)
            # alg_utils.further_sort_by(ordering, {x[1]: x[0] for x in  ordering})
            # self.further_sort(ordering, self.nodewise_overlays[final_node_order[-1]])

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

            if selected_index >= len(ordering) or (selected_index + 1 < len(ordering) and ordering[selected_index][0] == ordering[selected_index + 1][0]):
                new_labels = {n[1]: n[0] + len(final_node_order) for n in ordering}
                for j in range(0, len(final_node_order)):
                    new_labels[final_node_order[j]] = j
                more_refined = KTupleTest(self.G, k=self.K, external_labels=new_labels, mode="Servant")
                alg_utils.further_sort_by(ordering, more_refined.internal_labels)
                selected_index = 0

            final_node_order.append(ordering[selected_index][1])
            # print("Partitioning was: %s" % str(ordering))
            # print("Selected node %d, at which point there were %d labels" % (final_node_order[-1], len(final_node_order) + ordering[-1][0]))

            if len(ordering) > 1:
                #if selected_index + 1 < len(ordering) and ordering[selected_index][0] == ordering[selected_index + 1][0]:
                    #print("Chose the %dth node with a tie (1-indexed)." % (i+1))
                ordering.pop(selected_index)
        # print("The very final node ordering is:")
        # print(final_node_order)

        matrix = []
        for i in range(0, len(final_node_order)):
            next_row = []
            for j in range(i + 1, len(final_node_order)):
                if self.G.has_edge(final_node_order[i], final_node_order[j]):
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
