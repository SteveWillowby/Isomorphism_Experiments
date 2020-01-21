import networkx as nx
import graph_utils
import alg_utils
from weisfeiler_lehman import *

def k_tuple_check(G1, G2, k):
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
    labels = KTupleTest(G3, k=k, mode="Servant").internal_labels
    return labels[G1_max] == labels[G1_max + G2_max + 1]

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
        for i in range(1, max_tuple_size + 1):
            tuple_candidates = alg_utils.get_all_k_tuples(len(self.nodes), i)
            for candidate in tuple_candidates:
                # induced = graph_utils.induced_subgraph(self.G, candidate)
                # print("These nodes: %s induced these nodes and edges: %s, %s" % (str(candidate), str(list(induced.nodes())), str(list(induced.edges()))))
                comp = node_to_component_mapping[candidate[0]]
                all_same_comp = True
                for j in range(1, len(candidate)):
                    if node_to_component_mapping[candidate[j]] != comp:
                        all_same_comp = False
                        break
                # if True or nx.connected.is_connected(graph_utils.induced_subgraph(self.G, candidate)):
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
                    print("Took a total of %s rounds to first get the correct labels." % (counter))
                    print("There were a total of %d labels" % (len(set([new_labels[n] for n in self.G.nodes()]))))
                    # print(sorted([(new_labels[n], n) for n in self.nodes]))
                break
            WL(self.G, new_labels)
            self.internal_labels = new_labels
            counter += 1
            print(counter)

        if self.mode == "Master":
            self.set_canonical_form()

    def update_tuple_ids(self):
        # print("A")
        ids = []
        for i in range(0, len(self.tuples)):
            # print(float(i) / len(self.tuples))
            tup = self.tuples[i]
            if self.K == 0:
                i = (self.tuple_labels[tup], sorted([self.internal_labels[n] for n in self.mapping_to_neighbors[tup[0]]]))
            else:
                new_labels = [(self.internal_labels[n], n) for n in self.nodes]
                alg_utils.further_sort_by(new_labels, {n: n in tup for n in self.nodes})
                new_labels = {n: l for (l, n) in new_labels}
                i = (self.tuple_labels[tup], WL(self.G, new_labels, return_comparable_output=True)) 
            ids.append((i, tup))
        ids.sort()

        new_numeric_id = 0
        prev_identifier = ids[0][0]
        for (identifier, tup) in ids:
            if identifier != prev_identifier:
                new_numeric_id += 1
                prev_identifier = identifier
            self.tuple_labels[tup] = new_numeric_id

    def acquire_new_labels(self):
        # print("B")
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

    def full_comparison(self, other):
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
                for i in range(0, len(ordering)):
                    ordering[i] = (ordering[i][0] - ordering[0][0], ordering[i][1])
                #print("NOT READY!")
            # FROM HERE[A]....
            #print(ordering)
            # alg_utils.further_sort_by(ordering, {x[0]: x[1] for x in  ordering})
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
                if self.G.has_edge(final_node_order[i], final_node_order[j]):
                    next_row.append(1)
                else:
                    next_row.append(0)
            matrix.append(next_row)

        self.final_node_order = final_node_order
        self.ordered_labels = [self.external_labels[n] for n in final_node_order]
        self.matrix = matrix
