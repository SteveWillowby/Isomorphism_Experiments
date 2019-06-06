import networkx as nx

class CanonicalDescription:

    def __init__(self, G):
        self.sets_to_labels = {}
        self.label_counts = {}
        self.next_label = 0
        self.nodes = list(G.nodes)
        self.mapping_to_labels = {n: 0 for n in self.nodes}
        self.mapping_to_neighbors = {n: set(G.neighbors(n)) for n in self.nodes}

        while True:
            sorted_multisets = self.get_new_id_multisets_in_order()
            new_labels = self.assign_new_labels_for_sorted_multisets(sorted_multisets)
            if self.are_new_labels_effectively_the_same(new_labels):
                break
            self.mapping_to_labels = new_labels

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
