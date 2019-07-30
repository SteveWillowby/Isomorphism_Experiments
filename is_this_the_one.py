import networkx as nx

class NodeViaPaths:

    def __init__(self, G, node):
        self.G = G
        self.layer_nodes = [set([node])]
        self.layer_node_to_class = [{node: 0}]
        self.layer_next_class = [1]
        self.num_layers = 1

        self.found_nodes = set([node])
        self.neighbor_sets = {node: set(self.G.neighbors(node))}

        while not self.done():
            self.add_layer()
            change = True
            while change:
                change = False
                l = self.num_layers - 2
                while l > 0:
                    if not self.sort_layer_from_direction(l, 1): # Above
                        break
                    change = True
                    l -= 1
                l += 1
                while l < self.num_layers:
                    if self.sort_layer_from_direction(l, -1): # Below
                        change = True
                    l += 1

    def new_neighbors(self):
        new_node_set = set()
        last_layer = self.layer_nodes[-1]
        for node in last_layer:
            for n in self.neighbor_sets[node]:
                new_node_set.add(n)
                if n not in self.found_nodes:
                    self.neighbor_sets[n] = set(self.G.neighbors(n))
                    self.found_nodes.add(n)
        return new_node_set

    def add_layer(self):
        new_node_set = self.new_neighbors()

        self.layer_nodes.append(new_node_set)
        self.layer_node_to_class.append({n: 0 for n in new_node_set})
        self.layer_next_class.append(1)

        self.num_layers += 1
        self.sort_layer_from_direction(self.num_layers - 1, -1)

    def level_node_comparison(self, nl_a, nl_b):
        if nl_a[1] < nl_b[1]:
            return -1
        if nl_a[1] > nl_b[1]:
            return 1
        min_len = min(len(nl_a), len(nl_b))
        for i in range(2, min_len):
            if nl_a[i] < nl_b[i]:
                return -1
            if nl_a[i] > nl_b[i]:
                return 1
        if len(nl_a) > len(nl_b): # More connections move you further front rather than fewer.
            return -1
        if len(nl_a) < len(nl_b):
            return 1
        return 0

    def sort_layer_from_direction(self, l, direction):
        node_list = [[n, c] for n, c in self.layer_node_to_class[l].items()]
        for i in range(0, len(node_list)):
            node = node_list[i][0]
            other_layer_neighbors = self.neighbor_sets[node] & self.layer_nodes[l + direction] # Above vs Below
            other_layer_classes = [self.layer_node_to_class[l + direction][n] for n in other_layer_neighbors] # Above vs Below
            other_layer_classes.sort()
            node_list[i] = node_list[i] + other_layer_classes
        node_list.sort(cmp=self.level_node_comparison)

        any_real_change = False # Records if a class was split
        change_points = []
        for i in range(1, len(node_list)):
            if self.level_node_comparison(node_list[i - 1], node_list[i]) != 0:
                change_points.append(i)
                if node_list[i - 1][1] == node_list[i][1]:
                    any_real_change = True
        change_points.append(len(node_list))
        i = 0
        for c in change_points:
            while i < c:
                self.layer_node_to_class[l][node_list[i][0]] = self.layer_next_class[l]
                i += 1
            self.layer_next_class[l] += 1

        return any_real_change

    def done(self):
        # Have we found all the nodes we can?
        for node, c in self.layer_node_to_class[self.num_layers - 1].items():
            for n in self.neighbor_sets[node]:
                if n not in self.found_nodes:
                    return False

        # Are the last and 3rd-to-last layers equal in nodes and classification?
        # If so, we can stop.
        if self.num_layers > 2:
            ultimate_nodes = [n for n, c in self.layer_node_to_class[self.num_layers - 1].items()]
            antepenultimate_nodes = [n for n, c in self.layer_node_to_class[self.num_layers - 3].items()]
            if len(ultimate_nodes) == len(antepenultimate_nodes): # This also implies they have the same contents.
                ultimate_nodes.sort()
                antepenultimate_nodes.sort()
                ultimate_classes = self.layer_node_to_class[self.num_layers - 1]
                antepenultimate_classes = self.layer_node_to_class[self.num_layers - 3]
                u_class_identifiers = {}
                a_class_identifiers = {}
                found_discrepancy = False
                for i in range(0, len(ultimate_nodes)):
                    u_node = ultimate_nodes[i]
                    u_class = ultimate_classes[u_node]
                    a_node = antepenultimate_nodes[i]
                    a_class = antepenultimate_classes[a_node]
                    if u_class not in u_class_identifiers:
                        u_class_identifiers[u_class] = u_node
                    if a_class not in a_class_identifiers:
                        a_class_identifiers[a_class] = a_node

                    if u_class_identifiers[u_class] != a_class_identifiers[a_class]:
                        found_discrepancy = True
                        break

                if not found_discrepancy:
                    return True

        # Have we done enough iterations?
        return self.num_layers >= len(self.found_nodes)

    def comparison(self, node_a, node_b):
        if node_a.num_layers < node_b.num_layers:
            return -1
        if node_a.num_layers > node_b.num_layers:
            return 1
        node_a_classes = [[(c, n) for n, c in layer.items()] for layer in node_a.layer_node_to_class]
        node_b_classes = [[(c, n) for n, c in layer.items()] for layer in node_b.layer_node_to_class]

        # Compare labels
        for i in range(0, node_a.num_layers):
            node_a_classes[i].sort()
            node_b_classes[i].sort()
            if len(node_a_classes[i]) < len(node_b_classes[i]):
                return -1
            if len(node_a_classes[i]) > len(node_b_classes[i]):
                return 1
            for j in range(0, len(node_a_classes[i])):
                if node_a_classes[i][j][0] < node_b_classes[i][j][0]:
                    return -1
                if node_a_classes[i][j][0] > node_b_classes[i][j][0]:
                    return 1

        # Compare label definitions
        for i in range(0, node_a.num_layers - 1):
            for j in range(0, len(node_a_classes[i])):
                rep_a = node_a_classes[i][j][1]
                rep_b = node_b_classes[i][j][1]
                rep_a_neighbors = node_a.neighbor_sets[rep_a]
                rep_b_neighbors = node_b.neighbor_sets[rep_b]
                rep_a_classes = [node_a.layer_node_to_class[i + 1][n] for n in rep_a_neighbors]
                rep_b_classes = [node_b.layer_node_to_class[i + 1][n] for n in rep_b_neighbors]
                rep_a_classes.sort()
                rep_b_classes.sort()
                if len(rep_a_classes) < len(rep_b_classes):
                    return -1
                if len(rep_a_classes) > len(rep_b_classes):
                    return 1
                for k in range(0, len(rep_a_classes)):
                    if rep_a_classes[k] < rep_b_classes[k]:
                        return -1
                    if rep_a_classes[k] > rep_b_classes[k]:
                        return 1
        return 0

class PathGraph:
    
    def __init__(self, G):
        node_ids = list(G.nodes())
        self.node_reps = [NodeViaPaths(G, n) for n in node_ids]
        self.node_reps.sort(cmp=self.node_reps[0].comparison)

    def equal(self, another):
        if len(self.node_reps) != len(another.node_reps):
            False
        for i in range(0, len(self.node_reps)):
            if self.node_reps[i].comparison(self.node_reps[i], another.node_reps[i]) != 0:
                return False
        return True
