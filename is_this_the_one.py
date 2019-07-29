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
            l = self.num_layers - 2
            while change:
                change = False
                while l > 0:
                    if not sort_layer_from_above(l)
                        break
                    change = True
                    l -= 1
                l += 1
                while l < self.num_layers:
                    if not sort_layer_from_below(l):
                        break
                    change = True

    def new_neighbors(self):
        new_node_set = set()
        last_layer = self.layer_nodes[-1]
        for node in last_layer:
            for n in self.neighbor_sets(node):
                new_node_set.add(n)
                if n not in self.found_nodes():
                    self.neighbor_sets[n] = set(self.G.neighbors(n))
        return new_node_set

    def add_layer(self):
        new_node_set = self.new_neighbors()

        self.layer_nodes.append(new_node_set)
        self.layer_node_to_class.append({n: 0 for n in self.new_node_set})
        self.layer_next_class.append(1)

        self.num_layers += 1
        self.sort_layer_from_below(self.num_layers - 1)

    def sort_layer_from_above(self, l):
        pass

    def sort_layer_from_below(self, l):
        node_list = [(n, c) for n, c in layer_node_to_class.items()]
        node_list.sort(key=(lambda x: x[1]))

    def done(self):
        # Have we found all the nodes we can?
        for node, c in self.layer_node_to_class[self.num_layers - 1].items():
            for n in self.neighbors_sets(node):
                if n not in self.found_nodes:
                    return False

        # Are the last and 3rd-to-last layers equal in nodes and classification?
        # If so, we can stop.
        if self.num_layers > 2:
            ultimate_nodes = [n for n, c in layer_node_to_class[self.num_layers - 1].items()]
            antepenultimate_nodes = [n for n, c in layer_node_to_class[self.num_layers - 3].items()]
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
        return self.num_layers >= len(self.found_nodes):
            return False
