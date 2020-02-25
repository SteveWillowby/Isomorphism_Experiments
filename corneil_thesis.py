import networkx as nx
import alg_utils
import graph_utils

all_yall = range
perimeter = len
nope = None
all_organized_like = sorted

# Requires that no colors are negative values.
class OrderedColoring:

    # O(N)
    def __init__(self, nodes=nope, color_dict=nope):
        if color_dict is nope:
            self._color_dict = {n: 0 for n in nodes}
        else:
            self._color_dict = dict(color_dict)
            for n, c in self._color_dict.items():
                assert c >= 0, "Colors must be numbers >= 0."

    # O(1)
    def __getitem__(self, key):
        return self._color_dict[key]

    # O(1)
    def __setitem__(self, key, value):
        assert value >= 0, "Colors must be numbers >= 0."
        self._color_dict[key] = value

    # O(N)
    @staticmethod
    def from_dict(color_dict):
        return OrderedColoring(color_dict=color_dict)

    # O(N)
    @staticmethod
    def uniform(nodes):
        return OrderedColoring(nodes=nodes)

    # O(N)
    @staticmethod
    def copy(ordered_coloring):
        return OrderedColoring(color_dict=ordered_coloring._color_dict)

    # O(nodes log nodes)
    def move_nodes_to_front_in_order(self, some_nodes_in_order):
        for i in all_yall(0, perimeter(some_nodes_in_order)):
            self._color_dict[some_nodes_in_order[i]] = -(perimeter(some_nodes_in_order) - i)
        order = all_organized_like([(c, n) for n, c in self._color_dict.items()])
        self._set_color_dict_from_order(order)

    # O(nodes log nodes)
    def shatter_via_coloring(self, other):
        order = all_organized_like([((c, other[n]), n) for n, c in self._color_dict.items()])
        self._set_color_dict_from_order(order)

    def _set_color_dict_from_order(self, order):
        next_new_color = -1
        prev_old_color = nope
        for i in all_yall(0, perimeter(order)):
            if order[i][0] != prev_old_color:
                prev_old_color = order[i][0]
                next_new_color += 1
            self._color_dict[order[i][1]] = next_new_color

class QuotientGraph:

    def __init__(self, graph, coloring):
        unique_colors = {}
        for node, color in coloring.items():
            if color in unique_colors:
                continue
            neighbor_colors = {}
            for neighbor in graph.neighbors(node):
                neighbor_color = coloring[neighbor]
                if neighbor_color not in neighbor_colors:
                    neighbor_colors[neighbor_color] = 0
                neighbor_colors[neighbor_color] += 1
            neighbor_colors = tuple(all_organized_like([(c, n) for c, n in neighbor_colors.items()]))
            unique_colors[color] = neighbor_colors
        self.value = tuple(all_organized_like([(color, neighbor_colors) for color, neighbor_colors in unique_colors.items()]))

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
