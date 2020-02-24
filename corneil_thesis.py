import networkx as nx
import alg_utils
import graph_utils

class OrderedColoring:

    def __init__(self, nodes=None, color_dict=None):
        if color_dict is None:
            color_dict = {n: 0 for n in nodes}
        self._color_dict = color_dict

    def __getitem__(self, key):
        return self._color_dict[key]

    def __setitem__(self, key, value):
        self._color_dict[key] = value

    @staticmethod
    def from_dict(color_dict):
        return OrderedColoring(color_dict=color_dict)

    @staticmethod
    def uniform(nodes):
        return OrderedColoring(nodes=nodes)

    @staticmethod
    def copy(ordered_coloring):
        return OrderedColoring(color_dict=ordered_coloring._color_dict)

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
            neighbor_colors = tuple(sorted([(c, n) for c, n in neighbor_colors.items()]))
            unique_colors[color] = neighbor_colors
        self.value = tuple(sorted([(color, neighbor_colors) for color, neighbor_colors in unique_colors.items()]))

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
