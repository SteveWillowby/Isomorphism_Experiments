import networkx as nx
import time
from automorph_method_v3 import *

for size_limit in range(10, 21):
    next_size = 4.0
    G = nx.Graph()
    for i in range(0, size_limit):
        G.add_node(i)
        if i + 1 >= next_size:
            for j in range(0, i):
                G.add_edge(j, i)
            next_size += 2.5
            # print(len(G.edges()) - (i * (i - 1)) / 4)

    start = time.time()
    GG = PartialGGraph(G)
    end = time.time()
    print("Size %s: Time: %s" % (size_limit, end - start))
