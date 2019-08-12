import networkx as nx
from automorph_method_v3 import *
from is_this_the_one import *
from Gods_way_is_the_best_way import *

G1 = nx.Graph()
for i in range(0, 17):
    G1.add_node(i)

G1.add_edge(0,1)
G1.add_edge(0,2)
G1.add_edge(0,3)
G1.add_edge(0,4)
G1.add_edge(0,5)
G1.add_edge(0,6)
G1.add_edge(0,7)
G1.add_edge(0,8)

G1.add_edge(1,9)
G1.add_edge(2,10)
G1.add_edge(3,11)
G1.add_edge(4,12)
G1.add_edge(5,13)
G1.add_edge(6,14)
G1.add_edge(7,15)
G1.add_edge(8,16)

G1.add_edge(9,10)
G1.add_edge(10,11)
G1.add_edge(11,12)

G1.add_edge(13,14)
G1.add_edge(14,15)
G1.add_edge(15,16)

G2 = nx.Graph(G1)

G1.add_edge(9,12)
G1.add_edge(13,16)
G2.add_edge(9,16)
G2.add_edge(12,13)

#GG1 = GGraph(G1, first_layer=True)
#GG2 = GGraph(G2, first_layer=True)

#print(GG1.graph_comparison(GG1,GG2))

# Graph 2

G1 = nx.Graph()
for i in range(0, 14):
    G1.add_node(i)

G1.add_edge(0,1)
G1.add_edge(1,2)
G1.add_edge(2,3)
G1.add_edge(3,4)

G1.add_edge(5,6)
G1.add_edge(6,7)
G1.add_edge(7,8)
G1.add_edge(8,9)

G1.add_edge(10,11)
G1.add_edge(10,12)
G1.add_edge(10,13)
G1.add_edge(11,12)
G1.add_edge(11,13)
G1.add_edge(12,13)

G1.add_edge(10,0)
G1.add_edge(10,4)
G1.add_edge(11,5)
G1.add_edge(11,9)

G2 = nx.Graph(G1)

G1.add_edge(12,0)
G1.add_edge(12,5)
G1.add_edge(13,4)
G1.add_edge(13,9)

G2.add_edge(12,0)
G2.add_edge(12,4)
G2.add_edge(13,5)
G2.add_edge(13,9)

GG1 = GGraph(G1, first_layer=True)
label_graphs = [g for l, g in GG1.automorphism_orbits[0][0].label_graphs.items()]
label_graphs.sort(cmp=GG1.graph_comparison)
count = 1
for i in range(0, len(label_graphs)):
    comp = GG1.graph_comparison(label_graphs[i - 1], label_graphs[i])
    if comp != 0:
        count += 1
print("Unique labels: %s" % count)

GG2 = GGraph(G2, first_layer=True)

print(GG1.graph_comparison(GG1,GG2))

G1 = nx.Graph()
G1.add_node(0)
G1.add_node(1)
G1.add_node(2)
G1.add_edge(0,1)
G1.add_edge(0,2)
GG1 = GGraph(G1, first_layer=True)
