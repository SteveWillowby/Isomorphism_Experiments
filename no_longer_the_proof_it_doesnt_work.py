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

GG1 = GGraph(G1, first_layer=True)
GG2 = GGraph(G2, first_layer=True)

print(GG1.graph_comparison(GG1,GG2))
"""
Test1 = NodeViaPaths(G1, 0)
Test2 = NodeViaPaths(G2, 0)

print(Test1.layer_nodes)
node_to_class = [(c, n) for n, c in Test1.layer_node_to_class[-1].items()]
node_to_class.sort()
print(node_to_class)
print(Test1.layer_next_class[-1])
print(Test1.num_layers)
print("\n")
print(Test2.layer_nodes)
node_to_class = [(c, n) for n, c in Test2.layer_node_to_class[-1].items()]
node_to_class.sort()
print(node_to_class)
print(Test2.layer_next_class[-1])
print(Test2.num_layers)
print("\n")

PG1 = PathGraph(G1)
PG2 = PathGraph(G2)
print(PG1.equal(PG2))
"""
