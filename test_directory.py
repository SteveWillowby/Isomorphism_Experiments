import networkx as nx
import graph_utils
import sys
from k_tuple_test import *
from run_nauty import *
from os import listdir
from os.path import isfile, join
import time

directory = sys.argv[1]
batch_size = 2
if len(sys.argv) > 2:
    batch_size = int(sys.argv[2])

file_names = sorted([join(directory, f) for f in listdir(directory) if isfile(join(directory, f))])

COMPARE_IDENTICAL_GRAPHS = True
TIME_LIMIT_ISH = 240 # If a pair takes more than this many seconds for both ours and Nauty combined, stop comparing after the pair finishes.

next_start = 0
while next_start < len(file_names):
    graphs = []
    for j in range(0, batch_size):
        graphs.append((graph_utils.permute_node_labels(graph_utils.zero_indexed_graph(nx.Graph(nx.read_adjlist(file_names[next_start + j])))),\
            file_names[next_start + j]))
    next_start += batch_size
    for i in range(0, batch_size):
        for j in range(i + (1 - int(COMPARE_IDENTICAL_GRAPHS)), batch_size):
            G1 = graphs[i][0]
            G2 = graphs[j][0]
            print("\nComparing graphs: %s vs. %s" % (graphs[i][1].split("/")[-1], graphs[j][1].split("/")[-1]))
            if len(G1.nodes()) == 0 or len(G2.nodes()) == 0:
                print("At least one graph is empty. Continuing.")
                continue
            t1 = time.time()
            prediction= k_tuple_check(G1, G2) # exact_k=2
            t2 = time.time()
            print("Got prediction: %s" % prediction)
            print("Running Nauty...")
            t3 = time.time()
            actual = nauty_isomorphism_check(G1, G2)
            t4 = time.time()

            if prediction == actual:
                print("Correct!")
            else:
                print("Incorrect!")
                exit(1)

            if t2 - t1 < t4 - t3:
                print("Faster. (%d s)" % ((t4 - t3) - (t2 - t1)))
            elif t4 - t3 < t2 - t1:
                print("Slower. (%d s)" % ((t2 - t1) - (t4 - t3)))
            else:
                print("Tied")

            if (t4 - t3) + (t2 - t1) > TIME_LIMIT_ISH:
                print("Quitting because it's starting to take too long.")
                exit(0)

print("\nFinished Successfully")
