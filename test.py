import networkx as nx
from networkx import utils
# from networkx.algorithms.bipartite.generators import configuration_model
from networkx.algorithms import isomorphism
from networkx.algorithms.components import is_connected
from some_srgs import *
from faster_neighbors_revisited import *
from neighbors_revisited import *
from rudolf_mathon import *
from paths import *
from run_nauty import *
from k_tuple_test import *
from corneil_thesis import *
import graph_utils
import alg_utils
import time
from edge_xor_estimates import xor_estimate
from random_bijections import overlap_comparison_generic_bound, overlap_comparison_bayesian_bound

A1 = graph_from_srg_string(GRAPH_STRING_A1)
A2 = graph_from_srg_string(GRAPH_STRING_A2)
A3 = graph_from_srg_string(GRAPH_STRING_A3)
A4 = graph_from_srg_string(GRAPH_STRING_A4)
G_25_12 = [graph_from_srg_string(s) for s in GS_25_12_5_6]
G_25_12_COMP = []
for i in range(0, len(G_25_12)):
    for j in range(i, len(G_25_12)):
        G_25_12_COMP.append((G_25_12[i], G_25_12[j]))

SMALL_SRS_COMP = [(GS_5_2_0_1[0], GS_5_2_0_1[0]), (GS_9_4_1_2[0], GS_9_4_1_2[0]), (GS_10_3_0_1[0], GS_10_3_0_1[0]), (GS_13_6_2_3[0], GS_13_6_2_3[0]),\
    (GS_15_6_1_3[0], GS_15_6_1_3[0]), (GS_16_5_0_2[0], GS_16_5_0_2[0]), (GS_16_6_2_2[0], GS_16_6_2_2[0]), (GS_16_6_2_2[1], GS_16_6_2_2[1]),\
    (GS_16_6_2_2[0], GS_16_6_2_2[1]), (GS_17_8_3_4[0], GS_17_8_3_4[0]), (GS_21_10_3_6[0], GS_21_10_3_6[0])]
SMALL_SRS_COMP = [(graph_from_srg_string(a), graph_from_srg_string(b)) for (a, b) in SMALL_SRS_COMP]

# test = PathSteps(A2)

Pet = graph_utils.peterson_graph()
Gen1 = graph_utils.gen_graph_1()
Gen1Cycles = graph_utils.gen_graph_1_cycles()
M2 = graph_utils.miyazaki_graph(2)
M3 = graph_utils.miyazaki_graph(3)
M4 = graph_utils.miyazaki_graph(4)
M5 = graph_utils.miyazaki_graph(5)
M10 = graph_utils.miyazaki_graph(10)
M100 = graph_utils.miyazaki_graph(100)

RM_25_related_ER = nx.gnm_random_graph(25, 60)
seq = [min(i + 1, max(0, 120 - int((i * (i + 1)) / 2))) for i in range(0, 25)]
print(seq)
RM_25_related_degree_seq = graph_utils.make_graph_with_degree_sequence(seq)
RM_A25 = graph_utils.zero_indexed_graph(Rudolf_Mathon_A25())
RM_B25 = graph_utils.zero_indexed_graph(Rudolf_Mathon_B25())
# RM_A25 = graph_utils.graph_complement(RM_A25)  # Taking complement does not change the k for these graphs.
# RM_B25 = graph_utils.graph_complement(RM_B25)
RM_A35 = graph_utils.zero_indexed_graph(Rudolf_Mathon_A35())
RM_B35 = graph_utils.zero_indexed_graph(Rudolf_Mathon_B35())
RM_E72_A35 = graph_utils.zero_indexed_graph(Rudolf_Mathon_E72_A35())
RM_E72_B35 = graph_utils.zero_indexed_graph(Rudolf_Mathon_E72_B35())

#print("Is peterson 3SR? %s" % graph_utils.is_3_SR(Pet))
#print("Is Gen1 3SR? %s" % graph_utils.is_3_SR(Gen1))
#print("Is Gen1Cycles 3SR? %s" % graph_utils.is_3_SR(Gen1Cycles))
#print("Is M3 3SR? %s" % graph_utils.is_3_SR(M3))
#print("Is A1 3SR? %s" % graph_utils.is_3_SR(A1))
#print("Is A2 3SR? %s" % graph_utils.is_3_SR(A2))
#print("Is A3 3SR? %s" % graph_utils.is_3_SR(A3))
#print("Is A4 3SR? %s" % graph_utils.is_3_SR(A4))
#print("Is RM_A35 3SR? %s" % graph_utils.is_3_SR(RM_A35))
#print("Is RM_E72_A35 3SR? %s" % graph_utils.is_3_SR(RM_E72_A35))

# COMPARISONS = [(Gen1, Gen1), (Gen1Cycles, Gen1Cycles), (Pet, Pet),(M2, M2),(M3,M3), (M4,M4),(M5, M5),(M10,M10),(M100,M100)]
# COMPARISONS = [(A2, A2), (A1, A3), (A2, A2), (A1,A2),(A1,A3),(A1,A4),(A2,A4),(A3,A4)]
# COMPARISONS = G_25_12_COMP
# COMPARISONS = SMALL_SRS_COMP
# COMPARISONS = [(RM_A25, RM_B25), (RM_B25, RM_B25), (RM_A25, RM_A25), (RM_A35, RM_A35), (RM_B35, RM_B35), (RM_A35, RM_B35)]
# JS1_RM_A25 = graph_utils.Justus_square_1(RM_A25)
# JS1_JS1_RM_A25 = graph_utils.Justus_square_1(JS1_RM_A25)
COMPARISONS = [(RM_A25, RM_25_related_degree_seq), (RM_A25, RM_25_related_ER), (RM_A25, RM_B25), (RM_A25, RM_A25)] #, (JS1_JS1_RM_A25, JS1_JS1_RM_A25)]
# COMPARISONS = [(RM_E72_A35, RM_E72_A35), (RM_E72_B35, RM_E72_B35), (RM_E72_A35, RM_E72_B35)]

# print(l_nodes_k_dim_WL_coloring(RM_A25, 3, 1))
# print(k_dim_WL_coloring(RM_A25, 3))

bench_d3_a = nx.read_adjlist("benchmark_graphs/cfi-rigid-d3/converted/cfi-rigid-d3-3600-01-1.edge_list", create_using=nx.Graph, nodetype=int)
bench_d3_a = graph_utils.zero_indexed_graph(bench_d3_a)
bench_d3_b = nx.read_adjlist("benchmark_graphs/cfi-rigid-d3/converted/cfi-rigid-d3-3600-01-2.edge_list", create_using=nx.Graph, nodetype=int)
#base_0100_a = nx.read_adjlist("sat_cfi_dim/converted/sat_cfi_base_0100_a.edge_list", create_using=nx.Graph, nodetype=int)
#base_0100_b = nx.read_adjlist("sat_cfi_dim/converted/sat_cfi_base_0100_b.edge_list", create_using=nx.Graph, nodetype=int)
base_2000_a = nx.read_adjlist("sat_cfi_dim/converted/sat_cfi_base_2000_a.edge_list", create_using=nx.Graph, nodetype=int)
base_2000_b = nx.read_adjlist("sat_cfi_dim/converted/sat_cfi_base_2000_b.edge_list", create_using=nx.Graph, nodetype=int)
#base_0100_a = nx.Graph(base_0100_a)
#base_0100_b = nx.Graph(base_0100_b)
base_2000_a = nx.Graph(base_2000_a)
base_2000_b = nx.Graph(base_2000_b)

if False:
    edge_types = {}
    for (a, b) in bench_d3_a.edges():
        edge_types[(a, b)] = 0
        edge_types[(b, a)] = 0

    coloring = {n: 0 for n in bench_d3_a.nodes()}
    start_time = time.time()
    for i in range(0, 30):
        init_coloring = WLColoringWithEdgeTypes(bench_d3_a, coloring, edge_types, init_active_set=set([0])).coloring
    #the_c = [(0, n) for n in range(0, len(coloring))]
    #alg_utils.further_sort_by(the_c, init_coloring)
    #print([x[1] for x in sorted([(n, c) for n, c in the_c])])
    print("Old Code's Time")
    print(time.time() - start_time)

    start_time = time.time()
    for i in range(0, 30):
        coloring = [0 for i in range(0, len(bench_d3_a.nodes()))]
        WL(bench_d3_a, coloring, edge_types=edge_types, init_active_set=set([0]))
    #print(coloring)
    print("New Code's Time")
    print(time.time() - start_time)

# COMPARISONS = [(base_2000_a, graph_utils.permute_node_labels(base_2000_b)), (base_2000_a, graph_utils.permute_node_labels(base_2000_a))]
# COMPARISONS = [(bench_d3_a, bench_d3_b), (bench_d3_a, bench_d3_a)]

GWat = nx.Graph()
GWat.add_node(0)
GWat.add_node(1)
GWat.add_node(2)
GWat.add_node(3)
GWat.add_node(4)
GWat.add_node(5)
GWat.add_edge(0, 1)
GWat.add_edge(1, 2)
GWat.add_edge(3, 4)
GWat.add_edge(4, 5)
GWuh = nx.Graph(GWat)
GWat.add_edge(0, 2)
GWat.add_edge(3, 5)
GWuh.add_edge(0, 3)
GWuh.add_edge(2, 5)
# COMPARISONS = [(GWat, GWuh)]

for i in range(0, len(COMPARISONS)):
    #print("Creating Pairs of Graphs")
    good = False
    while not good:
        # Generate first G
        using_sequence = False
        #sequence = [2, 2, 2, 2, 6, 4, 4, 4, 4]  # Set sequence
        #G=nx.configuration_model(sequence)

        G=nx.erdos_renyi_graph(100,0.4)
        #G=nx.watts_strogatz_graph(100,3,0.3)
        #G=nx.barabasi_albert_graph(10,2)

        G=nx.Graph(G)
        G.remove_edges_from([(n, n) for n in G.nodes()])
        if not is_connected(G):
            print("Bad: G disconnected")
            continue
        good = True
        # G_prime = graph_utils.make_graph_with_same_degree_dist(G)
        # G_prime = graph_utils.permute_node_labels(G)

    (G, G_prime) = COMPARISONS[i]
    G_prime = graph_utils.permute_node_labels(G_prime)

    # G3 = graph_utils.graph_union(G, G_prime)
    # thing1 = FasterNeighborsRevisited(G3)
    #print("Starting prediction")
    #c_desc_G = KTupleTest(G, k=2)
    #c_desc_G = NeighborsRevisited(G)
    print("...")
    #c_desc_G_prime = NeighborsRevisited(G_prime)
    #c_desc_G_prime = KTupleTest(G_prime, k=2)
    #print("...")
    #predict_iso = c_desc_G == c_desc_G_prime
    print("Starting our prediction...")
    predict_iso = overlap_comparison_bayesian_bound(G, G_prime)

    # predict_iso = k_dim_WL_test(G, G_prime, 2)
    print("Got prediction: %s" % predict_iso)
    # print(c_desc_G.mapping_to_labels)
    print("Running Nauty...")
    actual_iso = nauty_isomorphism_check(G, G_prime)
    print("Nauty Finished")

    # Get actual result
    #GM = isomorphism.GraphMatcher(G, G_prime)
    #actual_iso = GM.is_isomorphic()
    # actual_iso = predict_iso

    if predict_iso == actual_iso:
        print("\nCorrect!")
        print(actual_iso)
    else:
        print("Incorrect!")
        print("Actual:")
        print(actual_iso)
