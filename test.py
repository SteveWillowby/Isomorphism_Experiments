import networkx as nx
from networkx import utils
# from networkx.algorithms.bipartite.generators import configuration_model
from networkx.algorithms import isomorphism
from networkx.algorithms.components import is_connected
from some_srgs import *
from faster_neighbors_revisited import *
from neighbors_revisited import *
from miyazaki_graphs import *
from paths import *
from run_nauty import *
import graph_utils

def make_graph_with_same_degree_dist(G):
    G_sequence = list(d for n, d in G.degree())
    G_sequence.sort()
    sorted_G_sequence = list((d, n) for n, d in G.degree())
    sorted_G_sequence.sort(key=lambda tup: tup[0])
    done = False
    while not done:
        G_prime = nx.configuration_model(G_sequence)
        G_prime = nx.Graph(G_prime)
        G_prime.remove_edges_from(G_prime.selfloop_edges())
        tries = 10
        while tries > 0 and (len(G.edges()) != len(G_prime.edges())):
            sorted_G_prime_sequence = list((d, n) for n, d in G_prime.degree())
            sorted_G_prime_sequence.sort(key=lambda tup: tup[0])
            #print("Sorted G_sequence:")
            #print(sorted_G_sequence)
            #print("Sorted G_prime_sequence:")
            #print(sorted_G_prime_sequence)
            missing = []
            for i in range(0, len(G.nodes())):
                while sorted_G_sequence[i][0] > sorted_G_prime_sequence[i][0]:
                    missing.append(sorted_G_prime_sequence[i][1])
                    sorted_G_prime_sequence[i] = (sorted_G_prime_sequence[i][0] + 1, sorted_G_prime_sequence[i][1])
            missing = np.random.permutation(missing)
            if len(missing) % 2 != 0:
                print("Sanity issue! Alert!")
            #print("Edges before:")
            #print(G_prime.edges())
            #print("Missing:")
            #print(missing)
            for i in range(0, int(len(missing) / 2)):
                G_prime.add_edge(missing[2*i], missing[2*i + 1])
            G_prime = nx.Graph(G_prime)
            G_prime.remove_edges_from(G_prime.selfloop_edges())
            #print("Edges after:")
            #print(G_prime.edges())
            #if not is_connected(G_prime):
                #print("Bad: G_prime disconnected")
            tries -= 1
        if not is_connected(G_prime):
            pass
        elif len(G.edges()) == len(G_prime.edges()):
            #print("Graph creation successful")
            done = True
    return G_prime

def peterson_graph():
    G = nx.Graph()
    for i in range(0, 10):
        G.add_node(i)
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(3, 4)
    G.add_edge(4, 0)

    G.add_edge(0, 5)
    G.add_edge(1, 6)
    G.add_edge(2, 7)
    G.add_edge(3, 8)
    G.add_edge(4, 9)

    G.add_edge(5, 7)
    G.add_edge(7, 9)
    G.add_edge(9, 6)
    G.add_edge(6, 8)
    G.add_edge(8, 5)

    return G

A1 = graph_from_srg_string(GRAPH_STRING_A1)
A2 = graph_from_srg_string(GRAPH_STRING_A2)
A3 = graph_from_srg_string(GRAPH_STRING_A3)
A4 = graph_from_srg_string(GRAPH_STRING_A4)

# test = PathSteps(A2)

Pet = peterson_graph()
M2 = miyazaki_graph(2)
M3 = miyazaki_graph(3)
M4 = miyazaki_graph(4)
M5 = miyazaki_graph(5)
M10 = miyazaki_graph(10)
M100 = miyazaki_graph(100)

COMPARISONS = [(Pet, Pet),(M2, M2),(M3,M3),(M4,M4),(M5, M5),(M10,M10),(M100,M100)]
#COMPARISONS = [(A1,A2),(A1,A3),(A1,A4),(A2,A3),(A2,A4),(A3,A4)]

bench_d3_a = nx.read_adjlist("benchmark_graphs/cfi-rigid-d3/cfi-rigid-d3-3600-01-1.edge_list", create_using=nx.Graph, nodetype=int)
bench_d3_b = nx.read_adjlist("benchmark_graphs/cfi-rigid-d3/cfi-rigid-d3-3600-01-2.edge_list", create_using=nx.Graph, nodetype=int)
base_0100_a = nx.read_adjlist("sat_cfi_dim/sat_cfi_base_0100_a.edge_list", create_using=nx.Graph, nodetype=int)
base_0100_b = nx.read_adjlist("sat_cfi_dim/sat_cfi_base_0100_b.edge_list", create_using=nx.Graph, nodetype=int)
base_1000_a = nx.read_adjlist("sat_cfi_dim/sat_cfi_base_8000_a.edge_list", create_using=nx.Graph, nodetype=int)
base_1000_b = nx.read_adjlist("sat_cfi_dim/sat_cfi_base_8000_b.edge_list", create_using=nx.Graph, nodetype=int)
base_0100_a = nx.Graph(base_0100_a)
base_0100_b = nx.Graph(base_0100_b)
base_1000_a = nx.Graph(base_1000_a)
base_1000_b = nx.Graph(base_1000_b)

#COMPARISONS = [(base_0100_a, graph_utils.permute_node_labels(base_0100_a)), (base_1000_a, graph_utils.permute_node_labels(base_1000_a))]
#COMPARISONS = [(bench_d3_a, bench_d3_b), (bench_d3_a, bench_d3_a)]

for i in range(0, len(COMPARISONS)):
    #print("Creating Pairs of Graphs")
    """
    good = False
    while not good:
        # Generate first G
        using_sequence = False
        #sequence = [2, 2, 2, 2, 6, 4, 4, 4, 4]  # Set sequence
        #G=nx.configuration_model(sequence)

        # G=nx.erdos_renyi_graph(100,0.4)
        G=nx.watts_strogatz_graph(100,3,0.3)
        #G=nx.barabasi_albert_graph(10,2)

        G=nx.Graph(G)
        G.remove_edges_from(G.selfloop_edges())
        if not is_connected(G):
            print("Bad: G disconnected")
            continue
        good = True
        G_prime = make_graph_with_same_degree_dist(G)
        # G_prime = graph_utils.permute_node_labels(G)
    """

    (G, G_prime) = COMPARISONS[i]
    G_prime = graph_utils.permute_node_labels(G_prime)
    #print("Starting prediction")
    #c_desc_G = FasterNeighborsRevisited(G)
    #print("...")
    #c_desc_G_prime = FasterNeighborsRevisited(G_prime)
    #print("...")
    #predict_iso = c_desc_G == c_desc_G_prime
    print("Starting our prediction...")
    predict_iso = paths_comparison(G, G_prime)
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

