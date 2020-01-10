# -*- coding: utf-8 -*-
"""
Wrapper around dreadnaut that computes the orbits of a graph.

NOTE: Must have installed `dreandaut`. The location of the binary can be passed
      as an argument to `nauty_compute_automorphisms`.

Author: Jean-Gabriel Young <info@jgyoung.ca>
"""
import subprocess
import networkx as nx
from os import remove

def _build_dreadnaut_file(g):
    """Prepare file to pass to dreadnaut.

    Warning
    -------
    Assumes that the nodes are represented by the 0 indexed integers.
    """
    # dreadnaut options
    file_content = ["As"]  # sparse mode
    file_content.append("-a")  # do not print out automorphisms
    file_content.append("-m")  # do not print out level markers
    # specify graph structure
    file_content.append("n=" + str(g.number_of_nodes()) + " g")
    for v in g.nodes():
          line = " " + str(v) + " : "
          for nb in g.neighbors(v):
              if v < nb:
                  line += str(nb) + " "
          line += ";"
          file_content.append(line)
    # add nauty command
    file_content.append(".")
    file_content.append("+c")
    file_content.append("x")
    file_content.append("o")
    # file_content.append("b")
    # print(file_content)
    return file_content


def nauty_compute(g, tmp_path="/tmp/dreadnaut.txt", dreadnaut_call="Nauty_n_Traces/nauty26r12/dreadnaut"):
    # get dreadnaut command file
    file_content = _build_dreadnaut_file(g)
    # write to tmp_path
    with open(tmp_path, 'w') as f:
        print("\n".join(file_content), file=f)
    # call dreadnaut
    proc = subprocess.run([dreadnaut_call],
                          input=b"< " + tmp_path.encode(),
                          stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL)
    res = proc.stdout.decode()
    print(res)
    lines = res.strip().split("\n")
    info = lines[0]
    orbits = lines[2]
    G = lines[3:]
    # = res.strip().split("\n", 2)
    #print(info)
    # ~~~~~~~~~~~~~~
    # Extract high level info from captured output
    # ~~~~~~~~~~~~~~
    num_orbits = int(info.split(" ")[0])
    num_gen = int(info.split(" ")[3])
    # ~~~~~~~~~~~~~~
    # Extract orbits
    # ~~~~~~~~~~~~~~
    # This big list comprehension splits all orbits into their own sublist, and
    # each of these orbits into individual components (as string).
    # There is still some post-processing to do since some of them are in the
    # compact notation X:X+n when the n+1 nodes of the orbits are contiguous.
    X = [_.strip().split(" (")[0].split(" ")
         for _ in orbits.replace("\n   ",'').strip().split(";")[:-1]]
    for i, orbit in enumerate(X):
        final_orbit = []
        for elem in orbit:
            if ":" in elem:
                _ = elem.split(":")
                final_orbit += range(int(_[0]), int(_[1]) + 1)
            else:
                final_orbit += [int(elem)]
        X[i] = final_orbit
    # garbage collection
    remove(tmp_path)
    return num_orbits, num_gen, X, G

def nauty_isomorphism_check(G1, G2):
    G3 = nx.Graph()
    G1_nodes = list(G1.nodes())
    G1_max = G1_nodes[0]
    for node in G1_nodes:
        if node > G1_max:
            G1_max = node
        G3.add_node(node)
    for (a, b) in G1.edges():
        G3.add_edge(a, b)
    G1_max += 1
    G3.add_node(G1_max)
    for node in G1_nodes:
        G3.add_edge(node, G1_max)

    G4 = nx.Graph(G3)

    G2_start = G1_max + 1
    G2_nodes = list(G2.nodes())
    G2_max = G2_nodes[0]
    for node in G2_nodes:
        if node > G2_max:
            G2_max = node
        G3.add_node(G2_start + node)
    G2_max += 1
    for (a, b) in G2.edges():
        G3.add_edge(a + G2_start, b + G2_start)
    G3.add_node(G2_max + G2_start)
    for node in G2_nodes:
        G3.add_edge(node + G2_start, G2_max + G2_start)

    n1, _, _, _ = nauty_compute(G3)
    n2, _, _, _ = nauty_compute(G4)
    print(n1)
    print(n2)
    #print(CG2)
    return n1 == n2

"""
if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    # declare networkx graph
    g = nx.Graph()
    g.add_node(0)
    g.add_node(1)
    g.add_node(2)
    g.add_edge(1,2)
    colors = [None for i in range(g.number_of_nodes())]

    # orbits and generators of the graph
    num_orbits, num_gen, X = nauty_compute_automorphisms(g)
    print("Graph:\t\t", "num_orbits=" +str(num_orbits),  "num_gen=" +str(num_gen))
     
    # Plot
    colors = [None for i in range(g.number_of_nodes())]
    for idx, orbit in enumerate(X):
        for v in orbit:
            colors[v] = idx
    nx.draw(g, node_color=colors, linewidths=2, width=2, edge_color='gray', edgecolors='k')
    plt.show()
"""
