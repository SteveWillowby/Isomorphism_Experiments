import networkx as nx
import matplotlib.pyplot as plt

def gadget_3(num):
    G = nx.Graph()
    G.add_node("%d_ttop" % num)
    G.add_node("%d_btop" % num)
    G.add_node("%d_tbot" % num)
    G.add_node("%d_bbot" % num)
    G.add_node("%d_tmid" % num)
    G.add_node("%d_bmid" % num)
    G.add_node("%d_0" % num)
    G.add_node("%d_1" % num)
    G.add_node("%d_2" % num)
    G.add_node("%d_3" % num)

    G.add_edge("%d_ttop" % num, "%d_0" % num)
    G.add_edge("%d_tbot" % num, "%d_0" % num)
    G.add_edge("%d_tmid" % num, "%d_0" % num)

    G.add_edge("%d_btop" % num, "%d_1" % num)
    G.add_edge("%d_bbot" % num, "%d_1" % num)
    G.add_edge("%d_tmid" % num, "%d_1" % num)

    G.add_edge("%d_btop" % num, "%d_2" % num)
    G.add_edge("%d_tbot" % num, "%d_2" % num)
    G.add_edge("%d_bmid" % num, "%d_2" % num)

    G.add_edge("%d_ttop" % num, "%d_3" % num)
    G.add_edge("%d_bbot" % num, "%d_3" % num)
    G.add_edge("%d_bmid" % num, "%d_3" % num)

    return G

def miyazaki_graph(k):
    G = nx.Graph()
    for i in range(0, 2*k):
        next_gadget = gadget_3(i)
        for node in next_gadget.nodes():
            G.add_node(node)
        for edge in next_gadget.edges():
            G.add_edge(edge[0], edge[1])
        if i == 0:
            G.add_edge("0_ttop", "0_bbot")
            G.add_edge("0_btop", "0_tbot")
        elif i == (2*k - 1):
            G.add_edge("%d_ttop" % i, "%d_bbot" % i)
            G.add_edge("%d_btop" % i, "%d_tbot" % i)

        if i > 0:
            if int(i / 2) * 2 == i:
                G.add_edge("%d_ttop" % (i - 1), "%d_ttop" % i)
                G.add_edge("%d_btop" % (i - 1), "%d_btop" % i)
                G.add_edge("%d_tbot" % (i - 1), "%d_tbot" % i)
                G.add_edge("%d_bbot" % (i - 1), "%d_bbot" % i)
            else:
                G.add_edge("%d_tmid" % (i - 1), "%d_tmid" % i)
                G.add_edge("%d_bmid" % (i - 1), "%d_bmid" % i)

    node_relabeling = list(G.nodes())
    node_relabeling.sort()
    node_relabeling = {node_relabeling[i]: i for i in range(0, len(node_relabeling))}

    G_int = nx.Graph()
    for i in range(0, len(node_relabeling)):
        G_int.add_node(i)
    for edge in G.edges():
        G_int.add_edge(node_relabeling[edge[0]], node_relabeling[edge[1]])

    #nx.draw_networkx(G, node_color='blue', node_size=100)#,nodelist= labels=labels, edgelist=, edge_color=edge_colors, pos=positions)
    #plt.title("hi")
    #plt.draw()
    #plt.show()

    return G_int
