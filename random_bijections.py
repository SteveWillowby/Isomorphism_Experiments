# Math:
# Chebyschev's Inequality:
#       Let X be an (integrable?) variable with expected value u and finite non-
#       zero variance o^2. Then for any real number k > 0,
#       Pr(|X - u| >= ko) <= 1/k^2
#
# Markov's Inequality:
#       Let X be a non-negative random variable and a > 0, then,
#       Pr(X >= a) <= E(X) / a

import numpy as np
import math
from graph_utils import zero_indexed_graph
from matplotlib import pyplot as plt

# Assumes G1 and G2 are zero-indexed and have the same number of nodes.
def random_overlap_count(G1, G2):
    p = np.random.permutation([i for i in range(0, len(G1.nodes()))])
    overlap = 0
    for (a, b) in G1.edges():
        if G2.has_edge(p[a], p[b]):
            overlap += 1
    return overlap

def overlap_comparison(G1, G2):
    print("Running overlap comparison...")
    G1 = zero_indexed_graph(G1)
    G2 = zero_indexed_graph(G2)

    G1_nodes = list(G1.nodes())
    G2_nodes = list(G2.nodes())
    if len(G1_nodes) != len(G2_nodes):
        return False
    if len(G1.edges()) != len(G2.edges()):
        return False

    num_measurements = (len(G1.edges())**4) * 1
    G1G1_values = {i: 0 for i in range(0, len(G1.edges()) + 1)}
    G1G2_values = {i: 0 for i in range(0, len(G1.edges()) + 1)}
    print("Target: %d" % num_measurements)
    for i in range(0, num_measurements):
        G1G1_val = random_overlap_count(G1, G1)
        G1G2_val = random_overlap_count(G1, G2)

        G1G1_values[G1G1_val] += 1
        G1G2_values[G1G2_val] += 1

    fig, ax = plt.subplots()
    G1G1_bars = ax.bar([i - 0.25 for i in range(0, len(G1.edges()) + 1)], \
                       [G1G1_values[i] for i in range(0, len(G1.edges()) + 1)], \
                       width=0.2, label='G1G1') 
    G1G2_bars = ax.bar([i + 0.25 for i in range(0, len(G1.edges()) + 1)], \
                       [G1G2_values[i] for i in range(0, len(G1.edges()) + 1)], \
                       width=0.2, label='G1G2')

    ax.set_ylabel('Occurrences')
    ax.set_title('Overlap Histograms')
    ax.legend()

    plt.show()
    plt.close()

    # Representing each total count for overlap O independently as produced by
    #   a sum of indicator variables which equal 1 if the random overlap equals
    #   O and 0 otherwise.

    pessimistic_prob_estimates = \
        [float(G1G1_values[i] + G1G2_values[i]) / (2.0 * num_measurements) for \
            i in range(0, len(G1.edges()) + 1)]

    pessimistic_expected_total_values = \
        [float(G1G1_values[i] + G1G2_values[i]) / 2.0 \
            for i in range(0, len(G1.edges()) + 1)]

    pessimistic_indicator_variances = \
        [pessimistic_prob_estimates[i] * (1.0 - pessimistic_prob_estimates[i]) \
            for i in range(0, len(G1.edges()) + 1)]

    pessimistic_total_sigmas = \
        [math.sqrt(num_measurements * pessimistic_indicator_variances[i]) for \
            i in range(0, len(G1.edges()) + 1)]

    pessimistic_diffs_from_means = \
        [abs(G1G1_values[i] - pessimistic_expected_total_values[i]) for \
            i in range(0, len(G1.edges()) + 1)]

    # Pr(|X - u| >= ko) <= 1/k^2
    pessimistic_chebyschev_non_chances = \
        [None for i in range(0, len(G1.edges()) + 1)]
    for i in range(0, len(G1.edges()) + 1):
        print("Overlap: %d" % i)
        print("    Diff: %f" % pessimistic_diffs_from_means[i])
        print("    Sigma: %f" % pessimistic_total_sigmas[i])

        if pessimistic_diffs_from_means[i] == 0.0:
            pessimistic_chebyschev_non_chances[i] = 1.0
        else:
            pessimistic_chebyschev_non_chances[i] = \
                1.0 / ((pessimistic_diffs_from_means[i] / \
                         pessimistic_total_sigmas[i])**2.0)

        print("    %s" % str(pessimistic_chebyschev_non_chances[i]))

    total_p_thing = 1.0
    for i in range(0, len(G1.edges()) + 1):
        c = min(pessimistic_chebyschev_non_chances[i], 1.0)
        # print(c)
        # if c < 0.01:
        #     return False
        total_p_thing *= c
    print("Total p thing: %f" % total_p_thing)
    return total_p_thing >= 0.01
