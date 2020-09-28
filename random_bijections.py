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
import bigfloat

# Assumes G1 and G2 are zero-indexed and have the same number of nodes.
def random_overlap_count(G1, G2):
    p = np.random.permutation([i for i in range(0, len(G1.nodes()))])
    overlap = 0
    for (a, b) in G1.edges():
        if G2.has_edge(p[a], p[b]):
            overlap += 1
    return overlap

# Assumes G1 and G2 are zero-indexed and have the same number of nodes.
def get_S_overlap_samples(S, G1, G2):
    values = {i: 0 for i in range(0, len(G1.edges()) + 1)}
    for i in range(0, S):
        val = random_overlap_count(G1, G2)

        values[val] += 1
    return values

def overlap_comparison_01(G1, G2):
    print("Running overlap comparison 01...")
    G1 = zero_indexed_graph(G1)
    G2 = zero_indexed_graph(G2)

    G1_nodes = list(G1.nodes())
    G2_nodes = list(G2.nodes())
    if len(G1_nodes) != len(G2_nodes):
        return False
    if len(G1.edges()) != len(G2.edges()):
        return False

    num_measurements = (len(G1.nodes())**5) * 1
    print("Target: %d" % num_measurements)
    G1G1_values = get_S_overlap_samples(num_measurements, G1, G1)
    G1G2_values = get_S_overlap_samples(num_measurements, G1, G2)

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

    # plt.show()
    plt.close()

    best_suggestion_they_are_different = 0.0
    total_sameness = 1.0
    for i in range(0, len(G1.edges())):
        p_bound = prob_same_bound_favorable_to_saying_same(\
            num_measurements, G1G1_values[i], G1G2_values[i])
        assert p_bound <= 1.0
        if p_bound < 1.0:
            print("%d: %f" % (i, p_bound))
        if 1.0 - p_bound > best_suggestion_they_are_different:
            best_suggestion_they_are_different = 1.0 - p_bound
        total_sameness *= p_bound
    
    print("Total Sameness: %f" % total_sameness)
    return not (best_suggestion_they_are_different >= 0.99)

    """
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
    return total_p_thing >= 0.5
    """

def overlap_comparison_02(G1, G2):
    print("Running overlap comparison 02...")
    G1 = zero_indexed_graph(G1)
    G2 = zero_indexed_graph(G2)

    G1_nodes = list(G1.nodes())
    G2_nodes = list(G2.nodes())
    if len(G1_nodes) != len(G2_nodes):
        return False
    if len(G1.edges()) != len(G2.edges()):
        return False

    total_trials = len(G1_nodes)**0
    trial_size = len(G1_nodes)**5

    print("Taking %d trials of size 3 * %d for a total of %d samples." % \
        (total_trials, trial_size, total_trials * 3 * trial_size))

    same_count = 0
    iso_count = 0
    non_iso_count = 0
    for trial in range(0, total_trials):
        G1G1_ref_values =  get_S_overlap_samples(trial_size, G1, G1)
        G1G1_comp_values = get_S_overlap_samples(trial_size, G1, G1)
        G1G2_comp_values = get_S_overlap_samples(trial_size, G1, G2)

        JSD_1111 = jensen_shannon_divergence_for_counts(G1G1_ref_values, \
                                                        G1G1_comp_values)
        JSD_1112 = jensen_shannon_divergence_for_counts(G1G1_ref_values, \
                                                        G1G2_comp_values)
        print("%f vs %f" % (JSD_1111, JSD_1112))
        if JSD_1111 == JSD_1112:
            same_count += 1
        elif JSD_1111 > JSD_1112:
            iso_count += 1
        else:
            non_iso_count += 1

    print("      ISO count: %d" % iso_count)
    print("  Non-ISO count: %d" % non_iso_count)
    print("     Same count: %d" % same_count)

    return non_iso_count / float(iso_count + non_iso_count) < 0.6

def overlap_comparison_03(G1, G2):
    print("Running overlap comparison 03...")
    G1 = zero_indexed_graph(G1)
    G2 = zero_indexed_graph(G2)

    G1_nodes = list(G1.nodes())
    G2_nodes = list(G2.nodes())
    if len(G1_nodes) != len(G2_nodes):
        return False
    if len(G1.edges()) != len(G2.edges()):
        return False

    num_measurements = len(G1.nodes())**4
    print("Taking 2 * %d samples for a total of %d." % \
        (num_measurements, 2*num_measurements))
    G1G1_values = get_S_overlap_samples(num_measurements, G1, G1)
    G1G2_values = get_S_overlap_samples(num_measurements, G1, G2)

    bf_context = bigfloat.Context(precision=200)

    big_bound = bigfloat.BigFloat(1.0, context=bf_context)

    for o in range(0, len(G1.edges())):
        C1 = float(G1G1_values[o])
        C2 = float(G1G2_values[o])
        S = float(num_measurements)
        over_one_stack = []
        under_one_stack = []
        bound = 1.0
        if C1 + C2 > 0.0:
            print("C1: %d C2: %d S: %d" % (int(C1), int(C2), int(S)))
        for i in range(int(C1), int(S)):
            val = (1.0 - ((C1 + C2) / (2 * S))) / (((i + 1) - C1) / (i + 1))
            if val > 1.0:
                over_one_stack.append(val)
            else:
                under_one_stack.append(val)
            bound *= val
        for i in range(int(C2), int(S)):
            val = (1.0 - ((C1 + C2) / (2 * S))) / (((i + 1) - C2) / (i + 1))
            if val > 1.0:
                over_one_stack.append(val)
            else:
                under_one_stack.append(val)
            bound *= val
        for i in range(0, int(C1)):
            val = ((C1 + C2) / (2 * S)) / ((i + 1) / (i + 1))
            if val > 1.0:
                over_one_stack.append(val)
            else:
                under_one_stack.append(val)
            bound *= val
        for i in range(0, int(C2)):
            val = ((C1 + C2) / (2 * S)) / ((i + 1) / (i + 1))
            if val > 1.0:
                over_one_stack.append(val)
            else:
                under_one_stack.append(val)
            bound *= val
        bound *= ((S + 1)**2)
        bound = bound / (1.0 + bound)
        if C1 + C2 > 0.0:
            print("Bound for overlap of %d: %f" % (o, bound))

        # Compute another way to minimize floating point error issues.
        alt_bound = (S + 1)**2
        over_one_stack_copy = list(over_one_stack)
        under_one_stack_copy = list(under_one_stack)
        while len(over_one_stack) > 0 or len(under_one_stack) > 0:
            if len(over_one_stack) == 0:
                alt_bound *= under_one_stack.pop()
                continue
            if len(under_one_stack) == 0:
                alt_bound *= over_one_stack.pop()
                continue
            a = over_one_stack[-1]
            b = under_one_stack[-1]
            a_diff = abs(1000.0 - (alt_bound * a))
            b_diff = abs(1000.0 - (alt_bound * b))
            if a_diff < b_diff:
                alt_bound *= over_one_stack.pop()
            else:
                alt_bound *= under_one_stack.pop()
        if C1 + C2 > 0.0:
            print("Alt-Bound for overlap of %d: %f (%f)" % (o, alt_bound / (1.0 + alt_bound), alt_bound))
        alt_bound = alt_bound / (1.0 + alt_bound)

        over_one_stack = over_one_stack_copy
        under_one_stack = under_one_stack_copy

        other_alt_bound = bigfloat.BigFloat((S + 1)**2, context=bf_context)
        while len(over_one_stack) > 0 or len(under_one_stack) > 0:
            if len(over_one_stack) == 0:
                other_alt_bound *= under_one_stack.pop()
                continue
            if len(under_one_stack) == 0:
                other_alt_bound *= over_one_stack.pop()
                continue
            a = over_one_stack[-1]
            b = under_one_stack[-1]
            a_diff = bigfloat.abs(1.0 - (other_alt_bound * a))
            b_diff = bigfloat.abs(1.0 - (other_alt_bound * b))
            if a_diff < b_diff:
                other_alt_bound *= over_one_stack.pop()
            else:
                other_alt_bound *= under_one_stack.pop()
        if C1 + C2 > 0.0:
            print("Other-Alt-Bound for overlap of %d: %f (%f)" % (o, other_alt_bound / (1.0 + other_alt_bound), other_alt_bound))
        other_alt_bound = other_alt_bound / (1.0 + other_alt_bound)

        big_bound *= other_alt_bound
        if big_bound < 0.0001:
            print("Big bound < 0.0001. Returning False.")

    print("Total big bound: %f" % big_bound)

    result = big_bound >= 0.0001
    print("Returning %s" % str(result))
    return result

# Input:
#   S -- total number of samples taken
#   c1 -- one count value
#   c2 -- another count value
#
# Output:
#   A conditional probability p as favorable to saying the counts come from the
#       same distribution, such that, in effect:
#       Prob(get this result | come from same distribution) <= p
#
#   Thus if p is very low this can be used to "suggest" that the distributions
#       are different.
def prob_same_bound_favorable_to_saying_same(S, c1, c2):
    cmax = max(c1, c2)
    cmin = min(c1, c2)
    pmin = (S*(1.0 + 2*cmin) + math.sqrt(S**2 * (1 + 2*cmin)**2 - 4*(S**2 + S) * cmin**2)) / \
            (2*(S**2 + S))
    pmax = (S*(1.0 + 2*cmax) - math.sqrt(S**2 * (1 + 2*cmax)**2 - 4*(S**2 + S) * cmax**2)) / \
            (2*(S**2 + S))
    # If bounds overlap, cannot do better than 1.0.
    if pmin >= pmax:
        return 1.0

    # Assume best at one of the bounds, where one of the 2 k's equals 1.
    pA = (S**2 * pmin**2 * (1.0 - pmin)**2) / ((cmin - S*pmin)**2 * (cmax - S*pmin)**2)
    pB = (S**2 * pmax**2 * (1.0 - pmax)**2) / ((cmin - S*pmax)**2 * (cmax - S*pmax)**2)
    pA_test = (S * pmin * (1.0 - pmin)) / (cmin - S*pmin)**2
    pB_test = (S * pmax * (1.0 - pmax)) / (cmax - S*pmax)**2
    # print("%f vs pmin test: %f" % (1.0, pA_test))
    # print("%f vs pmax test: %f" % (1.0, pB_test))
    pA_test_2 = (S * pmin * (1.0 - pmin)) / (cmax - S*pmin)**2
    pB_test_2 = (S * pmax * (1.0 - pmax)) / (cmin - S*pmax)**2
    # print("%f vs expected pA: %f" % (pA, pA_test * pA_test_2))
    # print("%f vs expected pB: %f" % (pB, pB_test * pB_test_2))
    return max(pA, pB)

# Note: Allows dicts with different total counts.
def jensen_shannon_divergence_for_counts(count_dict_1, count_dict_2):

    dict_1_total = sum([count for key, count in count_dict_1.items()])
    dict_2_total = sum([count for key, count in count_dict_2.items()])

    dict_1_probs = {key: count / float(dict_1_total) for \
        key, count in count_dict_1.items()}
    dict_2_probs = {key: count / float(dict_2_total) for \
        key, count in count_dict_2.items()}

    return jensen_shannon_divergence(dict_1_probs, dict_2_probs)

def jensen_shannon_divergence(prob_dict_1, prob_dict_2):

    all_keys = set([key for key, prob in prob_dict_1.items()] + \
                   [key for key, prob in prob_dict_2.items()])

    m_probs = {}
    for key in all_keys:
        p1 = 0.0
        p2 = 0.0
        if key in prob_dict_1:
            p1 = prob_dict_1[key]
            assert p1 <= 1.0
        if key in prob_dict_2:
            p2 = prob_dict_2[key]
            assert p2 <= 1.0
        m_probs[key] = (p1 + p2) / 2.0

    JSD = 0.0
    for key, p in prob_dict_1.items():
        if p == 0.0:
            continue
        JSD += 0.5 * p * math.log(p / m_probs[key], 2.0)
    for key, p in prob_dict_2.items():
        if p == 0.0:
            continue
        JSD += 0.5 * p * math.log(p / m_probs[key], 2.0)

    return JSD
