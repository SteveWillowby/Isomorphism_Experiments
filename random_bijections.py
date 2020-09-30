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

    exponent = 4
    num_measurements = len(G1.nodes())**exponent
    print("Taking 2 * V^%d = %d total samples."  % \
        (exponent, 2*num_measurements))
    G1G1_values = get_S_overlap_samples(num_measurements, G1, G1)
    G1G2_values = get_S_overlap_samples(num_measurements, G1, G2)

    S = float(num_measurements)

    max_bits_needed_in_worst_code = \
        int(math.ceil(4 * (S + 1) * math.log((S + 1), 2)))

    bf_context = bigfloat.Context(precision=max_bits_needed_in_worst_code)

    big_bound = bigfloat.BigFloat(1.0, context=bf_context)

    saved_results = {}

    for o in range(0, len(G1.edges())):
        C1 = float(G1G1_values[o])
        C2 = float(G1G2_values[o])

        if (C1, C2) in saved_results:
            print("Using saved result.")
            bound = saved_results[(C1, C2)]
        elif (C2, C1) in saved_results:
            print("Using saved result.")
            bound = saved_results[(C2, C1)]
        else:
            print("Computing result...")

            bound = bigfloat_03_bound_estimate(C1, C2, S, \
                bf_context=bf_context)

            # print("Bound for C1 = %d, C2 = %d: %f" % (int(C1), int(C2), float(other_alt_bound)))

            if C1 + C2 > 0.0:
                print("Bound for overlap of %d: %f (%f)" % (o, bound, bound / (1.0 - bound)))

            saved_results[(C1, C2)] = bound

        if bound < 0.5:
            big_bound *= bound
        if big_bound < 0.0001:
            print("Big bound %s < 0.0001. Returning False." % (float(big_bound)))
            return False

    print("Total big bound: %f" % float(big_bound))

    result = bool(big_bound >= 0.0001)
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
    pA_test_2 = (S * pmin * (1.0 - pmin)) / (cmax - S*pmin)**2
    pB_test_2 = (S * pmax * (1.0 - pmax)) / (cmin - S*pmax)**2
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

def bigfloat_03_bound_estimate(C1, C2, S, bf_context=None):
    if bf_context is None:
        bf_context = bigfloat.Context(precision=300)

    if True:
        C1_C2 = bigfloat.BigFloat(C1 + C2, context=bf_context)
        S2x = bigfloat.BigFloat(2 * S, context=bf_context)
        p = C1_C2 / S2x

        # print("C1 C2 = (%f, %f)" % (C1, C2))
        bound = bigfloat_prob_of_count_given_p(C1, p, S, bf_context=bf_context)
        # print("Bound p1: %s" % bound)
        bound *= bigfloat_prob_of_count_given_p(C2, p, S, bf_context=bf_context)
        # print("Bound p2: %s" % bound)
        bound *= bigfloat.BigFloat((S + 1)**2, context=bf_context)
        # print("Bound p3: %s" % bound)

        bound = bound / (1.0 + bound)
        # print("Bound p4: %s" % bound)
        assert float(bound) != float('nan')
        return bound

    if vals_under is None:
        C1 = float(C1)
        C2 = float(C2)
        S = float(S)
        vals_under = []
        vals_over = []

        C1_C2 = bigfloat.BigFloat(C1 + C2, context=bf_context)
        S2x = bigfloat.BigFloat(2.0 * S, context=bf_context)
        f1 = C1_C2 / S2x
        f2 = 1.0 - f1

        for i in range(int(C1), int(S)):
            val = f2 / \
                (bigfloat.BigFloat((i + 1) - C1, context=bf_context) / (i + 1))
            if val > 1.0:
                vals_over.append(val)
            else:
                vals_under.append(val)
        for i in range(int(C2), int(S)):
            val = f2 / \
                (bigfloat.BigFloat((i + 1) - C2, context=bf_context) / (i + 1))
            if val > 1.0:
                vals_over.append(val)
            else:
                vals_under.append(val)
        for i in range(0, int(C1)):
            val = f1  # / ((i + 1) / (i + 1))
            if val > 1.0:
                vals_over.append(val)
            else:
                vals_under.append(val)
        for i in range(0, int(C2)):
            val = f1  # / ((i + 1) / (i + 1))
            if val > 1.0:
                vals_over.append(val)
            else:
                vals_under.append(val)

    bound = bigfloat.BigFloat((S + 1)**2, context=bf_context)

    while len(vals_over) > 0 or len(vals_under) > 0:
        if len(vals_over) == 0:
            assert vals_under[-1] > 0.0
            bound *= vals_under.pop()
            continue
        if len(vals_under) == 0:
            assert vals_over[-1] > 0.0
            bound *= vals_over.pop()
            continue
        a = vals_over[-1]
        b = vals_under[-1]
        v1 = float(bound) * a
        v2 = float(bound) * b
        assert a > 0.0
        assert b > 0.0
        a_diff = abs(1.0 - v1)
        b_diff = abs(1.0 - v1)
        if a_diff < b_diff:
            bound *= vals_over.pop()
        else:
            bound *= vals_under.pop()
    bound = bound / (1.0 + bound)
    return bound

def bigfloat_choose(A, B, bf_context=None):
    if bf_context is None:
        bf_context = bigfloat.Context(precision=200)

    A_choose_B = bigfloat.BigFloat(1.0, context=bf_context)
    B = min(B, A - B)
    for i in range(0, int(B)):
        A_choose_B *= (A - i)
        A_choose_B /= (i + 1)

    return A_choose_B

def bigfloat_prob_of_count_given_p(C, p, S, bf_context=None):
    assert float(p) <= 1.0
    assert float(C) <= float(S)

    if bf_context is None:
        bf_context = bigfloat.Context(precision=200)

    zero = bigfloat.BigFloat(0.0, context=bf_context)
    one = bigfloat.BigFloat(1.0, context=bf_context)

    # Handle p == 0 and p == 1 with special cases due to pow issues.
    if p == zero:
        if int(C) == 0:
            return one
        else:
            return zero
    elif p == one:
        if int(C) == int(S):
            return one
        else:
            return zero

    C = bigfloat.BigFloat(C, context=bf_context)
    S = bigfloat.BigFloat(S, context=bf_context)
    p = bigfloat.BigFloat(p, context=bf_context)

    prob = bigfloat.pow(p, C)
    # Check to see if the bigfloat ran out of resolution:
    if zero == prob:
        print("Not enough bigfloat bits for pow(%f, %d). Using slow method..." % (p, C))
        return bigfloat_slow_prob_of_count_given_p(C, p, S, bf_context=bf_context)

    prob *= bigfloat_choose(S, C, bf_context=bf_context)
    # Check to see if the bigfloat ran out of resolution:
    if bigfloat.is_inf(prob):
        print("Not enough bigfloat bits for %d choose %d. Using slow method..." % (S, C))
        return bigfloat_slow_prob_of_count_given_p(C, p, S, bf_context=bf_context)

    prob *= bigfloat.pow(1.0 - p, S - C)
    # Check to see if the bigfloat ran out of resolution:
    if zero == prob:
        print("Not enough bigfloat bits for pow(1.0 - %f, %d). Using slow method..." % (1.0 - p, S - C))
        return bigfloat_slow_prob_of_count_given_p(C, p, S, bf_context=bf_context)

    if float(prob) > 1.0:
        print("Error! Got prob > 1.0 from params C = %f, p = %f, S = %f" % (C, p, S))
        assert float(prob) <= 1.0
    return prob

def bigfloat_slow_prob_of_count_given_p(C, p, S, bf_context=None):
    assert float(p) <= 1.0
    assert float(C) <= float(S)

    if bf_context is None:
        bf_context = bigfloat.Context(precision=200)

    C = bigfloat.BigFloat(C, context=bf_context)
    S = bigfloat.BigFloat(S, context=bf_context)
    p = bigfloat.BigFloat(p, context=bf_context)
    p_not = 1.0 - p

    C_min = bigfloat.min(C, S - C)
    under_one_vals = [p for i in range(0, int(C))] + \
                     [p_not for i in range(0, int(S) - int(C))]
    over_one_vals = [(S - i) / (C_min - (i + 1)) for i in range(0, int(C_min))]

    result = bigfloat.BigFloat(1.0, context=bf_context)

    while len(over_one_vals) > 0 or len(under_one_vals) > 0:
        if len(over_one_vals) == 0:
            result *= under_one_vals.pop()
            continue
        if len(under_one_vals) == 0:
            result *= over_one_vals.pop()
            continue
        a = over_one_vals[-1]
        b = under_one_vals[-1]
        v1 = float(result) * a
        v2 = float(result) * b
        assert a > 0.0
        assert b > 0.0
        a_diff = abs(1.0 - v1)
        b_diff = abs(1.0 - v1)
        if a_diff < b_diff:
            result *= over_one_vals.pop()
        else:
            result *= under_one_vals.pop()

    if float(result) > 1.0:
        print("Error! Got prob > 1.0 from params C = %f, p = %f, S = %f" % (C, p, S))
        assert float(result) <= 1.0

    return result

if __name__ == "__main__":
    bf_context = bigfloat.Context(precision=400)
    p1p1_total = bigfloat.BigFloat(0.0, context=bf_context)
    p1p2_total = bigfloat.BigFloat(0.0, context=bf_context)
    two = bigfloat.BigFloat(2.0, context=bf_context)

    p1 = bigfloat.BigFloat(1.0 / 3.0, context=bf_context)
    offset = bigfloat.pow(2.0, -1.0, context=bf_context)
    p2 = p1 + offset

    sample_size = 10
    for i in range(0, sample_size + 1):

        p1_i_prob = bigfloat_prob_of_count_given_p(i, p1, sample_size, \
            bf_context=bf_context)

        for j in range(0, sample_size + 1):
            # print((i, j))
            bound = bigfloat_03_bound_estimate(C1=i, C2=j, S=sample_size, \
                bf_context=bf_context)

            other_bound = bigfloat_03_bound_estimate(C1=j, C2=i, S=sample_size, \
                bf_context=bf_context)

            assert bound == other_bound

            p1_j_prob = bigfloat_prob_of_count_given_p(j, p1, sample_size, \
                bf_context=bf_context)

            p2_j_prob = bigfloat_prob_of_count_given_p(j, p2, sample_size, \
                bf_context=bf_context)

            if 0.5 <= bound:

                p1p1_total += p1_i_prob * p1_j_prob

            if (p1_j_prob / p2_j_prob) <= bound:

                p1p2_total += p1_i_prob * p2_j_prob

        print(float(i) / sample_size)

    print("P1P1 Total where S = %d, p1 = %f: %s" % (sample_size, float(p1), str(p1p1_total)))

    print("P1P2 Total where S = %d, p1 = %f, p2 = p1 + %f: %s" % (sample_size, float(p1), float(offset), str(p1p2_total)))
