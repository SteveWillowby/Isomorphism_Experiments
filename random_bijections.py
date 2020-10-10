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

    exponent = 2
    num_measurements = len(G1.nodes())**exponent
    print("Taking 2 * V^%d = %d total samples."  % \
        (exponent, 2*num_measurements))

    S = float(num_measurements)

    max_bits_needed_in_worst_code = \
        int(math.ceil((S + 1) * math.log((S + 1), 2)))

    if max_bits_needed_in_worst_code > bigfloat.PRECISION_MAX:
        print("Warning - want to use more bits than bigfloat limit of %d." % \
            bigfloat.PRECISION_MAX)
    bits = min(max_bits_needed_in_worst_code, bigfloat.PRECISION_MAX)

    bf_context = bigfloat.Context(precision=bits, \
        emax=bits, emin=-bits)
    bigfloat.setcontext(bf_context)

    min_bound = bigfloat.BigFloat(1.0)

    G1G1_values = get_S_overlap_samples(num_measurements, G1, G1)
    G1G2_values = get_S_overlap_samples(num_measurements, G1, G2)

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

            bound = bigfloat_04_fast_bound_estimate(C1, C2, S)
            # bound_B = bigfloat_03_fast_bound_estimate(C1, C2, S)

            # print("Bound for C1 = %d, C2 = %d: %f" % (int(C1), int(C2), float(other_alt_bound)))

            if C1 + C2 > 0.0:
                print("Bound for overlap of %d with C1 = %d, C2 = %d: %f" % \
                    (o, C1, C2, bound))

            saved_results[(C1, C2)] = bound

        if bound < min_bound:
            min_bound = bound
        if bound < 0.0001:
            print("Bound %s < 0.0001. Returning False." % (float(min_bound)))
            return False

    print("Min bound: %f" % float(min_bound))

    result = bool(min_bound >= 0.0001)
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

def bigfloat_03_bound_estimate(C1, C2, S):
    C1_C2 = bigfloat.BigFloat(C1 + C2)
    S2x = bigfloat.BigFloat(2 * S)
    p = C1_C2 / S2x

    print("  Computing prob of count %d given prob %f..." % (C1, p))
    bound = bigfloat_prob_of_count_given_p(C1, p, S)
    print("  Computing prob of count %d given prob %f..." % (C1, p))
    bound *= bigfloat_prob_of_count_given_p(C2, p, S)
    print("  Remainder of bound...")
    bound *= 0.5  # Prior that prob(same) = 0.5

    assert float(bound) != float('nan')

    if bound >= 1.0:
        return bigfloat.BigFloat(1.0)

    m = bigfloat.max((S + 1)**2.0, bigfloat.sqrt(((S + 1.0)**2.0 - 1.0) / bound))

    bound = (((S + 1.0)**2.0 - 1.0) / m)

    if bound >= 1.0:
        return bigfloat.BigFloat(1.0)

    return bound

def bigfloat_03_fast_bound_estimate(C1, C2, S):
    C1_C2 = bigfloat.BigFloat(C1 + C2)
    S2x = bigfloat.BigFloat(2 * S)
    p = C1_C2 / S2x

    print("  Computing prob of count %d given prob %f..." % (C1, p))
    bound = bigfloat_fast_prob_of_count_given_p(C1, p, S)
    print("  Computing prob of count %d given prob %f..." % (C1, p))
    bound *= bigfloat_fast_prob_of_count_given_p(C2, p, S)
    print("  Remainder of bound...")
    bound *= 0.5  # Prior that prob(same) = 0.5

    assert float(bound) != float('nan')

    if bound >= 1.0:
        return bigfloat.BigFloat(1.0)

    m = bigfloat.max((S + 1)**2.0, bigfloat.sqrt(((S + 1.0)**2.0 - 1.0) / bound))

    bound = (((S + 1.0)**2.0 - 1.0) / m)

    if bound >= 1.0:
        return bigfloat.BigFloat(1.0)

    return bound

def bigfloat_04_fast_bound_estimate(C1, C2, S):
    C1_C2 = bigfloat.BigFloat(C1 + C2)
    S2x = bigfloat.BigFloat(2 * S)
    p = C1_C2 / S2x

    print("  Computing prob of count %d given prob %f..." % (C1, p))
    bound = bigfloat_prob_of_count_given_p(C1, p, S)
    print("  Computing prob of count %d given prob %f..." % (C1, p))
    bound *= bigfloat_prob_of_count_given_p(C2, p, S)
    print("  Remainder of bound...")
    bound *= 0.5  # Prior that prob(same) = 0.5

    assert float(bound) != float('nan')

    if bound >= 1.0:
        return bigfloat.BigFloat(1.0)

    bound = bigfloat_04_fast_balancer(bound, S)

    if bound >= 1.0:
        return bigfloat.BigFloat(1.0)

    return bound

def bigfloat_04_fast_balancer(value, S, min_denom=bigfloat.BigFloat(0.00000000001), num_values_tried=bigfloat.BigFloat(1000)):
    max_denom = value
    if min_denom > max_denom:
        temp = min_denom
        min_denom = max_denom
        max_denom = temp
    increment = bigfloat.abs(value - min_denom) / num_values_tried
    #print("Starting at value: %f" % max_denom)
    #print("Decreasing to %f in increments of %f" % (min_denom, increment))
    denom = min_denom
    best_diff = None
    best_balanced = None
    while denom < max_denom:
        #print("   %f" % denom)
        prob_x_over_denom = bigfloat_prob_counts_over_threshold_at_least(denom, S)
        diff = bigfloat.abs((1.0 - (value / denom)) - prob_x_over_denom)
        if best_diff is None:
            best_diff = diff
            best_balanced = 1.0 - prob_x_over_denom
        elif diff < best_diff:
            best_diff = diff
            best_balanced = 1.0 - prob_x_over_denom
        else:
            break
        denom += increment
    return best_balanced

def bigfloat_prob_counts_over_threshold_at_least(threshold, S):
    prob_single_count_over_t_at_least = \
        bigfloat_prob_count_over_threshold_at_least_slow(threshold, S)
    prob_both_counts_over_t_at_least = 1.0 - (1.0 - prob_single_count_over_t_at_least)**2.0
    return prob_both_counts_over_t_at_least

def bigfloat_prob_count_over_threshold_at_least_slow(threshold, S):
    total = bigfloat.BigFloat(0.0)
    C = 0
    main_pow = bigfloat_fast_exact_pow(bigfloat.BigFloat(0.5), S)
    assert main_pow > 0.0
    prob = main_pow * bigfloat_fast_choose(S, C)
    # TODO: Make this a binary search and then apply a bound formula.
    while prob < threshold:
        #print("                %d, %f" % (C, total))
        total += prob
        C += 1
        prob = main_pow * bigfloat_fast_choose(S, C)
    return 1.0 - (2.0 * total)

def bigfloat_fast_choose_large(A, B):
    return bigfloat_fast_factorial_large(A) / \
        (bigfloat_fast_factorial_small(B) * \
         bigfloat_fast_factorial_small(A - B))

def bigfloat_fast_choose_small(A, B):
    return bigfloat_fast_factorial_small(A) / \
        (bigfloat_fast_factorial_large(B) * \
         bigfloat_fast_factorial_large(A - B))

def bigfloat_fast_choose(A, B):
    A = bigfloat.BigFloat(A)
    B = bigfloat.BigFloat(B)
    if B == 0.0 or B == A:
        return bigfloat.BigFloat(1.0)
    if B == 1.0 or B == (A - 1.0):
        return A

    pi = bigfloat.const_pi()
    first_part = bigfloat.sqrt(A / (2.0 * pi * B * (A - B)))
    second_part = bigfloat_fast_exact_pow(A / B, B)
    third_part = bigfloat_fast_exact_pow(A / (A - B), A - B)
    return first_part * second_part * third_part

    return bigfloat_fast_factorial_small(A) / \
        (bigfloat_fast_factorial_small(B) * \
         bigfloat_fast_factorial_small(A - B))

# Uses Stirling's Approximation
def bigfloat_fast_factorial_large(X):
    e = bigfloat.exp(1.0)
    return e * bigfloat_fast_exact_pow(X, X + 0.5) * bigfloat_fast_exact_pow(e, -X)

# Uses Stirling's Approximation
def bigfloat_fast_factorial_small(X):
    e = bigfloat.exp(1.0)
    pi = bigfloat.const_pi()
    result = bigfloat.sqrt(2.0 * pi) * bigfloat_fast_exact_pow(X, X + 0.5) * bigfloat_fast_exact_pow(e, -X)
    return result

def bigfloat_fast_exact_pow(X, Y):
    sign = 1.0 - 2.0 * float(int(Y < 0.0))
    Y = bigfloat.abs(Y)
    base = bigfloat.floor(Y)
    extra = Y - base
    addendum = bigfloat.pow(X, sign * extra)
    if base == 0.0:
        return addendum
    exps = [bigfloat.BigFloat(1)]
    vals = [bigfloat.pow(X, sign)]
    while exps[-1] < base:
        exps.append(2 * exps[-1])
        vals.append(vals[-1] * vals[-1])
    total_result = addendum
    total_exp = bigfloat.BigFloat(0.0)
    for i in range(0, len(exps)):
        idx = len(exps) - (i + 1)
        exp = exps[idx]
        if total_exp + exp <= base:
            total_exp += exp
            total_result = total_result * vals[idx]
        if total_exp == base:
            break
    return total_result

def bigfloat_choose(A, B):
    A_choose_B = bigfloat.BigFloat(1.0)

    B = min(B, A - B)
    for i in range(0, int(B)):
        A_choose_B *= (A - i)
        A_choose_B /= (i + 1)

    return A_choose_B

def bigfloat_prob_of_count_given_p(C, p, S):
    assert float(p) <= 1.0
    assert float(C) <= float(S)

    zero = bigfloat.BigFloat(0.0)
    one = bigfloat.BigFloat(1.0)

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

    C = bigfloat.BigFloat(C)
    S = bigfloat.BigFloat(S)
    p = bigfloat.BigFloat(p)

    print("    Computing %f^%d..." % (p, C))
    prob = bigfloat.pow(p, C)
    # Check to see if the bigfloat ran out of resolution:
    if zero == prob:
        print("Not enough bigfloat bits for pow(%f, %d). Using slow method..." % (p, C))
        return bigfloat_slow_prob_of_count_given_p(C, p, S)

    print("    Computing %d choose %d..." % (S, C))
    prob *= bigfloat_choose(S, C)
    # Check to see if the bigfloat ran out of resolution:
    if bigfloat.is_inf(prob):
        print("Not enough bigfloat bits for %d choose %d. Using slow method..." % (S, C))
        print(prob.precision)
        return bigfloat_slow_prob_of_count_given_p(C, p, S)

    print("    Computing %f^%d" % (1.0 - p, S - C))
    prob *= bigfloat.pow(1.0 - p, S - C)
    # Check to see if the bigfloat ran out of resolution:
    if zero == prob:
        print("Not enough bigfloat bits for pow(1.0 - %f, %d). Using slow method..." % (1.0 - p, S - C))
        return bigfloat_slow_prob_of_count_given_p(C, p, S)

    if float(prob) > 1.0:
        print("Error! Got prob > 1.0 from params C = %f, p = %f, S = %f" % (C, p, S))
    assert 0.0 <= float(prob) and float(prob) <= 1.0
    assert float(prob) != float('inf') and float(prob) != float('-inf') and float(prob) != float('nan')

    return prob

def bigfloat_fast_prob_of_count_given_p(C, p, S):
    assert 0.0 <= p
    assert p <= 1.0
    assert float(C) <= float(S)

    zero = bigfloat.BigFloat(0.0)
    one = bigfloat.BigFloat(1.0)

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

    C = bigfloat.BigFloat(C)
    S = bigfloat.BigFloat(S)
    p = bigfloat.BigFloat(p)

    print("    Estimating %f^%d..." % (p, C))
    prob = bigfloat_fast_exact_pow(p, C)
    # Check to see if the bigfloat ran out of resolution:
    if zero == prob:
        print("Not enough bigfloat bits for pow(%f, %d). Using slow method..." % (p, C))
        return bigfloat_slow_prob_of_count_given_p(C, p, S)

    print("    Estimating %d choose %d..." % (S, C))
    bf_fc = bigfloat_fast_choose(S, C)
    assert bf_fc > 0
    prob *= bf_fc
    # Check to see if the bigfloat ran out of resolution:
    if bigfloat.is_inf(prob):
        print("Not enough bigfloat bits for %d choose %d. Using slow method..." % (S, C))
        print(prob.precision)
        return bigfloat_slow_prob_of_count_given_p(C, p, S)

    print("    Estimating %f^%d..." % (1.0 - p, S - C))
    prob *= bigfloat_fast_exact_pow(1.0 - p, S - C)
    # Check to see if the bigfloat ran out of resolution:
    if zero == prob:
        print("Not enough bigfloat bits for pow(1.0 - %f, %d). Using slow method..." % (1.0 - p, S - C))
        return bigfloat_slow_prob_of_count_given_p(C, p, S)

    if float(prob) > 1.0:
        print("Error! Got prob > 1.0 from params C = %f, p = %f, S = %f" % (C, p, S))
    assert 0.0 <= float(prob)
    assert float(prob) <= 1.1  # NOTE! 1.1 rather than 1.0 to allow for some "slop" in the probs.
    assert float(prob) != float('inf') and float(prob) != float('-inf') and float(prob) != float('nan')

    return prob

def bigfloat_slow_prob_of_count_given_p(C, p, S):
    assert float(p) <= 1.0
    assert float(C) <= float(S)

    # if bf_context is None:
    #     bf_context = bigfloat.Context(precision=200, emax=200, emin=-200)

    C = bigfloat.BigFloat(C)
    S = bigfloat.BigFloat(S)
    p = bigfloat.BigFloat(p)
    p_not = 1.0 - p

    C_min = bigfloat.min(C, S - C)
    under_one_vals = [p for i in range(0, int(C))] + \
                     [p_not for i in range(0, int(S) - int(C))]
    over_one_vals = [(S - i) / (C_min - (i + 1)) for i in range(0, int(C_min))]

    result = bigfloat.BigFloat(1.0)

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
    bf_context = bigfloat.Context(precision=1000, emax=100000, emin=-100000)
    bigfloat.setcontext(bf_context)
    two = bigfloat.BigFloat(2.0)
    print(two.precision)
    print(bigfloat.getcontext().emax)
    print(bigfloat.getcontext().emin)

    probs = [0.0, 1.0 / 8.0, 1.0 / 3.0, 1.0 / 2.0, 2.0 / 3.0, 7.0 / 8.0, 1.0]
    offsets = [0.75, -0.75, 0.5, -0.5, 0.125, -0.125, 1.0 / 256.0, -1.0 / 256, 1.0 / (2.0**30.0), -1.0 / (2.0**30.0)]
    sample_sizes = [10, 100, 1000, 10000]

    old_prob = None
    for sample_size in sample_sizes:

        # First, compute all "half-probs" that would be used for bounds.
        all_probs = set()
        for i in range(0, sample_size + 1):
            for j in range(i, sample_size + 1):
                all_probs.add(bigfloat.BigFloat(i + j) / \
                               bigfloat.BigFloat(2 * sample_size))

        # Second, add the probs that will be used directly.
        for prob in probs:
            p1 = bigfloat.BigFloat(prob)
            all_probs.add(p1)

            for offset in offsets:
                p2 = p1 + bigfloat.BigFloat(offset)
                if p2 > 1.0 or p2 < 0.0:
                    continue

                all_probs.add(p2)

        # Third, compute all conditional probs for values.
        conditional_probs = {}
        for prob in all_probs:
            for value in range(0, sample_size + 1):
                cp = bigfloat_prob_of_count_given_p(value, prob, sample_size)
                conditional_probs[(prob, value)] = cp

        # Fourth, actually get the bounds tests.
        for prob in probs:
            for offset in offsets:
                p1 = bigfloat.BigFloat(prob)
                p2 = p1 + bigfloat.BigFloat(offset)
                p1p1_total = bigfloat.BigFloat(0.0)
                p1p2_total = bigfloat.BigFloat(0.0)

                if p2 > 1.0 or p2 < 0.0:
                    continue

                for C1 in range(0, sample_size + 1):
                    for C2 in range(0, sample_size + 1):
                        mid_prob = bigfloat.BigFloat(C1 + C2) / \
                            bigfloat.BigFloat(2 * sample_size)

                        raw_bound = float((sample_size + 1)**2) * \
                            conditional_probs[(mid_prob, C1)] * \
                            conditional_probs[(mid_prob, C2)]

                        if 1.0 <= raw_bound:
                            p1p1_total += conditional_probs[(p1, C1)] * \
                                          conditional_probs[(p1, C2)]

                        real_raw_bound = conditional_probs[(p1, C2)] / \
                            conditional_probs[(p2, C2)]
                        if real_raw_bound <= raw_bound:
                            p1p2_total += conditional_probs[(p1, C1)] * \
                                          conditional_probs[(p2, C2)]

                if old_prob is None or old_prob != prob:
                    print("------------------------")
                    print("P1P1 Total where S = %d, p1 = %f: %s" % \
                        (sample_size, float(p1), str(p1p1_total)[0:15] + "..." + str(p1p1_total)[-7:]))
                    old_prob = prob

                print("P1P2 Total where S = %d, p1 = %f, p2 = p1 + %f: %s" % \
                    (sample_size, float(p1), float(offset), str(p1p2_total)[0:15] + "..." + str(p1p2_total)[-7:]))
