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

def overlap_comparison_generic_bound(G1, G2):
    print("Running overlap comparison with the generic bound...")
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
            bound = saved_results[(C1, C2)]
        elif (C2, C1) in saved_results:
            bound = saved_results[(C2, C1)]
        else:
            bound = bigfloat_03_fast_bound_estimate(C1, C2, S)

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

def overlap_comparison_bayesian_bound(G1, G2):
    print("Running overlap comparison with bayesian bound...")
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
            bound = saved_results[(C1, C2)]
        elif (C2, C1) in saved_results:
            bound = saved_results[(C2, C1)]
        else:
            (bound, diagnostic) = bigfloat_04_fast_bound_estimate(C1, C2, S)

            if C1 + C2 > 0.0:
                print("Bound for overlap of %d with C1 = %d, C2 = %d, diagnostic = %f: %f" % \
                    (o, C1, C2, diagnostic, bound))

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

    # print("  Computing prob of count %d given prob %f..." % (C1, p))
    bound = bigfloat_prob_of_count_given_p(C1, p, S)
    # print("  Computing prob of count %d given prob %f..." % (C1, p))
    bound *= bigfloat_prob_of_count_given_p(C2, p, S)
    # print("  Remainder of bound...")
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

    # print("  Computing prob of count %d given prob %f..." % (C1, p))
    bound = bigfloat_fast_prob_of_count_given_p(C1, p, S)
    # print("  Computing prob of count %d given prob %f..." % (C1, p))
    bound *= bigfloat_fast_prob_of_count_given_p(C2, p, S)
    # print("  Remainder of bound...")
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

    # print("  Computing prob of count %d given prob %f..." % (C1, p))
    bound = bigfloat_fast_prob_of_count_given_p(C1, p, S)
    # print("  Computing prob of count %d given prob %f..." % (C1, p))
    bound *= bigfloat_fast_prob_of_count_given_p(C2, p, S)
    # print("  Remainder of bound...")
    bound *= 0.5  # Prior that prob(same) = 0.5

    assert float(bound) != float('nan')

    if bound >= 1.0:
        return (bigfloat.BigFloat(1.0), bound)

    diagnostic_value = bound
    bound = bigfloat_04_fast_balancer(bound, S)

    if bound >= 1.0:
        return (bigfloat.BigFloat(1.0), diagnostic_value)

    return (bound, diagnostic_value)

hacky_array_for_speedy_code = []

def bigfloat_04_fast_balancer(value, S):
    upper = int(S / 2) + 1
    lower = 0
    mid = lower + int((upper - lower) / 2)

    results = hacky_array_for_speedy_code
    if len(results) < upper:
        print("THIS SHOULD ONLY HAPPEN ONCE!")
        while len(results) < upper:
            results.append(None)

    best_k = None
    best_diff = None 
    while lower + 1 < upper:
        k = mid
        if results[k] is None:
            results[k] = bigfloat_04_fast_bound_pair(value, k, S)
        (inner_bound_const, outer_bound) = results[k]
        # print("For a value of %f, k of %d, we get: (%f, %f)" % (value, k, inner_bound, outer_bound))

        inner_strength = 1.0 - (value * inner_bound_const)
        outer_strength = outer_bound
        diff = bigfloat.abs(inner_strength - outer_strength)

        if diff == 0.0:
            return value * inner_bound_const

        if best_diff is None or diff < best_diff:
            best_k = k
            best_diff = diff

        if inner_strength > outer_strength:
            # Need to decrease k to make outer bound stronger.
            upper = mid
        else:
            # Need to increase k to make inner bound stronger.
            lower = mid
        mid = lower + int((upper - lower) / 2)
        
    one = bigfloat.BigFloat(1.0)
    print("      Best k = %d" % best_k)
    (inner_bound_const, outer_bound) = results[best_k]
    return bigfloat.max(bigfloat.min(one, value * inner_bound_const), 1.0 - outer_bound)

def bigfloat_04_fast_bound_pair(value, k, S):
    k = bigfloat.BigFloat(k)
    e = bigfloat.exp(1.0)
    pi = bigfloat.const_pi()
    inner_bound_const = bigfloat.BigFloat(1.0)
    inner_bound_const *= bigfloat.sqrt(2.0 * pi)
    inner_bound_const *= e * e
    inner_bound_const *= bigfloat_fast_exact_pow(0.5 * S, S)
    inner_bound_const /= bigfloat_fast_exact_pow(k, k)
    inner_bound_const /= bigfloat_fast_exact_pow(S - k, S - k)
    inner_bound_const *= bigfloat.sqrt(S / (k * (S - k)))
    inner_bound_const *= inner_bound_const
    inner_bound_const = 1.0 / inner_bound_const

    outer_bound = bigfloat_fast_exact_pow(0.5 * (S / k), k)
    outer_bound *= bigfloat_fast_exact_pow(0.5 / (1.0 - (k / S)), S - k)
    outer_bound = 1.0 - (2.0 * outer_bound)
    outer_bound *= outer_bound

    return (inner_bound_const, outer_bound)

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

def bigfloat_fast_choose_large(A, B):
    A = bigfloat.BigFloat(A)
    B = bigfloat.BigFloat(B)
    if B == 0.0 or B == A:
        return bigfloat.BigFloat(1.0)
    if B == 1.0 or B == (A - 1.0):
        return A

    pi = bigfloat.const_pi()
    e = bigfloat.exp(1.0)
    first_part = bigfloat.sqrt(A / (B * (A - B))) * (e / (2.0 * pi))
    second_part = bigfloat_fast_exact_pow(A / B, B)
    third_part = bigfloat_fast_exact_pow(A / (A - B), A - B)
    return first_part * second_part * third_part

def bigfloat_fast_choose_small(A, B):
    A = bigfloat.BigFloat(A)
    B = bigfloat.BigFloat(B)
    if B == 0.0 or B == A:
        return bigfloat.BigFloat(1.0)
    if B == 1.0 or B == (A - 1.0):
        return A

    pi = bigfloat.const_pi()
    e = bigfloat.exp(1.0)
    first_part = bigfloat.sqrt(2.0 * pi * A / (B * (A - B))) / (e * e)
    second_part = bigfloat_fast_exact_pow(A / B, B)
    third_part = bigfloat_fast_exact_pow(A / (A - B), A - B)
    return first_part * second_part * third_part

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
    if Y == 0.0:
        return bigfloat.BigFloat(1.0)
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

    # print("    Computing %f^%d..." % (p, C))
    prob = bigfloat.pow(p, C)
    # Check to see if the bigfloat ran out of resolution:
    if zero == prob:
        print("Not enough bigfloat bits for pow(%f, %d). Using slow method..." % (p, C))
        return bigfloat_slow_prob_of_count_given_p(C, p, S)

    # print("    Computing %d choose %d..." % (S, C))
    prob *= bigfloat_choose(S, C)
    # Check to see if the bigfloat ran out of resolution:
    if bigfloat.is_inf(prob):
        print("Not enough bigfloat bits for %d choose %d. Using slow method..." % (S, C))
        print(prob.precision)
        return bigfloat_slow_prob_of_count_given_p(C, p, S)

    # print("    Computing %f^%d" % (1.0 - p, S - C))
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
    bf_fc = bigfloat_fast_choose_large(S, C)
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

def test_A():
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

def test_sum_of_binomials():
    bf_context = bigfloat.Context(precision=1020, emax=100000, emin=-100000)
    bigfloat.setcontext(bf_context)
    S = 200

    centered_offset = bigfloat.BigFloat(1.0) / bigfloat.BigFloat(9.0)

    offset_1 = bigfloat.BigFloat(1.0)  / bigfloat.BigFloat(10.0)
    offset_2 = bigfloat.BigFloat(-1.0) / bigfloat.BigFloat(8.0)

    p1 = 0.5 + centered_offset
    p2 = 0.5 - centered_offset
    p3 = 0.5 + offset_1
    p4 = 0.5 + offset_2

    probs_centered = [0.5 * bigfloat_prob_of_count_given_p(i, p1, S) + \
                      0.5 * bigfloat_prob_of_count_given_p(i, p2, S) for i in range(0, S + 1)]
    probs_offset = [0.5 * bigfloat_prob_of_count_given_p(i, p3, S) + \
                    0.5 * bigfloat_prob_of_count_given_p(i, p4, S) for i in range(0, S + 1)]

    x_axis = [i for i in range(0, S + 1)]
    plt.scatter(x_axis, probs_centered)
    plt.scatter(x_axis, probs_offset)
    plt.title("S = %d" % S)
    plt.show()

    plt.close()  # .clf()

    sorted_probs_lists = [sorted(probs_centered), sorted(probs_offset)]
    cdf_pairs = [[], []]
    for arr_idx in range(0, 2):
        sorted_probs = sorted_probs_lists[arr_idx]
        cdf = cdf_pairs[arr_idx]
        total_here_or_less = bigfloat.BigFloat(0.0)
        total_here_or_more = bigfloat.BigFloat(1.0)
        for i in range(0, len(sorted_probs)):
            if i == 0:
                cdf.append((bigfloat.BigFloat(0.0), bigfloat.BigFloat(0.0), bigfloat.BigFloat(1.0)))
            elif sorted_probs[i] != sorted_probs[i - 1]:
                cdf.append((sorted_probs[i - 1], total_here_or_less, total_here_or_more))
                total_here_or_more = 1.0 - total_here_or_less

            total_here_or_less += sorted_probs[i]

        cdf.append((sorted_probs[-1], total_here_or_less, total_here_or_more))
        cdf.append((bigfloat.BigFloat(1.0), bigfloat.BigFloat(1.0), bigfloat.BigFloat(0.0)))

    cdf_centered = cdf_pairs[0]
    cdf_offset = cdf_pairs[1]
    x_axis_centered = [cdf_centered[i][0] for i in range(0, len(cdf_centered))]
    y_axis_centered = [cdf_centered[i][1] for i in range(0, len(cdf_centered))]
    x_axis_offset = [cdf_offset[i][0] for i in range(0, len(cdf_offset))]
    y_axis_offset = [cdf_offset[i][1] for i in range(0, len(cdf_offset))]
    plt.plot(x_axis_centered, y_axis_centered)
    plt.plot(x_axis_offset, y_axis_offset)
    plt.title("S = %d" % S)
    plt.show()

    # Now, check to find the highest p_thresh such that:
    #   Forall p_t <= p_thres: P(P(x) >= p_t | 50-50) <= P(P(x) >= p_t | offset)
    next_centered_idx = 0
    next_offset_idx = 0

    highest_pthresh_geq = bigfloat.BigFloat(0.0)  # geq is for the inner inequality to be '>=' as in above comment

    p_pt_centered_geq = bigfloat.BigFloat(1.0)
    p_pt_offset_geq = bigfloat.BigFloat(1.0)
    pt_centered_prev = bigfloat.BigFloat(0.0)
    pt_offset_prev = bigfloat.BigFloat(0.0)

    start_p_thresh = bigfloat.BigFloat(1.0) / bigfloat.BigFloat(S * S)

    while next_centered_idx < len(cdf_centered) and next_offset_idx < len(cdf_offset):
        pt_centered = cdf_centered[next_centered_idx][0]
        p_pt_centered_gr = 1.0 - cdf_centered[next_centered_idx][1]
        pt_offset = cdf_offset[next_offset_idx][0]
        p_pt_offset_gr = 1.0 - cdf_offset[next_offset_idx][1]

        pt_in_question = bigfloat.min(pt_centered, pt_offset)

        if pt_centered >= pt_offset and p_pt_centered_geq > p_pt_offset_geq and highest_pthresh_geq > start_p_thresh:
            print("First trailing false > %f with:" % start_p_thresh)
            print("  pt_centered of        %s" % pt_centered)
            print("  pt_offset of       %s" % pt_offset)
            print("  pt_centered_prev of   %s" % pt_centered_prev)
            print("  pt_offset_prev of  %s" % pt_offset_prev)
            print("  p_pt_centered_geq of  %s" % p_pt_centered_geq)
            print("  p_pt_offset_geq of %s" % p_pt_offset_geq)
            print("  p_pt_centered_gr of   %s" % p_pt_centered_gr)
            print("  p_pt_offset_gr of  %s" % p_pt_offset_gr)
            break
            
        highest_pthresh_geq = pt_in_question

        if pt_centered == pt_offset:
            print("A")
            next_centered_idx += 1
            next_offset_idx += 1

            pt_centered_prev = pt_centered
            pt_offset_prev = pt_offset
            p_pt_centered_geq = p_pt_centered_gr
            p_pt_offset_geq = p_pt_offset_gr
        elif pt_centered < pt_offset:
            print("B")
            next_centered_idx += 1
            pt_centered_prev = pt_centered
            p_pt_centered_geq = p_pt_centered_gr
        else:
            print("C")
            next_offset_idx += 1
            pt_offset_prev = pt_offset
            p_pt_offset_geq = p_pt_offset_gr


    print("    The highest p_thresh we found such that:")
    print("        Forall p_t <= p_thresh: P(P(x) >= p_t | 50-50) <= P(P(x) >= p_t | offset)")
    print("    was p_thresh = %s" % highest_pthresh_geq)

def one_over_S_binomial_bound_test():
    bf_context = bigfloat.Context(precision=2000, emax=100000, emin=-100000)
    bigfloat.setcontext(bf_context)
    S = 200

    threshold = bigfloat.BigFloat(1.0) / bigfloat.BigFloat(2 * S)

    print("Compare to confidence of %f" % (1.0 - (S - 1) * threshold))

    fixed_second_p = bigfloat.BigFloat(0.4)

    for p1 in [1.0, 0.9, 0.8, 0.7, 0.65, 0.6, 0.55, 0.5]:
        p1 = bigfloat.BigFloat(p1)
        # p2 = 1.0 - p1
        p2 = fixed_second_p
        lower_bound = bound_on_odds_at_least_threshold_with_shared_binoms_limited(threshold, p1, p2, S)
        print("Lower bound for threshold %f with probs %f, %f: %s" % (threshold, p1, p2, lower_bound))

    p1_vals = []
    bound_vals = []

    fixed_second_p = 1.0 - fixed_second_p

    num_vals = 10000

    print("\nPercent done:\n")

    for i in range(0, num_vals + 1):
        p = bigfloat.BigFloat(i) / bigfloat.BigFloat(num_vals)
        print("%s" % str(p * 100)[:5])
        p1_vals.append(p)
        bound_vals.append(bound_on_odds_at_least_threshold_with_shared_binoms(threshold, p, fixed_second_p, S))


    """
    low = bigfloat.BigFloat(0.0)
    high = bigfloat.BigFloat(0.5)
    mid = (low + high) / 2.0
    p1_vals.append(low)
    # bound_vals.append(bound_on_odds_at_least_threshold_with_shared_binoms_limited(threshold, low, 1.0 - low, S))
    bound_vals.append(bound_on_odds_at_least_threshold_with_shared_binoms_limited(threshold, low, fixed_second_p, S))
    p1_vals.append(high)
    # bound_vals.append(bound_on_odds_at_least_threshold_with_shared_binoms_limited(threshold, high, 1.0 - high, S))
    bound_vals.append(bound_on_odds_at_least_threshold_with_shared_binoms_limited(threshold, high, fixed_second_p, S))
    p1_vals.append(mid)
    # bound_vals.append(bound_on_odds_at_least_threshold_with_shared_binoms_limited(threshold, mid, 1.0 - mid, S))
    bound_vals.append(bound_on_odds_at_least_threshold_with_shared_binoms_limited(threshold, mid, fixed_second_p, S))

    mid_score = bound_vals[-1]

    min_gap = 0.00001
    while high - low > min_gap:
        print("Remaining Gap: %f" % ((high - low) - min_gap))
        candidate_mid_low = (low + mid) / 2.0
        candidate_mid_high = (mid + high) / 2.0
        # score_mid_low = bound_on_odds_at_least_threshold_with_shared_binoms_limited(threshold, candidate_mid_low, 1.0 - candidate_mid_low, S)
        # score_mid_high = bound_on_odds_at_least_threshold_with_shared_binoms_limited(threshold, candidate_mid_high, 1.0 - candidate_mid_high, S)
        score_mid_low = bound_on_odds_at_least_threshold_with_shared_binoms_limited(threshold, candidate_mid_low, fixed_second_p, S)
        score_mid_high = bound_on_odds_at_least_threshold_with_shared_binoms_limited(threshold, candidate_mid_high, fixed_second_p, S)

        p1_vals.append(candidate_mid_low)
        bound_vals.append(score_mid_low)
        p1_vals.append(candidate_mid_high)
        bound_vals.append(score_mid_high)

        if score_mid_low < mid_score and score_mid_high < mid_score:
            print("Explosion!")
        if score_mid_low < score_mid_high:
            mid_score = score_mid_low
            high = mid
            mid = candidate_mid_low
        else:
            mid_score = score_mid_high
            low = mid
            mid = candidate_mid_high

    print("And the winner is... %f" % mid)
    print("   ...with a bound of %f" % mid_score)
    """

    paired = sorted([(p1_vals[i], bound_vals[i]) for i in range(0, len(p1_vals))])
    p1_vals = [paired[i][0] for i in range(0, len(paired))]
    bound_vals = [paired[i][1] for i in range(0, len(paired))]
    plt.plot(p1_vals, bound_vals)
    plt.show()

def bound_on_odds_at_least_threshold_with_shared_binoms(thresh, p1, p2, S):
    probs = [0.5 * bigfloat_prob_of_count_given_p(i, p1, S) + \
             0.5 * bigfloat_prob_of_count_given_p(i, p2, S) for i in range(0, int(S) + 1)]
    total = bigfloat.BigFloat(0.0)
    for p in probs:
        if p >= thresh:
            total += p
    return total

def bound_on_odds_at_least_threshold_with_shared_binoms_limited(thresh, p1, p2, S):
    probs_1 = [0.5 * bigfloat_prob_of_count_given_p(i, p1, S) for i in range(0, int(S) + 1)]
    probs_2 = [0.5 * bigfloat_prob_of_count_given_p(i, p2, S) for i in range(0, int(S) + 1)]

    half_thresh = thresh / 2.0
    total = bigfloat.BigFloat(0.0)
    for i in range(0, int(S) + 1):
        prob_1 = probs_1[i]
        prob_2 = probs_2[i]
        if prob_1 >= thresh or prob_2 >= thresh or \
            (prob_1 >= half_thresh and prob_2 >= half_thresh):
            total += prob_1 + prob_2
    return total

def single_binomial_test():
    bf_context = bigfloat.Context(precision=2000, emax=100000, emin=-100000)
    bigfloat.setcontext(bf_context)

    p1 = bigfloat.BigFloat(1.0) / bigfloat.BigFloat(2.0)
    p2 = bigfloat.BigFloat(1.0) / bigfloat.BigFloat(3.0)

    S = 10

    thresholds = []
    p1_bounds = []
    p2_bounds = []

    thresholds_to_try = 10
    for t_idx in range(0, thresholds_to_try):
        print("##############")
        threshold = bigfloat.BigFloat(t_idx) / thresholds_to_try
        p1_bound = latest_and_greatest_binomial_outer_bound(threshold, p1, S)
        p2_bound = latest_and_greatest_binomial_outer_bound(threshold, p2, S)
        thresholds.append(threshold)
        p1_bounds.append(p1_bound)
        p2_bounds.append(p2_bound)

    plt.plot(thresholds, p1_bounds)
    plt.plot(thresholds, p2_bounds)
    plt.show()

def latest_and_greatest_binomial_outer_bound(thresh, p, S):
    thresh = bigfloat.BigFloat(thresh)
    p = bigfloat.BigFloat(p)
    S = bigfloat.BigFloat(S)

    fake_k = find_fake_k_for_thresh(thresh, p, S)

    outer_bound = bigfloat_fast_exact_pow(p * (S / fake_k), fake_k)
    outer_bound *= bigfloat_fast_exact_pow((1.0 - p) / (1.0 - (fake_k / S)), S - fake_k)
    return 1.0 - 2.0 * outer_bound

# Called "fake" because it may be a non-integer.
def find_fake_k_for_thresh(thresh, p, S):
    func = (lambda x: lambda y: bigfloat.abs(x[0] - bigfloat_fast_prob_of_count_given_p(y, x[1], x[2])))((thresh, p, S))
    (k, _) = binary_min_finder(func, 0, S / 2.0)
    return k

# Assumes function is convex
def binary_min_finder(func, low, high, tol=0.0001):
    print("Low: %f, High: %f" % (low, high))
    low = bigfloat.BigFloat(low)
    high = bigfloat.BigFloat(high)
    mid = low + ((high - low) / 2.0)

    low_func = func(low)
    high_func = func(high)
    mid_func = func(mid)

    low_func >= mid_func or high_func >= mid_func

    best_arg = mid
    best_func = mid_func

    while high - low > tol:
        print("  Remaining: %f" % ((high - low) - tol))
        left_mid = low + ((mid - low) / 2.0)
        right_mid = mid + ((high - mid) / 2.0)

        left_mid_func = func(left_mid)
        right_mid_func = func(right_mid)

        if left_mid_func < right_mid_func and left_mid_func < best_func:
            best_func = left_mid_func
            best_arg = left_mid
        elif right_mid_func < best_func:
            best_func = right_mid_func
            best_arg = right_mid

        if mid_func < left_mid_func and mid_func < right_mid_func:
            high = right_mid
            low = left_mid
            high_func = right_mid_func
            low_func = left_mid_func
        elif mid_func < left_mid_func:
            assert mid_func >= right_mid_func
            low = mid
            low_func = mid_func
            mid = right_mid
            mid_func = right_mid_func
        else:
            # assert mid_func >= left_mid_func
            high = mid
            high_func = mid_func
            mid = left_mid
            mid_func = right_mid_func

    return (best_arg, best_func)

if __name__ == "__main__":
    # one_over_S_binomial_bound_test()
    single_binomial_test()
