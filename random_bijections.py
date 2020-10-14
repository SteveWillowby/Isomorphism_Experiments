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
        return bigfloat.BigFloat(1.0)

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
