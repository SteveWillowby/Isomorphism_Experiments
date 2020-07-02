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

def xor_estimate(G1, G2, desired_confidence=0.99):
    G1 = zero_indexed_graph(G1)
    G2 = zero_indexed_graph(G2)
    epsilon = 1.0 - desired_confidence

    G1_nodes = list(G1.nodes())
    G2_nodes = list(G2.nodes())
    if len(G1_nodes) != len(G2_nodes):
        return False
    if len(G1.edges()) != len(G2.edges()):
        return False

    num_measurements = 0
    start_k = 5
    max_k = int(len(G1.edges()) * len(G1.edges()) / 2000)
    percent_so_called = int(max_k / 25)
    G1G1_values = []
    G1G2_values = []
    for k in range(start_k, max_k + 1):
        if k % percent_so_called == 0:
            print("%f percent done!" % (100.0 * k*k / (max_k*max_k)))
        k_squared = k * k
        while num_measurements < k_squared:
            G1G1_values.append(random_xor_count(G1, G1, G1_nodes))
            G1G2_values.append(random_xor_count(G1, G2, G2_nodes))
            num_measurements += 1

        means_and_variances_11 = \
            [mean_and_sample_variance(G1G1_values[k*i:(k*(i+1))]) \
                for i in range(0, k)]
        means_and_variances_12 = \
            [mean_and_sample_variance(G1G2_values[k*i:(k*(i+1))]) \
                for i in range(0, k)]

        means_11 = [x[0] for x in means_and_variances_11]
        means_12 = [x[0] for x in means_and_variances_12]
        sample_variances_11 = [x[1] for x in means_and_variances_11]
        sample_variances_12 = [x[1] for x in means_and_variances_12]
        if is_new_sample_excluded_by_dist(means_11, means_12[-1], epsilon):
            print("G1-G1 mean of %f excludes G1-G2 sample mean of %f" % \
                (float(sum(means_11)) / k, means_12[-1]))
            return False
        if is_new_sample_excluded_by_dist(means_12, means_11[-1], epsilon):
            print("G1-G2 mean of %f excludes G1-G1 sample mean of %f" % \
                (float(sum(means_12)) / k, means_11[-1]))
            return False
        if is_new_sample_excluded_by_dist(sample_variances_11, sample_variances_12[-1], epsilon):
            print("G1-G1 mean sample variance of %f excludes G1-G2 sample sample variance of %f" % \
                (float(sum(sample_variances_11)) / k, sample_variances_12[-1]))
            return False
        if is_new_sample_excluded_by_dist(sample_variances_12, sample_variances_11[-1], epsilon):
            print("G1-G2 mean sample variance of %f excludes G1-G1 sample sample variance of %f" % \
                (float(sum(sample_varainces_12)) / k, sample_variances_11[-1]))
            return False

        (mean_mean_11, mean_sv_11) = mean_and_sample_variance(means_11)
        diff = abs(mean_mean_11 - means_12[-1])
        print(abs(variance_based_on_threshold(diff, epsilon) - mean_sv_11))

    (mean_mean_11, mean_sv_11) = mean_and_sample_variance(means_11)
    (mean_mean_12, mean_sv_12) = mean_and_sample_variance(means_12)
    (sv_mean_11, sv_sv_11) = mean_and_sample_variance(sample_variances_11)
    (sv_mean_12, sv_sv_12) = mean_and_sample_variance(sample_variances_12)
    print("Found to be the same with means of %f (%f, need %f), %f (%f, need %f) and sample variances of %f (%f, need %f), %f (%f, need %f)" % \
        (mean_mean_11, mean_sv_11, variance_based_on_threshold(abs(mean_mean_11 - means_12[-1]), epsilon), \
         mean_mean_12, mean_sv_12, variance_based_on_threshold(abs(mean_mean_12 - means_11[-1]), epsilon), \
         sv_mean_11,   sv_sv_11,   variance_based_on_threshold(abs(mean_sv_11 - sample_variances_12[-1]), epsilon), \
         sv_mean_12,   sv_sv_12,   variance_based_on_threshold(abs(mean_sv_12 - sample_variances_11[-1]), epsilon)))

    return True

def random_xor_count(G1, G2, G2_nodes):
    p = np.random.permutation([i for i in range(0, len(G1.nodes()))])
    matches = 0
    for (a, b) in G1.edges():
        if G2.has_edge(G2_nodes[p[a]], G2_nodes[p[b]]):
            matches += 1
    return (len(G1.edges()) - matches) * 2

def mean_and_sample_variance(values):
    mean = float(sum(values)) / len(values)
    sample_variance = \
        float(sum([abs(v - mean) for v in values])) / (len(values) - 1)
    return (mean, sample_variance)

def is_new_sample_excluded_by_dist(values, new_sample, epsilon):
    (mean, sample_variance) = mean_and_sample_variance(values)
    dist = abs(mean - new_sample)
    threshold = threshold_based_on_variance(sample_variance, epsilon)
    return dist > threshold

def threshold_based_on_variance(variance, epsilon):
    return math.sqrt((1.0 / epsilon) * variance)

def variance_based_on_threshold(threshold, epsilon):
    return threshold * threshold * epsilon
