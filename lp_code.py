from scipy.optimize import linprog
import networkx as nx
from networkx import utils
from networkx.algorithms.bipartite.generators import configuration_model
from networkx.algorithms import isomorphism
from networkx.algorithms.shortest_paths.unweighted import all_pairs_shortest_path_length
from networkx.algorithms.components import is_connected
import numpy as np

# IMPORTANT: By default bounds on a variable are 0 <= v < +inf

# c = [-1, 4]
# A = [[-3, 1], [1, 2], [0, -1]]
# b = [6, 4, 3]
# res = linprog(c, A_ub=A, b_ub=b, options={"disp": True})
# print(res)


def lp_without_maxs_or_goal(N, SPG, SPG_prime, travel_min):
    A = []
    b = []
    for i in range(0, N):
        for j in range(i + 1, N):
            for k in range(0, N):
                for l in range(k + 1, N):
                    if SPG[i][j] >= travel_min and SPG_prime[k][l] >= travel_min:
                        new_constraint = [0 for n in range(0, N * N)]  # w_ik and w_jl
                        new_constraint[N * i + k] = -1
                        new_constraint[N * j + l] = -1
                        A.append(new_constraint)
                        b.append(-abs(SPG[i][j] - SPG_prime[k][l]))
                        new_constraint = [0 for n in range(0, N * N)]  # w_il and w_jk
                        new_constraint[N * i + l] = -1
                        new_constraint[N * j + k] = -1
                        A.append(new_constraint)
                        b.append(-abs(SPG[i][j] - SPG_prime[k][l]))
    return (A, b)

def maxs_matrix(N):
    A = []
    for i in range(0, N):
        new_constraint = [1 if N*i <= j < N * (i+1) else 0 for j in range(0, N*N)]
        A.append(new_constraint)
    for i in range(0, N):
        new_constraint = [1 if j % N == i else 0 for j in range(0, N * N)]
        A.append(new_constraint)
    return A

def goal_vector(N, target_min=None):
    c = [1 for n in range(0, N * N)]
    if target_min is not None:  # If there's one node (in G, not G_prime) we really want to minimize
        for i in range(0, N):
            c[N * target_min + i] += N
    return c

def goal_vector_prime(N, target_min=None):
    c = [1 for n in range(0, N * N)]
    if target_min is not None:  # If there's one node (in G, not G_prime) we really want to minimize
        for i in range(0, N):
            c[N * i + target_min] += N
    return c

def weight_caps(N, result):
    caps = [0.01 for n in range(0, N)]  # TODO: The 0.01 is for slack. It should probably be a function of N.
    for i in range(0, N * N):
        caps[int(i / N)] += result.x[i]
    return caps

def weight_caps_prime(N, result):
    caps = [0.01 for n in range(0, N)]  # TODO: The 0.01 is for slack. It should probably be a function of N.
    for i in range(0, N * N):
        caps[i % N] += result.x[i]
    return caps

def check_for_min_travel(N, SPG, SPG_prime, min_travel):
    # First run the graphs against themselves
    (AG, bG) = lp_without_maxs_or_goal(N, SPG, SPG, min_travel)
    c = goal_vector(N)
    G_with_G = linprog(c, A_ub=AG, b_ub=bG, method="interior-point", options={"disp":False, "maxiter":N*N*N*N*100})

    (AG_prime, bG_prime) = lp_without_maxs_or_goal(N, SPG_prime, SPG_prime, min_travel)
    G_prime_with_G_prime = linprog(c, A_ub=AG_prime, b_ub=bG_prime, method="interior-point", options={"disp":False, "maxiter":N*N*N*N*100})

    # First check: Are sums not equal?
    if abs(G_with_G.fun - G_prime_with_G_prime.fun) > 0.01:  # TODO: Replace this constant.
        return False
    print(G_with_G.fun)
    print(G_prime_with_G_prime.fun)

    print ("Going on to second round test.")
    # Then run them against each other, holding G's caps from before:
    (AGGP, bGGP) = lp_without_maxs_or_goal(N, SPG, SPG_prime, min_travel)
    A_weight_caps = maxs_matrix(N)
    A = AGGP + A_weight_caps
    b_G1_caps = weight_caps(N, G_with_G)
    b_G2_caps = [N*N for n in range(0, N)]
    b = bGGP + b_G1_caps + b_G2_caps
    G_with_G_prime = linprog(c, A_ub=A, b_ub=b, method="interior-point", options={"disp":False, "maxiter":N*N*N*N*100})

    # Second check: Are sums not equal or did it fail??
    if G_with_G_prime.status != 0 or abs(G_with_G.fun - G_with_G_prime.fun) > 0.01:  # TODO: Replace this constant.
        print("Second check failed.")
        #print(G_with_G_prime)
        return False

    # Use G_with_G_prime to get caps for G_prime, then see if it can use those with itself.
    A = AG_prime + A_weight_caps
    b_G1_caps = weight_caps_prime(N, G_with_G_prime)
    b = bG_prime + b_G1_caps + b_G2_caps
    G_prime_with_G_prime = linprog(c, A_ub=A, b_ub=b, method="interior-point", options={"disp":False, "maxiter":N*N*N*N*100})

    # Third check: Can G_prime match with itself using the limits from G_with_G_prime?
    if G_prime_with_G_prime.status != 0 or abs(G_with_G.fun - G_prime_with_G_prime.fun) > 0.01:  # TODO: Replace this constant.
        print("Third check failed.")
        return False

    return True #TODO: Remove this line.

    # Lastly, do the full operation.
    for i in range(0, N):
        # Measure G with itself where goal is to minimize sum_j w_ij
        c = goal_vector(N, i)
        A = AG + A_weight_caps
        b = bG + [N*N for n in range(0, 2*N)]
        G_with_G = linprog(c, A_ub=A, b_ub=b, method="interior-point", options={"disp":False, "maxiter":N*N*N*N*100})
        A = AGGP + A_weight_caps
        b_G1_caps = weight_caps(N, G_with_G)
        print(b_G1_caps[i])
        b = bGGP + b_G1_caps + b_G2_caps
        G_with_G_prime = linprog(c, A_ub=A, b_ub=b, method="interior-point", options={"disp":False, "maxiter":N*N*N*N*100})
        if G_with_G_prime.status != 0 or abs(G_with_G.fun - G_with_G_prime.fun) > 0.01:
            print("Fourth check failed.")
            return False

    # Need to add the reverse direction.

    return True

def lp_iso_check(G, G_prime):
    N = len(G.nodes())
    SPG = dict(all_pairs_shortest_path_length(G))
    SPG_prime = dict(all_pairs_shortest_path_length(G_prime))
    longest_sp = 0
    for i in range(0, N):
        for j in range(i + 1, N):
            if SPG[i][j] > longest_sp:
                longest_sp = SPG[i][j]
            if SPG_prime[i][j] > longest_sp:
                longest_sp = SPG_prime[i][j]
    for min_travel in range(0, longest_sp + 1):
        print("Round " + str(min_travel))
        if not check_for_min_travel(N, SPG, SPG_prime, min_travel):
            return False
    return True
