import bigfloat
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from random_bijections import *

# def bigfloat_prob_of_count_given_p(C, p, S):

def __best_m_and_evidence_strength_helper__(P_X_given_N, P_N, n, m):
    assert P_N != 1.0
    assert P_N != 0.0
    P_not_N = 1.0 - P_N
    chance_correct = (P_not_N * (1.0 - (n - 1) / m))
    if_correct = (P_X_given_N * P_N) / (P_X_given_N * P_N + (1.0 / m) * P_not_N)
    alpha = chance_correct * if_correct + (1.0 - chance_correct) * P_N
    return ((1.0 - alpha) * P_N) / (alpha * P_not_N)

def best_m_and_evidence_strength(C, coin_prob, n, P_N):
    P_X_given_N = bigfloat_prob_of_count_given_p(C, coin_prob, n - 1)
    if P_X_given_N >= 1.0 / n:
        return (None, 1.0)
    func = (lambda x: lambda y: 1.0 / __best_m_and_evidence_strength_helper__(x[0], x[1], x[2], y))((P_X_given_N, P_N, n))
    # return (best_arg, best_func)
    (best_m, best_func_val) = min_finder(func, n - 1)

    return (best_m, 1.0 / best_func_val)

# func must be convex
def min_finder(func, low_arg, tol=bigfloat.BigFloat(2.0**(-30))):
    prev_prev_arg = low_arg
    prev_arg = low_arg
    curr_arg = low_arg

    prev_func = func(curr_arg)
    curr_func = prev_func

    while curr_func <= prev_func:
        prev_prev_arg = prev_arg
        prev_arg = curr_arg
        curr_arg *= 2.0

        prev_func = curr_func
        curr_func = func(curr_arg)

    # return (best_arg, best_func)
    return binary_min_finder(func, low=prev_prev_arg, high=curr_arg, tol=tol)

if __name__ == "__main__":
    # Parameters for plots

    bf_context = bigfloat.Context(precision=2000, emax=100000, emin=-100000)
    bigfloat.setcontext(bf_context)

    start_P_N = bigfloat.pow(2.0, -20)
    end_P_N = bigfloat.BigFloat(1.0) - start_P_N
    # start_P_N = bigfloat.BigFloat(1.0) / 3.0
    # end_P_N = bigfloat.BigFloat(2.0) / 3.0
    P_N_increment = (end_P_N - start_P_N) / 10.0

    coin_prob = 0.5

    S = 10

    start_C = 0
    end_C = S
    C_increment = 1

    evidence_vals = []
    C = start_C
    C_vals = []
    while C <= end_C:
        C_vals.append(float(C))
        evidence_vals.append([])

        P_N_vals = []
        P_N = start_P_N
        while P_N <= end_P_N:
            P_N_vals.append(float(P_N))

            print("C: %d -- P_N: %f" % (C, P_N))
            (best_m, best_evidence_strength) = best_m_and_evidence_strength(C, coin_prob, S + 1, P_N)
            if best_m is None:
                best_m = "N/A"
            else:
                best_m = str(float(best_m))
            print("    Best m: %s -- Best evidence strength: %f" % (best_m, best_evidence_strength))
            evidence_vals[-1].append(float(best_evidence_strength))

            P_N += P_N_increment
        C += C_increment

    # Plotting Data

    # Have the "major" axis be C_vals and the "secondary" axis be P_N_vals
    P_N_vals_2d = np.array([P_N_vals for _ in C_vals])
    C_vals_2d = np.array([[C_vals[i] for _ in P_N_vals] for i in range(0, len(C_vals))])

    evidence_vals = np.array(evidence_vals)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(C_vals_2d, P_N_vals_2d, evidence_vals, cmap=cm.coolwarm,
               linewidth=0, antialiased=False)

    plt.suptitle("Log of Strength of Evidence Against Null (N)")
    plt.xlabel("Number of Heads H")
    plt.ylabel("Prior Value, P(N)")
    plt.title("P(N | not-H) / P(N | H)")

    plt.show()


    """
    num_m_values = 10
    start_m = bigfloat.BigFloat(100) * S
    end_m = 3.0 * start_m
    m_increment = (end_m - start_m) / (num_m_values - 1)

    axis_to_exclude = "m"

    # Collecting Data

    if axis_to_exclude == "m":
        m = start_m
        while m <= end_m:
            outer_bound_vals = []
            inner_bound_vals = []

            C = start_C
            C_vals = []
            while C <= end_C:
                C_vals.append(float(C))
                outer_bound_vals.append([])
                inner_bound_vals.append([])

                P_N_vals = []
                P_N = start_P_N
                while P_N <= end_P_N:
                    P_N_vals.append(float(P_N))

                    p = bigfloat_prob_of_count_given_p(C, coin_prob, S)
                    inner_bound = p / (p * P_N + (1.0 / m) * (1.0 - P_N))
                    outer_bound = (1.0 - P_N) * (1 - (S / m))

                    inner_bound_vals[-1].append(float(inner_bound))
                    outer_bound_vals[-1].append(float(outer_bound))

                    P_N += P_N_increment
                C += C_increment

            # Plotting Data

            # Have the "major" axis be C_vals and the "secondary" axis be P_N_vals
            P_N_vals_2d = np.array([P_N_vals for _ in C_vals])
            C_vals_2d = np.array([[C_vals[i] for _ in P_N_vals] for i in range(0, len(C_vals))])

            evidence_factor_vals = np.array([[math.log(1.0 / v) for v in inner_bound_vals_row] for inner_bound_vals_row in inner_bound_vals])
            inner_bound_vals = np.array(inner_bound_vals)
            outer_bound_vals = np.array(outer_bound_vals)

            fig = plt.figure()
            ax = fig.gca(projection='3d')

            surf = ax.plot_surface(C_vals_2d, P_N_vals_2d, evidence_factor_vals, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

            plt.suptitle("Strength of Evidence Against Null (N) _IF_ Actually Evidence (m = %d)" % m)
            plt.xlabel("Number of Heads H")
            plt.ylabel("Prior Value, P(N)")
            plt.title("log(P(N) / P(N | H))")

            plt.show()

            fig = plt.figure()
            ax = fig.gca(projection='3d')

            surf = ax.plot_surface(C_vals_2d, P_N_vals_2d, outer_bound_vals, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

            plt.suptitle("Confidence the Other Results were Actually Evidence (m = %d)" % m)
            plt.xlabel("Number of Heads H")
            plt.ylabel("Prior Value, P(N)")

            plt.show()

            m += m_increment
    """
