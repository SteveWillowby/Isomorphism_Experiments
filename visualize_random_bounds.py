import bigfloat
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from random_bijections import *

# def bigfloat_prob_of_count_given_p(C, p, S):

if __name__ == "__main__":
    # Parameters for plots

    bf_context = bigfloat.Context(precision=2000, emax=100000, emin=-100000)
    bigfloat.setcontext(bf_context)

    start_P_N = bigfloat.pow(2.0, -20)
    end_P_N = bigfloat.BigFloat(1.0)
    P_N_increment = (end_P_N - start_P_N) / 100.0

    coin_prob = 0.5

    S = 100

    start_C = 0
    end_C = S
    C_increment = 1

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
