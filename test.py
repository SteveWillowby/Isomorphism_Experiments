from scipy.optimize import linprog

# IMPORTANT: By default bounds on a variable are 0 <= v < +inf

c = [-1, 4]
A = [[-3, 1], [1, 2], [0, -1]]
b = [6, 4, 3]
res = linprog(c, A_ub=A, b_ub=b, options={"disp": True})
print(res)