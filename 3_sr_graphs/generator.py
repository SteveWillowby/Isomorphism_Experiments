# This must be run with python 3!
from pulp import *
import sys

def graph_with_n_nodes(NUM_NODES=7, force_1_regular=True):

    problem = LpProblem("Generate_a_3SR_Graph", LpMinimize)
    potential_edges = []
    threes = []
    for i in range(0, NUM_NODES):
        for j in range(i+1, NUM_NODES):
            potential_edges.append((i,j))
            for k in range(j+1, NUM_NODES):
                threes.append((i,j,k))

    edge_vars = LpVariable.dicts("Potential_Edge_Vars", potential_edges, cat="Binary")
    if force_1_regular:
        degree = LpVariable.dicts("Degree", ["degree"], cat="Continuous")

    # Here to speed things up and/or enforce degree regularity.
    prev_edges = []
    for i in range(0, NUM_NODES):
        edges = []
        for j in range(0, NUM_NODES):
            if i != j:
                edges.append((min(i,j), max(i,j)))
        if force_1_regular:
            problem += degree["degree"] == lpSum({key: edge_vars[key] for key in edges})
        elif i > 0:
            problem += lpSum({key: edge_vars[key] for key in prev_edges}) <= lpSum({key: edge_vars[key] for key in edges})
        prev_edges = edges

    is_triangle = LpVariable.dicts("Triangle_Vars", threes, cat="Binary")
    is_wedge = LpVariable.dicts("Wedge_Vars", threes, cat="Binary")
    is_edge = LpVariable.dicts("Edge_Vars", threes, cat="Binary")
    is_empty = LpVariable.dicts("Empty_Vars", threes, cat="Binary")
    #is_triangle = LpVariable.dicts("Triangle_Vars", threes, cat="Continuous", lowBound=0, upBound=1)
    #is_wedge = LpVariable.dicts("Wedge_Vars", threes, cat="Continuous", lowBound=0, upBound=1)
    #is_edge = LpVariable.dicts("Edge_Vars", threes, cat="Continuous", lowBound=0, upBound=1)
    #is_empty = LpVariable.dicts("Empty_Vars", threes, cat="Continuous", lowBound=0, upBound=1)

    # Constraints to force triple vars to correspond to edge vars.
    for (i, j, k) in threes:
        edge_dict = {key: edge_vars[key] for key in [(i,j), (i,k), (j,k)]}
        problem += lpSum(edge_dict) == is_triangle[(i,j,k)] * 3 + is_wedge[(i,j,k)] * 2 + is_edge[(i,j,k)]*1
        problem += is_triangle[(i,j,k)] + is_wedge[(i,j,k)] + is_edge[(i,j,k)] + is_empty[(i,j,k)] == 1
        problem += is_empty[(i,j,k)] <= 1 - edge_vars[(i,j)]
        problem += is_empty[(i,j,k)] <= 1 - edge_vars[(i,k)]
        problem += is_empty[(i,j,k)] <= 1 - edge_vars[(j,k)]

    # Force number of each 3-node component to be non-zero
    problem += lpSum(is_triangle) >= 1
    problem += lpSum(is_wedge) >= 1
    problem += lpSum(is_edge) >= 1
    problem += lpSum(is_empty) >= 1

    fours = []
    for i in range(0, NUM_NODES):
        for j in range(i+1, NUM_NODES):
            for k in range(j+1, NUM_NODES):
                for l in range(0, i):
                    fours.append((l, (i, j, k)))
                for l in range(i+1, j):
                    fours.append((l, (i, j, k)))
                for l in range(j+1, k):
                    fours.append((l, (i, j, k)))
                for l in range(k+1, NUM_NODES):
                    fours.append((l, (i, j, k)))

    to_all_triangle = LpVariable.dicts("To_All_Triangle", fours, cat="Binary")
    to_two_triangle = LpVariable.dicts("To_Two_Triangle", fours, cat="Binary")
    to_one_triangle = LpVariable.dicts("To_One_Triangle", fours, cat="Binary")
    to_all_wedge = LpVariable.dicts("To_All_Wedge", fours, cat="Binary")
    to_1_1_wedge = LpVariable.dicts("To_1_1_Wedge", fours, cat="Binary")
    to_2_wedge = LpVariable.dicts("To_2_Wedge", fours, cat="Binary")
    to_in_wedge = LpVariable.dicts("To_In_Wedge", fours, cat="Binary")
    to_out_wedge = LpVariable.dicts("To_Out_Wedge", fours, cat="Binary")
    to_all_edge = LpVariable.dicts("To_All_Edge", fours, cat="Binary")
    to_1_1_edge = LpVariable.dicts("To_1_1_Edge", fours, cat="Binary")
    to_2_edge = LpVariable.dicts("To_2_Edge", fours, cat="Binary")
    to_in_edge = LpVariable.dicts("To_In_Edge", fours, cat="Binary")
    to_out_edge = LpVariable.dicts("To_Out_Edge", fours, cat="Binary")
    to_all_empty = LpVariable.dicts("To_All_Empty", fours, cat="Binary")
    to_two_empty = LpVariable.dicts("To_Two_Empty", fours, cat="Binary")
    to_one_empty = LpVariable.dicts("To_One_Empty", fours, cat="Binary")
    """
    to_all_triangle = LpVariable.dicts("To_All_Triangle", fours, cat="Continuous", lowBound=0, upBound=1)
    to_two_triangle = LpVariable.dicts("To_Two_Triangle", fours, cat="Continuous", lowBound=0, upBound=1)
    to_one_triangle = LpVariable.dicts("To_One_Triangle", fours, cat="Continuous", lowBound=0, upBound=1)
    to_all_wedge = LpVariable.dicts("To_All_Wedge", fours, cat="Continuous", lowBound=0, upBound=1)
    to_1_1_wedge = LpVariable.dicts("To_1_1_Wedge", fours, cat="Continuous", lowBound=0, upBound=1)
    to_2_wedge = LpVariable.dicts("To_2_Wedge", fours, cat="Continuous", lowBound=0, upBound=1)
    to_in_wedge = LpVariable.dicts("To_In_Wedge", fours, cat="Continuous", lowBound=0, upBound=1)
    to_out_wedge = LpVariable.dicts("To_Out_Wedge", fours, cat="Continuous", lowBound=0, upBound=1)
    to_all_edge = LpVariable.dicts("To_All_Edge", fours, cat="Continuous", lowBound=0, upBound=1)
    to_1_1_edge = LpVariable.dicts("To_1_1_Edge", fours, cat="Continuous", lowBound=0, upBound=1)
    to_2_edge = LpVariable.dicts("To_2_Edge", fours, cat="Continuous", lowBound=0, upBound=1)
    to_in_edge = LpVariable.dicts("To_In_Edge", fours, cat="Continuous", lowBound=0, upBound=1)
    to_out_edge = LpVariable.dicts("To_Out_Edge", fours, cat="Continuous", lowBound=0, upBound=1)
    to_all_empty = LpVariable.dicts("To_All_Empty", fours, cat="Continuous", lowBound=0, upBound=1)
    to_two_empty = LpVariable.dicts("To_Two_Empty", fours, cat="Continuous", lowBound=0, upBound=1)
    to_one_empty = LpVariable.dicts("To_One_Empty", fours, cat="Continuous", lowBound=0, upBound=1)
    """
    
    # Must be a triangle to be a to_all_triangle, etc.
    for four in fours:
        problem += is_triangle[four[1]] >= to_all_triangle[four] + to_two_triangle[four] + to_one_triangle[four]
        problem += is_empty[four[1]] >= to_all_empty[four] + to_two_empty[four] + to_one_empty[four]
        problem += is_wedge[four[1]] >= to_all_wedge[four] + to_1_1_wedge[four] + to_2_wedge[four] + to_in_wedge[four] + to_out_wedge[four]
        problem += is_edge[four[1]] >= to_all_edge[four] + to_1_1_edge[four] + to_2_edge[four] + to_in_edge[four] + to_out_edge[four]
    
    # Constraints to force one of the to_a_b's with the right number of edges if is_b is set.
    for four in fours:
        (l, (i, j, k)) = four
        li = (min(l, i), max(l, i))
        lj = (min(l, j), max(l, j))
        lk = (min(l, k), max(l, k))
        problem += (3 * to_all_triangle[four]) + (2 * to_two_triangle[four]) + (1 * to_one_triangle[four]) <= \
            edge_vars[li] + edge_vars[lj] + edge_vars[lk]
        problem += edge_vars[li] + edge_vars[lj] + edge_vars[lk] <= \
            (3 * to_all_triangle[four]) + (2 * to_two_triangle[four]) + (1 * to_one_triangle[four]) + 3 - (3*is_triangle[(i, j, k)])
    
        problem += 3 * to_all_empty[four] + 2 * to_two_empty[four] + 1 * to_one_empty[four] <= \
            edge_vars[li] + edge_vars[lj] + edge_vars[lk]
        problem += edge_vars[li] + edge_vars[lj] + edge_vars[lk] <= \
            3 * to_all_empty[four] + 2 * to_two_empty[four] + 1 * to_one_empty[four] + 3*(1 - is_empty[(i, j, k)])
    
        problem += 3 * to_all_wedge[four] + 2 * to_1_1_wedge[four] + 2 * to_2_wedge[four] + to_in_wedge[four] + to_out_wedge[four] <= \
            edge_vars[li] + edge_vars[lj] + edge_vars[lk]
        problem += edge_vars[li] + edge_vars[lj] + edge_vars[lk] <= \
            3 * to_all_wedge[four] + 2 * to_1_1_wedge[four] + 2 * to_2_wedge[four] + to_in_wedge[four] + to_out_wedge[four] + 3*(1 - is_wedge[(i,j,k)])
    
        problem += 3 * to_all_edge[four] + 2 * to_1_1_edge[four] + 2 * to_2_edge[four] + to_in_edge[four] + to_out_edge[four] <= \
            edge_vars[li] + edge_vars[lj] + edge_vars[lk]
        problem += edge_vars[li] + edge_vars[lj] + edge_vars[lk] <= \
            3 * to_all_edge[four] + 2 * to_1_1_edge[four] + 2 * to_2_edge[four] + to_in_edge[four] + to_out_edge[four] + 3*(1 - is_edge[(i,j,k)])
    
    #TODO: Constraints for rule out _2_ and _1_1_, _in_ and _out_.
    for four in fours:
        (l, (i, j, k)) = four
        lij = sorted([l, i, j])
        lij = (lij[0], lij[1], lij[2])
        lik = sorted([l, i, k])
        lik = (lik[0], lik[1], lik[2])
        ljk = sorted([l, j, k])
        ljk = (ljk[0], ljk[1], ljk[2])
        problem += 3*to_2_wedge[four] <= 3 - (is_triangle[lij] + is_triangle[lik] + is_triangle[ljk])
        problem += to_1_1_wedge[four] <= is_triangle[lij] + is_triangle[lik] + is_triangle[ljk]
        problem += to_in_wedge[four] <= is_empty[lij] + is_empty[lik] + is_empty[ljk]
        problem += 3*to_out_wedge[four] <= 3 - (is_empty[lij] + is_empty[lik] + is_empty[ljk])
    
        problem += to_2_edge[four] <= is_triangle[lij] + is_triangle[lik] + is_triangle[ljk]
        problem += 3*to_1_1_edge[four] <= 3 - (is_triangle[lij] + is_triangle[lik] + is_triangle[ljk])
        problem += to_in_edge[four] <= is_wedge[lij] + is_wedge[lik] + is_wedge[ljk]
        problem += 3*to_out_edge[four] <= 3 - (is_wedge[lij] + is_wedge[lik] + is_wedge[ljk])
    
    nums = ["num_to_all_triangle", "num_to_two_triangle", "num_to_one_triangle",\
            "num_to_all_wedge", "num_to_1_1_wedge", "num_to_2_wedge", "num_to_in_wedge", "num_to_out_wedge",\
            "num_to_all_edge", "num_to_1_1_edge", "num_to_2_edge", "num_to_in_edge", "num_to_out_edge",\
            "num_to_all_empty", "num_to_two_empty", "num_to_one_empty"]
    
    MAX_POSSIBLE = NUM_NODES - 3
    num_vars = LpVariable.dicts("Num_To_All_Triangle", nums, cat="Continuous")
    for num in nums:
        problem += num_vars[num] >= 0
    
    for three in threes:
        (i, j, k) = three
        conditional_fours = []
        for l in range(0, NUM_NODES):
            if l != i and l != j and l != k:
                conditional_fours.append((l, three))
        problem += lpSum({to_all_triangle[key] for key in conditional_fours}) <= num_vars["num_to_all_triangle"] + MAX_POSSIBLE * (1-is_triangle[three])
        problem += num_vars["num_to_all_triangle"] <= lpSum({to_all_triangle[key] for key in conditional_fours}) + MAX_POSSIBLE * (1-is_triangle[three])
        problem += lpSum({to_two_triangle[key] for key in conditional_fours}) <= num_vars["num_to_two_triangle"] + MAX_POSSIBLE * (1-is_triangle[three])
        problem += num_vars["num_to_two_triangle"] <= lpSum({to_two_triangle[key] for key in conditional_fours}) + MAX_POSSIBLE * (1-is_triangle[three])
        problem += lpSum({to_one_triangle[key] for key in conditional_fours}) <= num_vars["num_to_one_triangle"] + MAX_POSSIBLE * (1-is_triangle[three])
        problem += num_vars["num_to_one_triangle"] <= lpSum({to_one_triangle[key] for key in conditional_fours}) + MAX_POSSIBLE * (1-is_triangle[three])
    
        problem += lpSum({to_all_empty[key] for key in conditional_fours}) <= num_vars["num_to_all_empty"] + MAX_POSSIBLE * (1-is_empty[three])
        problem += num_vars["num_to_all_empty"] <= lpSum({to_all_empty[key] for key in conditional_fours}) + MAX_POSSIBLE * (1-is_empty[three])
        problem += lpSum({to_two_empty[key] for key in conditional_fours}) <= num_vars["num_to_two_empty"] + MAX_POSSIBLE * (1-is_empty[three])
        problem += num_vars["num_to_two_empty"] <= lpSum({to_two_empty[key] for key in conditional_fours}) + MAX_POSSIBLE * (1-is_empty[three])
        problem += lpSum({to_one_empty[key] for key in conditional_fours}) <= num_vars["num_to_one_empty"] + MAX_POSSIBLE * (1-is_empty[three])
        problem += num_vars["num_to_one_empty"] <= lpSum({to_one_empty[key] for key in conditional_fours}) + MAX_POSSIBLE * (1-is_empty[three])
    
        problem += lpSum({to_all_wedge[key] for key in conditional_fours}) <= num_vars["num_to_all_wedge"] + MAX_POSSIBLE * (1-is_wedge[three])
        problem += num_vars["num_to_all_wedge"] <= lpSum({to_all_wedge[key] for key in conditional_fours}) + MAX_POSSIBLE * (1-is_wedge[three])
        problem += lpSum({to_1_1_wedge[key] for key in conditional_fours}) <= num_vars["num_to_1_1_wedge"] + MAX_POSSIBLE * (1-is_wedge[three])
        problem += num_vars["num_to_1_1_wedge"] <= lpSum({to_1_1_wedge[key] for key in conditional_fours}) + MAX_POSSIBLE * (1-is_wedge[three])
        problem += lpSum({to_2_wedge[key] for key in conditional_fours}) <= num_vars["num_to_2_wedge"] + MAX_POSSIBLE * (1-is_wedge[three])
        problem += num_vars["num_to_2_wedge"] <= lpSum({to_2_wedge[key] for key in conditional_fours}) + MAX_POSSIBLE * (1-is_wedge[three])
        problem += lpSum({to_in_wedge[key] for key in conditional_fours}) <= num_vars["num_to_in_wedge"] + MAX_POSSIBLE * (1-is_wedge[three])
        problem += num_vars["num_to_in_wedge"] <= lpSum({to_in_wedge[key] for key in conditional_fours}) + MAX_POSSIBLE * (1-is_wedge[three])
        problem += lpSum({to_out_wedge[key] for key in conditional_fours}) <= num_vars["num_to_out_wedge"] + MAX_POSSIBLE * (1-is_wedge[three])
        problem += num_vars["num_to_out_wedge"] <= lpSum({to_out_wedge[key] for key in conditional_fours}) + MAX_POSSIBLE * (1-is_wedge[three])
    
        problem += lpSum({to_all_edge[key] for key in conditional_fours}) <= num_vars["num_to_all_edge"] + MAX_POSSIBLE * (1-is_edge[three])
        problem += num_vars["num_to_all_edge"] <= lpSum({to_all_edge[key] for key in conditional_fours}) + MAX_POSSIBLE * (1-is_edge[three])
        problem += lpSum({to_1_1_edge[key] for key in conditional_fours}) <= num_vars["num_to_1_1_edge"] + MAX_POSSIBLE * (1-is_edge[three])
        problem += num_vars["num_to_1_1_edge"] <= lpSum({to_1_1_edge[key] for key in conditional_fours}) + MAX_POSSIBLE * (1-is_edge[three])
        problem += lpSum({to_2_edge[key] for key in conditional_fours}) <= num_vars["num_to_2_edge"] + MAX_POSSIBLE * (1-is_edge[three])
        problem += num_vars["num_to_2_edge"] <= lpSum({to_2_edge[key] for key in conditional_fours}) + MAX_POSSIBLE * (1-is_edge[three])
        problem += lpSum({to_in_edge[key] for key in conditional_fours}) <= num_vars["num_to_in_edge"] + MAX_POSSIBLE * (1-is_edge[three])
        problem += num_vars["num_to_in_edge"] <= lpSum({to_in_edge[key] for key in conditional_fours}) + MAX_POSSIBLE * (1-is_edge[three])
        problem += lpSum({to_out_edge[key] for key in conditional_fours}) <= num_vars["num_to_out_edge"] + MAX_POSSIBLE * (1-is_edge[three])
        problem += num_vars["num_to_out_edge"] <= lpSum({to_out_edge[key] for key in conditional_fours}) + MAX_POSSIBLE * (1-is_edge[three])
    
    problem.solve()
    print("Status:", LpStatus[problem.status])
    if str(LpStatus[problem.status]) == "Optimal":
        for i in range(0, NUM_NODES):
            row = []
            for j in range(0, NUM_NODES):
                if i == j:
                    row.append(0)
                else:
                    row.append(int(edge_vars[(min(i, j), max(i, j))].varValue))
            print(row)
        for num in nums:
            print("%s: %.1f" % (num, num_vars[num].varValue))
        for three in threes:
            if is_triangle[three].varValue > 0.0:
                print("Is triangle (%d, %d, %d): %f" % (three[0], three[1], three[2], is_triangle[three].varValue))
            if is_wedge[three].varValue > 0.0:
                print("Is wedge (%d, %d, %d): %f" % (three[0], three[1], three[2], is_wedge[three].varValue))
            if is_edge[three].varValue > 0.0:
                print("Is edge (%d, %d, %d): %f" % (three[0], three[1], three[2], is_edge[three].varValue))
            if is_empty[three].varValue > 0.0:
                print("Is empty (%d, %d, %d): %f" % (three[0], three[1], three[2], is_empty[three].varValue))
    sys.stdout.flush()
    
    #for three in threes:
    #    if is_triangle[three].varValue > 0.0:
    #        print("(%d, %d, %d) is triangle" % (three[0], three[1], three[2]))
    #    if is_wedge[three].varValue > 0.0:
    #        print("(%d, %d, %d) is wedge" % (three[0], three[1], three[2]))
    #    if is_edge[three].varValue > 0.0:
    #        print("(%d, %d, %d) is edge" % (three[0], three[1], three[2]))
    #    if is_empty[three].varValue > 0.0:
    #        print("(%d, %d, %d) is empty" % (three[0], three[1], three[2]))
    
    #for four in fours:
    #    print("(%d, (%d, %d, %d)) is to_in_wedge: %f, to_out_wedge: %f" % (four[0], four[1][0], four[1][1], four[1][2], \
    #        to_in_wedge[four].varValue, to_out_wedge[four].varValue))
    
for num_nodes in range(7, 30):
    print("Starting %s" % num_nodes)
    sys.stdout.flush()
    graph_with_n_nodes(num_nodes)

# Example from online below.

"""

# defining list of products
products = ['cola','peanuts', 'cheese', 'beer']
itemsets = ['x1','x2', 'x3']

#disctionary of the costs of each of the products is created
costs = {'cola' : 5, 'peanuts' : 3, 'cheese' : 1, 'beer' : 4 }

# dictionary of frequent itemsets
# ~~> This is hard to maintain - I would select a different data structure
# it gets really complicated below as you will see!
itemset_dict = { "x1" : (("cola", "peanuts"),10),
           "x2" : (("peanuts","cheese"),20),
           "x3" : (("peanuts","beer"),30)
           }

# Good practice to first define your problem
my_lp_program = LpProblem('My LP Problem', LpMaximize)  

# ~~>You do not need bounds for binary variables, they are automatically 0/1
products_var=LpVariable.dicts("Products", products, cat='Binary')
itemsets_var=LpVariable.dicts("Itemsets", itemsets, cat='Binary')

# ~~> Use an affine expression to define your objective.
# ~~> Even better, define two objects as LpAffineExpression and add them, 
# ~~> it keeps the code cleaner
my_lp_program += LpAffineExpression([(
    itemsets_var[x], itemset_dict[x][1])  for x in itemsets_var]) + \
    LpAffineExpression([(
    products_var[x], -costs[x])  for x in products_var])

# ~~> This is the right way to enter this constraint.
# ~~> I do not like the naming though..
my_lp_program += lpSum(products_var) <= 3, '1Constraint'

# ~~> Here are your constraints
counter = 1
for a in itemset_dict.keys():
    item = itemsets_var[a]
    for b in itemset_dict[a][0]:
        product = products_var[b]
        counter +=1
        my_lp_program += product  >= item, "{}Constraint".format(counter)

# ~~> Great that you export the lp! If you look at the file you can
# ~~> spot a lot of issues with the model during debugging
my_lp_program.writeLP("CheckLpProgram.lp")
my_lp_program.solve()

print("Status:", LpStatus[my_lp_program.status])

print("Total Optimum=", value(my_lp_program.objective))

for v in my_lp_program.variables():
    print(v.name, "=", v.varValue)
"""

