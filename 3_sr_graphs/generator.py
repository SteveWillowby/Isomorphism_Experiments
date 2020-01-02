# This must be run with python 3!
from pulp import *

problem = LpProblem("Generate_a_3SR_Graph", LpMinimize)

NUM_NODES = 15
potential_edges = []
is_a_triangle = []
is_a_wedge = []
is_an_edge = []
is_empty = []
for i in range(0, NUM_NODES):
    for j in range(i+1, NUM_NODES):
        potential_edges.append((i,j))
        for k in range(j+1, NUM_NODES):
            is_a_triangle.append((i,j,k))
            is_a_wedge.append((i,j,k))
            is_an_edge.append((i,j,k))
            is_empty.append((i,j,k))

potential_edge_vars = LpVariable.dicts("Potential_Edge_Vars", potential_edges, cat="Binary")

triangle_vars = LpVariable.dicts("Triangle_Vars", is_a_triangle, cat="Binary")
wedge_vars = LpVariable.dicts("Wedge_Vars", is_a_wedge, cat="Binary")
edge_vars = LpVariable.dicts("Edge_Vars", is_an_edge, cat="Binary")
empty_vars = LpVariable.dicts("Empty_Vars", is_empty, cat="Binary")

# Constraints to force triple vars to correspond to edge vars.
for i in range(0, NUM_NODES):
    for j in range(i+1, NUM_NODES):
        for k in range(j+1, NUM_NODES):
            edge_dict = {key: potential_edge_vars[key] for key in [(i,j), (i,k), (j,k)]}
            problem += lpSum(edge_dict) == triangle_vars[(i,j,k)] * 3 + wedge_vars[(i,j,k)] * 2 + edge_vars[(i,j,k)]*1
            problem += triangle_vars[(i,j,k)] + wedge_vars[(i,j,k)] + edge_vars[(i,j,k)] + empty_vars[(i,j,k)] == 1

# Force number of each 3-node component to be non-zero
problem += lpSum(triangle_vars) >= 1
problem += lpSum(wedge_vars) >= 1
problem += lpSum(edge_vars) >= 1
problem += lpSum(empty_vars) >= 1

to_all_triangle = LpVariable.dicts("To_All_Triangle", is_a_triangle, cat="Binary")
to_two_triangle = LpVariable.dicts("To_Two_Triangle", is_a_triangle, cat="Binary")
to_one_triangle = LpVariable.dicts("To_One_Triangle", is_a_triangle, cat="Binary")
to_all_wedge = LpVariable.dicts("To_All_Wedge", is_a_triangle, cat="Binary")
to_1_1_wedge = LpVariable.dicts("To_1_1_Wedge", is_a_triangle, cat="Binary")
to_2_wedge = LpVariable.dicts("To_2_Wedge", is_a_triangle, cat="Binary")
to_in_wedge = LpVariable.dicts("To_In_Wedge", is_a_triangle, cat="Binary")
to_out_wedge = LpVariable.dicts("To_Out_Wedge", is_a_triangle, cat="Binary")
to_all_edge = LpVariable.dicts("To_All_Edge", is_a_triangle, cat="Binary")
to_1_1_edge = LpVariable.dicts("To_1_1_Edge", is_a_triangle, cat="Binary")
to_2_edge = LpVariable.dicts("To_2_Edge", is_a_triangle, cat="Binary")
to_in_edge = LpVariable.dicts("To_In_Edge", is_a_triangle, cat="Binary")
to_out_edge = LpVariable.dicts("To_Out_Edge", is_a_triangle, cat="Binary")
to_all_empty = LpVariable.dicts("To_All_Empty", is_a_triangle, cat="Binary")
to_two_empty = LpVariable.dicts("To_Two_Empty", is_a_triangle, cat="Binary")
to_one_empty = LpVariable.dicts("To_One_Empty", is_a_triangle, cat="Binary")

# print(problem)
problem.solve()
for pe, pe_v in potential_edge_vars.items():
    print(pe)
    print(pe_v.varValue)

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

