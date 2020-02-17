def further_sort_by(l, d):
    for i in range(0, len(l)):
        l[i] = ((l[i][0], d[l[i][1]]), l[i][1])
    l.sort()
    new_id = 0
    prev_label = l[0][0]
    l[0] = (0, l[0][1])
    for i in range(1, len(l)):
        if l[i][0] != prev_label:
            new_id += 1
            prev_label = l[i][0]
        l[i] = (new_id, l[i][1])

def jointly_further_sort_by_and_compare(l1, d1, l2, d2):
    for i in range(0, len(l1)):
        l1[i] = ((l1[i][0], d1[l1[i][1]]), l1[i][1])
        l2[i] = ((l2[i][0], d2[l2[i][1]]), l2[i][1])
    l1.sort()
    l2.sort()
    all_equal = l1[0][0] == l2[0][0]
    new_id_1 = 0
    prev_label_1 = l1[0][0]
    new_id_2 = 0
    prev_label_2 = l2[0][0]
    l1[0] = (0, l1[0][1])
    l2[0] = (0, l2[0][1])
    for i in range(1, len(l1)):
        if l1[i][0] != l2[i][0]:
            all_equal = False
        if l1[i][0] != prev_label_1:
            new_id_1 += 1
            prev_label_1 = l1[i][0]
        if l2[i][0] != prev_label_2:
            new_id_2 += 1
            prev_label_2 = l2[i][0]
        l1[i] = (new_id_1, l1[i][1])
        l2[i] = (new_id_2, l2[i][1])
    return all_equal

# Returns False if the tuple cannot be incremented. True otherwise. Modifies t in place.
# n is the number of values a variable in the tuple may take. Values are zero-indexed.
def increment_k_tuple(t, n):
    idx_to_increment = len(t) - 1
    max_idx = n - 1
    while idx_to_increment >= 0 and t[idx_to_increment] == max_idx:
        max_idx -= 1
        idx_to_increment -= 1
    if idx_to_increment < 0:
        return False
    t[idx_to_increment] += 1
    for j in range(idx_to_increment + 1, len(t)):
        t[j] = t[j - 1] + 1
    return True

def get_all_k_tuples(n, k):
    current_tuple = [i for i in range(0, k)]
    stored_tuples = [tuple(current_tuple)]
    while increment_k_tuple(current_tuple, n):
        stored_tuples.append(tuple(current_tuple))
    return stored_tuples

def increment_k_permutation(p, n):
    idx_to_increment = len(p) - 1
    while idx_to_increment >= 0 and p[idx_to_increment] == n - 1:
        idx_to_increment -= 1
    if idx_to_increment < 0:
        return False
    p[idx_to_increment] += 1
    for j in range(idx_to_increment + 1, len(p)):
        p[j] = 0
    return True

def get_all_k_permutations(n, k):
    current_tuple = [0 for i in range(0, k)]
    stored_tuples = [tuple(current_tuple)]
    while increment_k_permutation(current_tuple, n):
        stored_tuples.append(tuple(current_tuple))
    return stored_tuples
