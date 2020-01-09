

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
