from heapdict import heapdict
import math
import scipy.sparse as sparse
import pprofile
import queue


def MSSP_ultimate(graph, pivots):
    weights = {}
    dists = {}
    I, J, V = sparse.find(graph)
    for e in range(len(I)):
        i, j, v = I[e], J[e], V[e]
        if i < j:
            weights[(i, j)] = weights[(j, i)] = 1 / v ** 2
            dists[(i, j)] = dists[(j, i)] = v

    n = graph.shape[0]
    graph_arr = graph.toarray()

    # initialize s for each pivot
    s = {}

    mark_for_region = [False] * n
    mark_for_weights = {}
    level = {}
    regions = {}
    region_assignment = [-1] * n
    q1 = queue.Queue()
    q2 = {}
    q3 = queue.Queue()

    dist = {}
    prev_dist = {}

    for p in pivots:
        regions[p] = [p]
        region_assignment[p] = p
        q1.put(p)
        q2[p] = queue.Queue()
        mark_for_weights[p] = [False] * n
        mark_for_weights[p][p] = True
        level[p] = [0] * n
        mark_for_region[p] = True
        s[p] = 1
        dist[p] = 0
        prev_dist[p] = 1

    while not q1.empty():
        index = q1.get()
        pivot_index = region_assignment[index]
        cur_level = level[pivot_index][index]

        if cur_level > dist[pivot_index]:
            dist[pivot_index] = cur_level

        compute_s(q2, pivot_index, level, dist, prev_dist, s)

        if pivot_index != index and graph[pivot_index, index] == 0 and (pivot_index, index) not in weights and (index, pivot_index) not in weights:
            compute_weight(index, pivot_index, weights, dists, level, s)

        neighbours = graph_arr[index].nonzero()[0]
        for neighbour in neighbours:
            if not mark_for_region[neighbour]:
                mark_for_region[neighbour] = True
                mark_for_weights[pivot_index][neighbour] = True
                level[pivot_index][neighbour] = cur_level + 1
                q1.put(neighbour)
                region_assignment[neighbour] = pivot_index
                regions[pivot_index].append(neighbour)
                q2[pivot_index].put(neighbour)
            if not mark_for_weights[pivot_index][neighbour]:
                mark_for_weights[pivot_index][neighbour] = True
                level[pivot_index][neighbour] = cur_level + 1
                q3.put((neighbour, pivot_index))

    while not q3.empty():
        index, pivot_index = q3.get()
        cur_level = level[pivot_index][index]

        if cur_level > dist[pivot_index]:
            dist[pivot_index] = cur_level

        compute_s(q2, pivot_index, level, dist, prev_dist, s)
        compute_weight(index, pivot_index, weights, dists, level, s)

        neighbours = graph_arr[index].nonzero()[0]
        for neighbour in neighbours:
            if not mark_for_weights[pivot_index][neighbour]:
                mark_for_weights[pivot_index][neighbour] = True
                level[pivot_index][neighbour] = cur_level + 1
                q3.put((neighbour, pivot_index))

    return dists, weights


def compute_s(q2, pivot_index, level, dist, prev_dist, s):
    if dist[pivot_index] % prev_dist[pivot_index] == 0 and dist[pivot_index] / prev_dist[pivot_index] == 2:
        while not q2[pivot_index].empty():
            regional_index = q2[pivot_index].get()
            if level[pivot_index][regional_index] <= prev_dist[pivot_index]:
                s[pivot_index] += 1
            else:
                q2[pivot_index].put(regional_index)
                break
        prev_dist[pivot_index] += 1


def compute_weight(index, pivot_index, weights, dists, level, s):
    weights[(index, pivot_index)] = s[pivot_index] / level[pivot_index][index] ** 2
    if (pivot_index, index) not in weights:
        weights[(pivot_index, index)] = 0
    dists[(pivot_index, index)] = dists[(index, pivot_index)] = level[pivot_index][index]