from heapdict import heapdict
import math
import scipy.sparse as sparse
import pprofile


def stress_partitioning(n, pivots, dists, region_assignment, region_size, tA, tC, oDist, cur_dist, graph_arr, weights, s, shortest_paths):
    profiler = pprofile.StatisticalProfile()
    with profiler():
        for i in range(len(tA)):
            index = tA[i]
            pivot_no = math.floor(index / n)
            pivot_index = pivots[pivot_no]
            v = index - pivot_no * n
            if region_assignment[v] < 0:
                region_assignment[v] = pivot_index
                region_size[pivot_index] += 1
            #                 pD[pivot_index].append(oDist)
            # pivot_index in this case represent the neighbour pivot of current vertex v
            # check if vertex assigned pivot is its neighbour pivot, if not, reassign the vertex to its neighbour
            if region_size[region_assignment[v]] > (region_size[pivot_index] + 1):
                region_size[region_assignment[v]] -= 1
                region_assignment[v] = pivot_index
                region_size[pivot_index] += 1
        for i in range(len(tC)):
            index = tC[i]
            pivot_no = math.floor(index / n)
            pivot_index = pivots[pivot_no]
            v = index - pivot_no * n
            if (oDist > 0) and (region_assignment[v] != pivot_index) and (v not in graph_arr[pivot_index].nonzero()[0]):
                weights[(v, pivot_index)] = s[pivot_index] / shortest_paths[pivot_index][v] ** 2
                if (pivot_index, v) not in weights:
                    weights[(pivot_index, v)] = 0
                dists[(pivot_index, v)] = dists[(v, pivot_index)] = shortest_paths[pivot_index][v]
        for key in region_size:
            for i in range(len(region_assignment)):
                if s[key] == region_size[key]:
                    break
                if region_assignment[i] == key and shortest_paths[key][i] <= cur_dist / 2 and key != i:
                    s[key] += 1
    profiler.print_stats()


def MSSP(graph, pivots, shortest_paths):
    weights = {}
    dists = {}
    I, J, V = sparse.find(graph)
    for e in range(len(I)):
        i, j, v = I[e], J[e], V[e]
        if i < j:
            weights[(i, j)] = weights[(j, i)] = 1 / v ** 2
            dists[(i, j)] = dists[(j, i)] = v
    # with profiler():
    n = graph.shape[0]
    num_of_pivots = len(pivots)
    graph_arr = graph.toarray()
    # initialize priority queue
    q = heapdict()
    # initialize pivot distance to every other vertices
    # pD = []
    # for p in pivots:
    #     pD.append(shortest_paths[p])

    # initialize marking for nodes
    mark = [False] * num_of_pivots * n

    # initialize cluster assignment for each vertex
    region_assignment = [-1] * n
    region_size = {}

    # initialize s for each pivot
    s = {}

    for i in range(num_of_pivots):
        q[i * n + pivots[i]] = 0.0
        s[pivots[i]] = 0
        region_size[pivots[i]] = 0

    # initialize distances
    cur_dist = 0
    pre_dist = 0
    tA = []
    tC = []

    while len(q) != 0:
        cur_index, cur_dist = q.popitem()
        if pre_dist != cur_dist:
            stress_partitioning(n, pivots, dists, region_assignment, region_size, tA, tC, pre_dist, cur_dist, graph_arr,
                                weights, s,
                                shortest_paths)
            pre_dist = cur_dist
            tA.clear()
            tC.clear()
            # continue

        pivot_num = math.floor(cur_index / n)
        v = cur_index - pivot_num * n
        mark[cur_index] = True
        tC.append(cur_index)
        # Assign to cluster if not yet assigned
        if region_assignment[v] < 0:
            tA.append(cur_index)
        neighbours = graph_arr[v].nonzero()[0]
        for w in neighbours:
            neighbour_index = cur_index - v + w
            if not mark[neighbour_index]:
                q[neighbour_index] = cur_dist + shortest_paths[w][v]

    stress_partitioning(n, pivots, dists, region_assignment, region_size, tA, tC, pre_dist, cur_dist, graph_arr, weights,
                        s, shortest_paths)
    # profiler.print_stats()
    return dists, weights

