from heapdict import heapdict
import math
import scipy.sparse as sparse
import pprofile
import queue


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


# def MSSP_unweighted(graph, pivots, shortest_paths):
#     weights = {}
#     dists = {}
#     I, J, V = sparse.find(graph)
#     for e in range(len(I)):
#         i, j, v = I[e], J[e], V[e]
#         if i < j:
#             weights[(i, j)] = weights[(j, i)] = 1 / v ** 2
#             dists[(i, j)] = dists[(j, i)] = v
#
#     n = graph.shape[0]
#     # num_of_pivots = len(pivots)
#     graph_arr = graph.toarray()
#
#     q1 = queue.Queue()
#     q2 = queue.Queue()
#
#     # initialize marking for nodes
#     mark = [False] * n
#
#     # initialize s for each pivot
#     s = {}
#     # initialize cluster assignment for each vertex
#     region_assignment = [-1] * n
#     # initialize region for each pivots
#     regions = {}
#     for p in pivots:
#         regions[p] = [p]
#         q1.put(p)
#         region_assignment[p] = p
#         s[p] = 1
#
#     q1.put(-1)
#
#     dist = 0
#     previous_dist = 1
#
#     while q1.qsize() > 1:
#         index = q1.get()
#         if index < 0:
#             q1.put(-1)
#             q2.put(-1)
#             dist += 1
#             if dist % previous_dist == 0 and dist / previous_dist == 2:
#                 previous_dist += 1
#                 while not q2.empty():
#                     regional_index = q2.get()
#                     if regional_index < 0:
#                         break
#                     cur_region = region_assignment[regional_index]
#                     s[cur_region] += 1
#             continue
#         mark[index] = True
#         pivot_index = region_assignment[index]
#         if s[pivot_index] != 0:
#             weights[(index, pivot_index)] = s[pivot_index] / shortest_paths[pivot_index][index] ** 2
#             if (pivot_index, index) not in weights:
#                 weights[(pivot_index, index)] = 0
#             dists[(pivot_index, index)] = dists[(index, pivot_index)] = shortest_paths[pivot_index][index]
#
#         neighbours = graph_arr[index].nonzero()[0]
#         for neighbour in neighbours:
#             if not mark[neighbour]:
#                 q1.put(neighbour)
#                 q2.put(neighbour)
#                 region_assignment[neighbour] = pivot_index
#                 regions[pivot_index].append(neighbour)
#     return dists, weights


def vertices_partition(graph, pivots):
    n = graph.shape[0]
    graph_arr = graph.toarray()

    q1 = queue.Queue()
    q2 = queue.Queue()

    # initialize marking for nodes
    mark = [False] * n

    # initialize cluster assignment for each vertex
    region_assignment = [-1] * n
    # initialize region for each pivots
    regions = {}
    for p in pivots:
        regions[p] = [p]
        q1.put(p)
        region_assignment[p] = p
        mark[p] = True

    q1.put(-1)

    while q1.qsize() > 1:
        index = q1.get()
        if index < 0:
            q1.put(-1)
            continue
        pivot_index = region_assignment[index]
        neighbours = graph_arr[index].nonzero()[0]
        for neighbour in neighbours:
            if not mark[neighbour]:
                mark[neighbour] = True
                q1.put(neighbour)
                q2.put(neighbour)
                region_assignment[neighbour] = pivot_index
                regions[pivot_index].append(neighbour)

    return regions, region_assignment


def MSSP_unweighted(graph, pivots, shortest_paths, regions, region_assignment):
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

    for p in pivots:
        q1 = queue.Queue()
        q2 = queue.Queue()
        # initialize marking for nodes
        mark = [False] * n
        # print("processing pivot ", p)
        mark[p] = True
        q1.put(p)
        s[p] = 1
        q1.put(-1)
        dist = 0
        previous_dist = 1
        while q1.qsize() > 1:
            index = q1.get()
            if index < 0:
                q1.put(-1)
                q2.put(-1)
                dist += 1
                if s[p] < len(regions[p]):
                    if dist % previous_dist == 0 and dist / previous_dist == 2:
                        previous_dist += 1
                        while not q2.empty():
                            regional_index = q2.get()
                            if regional_index < 0:
                                break
                            cur_region = region_assignment[regional_index]
                            s[cur_region] += 1
                continue

            if p != index and graph[p, index] == 0 and (p, index) not in weights and (index, p) not in weights:
                weights[(index, p)] = s[p] / shortest_paths[p][index] ** 2
                if(p, index) not in weights:
                    weights[(p, index)] = 0
                dists[(p, index)] = dists[(index, p)] = shortest_paths[p][index]

            neighbours = graph_arr[index].nonzero()[0]
            for neighbour in neighbours:
                if not mark[neighbour]:
                    mark[neighbour] = True
                    q1.put(neighbour)
                    if neighbour in regions[p]:
                        q2.put(neighbour)

    return dists, weights
