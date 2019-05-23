import scipy.io as io
import numpy as np
import scipy.sparse.csgraph as cs
import scipy.sparse as sparse
import math
import random
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import time
from numba import jit
from sklearn.cluster import KMeans
from scipy.spatial import distance


@jit
def k_means_max_min_sp(graph, n_pivots=200, sgd_iter=15, epsilon=0.1, k_means_iter=50, directed=False, unweighted=False):
    # get number of vertices
    n = graph.shape[0]

    shortest_paths = cs.shortest_path(graph, directed=directed, unweighted=unweighted)

    k_means = KMeans(n_clusters=int(n_pivots / 2), max_iter=k_means_iter).fit(shortest_paths)

    centroids = k_means.cluster_centers_

    pivots = []

    for c in centroids:
        euc_dist = distance.cdist([c], shortest_paths, 'euclidean')
        pivots.append(euc_dist.argmin())

    remaining_vertices = []
    for i in range(n):
        if i not in pivots:
            remaining_vertices.append(i)

    mins = []
    for i in range(n):
        mins.append([shortest_paths[pivots[0]][i], pivots[0]])

    for i in range(1, int(n_pivots / 2)):
        for j in range(n):
            temp = shortest_paths[pivots[i]][j]
            if temp < mins[j][0]:
                mins[j][0] = temp
                mins[j][1] = pivots[i]

    p0 = np.random.choice(remaining_vertices)
    pivots.append(p0)
    remaining_vertices.remove(p0)

    # update list of minimum distances to each pivots with the first pivot selected randomly for max/min sp
    for i in range(n):
        temp = shortest_paths[p0][i]
        if temp < mins[i][0]:
            mins[i][0] = temp
            mins[i][1] = p0

    # normal max/min sp
    for i in range(int(n_pivots / 2) + 1, n_pivots):
        argmax = 0
        for k in remaining_vertices:
            if mins[k][0] > mins[argmax][0]:
                argmax = k
        pivots.append(argmax)

        for j in range(n):
            temp = shortest_paths[pivots[i]][j]
            if temp < mins[j][0]:
                mins[j][0] = temp
                mins[j][1] = pivots[i]

    regions = {}
    for p in pivots:
        regions[p] = []

    for i in range(n):
        closest_pivot = mins[i][1]
        regions[closest_pivot].append(i)

    weights, dists = sssp(n, graph, shortest_paths, regions, pivots)

    constraints, weights, schedules = set_constraints(graph, weights, dists, sgd_iter, epsilon)

    return sgd_graph_layout(n, constraints, weights, schedules)


@jit
def k_means_sp(graph, n_pivots=200, sgd_iter=15, epsilon=0.1, k_means_iter=50, directed=False, unweighted=False):
    # get number of vertices
    n = graph.shape[0]

    shortest_paths = cs.shortest_path(graph, directed=directed, unweighted=unweighted)

    # randomly choose a pivot
    p0 = np.random.randint(0, n)
    pivots = [p0]

    # initialize list of minimum distances to each pivots
    mins = []
    for i in range(n):
        mins.append([shortest_paths[p0][i], p0])

    for i in range(1, n_pivots):
        # normal max/min sp:
        argmax = 0
        for k in range(1, n):
            if mins[k][0] > mins[argmax][0]:
                argmax = k
        pivots.append(argmax)

        for j in range(n):
            temp = shortest_paths[pivots[i]][j]
            if temp < mins[j][0]:
                mins[j][0] = temp
                mins[j][1] = pivots[i]

    pivots_shortest_paths = []
    for p in pivots:
        pivots_shortest_paths.append(shortest_paths[p])

    kmeans = KMeans(n_init=1, init=np.asarray(pivots_shortest_paths), n_clusters=n_pivots, max_iter=k_means_iter).fit(
        shortest_paths)
    centroids = kmeans.cluster_centers_

    pivots = []

    for c in centroids:
        euc_dist = distance.cdist([c], shortest_paths, 'euclidean')
        pivots.append(euc_dist.argmin())

    mins = []
    for i in range(n):
        mins.append([shortest_paths[pivots[0]][i], pivots[0]])

    for i in range(1, n_pivots):
        for j in range(n):
            temp = shortest_paths[pivots[i]][j]
            if temp < mins[j][0]:
                mins[j][0] = temp
                mins[j][1] = pivots[i]

    regions = {}
    for p in pivots:
        regions[p] = []

    for i in range(n):
        closest_pivot = mins[i][1]
        regions[closest_pivot].append(i)

    weights, dists = sssp(n, graph, shortest_paths, regions, pivots)

    constraints, weights, schedules = set_constraints(graph, weights, dists, sgd_iter, epsilon)

    return sgd_graph_layout(n, constraints, weights, schedules)


@jit
def k_means_layout(graph, n_pivots=200, sgd_iter=15, epsilon=0.1, k_means_iter=50, unweighted=False):
    # get number of vertices
    n = graph.shape[0]

    shortest_paths = cs.shortest_path(graph, unweighted=unweighted)

    kmeans = KMeans(n_clusters=n_pivots, random_state=0, max_iter=k_means_iter).fit(shortest_paths)
    centroids = kmeans.cluster_centers_

    pivots = []

    for c in centroids:
        euc_dist = distance.cdist([c], shortest_paths, 'euclidean')
        pivots.append(euc_dist.argmin())

    mins = []
    for i in range(n):
        mins.append([shortest_paths[pivots[0]][i], pivots[0]])

    for i in range(1, n_pivots):
        for j in range(n):
            temp = shortest_paths[pivots[i]][j]
            if temp < mins[j][0]:
                mins[j][0] = temp
                mins[j][1] = pivots[i]

    regions = {}
    for p in pivots:
        regions[p] = []

    for i in range(n):
        closest_pivot = mins[i][1]
        regions[closest_pivot].append(i)

    weights, dists = sssp(n, graph, shortest_paths, regions, pivots)

    constraints, weights, schedules = set_constraints(graph, weights, dists, sgd_iter, epsilon)

    return sgd_graph_layout(n, constraints, weights, schedules)


@jit
def max_min_random_sp(graph, n_pivots=200, sgd_iter=15, epsilon=0.1, unweighted=False):
    # get number of vertices
    n = graph.shape[0]

    # randomly choose a pivot
    p0 = np.random.randint(0, n)
    pivots = [p0]
    shortest_paths = cs.shortest_path(graph, unweighted=unweighted)

    # initialize list of minimum distances to each pivots
    mins = []
    for i in range(n):
        mins.append([shortest_paths[p0][i], p0])

    for i in range(1, n_pivots):
        total_probability = 0
        cumulative_probability = []
        for j in range(n):
            # probability is proportional to the shortest paths to the pivot
            total_probability += mins[j][0]
            cumulative_probability.append(total_probability)
        sample = np.random.uniform(0, total_probability)
        for j in range(n):
            if sample < cumulative_probability[j]:
                pivots.append(j)
                break

        for j in range(n):
            temp = shortest_paths[pivots[i]][j]
            if temp < mins[j][0]:
                mins[j][0] = temp
                mins[j][1] = pivots[i]

    # find the regions for each pivot
    regions = {}
    for p in pivots:
        regions[p] = []

    for i in range(n):
        closest_pivot = mins[i][1]
        regions[closest_pivot].append(i)

    weights, dists = sssp(n, graph, shortest_paths, regions, pivots)

    constraints, weights, schedules = set_constraints(graph, weights, dists, sgd_iter, epsilon)

    return sgd_graph_layout(n, constraints, weights, schedules)


@jit
def max_min_sp(graph, n_pivots=200, sgd_iter=15, epsilon=0.1, unweighted=False):
    # get number of vertices
    n = graph.shape[0]

    # randomly choose a pivot
    p0 = np.random.randint(0, n)
    pivots = [p0]
    shortest_paths = cs.shortest_path(graph, unweighted=unweighted)

    # initialize list of minimum distances to each pivots
    mins = []
    for i in range(n):
        mins.append([shortest_paths[p0][i], p0])

    for i in range(1, n_pivots):
        # normal max/min sp:
        argmax = 0
        for k in range(1, n):
            if mins[k][0] > mins[argmax][0]:
                argmax = k
        pivots.append(argmax)

        for j in range(n):
            temp = shortest_paths[pivots[i]][j]
            if temp < mins[j][0]:
                mins[j][0] = temp
                mins[j][1] = pivots[i]

    # find the regions for each pivot
    regions = {}
    for p in pivots:
        regions[p] = []

    for i in range(n):
        closest_pivot = mins[i][1]
        regions[closest_pivot].append(i)

    weights, dists = sssp(n, graph, shortest_paths, regions, pivots)

    constraints, weights, schedules = set_constraints(graph, weights, dists, sgd_iter, epsilon)

    return sgd_graph_layout(n, constraints, weights, schedules)


def mis_filtration(graph, n_pivots=200, sgd_iter=15, epsilon=0.1, unweighted=False):
    # get number of vertices
    n = graph.shape[0]
    node_num = n

    shortest_paths_unweighted = cs.shortest_path(graph, unweighted=True)
    if unweighted is True:
        shortest_paths = shortest_paths_unweighted
    else:
        shortest_paths = cs.shortest_path(graph, unweighted=False)

    layer_index = 1

    pre_pivots = []
    pivots = list(range(0, n))
    remaining_vertices = range(0, n)
    while node_num > n_pivots:
        pre_pivots = pivots
        p_temp = np.random.choice(pivots)
        pivots = []
        empty = False

        while empty is False:
            path_lengths = shortest_paths_unweighted[p_temp]
            pivots.append(p_temp)
            remaining_vertices = [i for i in remaining_vertices if
                                  path_lengths[i] > 2 ** layer_index and i not in pivots]
            if not remaining_vertices:
                empty = True
            else:
                p_temp = np.random.choice(remaining_vertices)
        layer_index += 1
        node_num = len(pivots)

    for p in pre_pivots:
        if p in pivots:
            pre_pivots.remove(p)

    while len(pivots) < n_pivots:
        p = np.random.choice(pre_pivots)
        pivots.append(p)
        pre_pivots.remove(p)

    p0 = pivots[0]

    mins = []
    for i in range(n):
        mins.append([shortest_paths[p0][i], p0])

    for i in range(1, n_pivots):
        for j in range(n):
            temp = shortest_paths[pivots[i]][j]
            if temp < mins[j][0]:
                mins[j][0] = temp
                mins[j][1] = pivots[i]

    # find the regions for each pivot
    regions = {}
    for p in pivots:
        regions[p] = []

    for i in range(n):
        closest_pivot = mins[i][1]
        regions[closest_pivot].append(i)

    weights, dists = sssp(n, graph, shortest_paths, regions, pivots)

    constraints, weights, schedules = set_constraints(graph, weights, dists, sgd_iter, epsilon)

    return sgd_graph_layout(n, constraints, weights, schedules)


@jit
def max_min_euclidean(graph, n_pivots=200, sgd_iter=15, epsilon=0.1, unweighted=False):
    # get number of vertices
    n = graph.shape[0]

    # randomly initialize the first pivot
    p0 = np.random.randint(0, n)
    pivots = [p0]

    # convert graph vertices to Euclidean Space using the shortest paths entries
    graph_arr = graph.toarray()

    shortest_dist = {p0: euclidean_dist(graph_arr, p0)}

    mins = []
    for i in range(n):
        mins.append([shortest_dist[p0][i], p0])
    for i in range(1, n_pivots):
        argmax = 0
        for k in range(1, n):
            if mins[k][0] > mins[argmax][0]:
                argmax = k
        pivots.append(argmax)

        shortest_dist[pivots[i]] = euclidean_dist(graph_arr, pivots[i])
        for j in range(n):
            temp = shortest_dist[pivots[i]][j]
            if temp < mins[j][0]:
                mins[j][0] = temp
                mins[j][1] = pivots[i]

    regions = {}
    for p in pivots:
        regions[p] = []

    for i in range(n):
        closest_pivot = mins[i][1]
        regions[closest_pivot].append(i)

    shortest_paths = cs.shortest_path(graph)

    weights, dists = sssp(n, graph, shortest_paths, regions, pivots)

    constraints, weights, schedules = set_constraints(graph, weights, dists, sgd_iter, epsilon)

    return sgd_graph_layout(n, constraints, weights, schedules)


@jit(nopython=True, fastmath=True, parallel=True)
def euclidean_dist(euc_coordinates, pivot):
    dist = []
    for i in range(euc_coordinates.shape[0]):
        dist.append(np.linalg.norm(euc_coordinates[pivot] - euc_coordinates[i]))
    return dist


@jit
def random_pivots(graph, n_pivots=200, sgd_iter=15, epsilon=0.1):
    # randomly choose a pivot
    n = graph.shape[0]
    pivots = random.sample(range(0, n), n_pivots)
    p0 = pivots[0]
    shortest_paths = cs.shortest_path(graph, directed=False, unweighted=True)

    # initialize list of minimum distances to each pivots
    mins = []
    for i in range(n):
        mins.append([shortest_paths[p0][i], p0])

    for i in range(1, n_pivots):
        for j in range(n):
            temp = shortest_paths[pivots[i]][j]
            if temp < mins[j][0]:
                mins[j][0] = temp
                mins[j][1] = pivots[i]

    # find the regions for each pivot
    regions = {}
    for p in pivots:
        regions[p] = []

    for i in range(n):
        closest_pivot = mins[i][1]
        regions[closest_pivot].append(i)

    weights, dists = sssp(n, graph, shortest_paths, regions, pivots)

    constraints, weights, schedules = set_constraints(graph, weights, dists, sgd_iter, epsilon)

    return sgd_graph_layout(n, constraints, weights, schedules)


def sssp(n, graph, shortest_paths, regions, pivots):
    # adjust the weights
    weights = {}
    dists = {}

    for p in pivots:
        for i in range(n):
            if graph[p, i] == 0 and p != i:
                # ignore neighbours
                s = sum(1 for j in regions[p] if shortest_paths[p][j] <= shortest_paths[p][i] / 2)
                w = 1 / shortest_paths[p][i] ** 2

                weights[(i, p)] = s * w
                if (p, i) not in weights:
                    weights[(p, i)] = 0

                dists[(p, i)] = dists[(i, p)] = shortest_paths[p][i]

        print('.', end='')

    return weights, dists


@jit
def set_constraints(graph, weights, dists, sgd_iter=15, epsilon=0.1):
    I, J, V = sparse.find(graph)
    for e in range(len(I)):
        i, j, v = I[e], J[e], V[e]
        if i < j:
            weights[(i, j)] = weights[(j, i)] = 1 / v ** 2
            dists[(i, j)] = dists[(j, i)] = v

    constraints = []
    for ij in dists.keys():
        i = ij[0]
        j = ij[1]
        if i < j:
            constraints.append((i, j, dists[ij]))

    w_max = 0
    w_min = math.inf

    for w in weights.values():
        if w != 0:
            w_max = max(w, w_max)
            w_min = min(w, w_min)

    eta_max = 1 / w_min
    eta_min = epsilon / w_max

    lambd = np.log(eta_max / eta_min) / (sgd_iter - 1)
    # eta = lambda t: eta_max*np.exp(-lambd*t)

    # set up the schedule as exponential decay
    schedule = []
    for i in range(sgd_iter):
        eta = eta_max * math.exp(-lambd * i)
        schedule.append(eta)

    return constraints, weights, schedule


@jit
def sgd_graph_layout(n, constraints, weights, schedules):
    # initialise an array of 2D positions
    X = np.random.rand(n, 2)

    for s in schedules:
        random.shuffle(constraints)

        for i, j, d in constraints:
            w_i = min(weights[(i, j)] * s, 1)
            w_j = min(weights[(j, i)] * s, 1)
            ij = X[i] - X[j]
            mag = np.linalg.norm(ij)
            m = ((d - mag) / 2) * (ij / mag)

            X[i] += w_i * m
            X[j] -= w_j * m

        print('.', end='')

    return X

