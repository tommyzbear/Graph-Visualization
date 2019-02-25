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
import src.sampling_scheme as ss


def compute_stress(dists, X):
    n = dists.shape[0]
    stress = 0
    for i in range(n):
        for j in range(i):
            dist = X[i] - X[j]
            magnitude = np.linalg.norm(dist)
            stress += (1 / dists[i][j] ** 2) * (dists[i][j] - magnitude) ** 2

    return stress/n**2


def draw_graph(X, graph):
    plt.axis('equal')
    ax = plt.axes()
    ax.set_xlim(min(X[:, 0]) - 10, max(X[:, 0]) + 10)
    ax.set_ylim(min(X[:, 1]) - 10, max(X[:, 1]) + 10)

    lines = []
    for i, j in zip(*graph.nonzero()):
        lines.append([X[i], X[j]])

    lc = mc.LineCollection(lines, linewidths=.3, colors='#0000007f')
    ax.add_collection(lc)

    plt.savefig(graph_name + '.svg', dpi=1000)
    plt.show()


graph_name = '1138_bus'
mat_data = io.loadmat(graph_name + '.mat')
graph = mat_data['Problem']['A'][0][0]

# graph

# get number of vertices
n = graph.shape[0]
num_of_pivots_arr = np.asarray([50, 100, 150, 200])
sgd_iter = 20
actual_dists = cs.shortest_path(graph, directed=False, unweighted=True)

random_stress = []
mis_stress = []
euc_stress = []
max_min_sp_stress = []
max_min_rand_sp_stress = []
k_means_stress = []
k_means_sp_stress = []
k_means_max_min_sp_stress = []

for num_of_pivots in num_of_pivots_arr:
    X = ss.random_pivots(graph, num_of_pivots, sgd_iter)
    stress = compute_stress(actual_dists, X)
    random_stress.append(stress)

    X = ss.mis_filtration(graph, num_of_pivots, sgd_iter, unweighted=True)
    stress = compute_stress(actual_dists, X)
    mis_stress.append(stress)

    X = ss.max_min_euclidean(graph, num_of_pivots, sgd_iter, unweighted=True)
    stress = compute_stress(actual_dists, X)
    euc_stress.append(stress)

    X = ss.max_min_sp(graph, num_of_pivots, sgd_iter, unweighted=True)
    stress = compute_stress(actual_dists, X)
    max_min_sp_stress.append(stress)

    X = ss.max_min_random_sp(graph, num_of_pivots, sgd_iter, unweighted=True)
    stress = compute_stress(actual_dists, X)
    max_min_rand_sp_stress.append(stress)

    X = ss.k_means_layout(graph, num_of_pivots, sgd_iter, unweighted=True)
    stress = compute_stress(actual_dists, X)
    k_means_stress.append(stress)

    X = ss.k_means_sp(graph, num_of_pivots, sgd_iter, unweighted=True)
    stress = compute_stress(actual_dists, X)
    k_means_sp_stress.append(stress)

    X = ss.k_means_max_min_sp(graph, num_of_pivots, sgd_iter, unweighted=True)
    stress = compute_stress(actual_dists, X)
    k_means_max_min_sp_stress.append(stress)

ax = plt.subplot(111)

ax.plot(num_of_pivots_arr, random_stress, 'o', label='random')
ax.plot(num_of_pivots_arr, mis_stress, 'o', label='MIS filtration')
ax.plot(num_of_pivots_arr, euc_stress, 'o', label='max/min Euclidean')
ax.plot(num_of_pivots_arr, max_min_sp_stress, 'o', label='max/min sp')
ax.plot(num_of_pivots_arr, max_min_rand_sp_stress, 'o', label='max/min random sp')
ax.plot(num_of_pivots_arr, k_means_stress, 'o', label='k-means')
ax.plot(num_of_pivots_arr, k_means_sp_stress, 'o', label='k-means sp')
ax.plot(num_of_pivots_arr, k_means_max_min_sp_stress, 'o', label='k_means + max/min sp')

ax.set_xlim(0, 250)
ax.set_ylim(0, 1)
# Shrink current axis by 20%
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
