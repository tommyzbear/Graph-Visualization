import scipy.io as io
import numpy as np
import scipy.sparse.csgraph as cs
import math
import random
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import time
from sklearn.cluster import KMeans
from scipy.spatial import distance
from tools.MSSP import MSSP, MSSP_unweighted, vertices_partition, MSSP_ultimate


graph_name = 'dwt_1005'
mat_data = io.loadmat(graph_name + '.mat')

graph = mat_data['Problem']['A'][0][0]
graph_arr = graph.toarray()

# get number of vertices
n = graph.shape[0]
num_of_pivots = 100

# maximum iteration for K_means
max_iter = 50

start_time = time.time()

shortest_paths = cs.shortest_path(graph)

# randomly choose a pivot
p0 = np.random.randint(0, n)
pivots = [p0]

# initialize list of minimum distances to each pivots
mins = []
for i in range(n):
    mins.append([shortest_paths[p0][i], p0])

for i in range(1, num_of_pivots):
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

max_iter = 50

kmeans = KMeans(n_init=1, init=np.asarray(pivots_shortest_paths), n_clusters=num_of_pivots, max_iter=max_iter).fit(shortest_paths)
centroids = kmeans.cluster_centers_

pivots = []

for c in centroids:
    euc_dist = distance.cdist([c], shortest_paths, 'euclidean')
    pivots.append(euc_dist.argmin())

# dists, weights = MSSP(graph, pivots, shortest_paths)
# regions, region_assignment = vertices_partition(graph, pivots)
# dists, weights = MSSP_unweighted(graph, pivots, regions)
dists, weights = MSSP_ultimate(graph, pivots)

constraints = []
for ij in dists.keys():
    i = ij[0]
    j = ij[1]
    if i < j:
        constraints.append((i, j, dists[ij]))

print("number of constraints found: ", len(constraints))

w_max = 0
w_min = math.inf

for w in weights.values():
    if w != 0:
        w_max = max(w, w_max)
        w_min = min(w, w_min)

c_max = 1 / w_min
c_min = 0.1 / w_max

num_iter = 15

lambd = np.log(c_min / c_max) / (num_iter - 1)
print("{} {} {}".format(w_max, w_min, lambd))

cool = lambda k: c_max * np.exp(lambd * k)

X = np.random.rand(n, 2)
for k in range(num_iter + 5):
    random.shuffle(constraints)
    c = cool(k - 1)

    for i, j, d in constraints:
        w_i = min(weights[(i, j)] * c, 1)
        w_j = min(weights[(j, i)] * c, 1)
        ij = X[i] - X[j]
        mag = np.linalg.norm(ij)
        m = ((d - mag) / 2) * (ij / mag)

        X[i] += w_i * m
        X[j] -= w_j * m

    print('.', end='')

end_time = time.time()

print("Computation time: %.2f" % (end_time - start_time))

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
