import sparse_layout as cpp
import numpy as np
import scipy.io as io
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import time
import scipy.sparse.csgraph as cs

graph_name = 'commanche_weighted'
mat_data = io.loadmat(graph_name + '.mat')
graph = mat_data['graph']
# graph_name = 'dwt_1005'
# mat_data = io.loadmat(graph_name + '.mat')
# graph = mat_data['Problem']['A'][0][0]

I, J, V = sparse.find(graph)

if len(I) != len(J):
    raise "length of edge indices I and J not equal"

n = max(max(I), max(J)) + 1
X = np.random.rand(n, 2)

start_time = time.time()
# cpp.sparse_layout_MSSP_unweightd(X, I, J, "max_min_random_sp", 200, 20, 0.1)
# cpp.sparse_layout_naive_unweighted(X, I, J, "max_min_random_sp", 200, 15, 0.1)
# cpp.layout_unweighted(X, I, J, 15, 0.1)
# cpp.layout_weighted(X, I, J, V, 20, 0.1)
# cpp.sparse_layout_naive_weighted(X, I, J, V, "max_min_random_sp", 700, 20, 0.1)
cpp.sparse_layout_MSSP_weightd(X, I, J, V, "random", 200, 20, 0.1)
end_time = time.time()

print("Computation time: %.2f" % (end_time - start_time))

plt.axis('equal')
ax = plt.axes()
ax.set_xlim(min(X[:, 0])-10, max(X[:, 0])+10)
ax.set_ylim(min(X[:, 1])-10, max(X[:, 1])+10)

lines = []
for i, j in zip(*graph.nonzero()):
    lines.append([X[i], X[j]])

lc = mc.LineCollection(lines, linewidths=.3, colors='#0000007f')
ax.add_collection(lc)

# plt.savefig(graph_name + '.svg', dpi=1000)
plt.savefig(graph_name + '.png', dpi=1000)
plt.show()

shortest_paths = cs.dijkstra(graph, unweighted=True)
stress = 0
for i in range(n):
    for j in range(i):
        pq = X[i] - X[j]
        mag = np.linalg.norm(pq)
        
        stress += (1/shortest_paths[i,j]**2) * (shortest_paths[i,j]-mag)**2
        

print('stress = {:.0f}'.format(stress))