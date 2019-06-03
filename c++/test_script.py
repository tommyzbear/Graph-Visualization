import sparse_layout as cpp
import numpy as np
import scipy.io as io
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import time

graph_name = 'dwt_1005'
mat_data = io.loadmat(graph_name + '.mat')
graph = mat_data['Problem']['A'][0][0]
# graph_name = 'commanche_dual'
# mat_data = io.loadmat(graph_name + '.mat')
# graph = mat_data['Problem']['A'][0][0]

I, J, V = sparse.find(graph)

if len(I) != len(J):
    raise "length of edge indices I and J not equal"

n = max(max(I), max(J)) + 1
X = np.random.rand(n, 2)

start_time = time.time()
cpp.sparse_layout_MSSP_unweightd(X, I, J, "max_min_euc", 200, 15, 1.0)
# cpp.sparse_layout_naive_unweighted(X, I, J, "max_min_euc", 200, 15, 1.0)
# cpp.layout_unweighted(X, I, J, 30, 1.0)
# cpp.layout_weighted(X, I, J, V, 30, 1.0)
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

plt.savefig(graph_name + '.svg', dpi=1000)
plt.show()