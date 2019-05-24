#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <time.h>
#include <numeric>
#include <algorithm>
#include <complex>
#include <map>
#include <limits>
#include <queue>
#include <unordered_set>
#include <unordered_map>

struct edge
{
    // NOTE: this will be used for 'invisible' edges too!
    int target;
    double weight;
    edge(int target, double weight) : target(target), weight(weight) {}
};

struct edge_comp
{
    bool operator()(const edge &lhs, const edge &rhs) const
    {
        return lhs.weight > rhs.weight;
    }
};

// Graph builders
std::vector<std::vector<int>> build_graph_unweighted(int n, int m, int *I, int *J);
std::vector<std::vector<edge>> build_graph_weighted(int n, int m, int *I, int *J, double *V);
template <typename T>
std::vector<T> buildGraphArray(int n, int m, int *I, int *J, T *V);

std::vector<std::vector<int>> build_graph_unweighted(int n, int m, int *I, int *J)
{
    // used to make graph undirected, in case it is not already
    std::vector<std::unordered_set<int>> undirected(n);
    std::vector<std::vector<int>> graph(n);

    for (int ij = 0; ij < m; ij++)
    {
        int i = I[ij], j = J[ij];
        if (i >= n || j >= n)
            throw "i or j bigger than n";

        if (undirected[j].find(i) == undirected[j].end()) // if edge not seen
        {
            undirected[i].insert(j);
            undirected[j].insert(i);
            graph[i].push_back(j);
            graph[j].push_back(i);
        }
    }
    return graph;
}

std::vector<std::vector<edge>> build_graph_weighted(int n, int m, int *I, int *J, double *V)
{
    // used to make graph undirected, in case graph is not already
    std::vector<std::unordered_map<int, double>> undirected(n);
    std::vector<std::vector<edge>> graph(n);

    for (int ij = 0; ij < m; ij++)
    {
        int i = I[ij], j = J[ij];
        if (i >= n || j >= n)
            throw "i or j bigger than n";

        double v = V[ij];
        if (v <= 0)
            throw "v less or equal 0";

        if (undirected[j].find(i) == undirected[j].end()) // if key not there
        {
            undirected[i].insert({j, v});
            undirected[j].insert({i, v});
            graph[i].push_back(edge(j, v));
            graph[j].push_back(edge(i, v));
        }
        else
        {
            if (undirected[j][i] != v)
                throw "graph weights not symmetric";
        }
    }
    return graph;
}

template <typename T>
std::vector<T> buildGraphArray(int n, int m, int *I, int *J, T *V)
{
    std::vector<std::unordered_map<int, T>> undirected(n);
    std::vector<T> graph(n * n, 0);

    for (int ij = 0; ij < m; ij++)
    {
        int i = I[ij], j = J[ij];
        if (i >= n || j >= n)
        {
            throw "i or j bigger than n";
        }

        double v = V[ij];
        if (v <= 0)
        {
            throw "v less or equal 0";
        }

        if (undirected[j].find(i) == undirected[j].end())
        {
            undirected[i].insert({j, v});
            undirected[j].insert({i, v});
            graph[i * n + j] = v;
            graph[j * n + i] = v;
        }
        else
        {
            if (undirected[j][i] != v)
            {
                throw "graph weights not symmetric";
            }
        }
    }

    return graph;
}
