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
#include "GraphBuilders.hpp"

std::vector<double> dijkstra(int n, int m, int source, int *I, int *J, double *V);
std::vector<int> bfs(int n, int m, int source, int *I, int *J);

// dijkstra algorithm to find shortest paths to every other vertices from source node
std::vector<double> dijkstra(int n, int m, int source, int *I, int *J, double *V)
{
    auto graph = build_graph_weighted(n, m, I, J, V);
    std::vector<bool> visited(n, false);
    std::vector<double> dist(n, std::numeric_limits<double>::max());
    std::vector<int> prev(n, -1);
    dist[source] = 0;

    std::priority_queue<edge, std::vector<edge>, edge_comp> queue;
    // queue.push(edge(source, 0));

    for (int i = 0; i < n; i++)
    {
        queue.push(edge(i, dist[i]));
    }

    while (!queue.empty())
    {
        int current = queue.top().target;
        double d_ij = queue.top().weight;
        queue.pop();

        if (!visited[current])
        {
            visited[current] = true;

            for (edge e : graph[current])
            {
                // here the edge is not 'invisible'
                int next = e.target;
                double weight = e.weight;

                if (dist[next] > d_ij + weight)
                {
                    dist[next] = d_ij + weight; // update tentative value of d
                    queue.push(edge(next, dist[next]));
                }
            }
        }
    }

    return dist;
}

std::vector<int> bfs(int n, int m, int source, int *I, int *J)
{
    auto graph = build_graph_unweighted(n, m, I, J);

    std::vector<int> d(n, -1); // distances from source
    std::queue<int> q;

    d[source] = 0;
    q.push(source);

    while (!q.empty())
    {
        int current = q.front();
        q.pop();
        for (int next : graph[current])
        {
            if (d[next] == -1)
            {
                q.push(next);
                int d_ij = d[current] + 1;
                d[next] = d_ij;
            }
        }
    }

    return d;
}