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
#include "ShortestPaths.hpp"

struct term
{
    int i, j;
    double d, wij, wji;
    term(int i, int j, double d, double wij, double wji) : i(i), j(j), d(d), wij(wij), wji(wji) {}
};

// Regions and terms computation
std::vector<term> naivePartition(int n, int m, int nPivots, std::vector<int> pivots, std::map<int, std::vector<int>> regions, int *I, int *J, double *V);
std::vector<term> multiSourcePartitionUnweighted(int n, int m, int nPivots, std::vector<int> pivots, int *I, int *J, double *V);
std::map<std::tuple<int, int>, term> dijkstraFindTerms(int n, int m, int *I, int *J, double *V);
std::vector<double> schedule(const std::vector<term> &terms, int t_max, double eps);

// SGD graph drawing
void sgd(double *X, std::vector<term> &terms, const std::vector<double> &etas);

void sgd(double *X, std::vector<term> &terms, const std::vector<double> &etas)
{
    // iterate through step sizes
    int iteration = 0;
    for (double eta : etas)
    {
        // shuffle terms
        std::random_shuffle(terms.begin(), terms.end());

        for (const term &t : terms)
        {
            // cap step size
            double wij = t.wij;
            double wji = t.wji;
            double mu_i = std::min(wij * eta, 1.0);
            double mu_j = std::min(wji * eta, 1.0);
            double d_ij = t.d;
            int i = t.i, j = t.j;

            double dx = X[i * 2] - X[j * 2], dy = X[i * 2 + 1] - X[j * 2 + 1];
            double mag = sqrt(dx * dx + dy * dy);

            double r = (mag - d_ij) / (2 * mag);
            double r_x = r * dx;
            double r_y = r * dy;

            X[i * 2] -= mu_i * r_x;
            X[i * 2 + 1] -= mu_i * r_y;
            X[j * 2] += mu_j * r_x;
            X[j * 2 + 1] += mu_j * r_y;
        }
        std::cerr << ++iteration << ", eta: " << eta << std::endl;
    }
}

std::vector<double> schedule(const std::vector<term> &terms, int tMax, double eps)
{
    double wMin = std::numeric_limits<double>::max(), wMax = 0;
    for (int i = 0; i < terms.size(); i++)
    {
        double wij = terms[i].wij;
        double wji = terms[i].wji;
        wMin = std::min(wMin, wij, wji);
        wMax = std::max(wMax, wij, wji);
    }

    double etaMax = 1.0 / wMin;
    double etaMin = 1.0 / wMax;

    double lambda = log(etaMax / etaMin) / (tMax - 1);

    // initialize step sizes
    std::vector<double> etas;
    etas.reserve(tMax);
    for (int t = 0; t < tMax; t++)
    {
        etas.push_back(etaMax * exp(-lambda * t));
    }

    return etas;
}

std::vector<term> naivePartition(int n, int m, int nPivots, std::vector<int> pivots, std::map<int, std::vector<int>> regions, int *I, int *J, double *V)
{
    std::map<std::tuple<int, int>, term> constraints = dijkstraFindTerms(n, m, I, J, V);
    std::vector<term> terms = {};

    for (int p : pivots)
    {
        std::vector<double> shortestPath = dijkstra(n, m, p, I, J, V);
        for (int i = 0; i < n; i++)
        {
            int sum = 0;
            for (int j : regions[p])
            {
                if (shortestPath[j] <= shortestPath[i] / 2)
                {
                    sum++;
                }
            }

            double wip = sum / std::pow(shortestPath[i], 2);
            constraints[{i, p}] = term(i, p, shortestPath[i], wip, 0);
            // terms.push_back(term(i, p, shortestPath[i], wip, 0));
        }
    }

    // availableTerms.insert(availableTerms.end(), terms.begin(), terms.end());
    for (std::map<std::tuple<int, int>, term>::iterator it = constraints.begin(); it != constraints.end(); it++)
    {
        terms.push_back(it->second);
    }

    return terms;
}

std::map<std::tuple<int, int>, term> dijkstraFindTerms(int n, int m, int *I, int *J, double *V)
{
    auto graph = build_graph_weighted(n, m, I, J, V);

    int nC2 = (n * (n - 1)) / 2;
    std::map<std::tuple<int, int>, term> constraints;
    // std::vector<term> terms;
    // terms.reserve(nC2);

    int terms_size_goal = 0; // to keep track of when to stop searching i<j

    for (int source = 0; source < n - 1; source++) // no need to do final vertex because i<j
    {
        std::vector<bool> visited(n, false);
        std::vector<double> d(n, std::numeric_limits<double>::max()); // init 'tentative' distances to infinity

        // I am not using a fibonacci heap. I AM NOT USING A FIBONACCI HEAP
        // edges are used 'invisibly' here
        std::priority_queue<edge, std::vector<edge>, edge_comp> pq;

        d[source] = 0;
        pq.push(edge(source, 0));

        terms_size_goal += n - source - 1; // this is how many terms exist for i<j

        while (!pq.empty() && constraints.size() <= terms_size_goal)
        {
            int current = pq.top().target;
            double d_ij = pq.top().weight;
            pq.pop();

            if (!visited[current]) // ignore redundant elements in queue
            {
                visited[current] = true;

                if (source < current) // only add terms for i<j
                {
                    double w_ij = 1.0 / (d_ij * d_ij);
                    constraints[{source, current}] = term(source, current, d_ij, w_ij, w_ij);
                }
                for (edge e : graph[current])
                {
                    // here the edge is not 'invisible'
                    int next = e.target;
                    double weight = e.weight;

                    if (d[next] > d_ij + weight)
                    {
                        d[next] = d_ij + weight; // update tentative value of d
                        pq.push(edge(next, d[next]));
                    }
                }
            }
        }
        if (constraints.size() != terms_size_goal)
        {
            throw "graph is not strongly connected";
        }
    }
    return constraints;
}

std::vector<term> multiSourcePartitionUnweighted(int n, int m, int nPivots, std::vector<int> pivots, int *I, int *J, int *V)
{
    std::vector<int> graph = buildGraphArray(n, m, I, J, V);
    std::map<std::tuple<int, int>, term> terms;

    for (int e = 0; e < sizeof(I); e++)
    {
        int i = I[e], j = J[e];
        double v = V[e];
        if (i < j)
        {
            double w = 1 / pow(v, 2);
            terms[std::tuple<int, int>(i, j)] = term(i, j, v, w, w);
        }
    }

    // initialize s for each pivot
    std::map<int, int> s;

    std::vector<bool> markForRegion(n, false);
    std::map<int, std::vector<bool>> markForWeights;
    std::map<int, std::vector<int>> level;
    std::map<int, std::vector<int>> regions;
    std::vector<int> regionAssignment(n, -1);
    std::queue<int> q1;
    std::map<int, std::queue<int>> q2;
    std::queue<std::tuple<int, int>> q3;

    std::map<int, int> dist;
    std::map<int, int> prevDist;

    for (int p : pivots)
    {
        regions[p] = std::vector<int>{p};
        regionAssignment[p] = p;
        q1.push(p);
        q2[p] = std::queue<int>();
        markForRegion[p] = true;
        markForWeights[p] = std::vector<bool>(n, false);
        markForWeights[p][p] = true;
        level[p] = std::vector<int>(n, 0);
        s[p] = 1;
        dist[p] = 0;
        prevDist[p] = 1;
    }

    while (!q1.empty())
    {
        int index = q1.front();
        q1.pop();
        int pivot_index = regionAssignment[index];
        int curLevel = level[pivot_index][index];

        if (curLevel > dist[pivot_index])
            dist[pivot_index] = curLevel;

        if ((dist[pivot_index] % prevDist[pivot_index] == 0) && (dist[pivot_index] / prevDist[pivot_index] == 2))
        {
            while (!q2[pivot_index].empty())
            {
                int regionalIndex = q2[pivot_index].front();
                if (level[pivot_index][regionalIndex] <= prevDist[pivot_index])
                {
                    s[pivot_index]++;
                    q2[pivot_index].pop();
                }
                else
                {
                    break;
                }
            }

            prevDist[pivot_index]++;
        }

        if ((pivot_index != index) && (graph[n * pivot_index + index] == 0) && (terms.find(std::tuple(pivot_index, index)) == terms.end()) && (terms.find(std::tuple<int, int>(index, pivot_index)) == terms.end()))
        {
            double wij = s[pivot_index] / pow(level[pivot_index][index], 2);
            terms[std::tuple(index, pivot_index)] = term(index, pivot_index, level[pivot_index][index], wij, 0);
        }

        for (int i = n * pivot_index; i < n * (pivot_index + 1); i++)
        {
            if (graph[i] != 0)
            {
                int neighbour = i - n * pivot_index;
                if (!markForRegion[neighbour])
                {
                    markForRegion[neighbour] = true;
                    markForWeights[pivot_index][neighbour] = true;
                    level[pivot_index][neighbour] = curLevel + 1;
                    q1.push(neighbour);
                    regionAssignment[neighbour] = pivot_index;
                    regions[pivot_index].push_back(neighbour);
                    q2[pivot_index].push(neighbour);
                }
                if (!markForWeights[pivot_index][neighbour])
                {
                    markForWeights[pivot_index][neighbour] = true;
                    level[pivot_index][neighbour] = curLevel + 1;
                    q3.push(std::tuple(neighbour, pivot_index));
                }
            }
        }
    }

    while (!q3.empty())
    {
        int index = std::get<0>(q3.front());
        int pivot_index = std::get<1>(q3.front());
        q3.pop();
        int curLevel = level[pivot_index][index];

        if (curLevel > dist[pivot_index])
            dist[pivot_index] = curLevel;

        if ((dist[pivot_index] % prevDist[pivot_index] == 0) && (dist[pivot_index] / prevDist[pivot_index] == 2))
        {
            while (!q2[pivot_index].empty())
            {
                int regionalIndex = q2[pivot_index].front();
                if (level[pivot_index][regionalIndex] <= prevDist[pivot_index])
                {
                    s[pivot_index]++;
                    q2[pivot_index].pop();
                }
                else
                {
                    break;
                }
            }

            prevDist[pivot_index]++;
        }

        double wij = s[pivot_index] / pow(level[pivot_index][index], 2);
        terms[std::tuple(index, pivot_index)] = term(index, pivot_index, level[pivot_index][index], wij, 0);

        for (int i = n * pivot_index; i < n * (pivot_index + 1); i++)
        {
            if (graph[i] != 0)
            {
                int neighbour = i - n * pivot_index;

                if (!markForWeights[pivot_index][neighbour])
                {
                    markForWeights[pivot_index][neighbour] = true;
                    level[pivot_index][neighbour] = curLevel + 1;
                    q3.push(std::tuple(neighbour, pivot_index));
                }
            }
        }
    }

    std::vector<term> terms_vec;

    for (std::map<std::tuple<int, int>, term>::iterator it = terms.begin(); it != terms.end(); it++)
    {
        terms_vec.push_back(it->second);
    }

    return terms_vec;
}
