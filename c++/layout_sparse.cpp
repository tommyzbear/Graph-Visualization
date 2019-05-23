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

struct term
{
    int i, j;
    double d, wij, wji;
    term(int i, int j, double d, double wij, double wji) : i(i), j(j), d(d), wij(wij), wji(wji) {}
};

std::vector<std::vector<int>> build_graph_unweighted(int n, int m, int *I, int *J);
std::vector<std::vector<edge>> build_graph_weighted(int n, int m, int *I, int *J, double *V);
template <typename T>
std::vector<T> buildGraphArray(int n, int m, int *I, int *J, T *V);

std::vector<int> randomPivots(int n, int nPivots, int *I, int *J);
std::vector<int> misFitration(int n, int m, int nPivots, int *I, int *J, double *V);
std::tuple<std::vector<int>, std::map<int, std::vector<int>>> maxMinEuclidean(int n, int m, int nPivots, int *I, int *J, double *V);
std::tuple<std::vector<int>, std::map<int, std::vector<int>>> maxMinSP(int n, int m, int nPivots, int *I, int *J, double *V);
std::tuple<std::vector<int>, std::map<int, std::vector<int>>> maxMinRandomSP(int n, int nPivots, int *I, int *J, double *V);
std::vector<int> kMeansLayout(int n, int nPivots, int *I, int *J, double *V);

template <typename T>
std::vector<double> euclideanDist(int n, std::vector<T> coordinates, int pivot);
template <class Iter_T, class Iter2_T>
double vectorDistance(Iter_T first, Iter_T last, Iter2_T first2);
std::vector<double> dijkstra(int n, int m, int source, int *I, int *J, double *V);
std::vector<int> bfs(int n, int m, int source, int *I, int *J);

std::vector<term> naivePartition(int n, int m, int nPivots, std::vector<int> pivots, std::map<int, std::vector<int>> regions, int *I, int *J, double *V);
std::vector<term> multiSourcePartition(int n, int m, int nPivots, std::vector<int> pivots, std::map<int, std::vector<int>> regions, std::vector<std::vector<int>> graph, int *I, int *J, double *V);
std::vector<term> multiSourcePartitionUnweighted(int n, int m, int nPivots, std::vector<int> pivots, int *I, int *J, double *V);
std::map<std::tuple<int, int>, term> dijkstraFindTerms(int n, int m, int *I, int *J, double *V);

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

std::vector<int> randomPivots(int n, int nPivots, int *I, int *J)
{
    std::srand(time(0));
    std::vector<int> pivots;

    for (int i = 0; i < nPivots; i++)
    {
        pivots.push_back(rand() % n);
    }

    return pivots;
}

std::vector<int> misFitration(int n, int m, int nPivots, int *I, int *J, double *V)
{
    int layerIndex = 1;

    std::vector<int> previousPivot{};
    std::vector<int> pivots(n);
    std::vector<int> remainingVertices(n);
    int numOfNodes = n;

    // fills the pivot list with consecutive integers starting from 0 to n-1
    std::iota(pivots.begin(), pivots.end(), 0);
    std::iota(remainingVertices.begin(), remainingVertices.end(), 0);

    while (numOfNodes > nPivots)
    {
        previousPivot = pivots;
        int pTemp = pivots[rand() % pivots.size()];
        pivots.clear();
        bool empty = false;
        while (!empty)
        {
            std::vector<int> pathLengths = bfs(n, m, pTemp, I, J);
            pivots.push_back(pTemp);
            std::vector<int> remainingVerticesTemp;
            for (int i = 0; i < sizeof(remainingVertices); i++)
            {
                if (pathLengths[remainingVertices[i]] > pow(2, layerIndex) && (std::find(pivots.begin(), pivots.end(), remainingVertices[i]) != pivots.end()))
                {
                    remainingVerticesTemp.push_back(remainingVertices[i]);
                }
            }
            empty = remainingVerticesTemp.empty();
            pTemp = remainingVerticesTemp[rand() % remainingVerticesTemp.size()];
        }
        layerIndex++;
        numOfNodes = pivots.size();
    }

    for (int i = 0; i < previousPivot.size(); i++)
    {
        if (std::find(pivots.begin(), pivots.end(), previousPivot[i]) != pivots.end())
        {
            previousPivot.erase(previousPivot.begin() + i);
        }
    }

    while (pivots.size() < nPivots)
    {
        int i = rand() % previousPivot.size();
        int p = previousPivot[i];
        pivots.push_back(p);
        previousPivot.erase(previousPivot.begin() + i);
    }

    return pivots;
}

std::tuple<std::vector<int>, std::map<int, std::vector<int>>> maxMinEuclidean(int n, int m, int nPivots, int *I, int *J, double *V)
{
    std::srand(time(0));

    int p0 = rand() % n;

    std::vector<int> pivots = {p0};

    std::vector<double> graph = buildGraphArray(n, m, I, J, V);

    std::map<int, std::vector<double>> shortestDist;

    shortestDist[p0] = euclideanDist(n, graph, p0);

    std::vector<std::tuple<double, int>> mins;
    for (int i = 0; i < n; i++)
    {
        mins.push_back({shortestDist[p0][i], p0});
    }

    for (int i = 1; i < nPivots; i++)
    {
        int argMax = 0;
        for (int k = 1; k < n; k++)
        {
            if (std::get<0>(mins[k]) > std::get<0>(mins[argMax]))
            {
                argMax = k;
            }
        }

        pivots.push_back(argMax);

        shortestDist[pivots[i]] = euclideanDist(n, graph, pivots[i]);
        for (int j = 0; j < n; j++)
        {
            double temp = shortestDist[pivots[i]][j];
            if (temp < std::get<0>(mins[j]))
            {
                mins[j] = {temp, pivots[i]};
            }
        }
    }

    std::map<int, std::vector<int>> regions;

    for (int p : pivots)
    {
        regions.insert(std::pair<int, std::vector<int>>(p, {}));
    }

    for (int i = 0; i < n; i++)
    {
        int closestPivot = std::get<1>(mins[i]);
        regions[closestPivot].push_back(i);
    }

    return {pivots, regions};
}

template <typename T>
std::vector<double> euclideanDist(int n, std::vector<T> coordinates, int pivot)
{
    std::vector<T> pivotCoordinate(coordinates.begin() + n * pivot, coordinates.begin() + n * (pivot + 1));
    std::vector<double> dist;

    for (int i = 0; i < n; i++)
    {
        std::vector<T> vertexCoordinate(coordinates.begin() + n * i, coordinates.begin() + n * (i + 1));
        dist.push_back(vectorDistance(pivotCoordinate.begin(), pivotCoordinate.end(), vertexCoordinate.begin()));
    }

    return dist;
}

template <class Iter_T, class Iter2_T>
double vectorDistance(Iter_T first, Iter_T last, Iter2_T first2)
{
    double result = 0.0;
    while (first != last)
    {
        double dist = (*first++) - (*first2++);
        result += dist * dist;
    }

    return result > 0.0 ? std::sqrt(result) : throw "accumulation of distances square is negative";
}

std::tuple<std::vector<int>, std::map<int, std::vector<int>>> maxMinSP(int n, int m, int nPivots, int *I, int *J, double *V)
{
    std::srand(time(0));

    int p0 = rand() % n;

    std::vector<int> pivots = {p0};

    std::vector<double> shortestPaths = dijkstra(n, m, p0, I, J, V);

    std::vector<std::tuple<double, int>> mins;

    for (int i = 0; i < n; i++)
    {
        mins.push_back({shortestPaths[i], p0});
    }

    for (int i = 1; i < nPivots; i++)
    {
        int argMax = 0;
        for (int k = 1; k < n; k++)
        {
            if (std::get<0>(mins[k]) > std::get<0>(mins[argMax]))
            {
                argMax = k;
            }
        }

        shortestPaths = dijkstra(n, m, argMax, I, J, V);

        pivots.push_back(argMax);

        for (int j = 0; j < n; j++)
        {
            double temp = shortestPaths[j];
            if (temp < std::get<0>(mins[j]))
            {
                mins[j] = {temp, pivots[i]};
            }
        }
    }

    std::map<int, std::vector<int>> regions;

    for (int p : pivots)
    {
        regions.insert(std::pair<int, std::vector<int>>(p, {}));
    }

    for (int i = 0; i < n; i++)
    {
        int closestPivot = std::get<1>(mins[i]);
        regions[closestPivot].push_back(i);
    }

    return {pivots, regions};
}

std::tuple<std::vector<int>, std::map<int, std::vector<int>>> maxMinRandomSP(int n, int m, int nPivots, int *I, int *J, double *V)
{
    std::srand(time(0));
    std::default_random_engine generator;

    int p0 = rand() % n;

    std::vector<int> pivots = {p0};

    std::vector<double> shortestPaths = dijkstra(n, m, p0, I, J, V);

    std::vector<std::tuple<double, int>> mins;
    for (int i = 0; i < n; i++)
    {
        mins.push_back({shortestPaths[i], p0});
    }

    for (int i = 1; i < nPivots; i++)
    {
        double totalProb = 0;
        std::vector<double> cumulativeProb = {};
        for (int j = 0; j < n; j++)
        {
            totalProb += std::get<0>(mins[j]);
            cumulativeProb.push_back(totalProb);
        }

        std::uniform_real_distribution<double> distribution(0.0, totalProb);
        double sample = distribution(generator);

        for (int j = 0; j < n; j++)
        {
            if (sample < cumulativeProb[j])
            {
                pivots.push_back(j);
                break;
            }
        }

        shortestPaths = dijkstra(n, m, pivots[i], I, J, V);

        for (int j = 0; j < n; j++)
        {
            double temp = shortestPaths[j];
            if (temp < std::get<0>(mins[j]))
            {
                mins[j] = {temp, pivots[i]};
            }
        }
    }

    std::map<int, std::vector<int>> regions;

    for (int p : pivots)
    {
        regions.insert(std::pair<int, std::vector<int>>(p, {}));
    }

    for (int i = 0; i < n; i++)
    {
        int closestPivot = std::get<1>(mins[i]);
        regions[closestPivot].push_back(i);
    }

    return {pivots, regions};
}

std::vector<int> kMeansLayout(int n, int nPivots, int *I, int *J, double *V)
{
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

std::vector<term> multiSourcePartition(int n, int m, int nPivots, std::vector<int> pivots, std::map<int, std::vector<int>> regions, std::vector<std::vector<int>> graph, int *I, int *J, double *V)
{
    std::map<int, int> regionAssignment;
    for (std::map<int, std::vector<int>>::iterator it = regions.begin(); it != regions.end(); it++)
    {
        for (int i : it->second)
        {
            regionAssignment[i] = it->first;
        }
    }

    std::map<std::tuple<int, int>, term> constraints = dijkstraFindTerms(n, m, I, J, V);
    std::vector<term> terms = {};
    std::map<int, int> s;

    for (int p : pivots)
    {
        std::vector<double> shortestPath = dijkstra(n, m, p, I, J, V);
        std::queue<int> q1;
        std::queue<int> q2;

        std::vector<bool> mark(n, false);

        mark[p] = true;
        q1.push(p);
        s[p] = 1;
        q1.push(-1);
        int dist = 0;
        int preDist = 1;
        while (q1.size() > 1)
        {
            int index = q1.front();
            q1.pop();

            if (index < 0)
            {
                q1.push(-1);
                q2.push(-1);
                dist++;
                if (s[p] < regions[p].size())
                {
                    if ((dist % preDist == 0) && (dist / preDist == 2))
                    {
                        preDist++;
                        while (!q2.empty())
                        {
                            int regionalIndex = q2.front();
                            q2.pop();
                            if (regionalIndex < 0)
                            {
                                break;
                            }

                            int currentRegion = regionAssignment[regionalIndex];
                            s[currentRegion]++;
                        }
                    }
                }

                continue;
            }

            if ((p != index) && (graph[p][index] == 0) && (constraints.find({p, index}) == constraints.end()) && (constraints.find({index, p}) == constraints.end()))
            {
                double wij = s[p] / std::pow(shortestPath[index], 2);
                auto tempTerm = term(index, p, shortestPath[index], wij, 0);
                constraints[{p, index}] = tempTerm;
            }

            for (int j : graph[p])
            {
                if ((j != 0) && (!mark[j]))
                {
                    mark[j] = true;
                    q1.push(j);
                    if (std::find(regions.begin(), regions.end(), j) != regions.end())
                    {
                        q2.push(j);
                    }
                }
            }
        }
    }

    for (std::map<std::tuple<int, int>, term>::iterator it = constraints.begin(); it != constraints.end(); it++)
    {
        terms.push_back(it->second);
    }

    return terms;
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
