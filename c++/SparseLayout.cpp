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
#include <iomanip>
#include <string.h>
// #include <ctime>
// #include <tbb/parallel_for.h>

enum samplingSchemeCode
{
    erandom,
    emisFiltration,
    emaxMinEuc,
    emaxMinSP,
    emaxMinRandomSP,
    ekMeans,
    ekMeansSP,
    ekMeansMaxMinSP
};

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

struct term
{
    int i, j;
    double d, wij, wji;
    term(int i, int j, double d, double wij, double wji) : i(i), j(j), d(d), wij(wij), wji(wji) {}
    term() = default;
};

// Graph builders
std::vector<std::vector<int>> buildGraphUnweighted(int n, int m, int *I, int *J);
std::vector<std::vector<edge>> buildGraphWeighted(int n, int m, int *I, int *J, double *V);
template <typename T>
std::vector<T> buildGraphArray(int n, int m, int *I, int *J, T *V);
std::vector<int> buildGraphArray(int n, int m, int *I, int *J);

// find shortest paths between source node to other vertices
std::vector<double> dijkstra(int n, int m, int source, int *I, int *J, double *V);
// find unweighted graph theoratic distances between source node to other vertices
std::vector<int> bfs(int n, int m, int source, int *I, int *J);
// Naive find terms
std::vector<term> bfs(int n, int m, int *I, int *J);
std::vector<term> dijkstra(int n, int m, int *I, int *J, double *V);

// Sampling schemes
void randomPivots(int n, int nPivots);
void misFitration(int n, int m, int nPivots, std::vector<int> &pivots, int *I, int *J);
void maxMinEuclidean(int n, int m, int nPivots, std::vector<int> &pivots, int *I, int *J);
void maxMinSP(int n, int m, int nPivots, std::map<int, std::vector<int>> &shortestPaths, std::vector<int> &pivots, std::map<int, std::vector<int>> &regions, int *I, int *J);
void maxMinSP(int n, int m, int nPivots, std::vector<int> &pivots, int *I, int *J);
void maxMinRandomSP(int n, int m, int nPivots, std::map<int, std::vector<int>> &shortestPaths, std::vector<int> &pivots, std::map<int, std::vector<int>> &regions, int *I, int *J);
void maxMinRandomSP(int n, int m, int nPivots, std::vector<int> &pivots, int *I, int *J);
// std::vector<int> kMeansLayout(int n, int nPivots, int *I, int *J, double *V);

template <typename T>
std::vector<double> euclideanDist(int n, std::vector<T> coordinates, int pivot);
template <class Iter_T, class Iter2_T>
double vectorDistance(Iter_T first, Iter_T last, Iter2_T first2);
samplingSchemeCode hashSamplingScheme(std::string const &inString);

// Regions and terms computation
// std::vector<term> naivePartition(int n, int m, int nPivots, std::vector<int> pivots, std::map<int, std::vector<int>> regions, int *I, int *J, double *V);
// std::vector<term> multiSourcePartitionUnweighted(int n, int m, int nPivots, std::vector<int> pivots, int *I, int *J, double *V);
// std::map<std::tuple<int, int>, term> dijkstraFindTerms(int n, int m, int *I, int *J, double *V);
std::vector<double> schedule(const std::vector<term> &terms, int t_max, double eps);

// SGD graph drawing
void sgd(double *X, std::vector<term> &terms, const std::vector<double> &etas);

// Layout
void layout_unweighted(int n, double *X, int m, int *I, int *J, int t_max, double eps);
void sparse_layout_naive_unweighted(int n, double *X, int m, int *I, int *J, char *sampling_scheme, int k, int t_max, double eps);
void sparse_layout_MSSP_unweightd(int n, double *X, int m, int *I, int *J, int k, int t_max, double eps);

std::vector<std::vector<int>> buildGraphUnweighted(int n, int m, int *I, int *J)
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

std::vector<std::vector<edge>> buildGraphWeighted(int n, int m, int *I, int *J, double *V)
{
    // used to make graph undirected, in case graph is not already
    std::vector<std::map<int, double>> undirected(n);
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
    std::vector<std::map<int, T>> undirected(n);
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

std::vector<int> buildGraphArray(int n, int m, int *I, int *J)
{
    std::vector<std::map<int, int>> undirected(n);
    std::vector<int> graph(n * n, 0);

    for (int ij = 0; ij < m; ij++)
    {
        int i = I[ij], j = J[ij];
        if (i >= n || j >= n)
        {
            throw "i or j bigger than n";
        }

        if (undirected[j].find(i) == undirected[j].end())
        {
            undirected[i].insert({j, 1});
            undirected[j].insert({i, 1});
            graph[i * n + j] = 1;
            graph[j * n + i] = 1;
        }
        else
        {
            if (undirected[j][i] != 1)
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
    auto graph = buildGraphWeighted(n, m, I, J, V);
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
    auto graph = buildGraphUnweighted(n, m, I, J);

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

std::vector<term> bfs(int n, int m, int *I, int *J)
{
    auto graph = buildGraphUnweighted(n, m, I, J);

    int nC2 = (n * (n - 1)) / 2;

    std::cerr << "Total terms = " << nC2 << std::endl;
    std::cerr << "Size of a term in bytes: " << sizeof(term) << std::endl;

    std::vector<term> terms;
    terms.reserve(nC2);

    int termSizeGoal = 0;

    for (int source = 0; source < n - 1; source++)
    {
        std::vector<int> d(n, -1);
        std::queue<int> q;

        d[source] = 0;
        q.push(source);

        termSizeGoal += n - source - 1;

        while (!q.empty() && terms.size() <= termSizeGoal)
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
                    if (source < next)
                    {
                        double w_ij = 1.0 / (d_ij * d_ij);
                        terms.push_back(term(source, next, d_ij, w_ij, w_ij));
                    }
                }
            }
        }

        if (terms.size() != termSizeGoal)
        {
            throw "graph is not strongly connected";
        }
    }

    return terms;
}

std::vector<term> dijkstra(int n, int m, int *I, int *J, double *V)
{
    auto graph = buildGraphWeighted(n, m, I, J, V);

    int nC2 = (n * (n - 1)) / 2;
    std::vector<term> terms;
    terms.reserve(nC2);

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

        while (!pq.empty() && terms.size() <= terms_size_goal)
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
                    terms.push_back(term(source, current, d_ij, w_ij, w_ij));
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
        if (terms.size() != terms_size_goal)
        {
            throw "graph is not strongly connected";
        }
    }
    return terms;
}

void randomPivots(int n, int nPivots, std::vector<int> &pivots)
{
    std::srand(time(0));

    for (int i = 0; i < nPivots; i++)
    {
        int p = rand() % n;
        pivots.push_back(p);
        std::cerr << "pivot no." << i << " is " << p << std::endl;
    }
}

void misFitration(int n, int m, int nPivots, std::vector<int> &pivots, int *I, int *J)
{
    int layerIndex = 1;

    std::vector<int> previousPivot;
    pivots = std::vector<int>(n);
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
            for (int i = 0; i < remainingVertices.size(); i++)
            {
                if (pathLengths[remainingVertices[i]] > pow(2, layerIndex) && (std::find(pivots.begin(), pivots.end(), remainingVertices[i]) == pivots.end()))
                {
                    remainingVerticesTemp.push_back(remainingVertices[i]);
                }
            }
            empty = remainingVerticesTemp.empty();
            if (!empty)
            {
                pTemp = remainingVerticesTemp[rand() % remainingVerticesTemp.size()];
            }
            remainingVertices = remainingVerticesTemp;
        }
        layerIndex++;
        numOfNodes = pivots.size();
    }

    if (pivots.size() < nPivots)
    {
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
    }
}

void maxMinEuclidean(int n, int m, int nPivots, std::vector<int> &pivots, int *I, int *J)
{
    std::srand(time(0));

    int p0 = rand() % n;

    pivots.push_back(p0);

    std::vector<int> graph = buildGraphArray(n, m, I, J);

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

    return result >= 0.0 ? std::sqrt(result) : throw "accumulation of distances square is negative";
}

void maxMinSP(int n, int m, int nPivots, std::map<int, std::vector<int>> &shortestPaths, std::vector<int> &pivots, std::map<int, std::vector<int>> &regions, int *I, int *J)
{
    std::srand(time(0));

    int p0 = rand() % n;

    pivots.push_back(p0);

    std::vector<std::tuple<double, int>> mins(n);

    shortestPaths[p0] = bfs(n, m, p0, I, J);

    for (int i = 0; i < n; i++)
    {
        mins[i] = {shortestPaths[p0][i], p0};
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

        shortestPaths[argMax] = bfs(n, m, argMax, I, J);
        pivots.push_back(argMax);

        for (int j = 0; j < n; j++)
        {
            double temp = shortestPaths[argMax][j];
            if (temp < std::get<0>(mins[j]))
            {
                mins[j] = {temp, argMax};
            }
        }
    }

    for (int p : pivots)
    {
        regions[p] = std::vector<int>{};
    }

    for (int i = 0; i < n; i++)
    {
        int closestPivot = std::get<1>(mins[i]);
        regions[closestPivot].push_back(i);
    }
}

void maxMinSP(int n, int m, int nPivots, std::vector<int> &pivots, int *I, int *J)
{
    std::srand(time(0));

    int p0 = rand() % n;

    pivots.push_back(p0);

    std::vector<std::tuple<double, int>> mins(n);
    std::map<int, std::vector<int>> shortestPaths;

    shortestPaths[p0] = bfs(n, m, p0, I, J);

    for (int i = 0; i < n; i++)
    {
        mins[i] = {shortestPaths[p0][i], p0};
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

        shortestPaths[argMax] = bfs(n, m, argMax, I, J);
        pivots.push_back(argMax);

        for (int j = 0; j < n; j++)
        {
            double temp = shortestPaths[argMax][j];
            if (temp < std::get<0>(mins[j]))
            {
                mins[j] = {temp, argMax};
            }
        }
    }
}

void maxMinRandomSP(int n, int m, int nPivots, std::map<int, std::vector<int>> &shortestPaths, std::vector<int> &pivots, std::map<int, std::vector<int>> &regions, int *I, int *J)
{
    std::srand(time(0));
    std::default_random_engine generator;

    int p0 = rand() % n;

    pivots.push_back(p0);

    shortestPaths[p0] = bfs(n, m, p0, I, J);

    std::vector<std::tuple<double, int>> mins;
    for (int i = 0; i < n; i++)
    {
        mins.push_back({shortestPaths[p0][i], p0});
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

        shortestPaths[pivots[i]] = bfs(n, m, pivots[i], I, J);

        for (int j = 0; j < n; j++)
        {
            double temp = shortestPaths[pivots[i]][j];
            if (temp < std::get<0>(mins[j]))
            {
                mins[j] = {temp, pivots[i]};
            }
        }
    }

    for (int p : pivots)
    {
        regions[p] = {};
    }

    for (int i = 0; i < n; i++)
    {
        int closestPivot = std::get<1>(mins[i]);
        regions[closestPivot].push_back(i);
    }
}

void maxMinRandomSP(int n, int m, int nPivots, std::vector<int> &pivots, int *I, int *J)
{
    std::srand(time(0));
    std::default_random_engine generator;
    std::map<int, std::vector<int>> shortestPaths;

    int p0 = rand() % n;

    pivots.push_back(p0);

    shortestPaths[p0] = bfs(n, m, p0, I, J);

    std::vector<std::tuple<double, int>> mins;
    for (int i = 0; i < n; i++)
    {
        mins.push_back({shortestPaths[p0][i], p0});
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

        shortestPaths[pivots[i]] = bfs(n, m, pivots[i], I, J);

        for (int j = 0; j < n; j++)
        {
            double temp = shortestPaths[pivots[i]][j];
            if (temp < std::get<0>(mins[j]))
            {
                mins[j] = {temp, pivots[i]};
            }
        }
    }
}

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
        if (wij != 0)
        {
            wMin = wij < wMin ? wij : wMin;
            wMax = wij > wMax ? wij : wMax;
        }
        if (wji != 0)
        {
            wMin = wji < wMin ? wji : wMin;
            wMax = wji > wMax ? wji : wMax;
        }
    }

    double etaMax = 1.0 / wMin;
    double etaMin = eps / wMax;

    std::cerr << "w_min = " << wMin << ", w_max = " << wMax << std::endl;

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

// std::vector<term> naivePartition(int n, int m, int nPivots, std::vector<int> pivots, std::map<int, std::vector<int>> regions, int *I, int *J, double *V)
// {
//     std::map<std::tuple<int, int>, term> constraints = dijkstraFindTerms(n, m, I, J, V);
//     std::vector<term> terms = {};

//     for (int p : pivots)
//     {
//         std::vector<double> shortestPath = dijkstra(n, m, p, I, J, V);
//         for (int i = 0; i < n; i++)
//         {
//             int sum = 0;
//             for (int j : regions[p])
//             {
//                 if (shortestPath[j] <= shortestPath[i] / 2)
//                 {
//                     sum++;
//                 }
//             }

//             double wip = sum / std::pow(shortestPath[i], 2);
//             constraints[{i, p}] = term(i, p, shortestPath[i], wip, 0);
//             // terms.push_back(term(i, p, shortestPath[i], wip, 0));
//         }
//     }

//     // availableTerms.insert(availableTerms.end(), terms.begin(), terms.end());
//     for (std::map<std::tuple<int, int>, term>::iterator it = constraints.begin(); it != constraints.end(); it++)
//     {
//         terms.push_back(it->second);
//     }

//     return terms;
// }

// std::map<std::tuple<int, int>, term> dijkstraFindTerms(int n, int m, int *I, int *J, double *V)
// {
//     auto graph = buildGraphWeighted(n, m, I, J, V);

//     int nC2 = (n * (n - 1)) / 2;
//     std::map<std::tuple<int, int>, term> constraints;
//     // std::vector<term> terms;
//     // terms.reserve(nC2);

//     int terms_size_goal = 0; // to keep track of when to stop searching i<j

//     for (int source = 0; source < n - 1; source++) // no need to do final vertex because i<j
//     {
//         std::vector<bool> visited(n, false);
//         std::vector<double> d(n, std::numeric_limits<double>::max()); // init 'tentative' distances to infinity

//         // I am not using a fibonacci heap. I AM NOT USING A FIBONACCI HEAP
//         // edges are used 'invisibly' here
//         std::priority_queue<edge, std::vector<edge>, edge_comp> pq;

//         d[source] = 0;
//         pq.push(edge(source, 0));

//         terms_size_goal += n - source - 1; // this is how many terms exist for i<j

//         while (!pq.empty() && constraints.size() <= terms_size_goal)
//         {
//             int current = pq.top().target;
//             double d_ij = pq.top().weight;
//             pq.pop();

//             if (!visited[current]) // ignore redundant elements in queue
//             {
//                 visited[current] = true;

//                 if (source < current) // only add terms for i<j
//                 {
//                     double w_ij = 1.0 / (d_ij * d_ij);
//                     constraints[{source, current}] = term(source, current, d_ij, w_ij, w_ij);
//                 }
//                 for (edge e : graph[current])
//                 {
//                     // here the edge is not 'invisible'
//                     int next = e.target;
//                     double weight = e.weight;

//                     if (d[next] > d_ij + weight)
//                     {
//                         d[next] = d_ij + weight; // update tentative value of d
//                         pq.push(edge(next, d[next]));
//                     }
//                 }
//             }
//         }
//         if (constraints.size() != terms_size_goal)
//         {
//             throw "graph is not strongly connected";
//         }
//     }
//     return constraints;
// }

// std::vector<term> multiSourcePartitionUnweighted(int n, int m, int nPivots, std::vector<int> pivots, int *I, int *J, int *V)
// {
//     std::vector<int> graph = buildGraphArray(n, m, I, J, V);
//     std::map<std::tuple<int, int>, term> terms;

//     for (int e = 0; e < sizeof(I); e++)
//     {
//         int i = I[e], j = J[e];
//         double v = V[e];
//         if (i < j)
//         {
//             double w = 1 / pow(v, 2);
//             terms[std::tuple<int, int>(i, j)] = term(i, j, v, w, w);
//         }
//     }

//     // initialize s for each pivot
//     std::map<int, int> s;

//     std::vector<bool> markForRegion(n, false);
//     std::map<int, std::vector<bool>> markForWeights;
//     std::map<int, std::vector<int>> level;
//     std::map<int, std::vector<int>> regions;
//     std::vector<int> regionAssignment(n, -1);
//     std::queue<int> q1;
//     std::map<int, std::queue<int>> q2;
//     std::queue<std::tuple<int, int>> q3;

//     std::map<int, int> dist;
//     std::map<int, int> prevDist;

//     for (int p : pivots)
//     {
//         regions[p] = std::vector<int>{p};
//         regionAssignment[p] = p;
//         q1.push(p);
//         q2[p] = std::queue<int>();
//         markForRegion[p] = true;
//         markForWeights[p] = std::vector<bool>(n, false);
//         markForWeights[p][p] = true;
//         level[p] = std::vector<int>(n, 0);
//         s[p] = 1;
//         dist[p] = 0;
//         prevDist[p] = 1;
//     }

//     while (!q1.empty())
//     {
//         int index = q1.front();
//         q1.pop();
//         int pivot_index = regionAssignment[index];
//         int curLevel = level[pivot_index][index];

//         if (curLevel > dist[pivot_index])
//             dist[pivot_index] = curLevel;

//         if ((dist[pivot_index] % prevDist[pivot_index] == 0) && (dist[pivot_index] / prevDist[pivot_index] == 2))
//         {
//             while (!q2[pivot_index].empty())
//             {
//                 int regionalIndex = q2[pivot_index].front();
//                 if (level[pivot_index][regionalIndex] <= prevDist[pivot_index])
//                 {
//                     s[pivot_index]++;
//                     q2[pivot_index].pop();
//                 }
//                 else
//                 {
//                     break;
//                 }
//             }

//             prevDist[pivot_index]++;
//         }

//         if ((pivot_index != index) && (graph[n * pivot_index + index] == 0) && (terms.find(std::tuple<int, int>(pivot_index, index)) == terms.end()) && (terms.find(std::tuple<int, int>(index, pivot_index)) == terms.end()))
//         {
//             double wij = s[pivot_index] / pow(level[pivot_index][index], 2);
//             terms[std::tuple<int, int>(index, pivot_index)] = term(index, pivot_index, level[pivot_index][index], wij, 0);
//         }

//         for (int i = n * pivot_index; i < n * (pivot_index + 1); i++)
//         {
//             if (graph[i] != 0)
//             {
//                 int neighbour = i - n * pivot_index;
//                 if (!markForRegion[neighbour])
//                 {
//                     markForRegion[neighbour] = true;
//                     markForWeights[pivot_index][neighbour] = true;
//                     level[pivot_index][neighbour] = curLevel + 1;
//                     q1.push(neighbour);
//                     regionAssignment[neighbour] = pivot_index;
//                     regions[pivot_index].push_back(neighbour);
//                     q2[pivot_index].push(neighbour);
//                 }
//                 if (!markForWeights[pivot_index][neighbour])
//                 {
//                     markForWeights[pivot_index][neighbour] = true;
//                     level[pivot_index][neighbour] = curLevel + 1;
//                     q3.push(std::tuple<int, int>(neighbour, pivot_index));
//                 }
//             }
//         }
//     }

//     while (!q3.empty())
//     {
//         int index = std::get<0>(q3.front());
//         int pivot_index = std::get<1>(q3.front());
//         q3.pop();
//         int curLevel = level[pivot_index][index];

//         if (curLevel > dist[pivot_index])
//             dist[pivot_index] = curLevel;

//         if ((dist[pivot_index] % prevDist[pivot_index] == 0) && (dist[pivot_index] / prevDist[pivot_index] == 2))
//         {
//             while (!q2[pivot_index].empty())
//             {
//                 int regionalIndex = q2[pivot_index].front();
//                 if (level[pivot_index][regionalIndex] <= prevDist[pivot_index])
//                 {
//                     s[pivot_index]++;
//                     q2[pivot_index].pop();
//                 }
//                 else
//                 {
//                     break;
//                 }
//             }

//             prevDist[pivot_index]++;
//         }

//         double wij = s[pivot_index] / pow(level[pivot_index][index], 2);
//         terms[std::tuple<int, int>(index, pivot_index)] = term(index, pivot_index, level[pivot_index][index], wij, 0);

//         for (int i = n * pivot_index; i < n * (pivot_index + 1); i++)
//         {
//             if (graph[i] != 0)
//             {
//                 int neighbour = i - n * pivot_index;

//                 if (!markForWeights[pivot_index][neighbour])
//                 {
//                     markForWeights[pivot_index][neighbour] = true;
//                     level[pivot_index][neighbour] = curLevel + 1;
//                     q3.push(std::tuple<int, int>(neighbour, pivot_index));
//                 }
//             }
//         }
//     }

//     std::vector<term> terms_vec;

//     for (std::map<std::tuple<int, int>, term>::iterator it = terms.begin(); it != terms.end(); it++)
//     {
//         terms_vec.push_back(it->second);
//     }

//     return terms_vec;
// }

void layout_unweighted(int n, double *X, int m, int *I, int *J, int t_max, double eps)
{
    try
    {
        std::vector<term> terms = bfs(n, m, I, J);
        std::vector<double> etas = schedule(terms, t_max, eps);
        sgd(X, terms, etas);
    }
    catch (const char *msg)
    {
        std::cerr << "Error: " << msg << std::endl;
    }
}

void layout_weighted(int n, double *X, int m, int *I, int *J, double *V, int t_max, double eps)
{
    try
    {
        std::vector<term> terms = dijkstra(n, m, I, J, V);
        std::vector<double> etas = schedule(terms, t_max, eps);
        sgd(X, terms, etas);
    }
    catch (const char *msg)
    {
        std::cerr << "Error: " << msg << std::endl;
    }
}

samplingSchemeCode hashSamplingScheme(const char *inString)
{
    if (strcmp(inString, "random") == 0)
    {
        std::cerr << "Using Random sampling scheme" << std::endl;
        return erandom;
    }
    if (strcmp(inString, "mis") == 0)
    {
        std::cerr << "Using MIS Filtration sampling scheme" << std::endl;
        return emisFiltration;
    }
    if (strcmp(inString, "max_min_euc") == 0)
    {
        std::cerr << "Using Max-Min Euclidean sampling scheme" << std::endl;
        return emaxMinEuc;
    }
    if (strcmp(inString, "max_min_sp") == 0)
    {
        std::cerr << "Using Max-Min SP sampling scheme" << std::endl;
        return emaxMinSP;
    }
    if (strcmp(inString, "max_min_random_sp") == 0)
    {
        std::cerr << "Using Max-Min Random SP sampling scheme" << std::endl;
        return emaxMinRandomSP;
    }
    if (strcmp(inString, "k_means") == 0)
    {
        std::cerr << "Using K-Means Layout sampling scheme" << std::endl;
        return ekMeans;
    }
    if (strcmp(inString, "k_means_sp") == 0)
    {
        std::cerr << "Using K-Means SP sampling scheme" << std::endl;
        return ekMeansSP;
    }
    if (strcmp(inString, "k_means_max_min_sp") == 0)
    {
        std::cerr << "Using K-Means Max-Min SP sampling scheme" << std::endl;
        return ekMeansMaxMinSP;
    }
}

void sparse_layout_naive_unweighted(int n, double *X, int m, int *I, int *J, char *sampling_scheme, int k, int t_max, double eps)
{
    try
    {
        std::vector<int> pivots;
        std::map<int, std::vector<int>> shortestPaths;
        auto graph = buildGraphUnweighted(n, m, I, J);
        std::map<int, std::vector<int>> regions;

        switch (hashSamplingScheme(sampling_scheme))
        {
        case erandom:
        {
            randomPivots(n, k, pivots);
            // std::clock_t start;
            // double duration;
            // start = std::clock();
            // tbb::parallel_for(0, (int)pivots.size(), [&](int i){
            //     shortestPaths[pivots[i]] = bfs(n, m, pivots[i], I, J);
            // });
            for (int p : pivots)
            {
                shortestPaths[p] = bfs(n, m, p, I, J);
            }
            // duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
            // std::cerr << "Duration is: " << duration << "s" << std::endl;
            // initialize shortest dist to closest pivot pair
            std::vector<std::tuple<int, int>> mins(n);
            int p0 = pivots[0];

            for (int i = 0; i < n; i++)
            {
                mins[i] = std::make_tuple(shortestPaths[p0][i], p0);
            }

            for (int i = 1; i < k; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    int temp = shortestPaths[pivots[i]][j];
                    if (temp < std::get<0>(mins[j]))
                    {
                        mins[j] = std::make_tuple(temp, pivots[i]);
                    }
                }
            }

            // Naive region partitions
            for (int p : pivots)
            {
                regions[p] = std::vector<int>{};
            }

            for (int i = 0; i < n; i++)
            {
                regions[std::get<1>(mins[i])].push_back(i);
            }

            break;
        }
        case emisFiltration:
        {
            misFitration(n, m, k, pivots, I, J);

            for (int p : pivots)
            {
                shortestPaths[p] = bfs(n, m, p, I, J);
            }

            // initialize shortest dist to closest pivot pair
            std::vector<std::tuple<int, int>> mins(n);
            int p0 = pivots[0];

            for (int i = 0; i < n; i++)
            {
                mins[i] = std::make_tuple(shortestPaths[p0][i], p0);
            }

            for (int i = 1; i < k; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    int temp = shortestPaths[pivots[i]][j];
                    if (temp < std::get<0>(mins[j]))
                    {
                        mins[j] = std::make_tuple(temp, pivots[i]);
                    }
                }
            }

            // Naive region partitions
            for (int p : pivots)
            {
                regions[p] = std::vector<int>{};
            }

            for (int i = 0; i < n; i++)
            {
                regions[std::get<1>(mins[i])].push_back(i);
            }

            break;
        }
        case emaxMinEuc:
        {
            maxMinEuclidean(n, m, k, pivots, I, J);
            for (int p : pivots)
            {
                shortestPaths[p] = bfs(n, m, p, I, J);
            }

            std::vector<std::tuple<int, int>> mins(n);
            int p0 = pivots[0];

            for (int i = 0; i < n; i++)
            {
                mins[i] = std::make_tuple(shortestPaths[p0][i], p0);
            }

            for (int i = 1; i < k; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    int temp = shortestPaths[pivots[i]][j];
                    if (temp < std::get<0>(mins[j]))
                    {
                        mins[j] = std::make_tuple(temp, pivots[i]);
                    }
                }
            }

            // Naive region partitions
            for (int p : pivots)
            {
                regions[p] = std::vector<int>{};
            }

            for (int i = 0; i < n; i++)
            {
                regions[std::get<1>(mins[i])].push_back(i);
            }

            break;
        }
        case emaxMinSP:
        {
            maxMinSP(n, m, k, shortestPaths, pivots, regions, I, J);
            break;
        }
        case emaxMinRandomSP:
        {
            maxMinRandomSP(n, m, k, shortestPaths, pivots, regions, I, J);
            break;
        }
        default:
            maxMinRandomSP(n, m, k, shortestPaths, pivots, regions, I, J);
            break;
        }

        std::map<std::tuple<int, int>, term> terms;

        // Naive adpated weights calculation
        for (int p : pivots)
        {
            for (int i = 0; i < n; i++)
            {
                if ((p != i) && (std::find(graph[p].begin(), graph[p].end(), i) == graph[p].end()))
                {
                    int s = 0;
                    for (int j : regions[p])
                    {
                        if (shortestPaths[p][j] <= shortestPaths[p][i] / 2)
                        {
                            s++;
                        }
                    }
                    // if (s == 0)
                    // {
                    //     std::cerr << "pivot = " << p << ", i = " << i << std::endl;
                    //     throw "s equals to 0";
                    // }
                    // std::cerr << p << " " << i << " s = " << s << std::endl;
                    double w = (double)s / (shortestPaths[p][i] * shortestPaths[p][i]);

                    // keep the key value i < j for convinience
                    if (p < i)
                    {
                        std::tuple<int, int> pi = std::make_tuple(p, i);
                        if (terms.find(pi) == terms.end())
                        {
                            term t = {p, i, shortestPaths[p][i], w, 0};
                            terms[pi] = t;
                        }
                        else
                        {
                            terms[pi].wji = w;
                        }
                    }
                    else
                    {
                        std::tuple<int, int> ip = std::make_tuple(i, p);
                        if (terms.find(ip) == terms.end())
                        {
                            term t = {i, p, shortestPaths[p][i], w, 0};
                            terms[ip] = t;
                        }
                        else
                        {
                            terms[ip].wji = w;
                        }
                    }
                }
            }
        }

        // Find all avaliable terms in graph
        for (int ij = 0; ij < m; ij++)
        {
            int i = I[ij];
            int j = J[ij];
            if (i < j)
            {
                std::tuple<int, int> key = std::make_tuple(i, j);
                term t = {i, j, 1.0, 1.0, 1.0};
                terms[key] = t;
            }
        }

        std::vector<term> terms_vec;
        for (std::map<std::tuple<int, int>, term>::iterator it = terms.begin(); it != terms.end(); it++)
        {
            terms_vec.push_back(it->second);
        }

        std::vector<double> etas = schedule(terms_vec, t_max, eps);
        sgd(X, terms_vec, etas);
    }
    catch (const char *msg)
    {
        std::cerr << "Error: " << msg << std::endl;
    }
}

void sparse_layout_MSSP_unweightd(int n, double *X, int m, int *I, int *J, char *sampling_scheme, int k, int t_max, double eps)
{
    try
    {
        std::vector<int> pivots;
        randomPivots(n, k, pivots);
        auto graph = buildGraphUnweighted(n, m, I, J);
        switch (hashSamplingScheme(sampling_scheme))
        {
        case erandom:
        {
            randomPivots(n, k, pivots);
            break;
        }
        case emisFiltration:
        {
            misFitration(n, m, k, pivots, I, J);
            break;
        }
        case emaxMinEuc:{
            maxMinEuclidean(n, m, k, pivots, I, J);
            break;
        }
        case emaxMinSP:{
            maxMinSP(n, m, k, pivots, I, J);
            break;
        }
        case emaxMinRandomSP:{
            maxMinRandomSP(n, m, k, pivots, I, J);
            break;
        }
        default:
            maxMinRandomSP(n, m, k, pivots, I, J);
            break;
        }

        std::map<std::tuple<int, int>, term> terms;

        // MSSP
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

            if ((pivot_index != index) && (std::find(graph[pivot_index].begin(), graph[pivot_index].end(), index) == graph[pivot_index].end()))
            {
                double w = (double)s[pivot_index] / (level[pivot_index][index] * level[pivot_index][index]);
                if (pivot_index < index)
                {
                    std::tuple<int, int> pi = std::make_tuple(pivot_index, index);
                    if (terms.find(pi) == terms.end())
                    {
                        term t = {pivot_index, index, level[pivot_index][index], w, 0};
                        terms[pi] = t;
                    }
                    else
                    {
                        terms[pi].wji = w;
                    }
                }
                else
                {
                    std::tuple<int, int> ip = std::make_tuple(index, pivot_index);
                    if (terms.find(ip) == terms.end())
                    {
                        term t = {index, pivot_index, level[pivot_index][index], w, 0};
                        terms[ip] = t;
                    }
                    else
                    {
                        terms[ip].wji = w;
                    }
                }
            }

            for (int neighbour : graph[index])
            {
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
                    q3.push(std::tuple<int, int>(neighbour, pivot_index));
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

            double w = (double)s[pivot_index] / (level[pivot_index][index] * level[pivot_index][index]);

            if (pivot_index < index)
            {
                std::tuple<int, int> pi = std::make_tuple(pivot_index, index);
                if (terms.find(pi) == terms.end())
                {
                    term t = {pivot_index, index, level[pivot_index][index], w, 0};
                    terms[pi] = t;
                }
                else
                {
                    terms[pi].wji = w;
                }
            }
            else
            {
                std::tuple<int, int> ip = std::make_tuple(index, pivot_index);
                if (terms.find(ip) == terms.end())
                {
                    term t = {index, pivot_index, level[pivot_index][index], w, 0};
                    terms[ip] = t;
                }
                else
                {
                    terms[ip].wji = w;
                }
            }

            for (int neighbour : graph[index])
            {
                if (!markForWeights[pivot_index][neighbour])
                {
                    markForWeights[pivot_index][neighbour] = true;
                    level[pivot_index][neighbour] = curLevel + 1;
                    q3.push(std::tuple<int, int>(neighbour, pivot_index));
                }
            }
        }
        // Find all avaliable terms in graph
        for (int ij = 0; ij < m; ij++)
        {
            int i = I[ij];
            int j = J[ij];
            if (i < j)
            {
                std::tuple<int, int> key = std::make_tuple(i, j);
                term t = {i, j, 1.0, 1.0, 1.0};
                terms[key] = t;
            }
        }

        std::vector<term> terms_vec;
        for (std::map<std::tuple<int, int>, term>::iterator it = terms.begin(); it != terms.end(); it++)
        {
            terms_vec.push_back(it->second);
        }

        std::vector<double> etas = schedule(terms_vec, t_max, eps);
        sgd(X, terms_vec, etas);
    }
    catch (const char *msg)
    {
        std::cerr << "Error: " << msg << std::endl;
    }
}