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

// Sampling schemes
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
