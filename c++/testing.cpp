#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <numeric>
#include <cmath>
#include <complex>
#include <algorithm>

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

int main()
{
    std::ifstream inputFile;
    std::vector<int> I, J;
    std::vector<double> D;

    inputFile.open("dwt_1005.txt");

    if (!inputFile)
    {
        std::cerr << "Unable to open specified text file";
        exit(1);
    }

    int i, j;
    double d;
    while (inputFile >> i >> j >> d)
    {
        I.push_back(i);
        J.push_back(j);
        D.push_back(d);
    }

    inputFile.close();

    for (i = 0; i < 10; i++)
    {
        std::cout << I[i] << " " << J[i] << " " << D[i] << std::endl;
    }

    std::vector<int> v1 = {1, 2, 3, 4, 5};
    std::vector<int> v2 = {2, 3, 4, 5, 6};

    // std::vector<int> diff(5);
    // std::transform(v1.begin(), v1.end(), v2.begin(), diff.begin(), std::minus<int>());
    // double result = std::norm(diff);

    std::cout << vectorDistance(v1.begin(), v1.end(), v2.begin());

    return 0;
}