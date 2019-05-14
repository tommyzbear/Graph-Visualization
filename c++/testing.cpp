#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

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

    return 0;
}