#ifndef B_SOLUTION_H
#define B_SOLUTION_H

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

class BSolution {
public:
    void build(int d, const vector<float>& base);
    void search(const vector<float>& query, int *res);

private:
    int dim;
    vector<vector<float>> base_vectors;
};
#endif //B_SOLUTION_H