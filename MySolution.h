#ifndef CPP_SOLUTION_H
#define CPP_SOLUTION_H

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

class Solution {
public:
    void build(int d, const vector<float>& base);
    void search(const vector<float>& query, int *res);

private:
    int dim;
    vector<vector<float>> base_vectors;
};
#endif //CPP_SOLUTION_H