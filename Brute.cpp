#include "Brute.h"
#include <climits>
#include <algorithm>

void BSolution::build(int d, const vector<float>& base) {
    dim = d;
    int N = (int)base.size() / dim;
    base_vectors.resize(N);

    for (int i = 0; i < N; ++i) {
        base_vectors[i].assign(base.begin() + i * dim, base.begin() + (i + 1) * dim);
    }
}

void BSolution::search(const vector<float>& query, int *res) {
    vector<pair<float, int>> dist_id;

    for (int i = 0; i < (int)base_vectors.size(); ++i) {
        const vector<float>& v = base_vectors[i];
        float dist = 0.0f;
        for (int j = 0; j < dim; ++j) {
            float diff = query[j] - v[j];
            dist += diff * diff;
        }
        dist_id.emplace_back(dist, i);
    }

    int k = min(10, (int)dist_id.size());
    if (k > 0) {
        std::partial_sort(dist_id.begin(), dist_id.begin() + k, dist_id.end());
        for (int i = 0; i < k; ++i) {
            res[i] = dist_id[i].second;
        }
    }
}
