#include "MySolution.h"

void Solution::build(int d, const vector<float>& base) {
    dim_ = d;   // 数据维数
    data_size_ = (size_t)dim_ * sizeof(float);  // 数据向量大小

    size_t n = base.size() / (size_t)dim_;  // 数据点数量
    // 分别代表：max_elements, M, random_seed, ef_construction, ef, data_size
    initHNSW(n, 16, 100, 200, 10,  data_size_);

    for (size_t i = 0; i < n; ++i) {
        const void* vec = (const void*)(&base[i * dim_]);
        insert(vec, -1, (labeltype)i); 
    }
}

void Solution::search(const vector<float>& query, int *res) {
    const size_t k = 10;

    // 用 searchKnn 取前 k 个
    auto pq = searchKnn(static_cast<const void*>(query.data()), k);

    // priority_queue 顶是“最远”，弹出后反转为从近到远
    std::vector<distPair> buf;
    buf.reserve(pq.size());
    while (!pq.empty()) { buf.push_back(pq.top()); pq.pop(); }
    std::reverse(buf.begin(), buf.end());

    size_t i = 0;
    for (; i < buf.size() && i < k; ++i) {
        tableint id = buf[i].second;
        res[i] = static_cast<int>(id);
    }
}