#include "MySolution.h"

// hnsw

void Solution::build(int d, const vector<float>& base) {
    dim_ = d;   // 数据维数
    data_size_ = (size_t)dim_ * sizeof(float);  // 数据向量大小

    size_t n = base.size() / (size_t)dim_;  // 数据点数量
    // 分别代表：max_elements, M, random_seed, ef_construction, ef, data_size
    initHNSW(n, 16, 100, 200, 350,  data_size_);

    for (size_t i = 0; i < n; ++i) {
        const void* vec = (const void*)(&base[i * dim_]);
        insert(vec, -1, (labeltype)i);
    }
}

void Solution::search(const vector<float>& query, int *res) {
    const size_t k = 10;

    auto pq = searchKnn(static_cast<const void*>(query.data()), k);

    vector<distPair> buf;
    buf.reserve(pq.size());
    while (!pq.empty()) { buf.push_back(pq.top()); pq.pop(); }
    reverse(buf.begin(), buf.end());

    size_t i = 0;
    for (; i < buf.size() && i < k; ++i) {
        tableint id = buf[i].second;
        res[i] = static_cast<int>(id);
    }
    
    
    // qbuf_.resize(dim_);
    // memcpy(qbuf_.data(), query.data(), sizeof(float) * dim_);

    // auto pq = searchKnn((const void*)qbuf_.data(), k);

    // std::vector<distPair> buf;
    // buf.reserve(pq.size());
    // while (!pq.empty()) { buf.push_back(pq.top()); pq.pop(); }
    // std::reverse(buf.begin(), buf.end());

    // size_t i = 0;
    // for (; i < buf.size() && i < k; ++i) res[i] = (int)buf[i].second;
    // for (; i < k; ++i) res[i] = -1;
}

/*
// kmeans

void Solution::build(int d, const vector<float>& base) {

    dim_ = d;
    data_size_ = (size_t)dim_ * sizeof(float);
    size_t n = base.size() / (size_t)dim_;
    n_points_ = n;
    base_ = base;
        
    vector<Point> points;
    for (size_t i = 0; i < n; ++i) {
        V vec(base.begin() + i * dim_, base.begin() + (i + 1) * dim_);
        points.emplace_back(vec, i);
    }
        
        // 自适应确定聚类数
    size_t adaptive_K = min(size_t(10), max(size_t(2), n / 100));
    initKmeans(points, n, adaptive_K);
}

void Solution::search(const vector<float>& query, int *res) {
    const size_t k = 10;

    if (centroids_.empty() || cluster_members_.empty()) {
            // 如果没有聚类结果，返回默认值
            for (size_t i = 0; i < k; ++i) {
                res[i] = -1; // 使用-1表示无效结果
            }
            return;
        }

        // 找到查询点最近的簇
        int nearest_cluster = findNearestCluster(query);
        
        // 找到要搜索的簇集合
        vector<int> search_clusters = findSearchClusters(nearest_cluster);
        
        // 在选定的簇中搜索最近邻
        vector<pair<float, int>> candidates = searchInClusters(query, search_clusters, k);
        
        for (size_t i = 0; i < k; ++i) {
            if (i < candidates.size()) {
                res[i] = candidates[i].second;
            } else {
                res[i] = -1; 
            }
        }
}
*/