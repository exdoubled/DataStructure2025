#include "MySolution.h"
#include <thread>
#include <atomic>

// hnsw

void Solution::build(int d, const vector<float>& base) {
    dim_ = d;   // 数据维数
    data_size_ = (size_t)dim_ * sizeof(float);  // 数据向量大小

    size_t n = base.size() / (size_t)dim_;  // 数据点数量
    
    size_t worker_count = min(std::thread::hardware_concurrency(), 64u); // 多线程线程数

    // 分别代表：max_elements, M, random_seed, ef_construction, ef, data_size, worker_count
    initHNSW(n, 16, 114514, 200, 350, data_size_, worker_count);

    


    // 单线程插入
    if (worker_count_ == 1) {
        for (size_t i = 0; i < n; ++i) {
            const void* vec = static_cast<const void*>(&base[i * dim_]);
            insert(vec, -1, static_cast<labeltype>(i));
        }
        return;
    }

    // 多线程插入
    std::atomic<size_t> next_index{0};
    auto worker = [this, &base, n, &next_index]() {
        while (true) {
            size_t idx = next_index.fetch_add(1, std::memory_order_relaxed);
            if (idx >= n) break;
            const void* vec = static_cast<const void*>(&base[idx * dim_]);
            insert(vec, -1, static_cast<labeltype>(idx));
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(worker_count_);
    for (size_t t = 0; t < worker_count_; ++t) {
        threads.emplace_back(worker);
    }
    for (auto &th : threads) {
        th.join();
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