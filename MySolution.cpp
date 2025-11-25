#include "MySolution.h"
#include <thread>
#include <atomic>

// hnsw

void Solution::build(int d, const vector<float>& base) {
    dim_ = d;   // 数据维数
    data_size_ = (size_t)dim_ * sizeof(float);  // 数据向量大小

    size_t n = base.size() / (size_t)dim_;  // 数据点数量
    
    size_t worker_count = min(std::thread::hardware_concurrency(), 32u); // 多线程线程数

    // ==================== HNSW 初始化 ====================
    // 分别代表：max_elements, M, random_seed, ef_construction, ef, data_size, worker_count
    initHNSW(n, 16, 114514, 900, 150, data_size_, worker_count);

    // 单线程插入
    if (worker_count_ == 1) {
        for (size_t i = 0; i < n; ++i) {
            const float* vec = &base[i * dim_];
            insert(vec, -1);
        }
        return;
    }

    // 多线程插入
    std::atomic<size_t> next_index{0};
    auto worker = [this, &base, n, &next_index]() {
        while (true) {
            size_t idx = next_index.fetch_add(1, std::memory_order_relaxed);
            if (idx >= n) break;
            const float* vec = &base[idx * dim_];
            insert(vec, -1);
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

    // ==================== ONNG 优化 ====================
    optimizeGraphDirectly(true);
    
    
}

void Solution::search(const vector<float>& query, int *res) {
    // SearchDistanceScope distance_scope(*this);
    const size_t k = 10;

    auto pq = searchKnn(query.data(), k);

    vector<distPair> buf;
    buf.reserve(pq.size());
    while (!pq.empty()) { buf.push_back(pq.top()); pq.pop(); }
    reverse(buf.begin(), buf.end());

    size_t i = 0;
    for (; i < buf.size() && i < k; ++i) {
        tableint id = buf[i].second;
        res[i] = static_cast<int>(id);
    }
}