#include "MySolution.h"
#include <thread>
#include <atomic>

// hnsw

void Solution::build(int d, const vector<float>& base) {
    // Use default config (backwards compatible)
    SolutionConfig default_cfg;
    buildWithConfig(d, base, default_cfg);
}

void Solution::buildWithConfig(int d, const vector<float>& base, const SolutionConfig& cfg) {
    // Store config
    config_ = cfg;
    
    dim_ = d;   // 数据维数
    data_size_ = (size_t)dim_ * sizeof(float);  // 数据向量大小

    size_t n = base.size() / (size_t)dim_;  // 数据点数量
    
    size_t worker_count = min(std::thread::hardware_concurrency(), 32u); // 多线程线程数

    // ==================== HNSW 初始化 ====================
    // Use config parameters
    initHNSW(n, cfg.M, cfg.random_seed, cfg.ef_construction, data_size_, worker_count); 
    ep_num_ = cfg.ep_num;

    // 单线程插入
    if (worker_count_ == 1) {
        for (size_t i = 0; i < n; ++i) {
            const float* vec = &base[i * dim_];
            insert(vec, -1);
        }
    } else {
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
    }

    // ==================== ONNG 优化 ====================
    if (cfg.enable_onng) {
        optimizeGraphDirectly(false, cfg.onng_out_degree, cfg.onng_in_degree, cfg.onng_min_edges);
    }

    // ==================== BFS 内存重排 ====================
    if (cfg.enable_bfs) {
        reorderNodesByBFS();
    }

    // 入口点应在所有可能改变物理 ID / 图结构的步骤之后重新初始化
    initRandomEpoints();
}

void Solution::search(const vector<float>& query, int *res) {
    searchWithK(query, res, config_.k);
}

void Solution::searchWithK(const vector<float>& query, int *res, size_t k) {
    // Increment query count for statistics
    incrementQueryCount();
    
    // Dispatch based on search method in config
    switch (config_.search_method) {
        case SearchMethod::FIXED_EF:
            searchKNN_FixedEF(query.data(), res, k, config_.ef_search);
            break;
        case SearchMethod::STATIC_GAMMA:
        case SearchMethod::DYNAMIC_GAMMA:
        default: {
            bool dynamic = (config_.search_method == SearchMethod::DYNAMIC_GAMMA);
            if (k == 10) {
                searchKNN(query.data(), res, config_.gamma, dynamic);
            } else {
                // k != 10 时使用固定 EF 回退
                searchKNN_FixedEF(query.data(), res, k, max(config_.ef_search, k * 10));
            }
            break;
        }
    }
}