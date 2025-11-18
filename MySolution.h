#ifndef CPP_SOLUTION_H
#define CPP_SOLUTION_H

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <queue>
#include <climits>
#include <unordered_set>
#include <random>
#include <cstring>
#include <utility>
#include <cstddef>
#include <cstdint>
#include <atomic>
#include <mutex>
#include <thread>
#include <memory>

// SIMD 指令集相关
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#define SOLUTION_PLATFORM_X86 1
#else
#define SOLUTION_PLATFORM_X86 0
#endif

#if SOLUTION_PLATFORM_X86
#    ifdef _MSC_VER
#        include <intrin.h>
#    endif
#    include <immintrin.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define SOLUTION_SIMD_TARGET_ATTR(x) __attribute__((target(x)))
#else
#define SOLUTION_SIMD_TARGET_ATTR(x)
#endif

using namespace std;

// HNSW
typedef unsigned int tableint;   // unsigned 和 float 都是 4 字节 
typedef pair<float, tableint> distPair; // 距离-节点ID对类型
typedef unsigned int linklistsizeint; // 邻居数量类型
typedef size_t labeltype;  // 标签类型
typedef float dist_t;      // 距离类型

// SIMD 支持检测
#if SOLUTION_PLATFORM_X86 && (defined(__GNUC__) || defined(__clang__))
#define SOLUTION_COMPILE_AVX512 1
#define SOLUTION_COMPILE_AVX 1
#define SOLUTION_COMPILE_SSE3 1
#define SOLUTION_COMPILE_SSE 1
#else
#if SOLUTION_PLATFORM_X86 && defined(__AVX512F__)
#define SOLUTION_COMPILE_AVX512 1
#else
#define SOLUTION_COMPILE_AVX512 0
#endif
#if SOLUTION_PLATFORM_X86 && defined(__AVX__)
#define SOLUTION_COMPILE_AVX 1
#else
#define SOLUTION_COMPILE_AVX 0
#endif
#if SOLUTION_PLATFORM_X86
#define SOLUTION_COMPILE_SSE3 1
#define SOLUTION_COMPILE_SSE 1
#else
#define SOLUTION_COMPILE_SSE3 0
#define SOLUTION_COMPILE_SSE 0
#endif
#endif

namespace simd_detail {
inline dist_t l2_distance_scalar(const float* a, const float* b, size_t dim) {
    dist_t dist = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dist_t diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

#if SOLUTION_PLATFORM_X86

// SIMD 调度结构体
struct SimdDispatch {
    using Kernel = dist_t (*)(const float*, const float*, size_t);
    Kernel kernel{l2_distance_scalar};
    bool has_avx512{false};
    bool has_avx{false};
    bool has_sse3{false};
    bool has_sse{false};
};

inline unsigned long long read_xcr0() {
#if defined(_MSC_VER)
    return _xgetbv(0);
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int eax = 0, edx = 0;
    __asm__ volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<unsigned long long>(edx) << 32) | eax;
#else
    return 0ull;
#endif
}

inline void cpuid_basic(int info[4], int func_id) {
#if defined(_MSC_VER)
    __cpuid(info, func_id);
#elif defined(__GNUC__) || defined(__clang__)
    int a = 0, b = 0, c = 0, d = 0;
    __asm__ volatile("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(func_id), "c"(0));
    info[0] = a;
    info[1] = b;
    info[2] = c;
    info[3] = d;
#else
    info[0] = info[1] = info[2] = info[3] = 0;
#endif
}

inline void cpuid_subfunction(int info[4], int func_id, int sub_func) {
#if defined(_MSC_VER)
    __cpuidex(info, func_id, sub_func);
#elif defined(__GNUC__) || defined(__clang__)
    int a = 0, b = 0, c = 0, d = 0;
    __asm__ volatile("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(func_id), "c"(sub_func));
    info[0] = a;
    info[1] = b;
    info[2] = c;
    info[3] = d;
#else
    info[0] = info[1] = info[2] = info[3] = 0;
#endif
}

#if SOLUTION_COMPILE_AVX512
SOLUTION_SIMD_TARGET_ATTR("avx512f")
inline dist_t l2_distance_avx512(const float* a, const float* b, size_t dim) {
    __m512 acc = _mm512_setzero_ps();
    size_t i = 0;
    size_t bound = dim & (~size_t(15));
    for (; i < bound; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 diff = _mm512_sub_ps(va, vb);
        acc = _mm512_add_ps(acc, _mm512_mul_ps(diff, diff));
    }
    dist_t sum = _mm512_reduce_add_ps(acc);
    for (; i < dim; ++i) {
        dist_t diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}
#endif

#if SOLUTION_COMPILE_AVX
SOLUTION_SIMD_TARGET_ATTR("avx")
inline dist_t l2_distance_avx(const float* a, const float* b, size_t dim) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    size_t bound = dim & (~size_t(7));
    for (; i < bound; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(diff, diff));
    }
    alignas(32) float buffer[8];
    _mm256_store_ps(buffer, acc);
    dist_t sum = buffer[0] + buffer[1] + buffer[2] + buffer[3] +
                 buffer[4] + buffer[5] + buffer[6] + buffer[7];
    for (; i < dim; ++i) {
        dist_t diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}
#endif

#if SOLUTION_COMPILE_SSE3
SOLUTION_SIMD_TARGET_ATTR("sse3")
inline dist_t l2_distance_sse3(const float* a, const float* b, size_t dim) {
    __m128 acc = _mm_setzero_ps();
    size_t i = 0;
    size_t bound = dim & (~size_t(3));
    for (; i < bound; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 diff = _mm_sub_ps(va, vb);
        acc = _mm_add_ps(acc, _mm_mul_ps(diff, diff));
    }
    acc = _mm_hadd_ps(acc, acc);
    acc = _mm_hadd_ps(acc, acc);
    dist_t sum = _mm_cvtss_f32(acc);
    for (; i < dim; ++i) {
        dist_t diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}
#endif

#if SOLUTION_COMPILE_SSE
SOLUTION_SIMD_TARGET_ATTR("sse2")
inline dist_t l2_distance_sse(const float* a, const float* b, size_t dim) {
    __m128 acc = _mm_setzero_ps();
    size_t i = 0;
    size_t bound = dim & (~size_t(3));
    for (; i < bound; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 diff = _mm_sub_ps(va, vb);
        acc = _mm_add_ps(acc, _mm_mul_ps(diff, diff));
    }
    alignas(16) float buffer[4];
    _mm_store_ps(buffer, acc);
    dist_t sum = buffer[0] + buffer[1] + buffer[2] + buffer[3];
    for (; i < dim; ++i) {
        dist_t diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}
#endif

inline SimdDispatch init_dispatch() {
    SimdDispatch dispatch;
    int info[4] = {0, 0, 0, 0};
    cpuid_basic(info, 0);
    int max_func = info[0];
    bool osxsave = false;

    if (max_func >= 1) {
        cpuid_basic(info, 1);
        dispatch.has_sse = (info[3] & (1 << 25)) != 0;
        dispatch.has_sse3 = dispatch.has_sse && ((info[2] & 1) != 0);
        osxsave = (info[2] & (1 << 27)) != 0;
        bool hw_avx = (info[2] & (1 << 28)) != 0;
        if (hw_avx && osxsave) {
            unsigned long long mask = read_xcr0();
            if ((mask & 0x6) == 0x6) {
                dispatch.has_avx = true;
            }
        }
    }

    if (max_func >= 7) {
        cpuid_subfunction(info, 7, 0);
        bool hw_avx512 = (info[1] & (1 << 16)) != 0;
        if (hw_avx512 && osxsave) {
            unsigned long long mask = read_xcr0();
            if ((mask & 0xE6ull) == 0xE6ull) {
                dispatch.has_avx512 = true;
            }
        }
    }

#if SOLUTION_COMPILE_AVX512
    if (dispatch.has_avx512) {
        dispatch.kernel = l2_distance_avx512;
        return dispatch;
    }
#endif
#if SOLUTION_COMPILE_AVX
    if (dispatch.has_avx) {
        dispatch.kernel = l2_distance_avx;
        return dispatch;
    }
#endif
#if SOLUTION_COMPILE_SSE3
    if (dispatch.has_sse3) {
        dispatch.kernel = l2_distance_sse3;
        return dispatch;
    }
#endif
#if SOLUTION_COMPILE_SSE
    if (dispatch.has_sse) {
        dispatch.kernel = l2_distance_sse;
        return dispatch;
    }
#endif

    dispatch.kernel = l2_distance_scalar;
    return dispatch;
}

inline const SimdDispatch& get_dispatch() {
    static const SimdDispatch dispatch = init_dispatch();
    return dispatch;
}

inline dist_t compute(const float* a, const float* b, size_t dim) {
    return get_dispatch().kernel(a, b, dim);
}

#else // SOLUTION_PLATFORM_X86

inline dist_t compute(const float* a, const float* b, size_t dim) {
    return l2_distance_scalar(a, b, dim);
}

#endif // SOLUTION_PLATFORM_X86

} // namespace simd_detail


class Solution {
public:
    // 图结构
    // 储存节点数据及第 0 层的邻居关系
    // 邻居数量，flag, 凑对齐，邻居节点 id，数据向量，标签
    // 对每一个 Node: size->flag->reserved->neighbors->data->label
    // size:2 bytes, flag:1 byte, reserved:1 byte

    char* data_level0_memory_{nullptr};
    
    // 二维数组，每一行代表一个节点从第 1 层到最高层的邻居关系
    // 邻居数量，凑对齐，邻居节点 id
    // 每一层存储格式：size->reserved->neighbors
    // size:2 bytes, reserved:2 bytes
    
    char** linkLists_{nullptr};

    std::vector<int> element_levels_; // 保持每个节点的层数
    mutable std::unique_ptr<std::mutex[]> link_list_locks_; // 每个节点的邻接锁，支持并发构建/查询
    std::atomic<bool> finished_build_{false}; // 构建完成后无需锁读邻接

    mutable std::unique_ptr<std::atomic<uint32_t>[]> visit_epoch_; // 访问标记
    mutable std::atomic<uint32_t> global_visit_epoch_{1};   

    size_t max_elements_{0}; // 最大节点数量
    std::atomic<size_t> cur_element_count{0}; // 当前节点数量
    size_t M_{0};               // 每个节点最大连接数
    size_t maxM_{0};           // 每个节点第 1 层及以上最大连接数
    size_t maxM0_{0};        // 每个节点第 0 层最大连接数
    size_t ef_construction_{0}; // 构建时候选列表大小
    size_t ef_{0};            // 查询时候选列表大小
    double reverse_size_{0.0};   // ln(M)
    double mult_{0.0};          // 论文中的 m_l
    int dim_{0};

    size_t data_size_{0}; // 数据向量大小
    size_t size_links_level0_{0}; // 第 0 层每个节点链接大小
    size_t size_data_per_element_{0}; // 每个节点数据大小
    size_t size_links_per_element_{0}; // 每个节点第 1 层及以上链接大小
    size_t offsetData_{0}, offsetLevel0_{0}, offsetLable_{0}; // 偏移量

    std::atomic<tableint> enterpoint_node_{static_cast<tableint>(-1)};  // 并行构建时的入口点
    std::atomic<int> maxlevel_{-1}; // 当前最大层数
    std::atomic<bool> has_enterpoint_{false}; // 并行构建时是否已有入口点
    size_t worker_count_{0}; // 构建时的工作线程数

    std::default_random_engine level_generator_; // 用于生成随机层数
    vector<float> qbuf_;

    void initHNSW(
        size_t max_elements,
        size_t M = 16,
        size_t random_seed = 100,
        size_t ef_construction = 200,
        size_t ef = 200,
        size_t data_size = 0,
        size_t worker_count = 1
    ){
        max_elements_ = max_elements;
        M_ = M;
        data_size_ = data_size;
        maxM_ = M_;
        maxM0_ = M_ * 2;
        ef_construction_ = std::max(ef_construction, M_);
        ef_ = ef;
        level_generator_.seed(random_seed);
        worker_count_ = worker_count;


        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
        offsetData_ = size_links_level0_;
        offsetLable_ = size_links_level0_ + data_size_;
        offsetLevel0_ = 0;

        data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");
        

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
        
        std::memset(linkLists_, 0, sizeof(void*) * max_elements_);
        
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        
        cur_element_count.store(0, std::memory_order_relaxed);
        mult_ = 1 / log(1.0 * M_);
        reverse_size_ = 1.0 / mult_;

        // 层数记录
        element_levels_.assign(max_elements_, 0);

        link_list_locks_.reset(new std::mutex[max_elements_]);
        finished_build_.store(false, std::memory_order_relaxed);

        visit_epoch_.reset(new std::atomic<uint32_t>[max_elements_]);
        for (size_t i = 0; i < max_elements_; ++i) {
            visit_epoch_[i].store(0u, std::memory_order_relaxed);
        }
        global_visit_epoch_.store(1u, std::memory_order_relaxed);

        enterpoint_node_.store(static_cast<tableint>(-1), std::memory_order_relaxed);
        maxlevel_.store(-1, std::memory_order_relaxed);
        has_enterpoint_.store(false, std::memory_order_relaxed);

    }

    void clear() {
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
        tableint existing = static_cast<tableint>(cur_element_count.load(std::memory_order_relaxed));
        for (tableint i = 0; i < existing; i++) {
            if (element_levels_[i] > 0)
                free(linkLists_[i]);
        }
        free(linkLists_);
        linkLists_ = nullptr;
        cur_element_count.store(0, std::memory_order_relaxed);
        enterpoint_node_.store(static_cast<tableint>(-1), std::memory_order_relaxed);
        maxlevel_.store(-1, std::memory_order_relaxed);
        has_enterpoint_.store(false, std::memory_order_relaxed);
        link_list_locks_.reset();
        finished_build_.store(false, std::memory_order_relaxed);
        visit_epoch_.reset();
        global_visit_epoch_.store(1u, std::memory_order_relaxed);
    }

    inline void seal() {
        finished_build_.store(true, std::memory_order_release);
    }

    // 获取下一个访问标记
    inline uint32_t nextVisitToken() const {
        uint32_t token = global_visit_epoch_.fetch_add(1u, std::memory_order_acq_rel);
        if (token == 0u) {
            auto *self = const_cast<Solution*>(this);
            if (!self->visit_epoch_) {
                return 1u;
            }
            for (size_t i = 0; i < max_elements_; ++i) {
                self->visit_epoch_[i].store(0u, std::memory_order_relaxed);
            }
            token = global_visit_epoch_.fetch_add(1u, std::memory_order_acq_rel);
        }
        return token;
    }


    // 获取节点数据指针
    inline char *getDataByInternalId(tableint id) const {
        return (data_level0_memory_ + id * size_data_per_element_ + offsetData_);
    }

    // 获取节点标签指针
    inline labeltype *getExternalLabelPtr(tableint id) const {
        return (labeltype *) (data_level0_memory_ + id * size_data_per_element_ + offsetLable_);
    }

    // 获取第 0 层的邻居列表指针，不直接使用，需通过 get_linklist_at_level 调用
    linklistsizeint *get_linklist0(tableint id) const {
        return (linklistsizeint *) (data_level0_memory_ + id * size_data_per_element_ + offsetLevel0_);
    }

    // 获取第 1 层及以上的邻居列表指针，不直接使用，需通过 get_linklist_at_level 调用
    // 因为第 0 层和第 1 层及以上存储位置不同
    linklistsizeint *get_linklist(tableint id, int level) const {
        return (linklistsizeint *) (linkLists_[id] + (level - 1) * size_links_per_element_);
    }

    // 获取指定层的邻居列表指针
    linklistsizeint *get_linklist_at_level(tableint id, int level) const {
        return level == 0 ? get_linklist0(id) : get_linklist(id, level);
    }

    // 获取目前 ListCount 即邻居
    unsigned short int getListCount(linklistsizeint * ptr) const {
        return *reinterpret_cast<unsigned short*>(ptr);
    }

    // 把目前 ListCount 设置为 size
    void setListCount(linklistsizeint * ptr, unsigned short int size) const {
        *reinterpret_cast<unsigned short*>(ptr) = size;
    }

    // 统一通过头部后偏移得到邻居数组指针（头部固定 4 字节）
    inline tableint* getNeighborsArray(linklistsizeint* header) const {
        return reinterpret_cast<tableint*>(reinterpret_cast<char*>(header) + sizeof(linklistsizeint));
    }


    // 使用锁保护，确保并发插入时的线程安全
    void collectNeighbors(tableint id, int level, std::vector<tableint>& out) const {
        if (id >= max_elements_) {
            out.clear();
            return;
        }
        auto read_neighbors = [this, id, level, &out]() {
            if (level > element_levels_[id]) {
                out.clear();
                return;
            }
            linklistsizeint *header = get_linklist_at_level(id, level);
            int size = getListCount(header);
            tableint *data = getNeighborsArray(header);
            out.assign(data, data + size);
        };
        if (!finished_build_.load(std::memory_order_acquire)) {
            if (!link_list_locks_) {
                out.clear();
                return;
            }
            std::lock_guard<std::mutex> guard(link_list_locks_[id]);
            read_neighbors();
        } else {
            read_neighbors();
        }
    }

    // 欧式距离
    dist_t L2Distance(const void * a, const void * b) const {
        const float* lhs = static_cast<const float*>(a);
        const float* rhs = static_cast<const float*>(b);
        return simd_detail::compute(lhs, rhs, static_cast<size_t>(dim_));
    }

    // 优先队列比较器，按照距离从小到大排序
   // 这里每个节点 pair 包含 (距离, 节点ID)
    struct CompareDist {
        bool operator()(const distPair& a, const distPair& b) {
            return a.first < b.first; 
        }
    };

    // 产生指数分布的层数
    int generateRandomLevel(double reverse_size){
        uniform_real_distribution<float> distribution(0.0f, 1.0f);
        float r = distribution(level_generator_);
        // 避免 r=0 导致 -log(0) 造成层数极大并引发巨额分配
        r = std::min(0.999999f, std::max(r, 1e-6f));
        double level = -log(r) * reverse_size;
        return (int)level;
    }

    // 在指定层搜索，返回至多 ef_limit 个近邻节点
    // 实现论文的 Algorithm 2，这里要求 query 是数据指针
    priority_queue<distPair, vector<distPair>, CompareDist> 
    searchLayer(tableint ep_id, const void * query, int layer, size_t ef_limit) const {
        priority_queue<distPair, vector<distPair>, CompareDist> top_candidates; // W
        priority_queue<distPair, vector<distPair>, CompareDist> candidate_set; // C

        dist_t lowerBound;
        dist_t dist = L2Distance(query, getDataByInternalId(ep_id));
        top_candidates.emplace(dist, ep_id);
        // 这里用负距离是为了让优先队列变成小顶堆
        candidate_set.emplace(-dist, ep_id);

        lowerBound = dist;

        if (!visit_epoch_) {
            auto *self = const_cast<Solution*>(this);
            self->visit_epoch_.reset(new std::atomic<uint32_t>[max_elements_]);
            for (size_t i = 0; i < max_elements_; ++i) {
                self->visit_epoch_[i].store(0u, std::memory_order_relaxed);
            }
            global_visit_epoch_.store(1u, std::memory_order_relaxed);
        }

        uint32_t token = nextVisitToken();
        visit_epoch_[ep_id].store(token, std::memory_order_relaxed);

        thread_local std::vector<tableint> neighbors_tls;
        auto &neighbors = neighbors_tls;
        neighbors.clear();
        if (neighbors.capacity() < maxM0_) neighbors.reserve(maxM0_);

        while(!candidate_set.empty()){
            auto currPair = candidate_set.top();
            if((-currPair.first) > lowerBound && top_candidates.size() == ef_limit){
                break;
            }
            candidate_set.pop();

            tableint curID = currPair.second;
            collectNeighbors(curID, layer, neighbors);

            for (size_t idx = 0; idx < neighbors.size(); ++idx) {
                tableint candidateID = neighbors[idx];
                if (visit_epoch_[candidateID].load(std::memory_order_relaxed) == token) continue;
                visit_epoch_[candidateID].store(token, std::memory_order_relaxed);

#if SOLUTION_PLATFORM_X86
                if (idx + 1 < neighbors.size()) {
                    _mm_prefetch(reinterpret_cast<const char*>(getDataByInternalId(neighbors[idx + 1])), _MM_HINT_T0);
                }
#endif

                dist_t d = L2Distance(query, getDataByInternalId(candidateID));

                if(top_candidates.size() < ef_limit || d < lowerBound){
                    candidate_set.emplace(-d, candidateID);

                    top_candidates.emplace(d, candidateID);

                    if(top_candidates.size() > ef_limit){
                        top_candidates.pop();
                    }
                    if(!top_candidates.empty()){
                        lowerBound = top_candidates.top().first;
                    }
                }
            }
        }
        return top_candidates;
    }


    // 启发式寻找 M 个邻居
    // 寻找到的邻居存回 topCandidates 中
    // 这里和论文的 Algorithm 4 实现不一样
    void selectNeighborsHeuristic(
        priority_queue<distPair, vector<distPair>, CompareDist>& topCandidates,
        const size_t M
    ){
        if (topCandidates.size() < M) {
            return;
        }

        priority_queue<distPair> queueClosest; 
        vector<distPair> returnList;

        // 通过负数距离导致堆顶是距离最小的元素
        while(!topCandidates.empty()) {
            queueClosest.emplace(-topCandidates.top().first, topCandidates.top().second);
            topCandidates.pop();
        }

        while(queueClosest.size()){
            if (returnList.size() >= M)
                break;
            distPair currentPair = queueClosest.top();
            dist_t dist_to_query = -currentPair.first;
            queueClosest.pop();
            bool good = true;

            for (distPair second_pair : returnList) {
                dist_t curdist =
                        L2Distance(getDataByInternalId(second_pair.second),
                                    getDataByInternalId(currentPair.second));
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                returnList.push_back(currentPair);
            }
        }
        for(auto currPair : returnList){
            topCandidates.emplace(-currPair.first, currPair.second);
        }
    }

    // 将新节点和已有节点互相连接
    tableint mutuallyConnectNewElement(
        const void * vec,
        tableint cur_c,
        priority_queue<distPair, vector<distPair>, CompareDist>& top_candidates,
        int level,
        bool isUpdate
    ){
        size_t Mcurmax = level ? maxM_ : maxM0_;
        // 多线程实现，使用 thread_local 变量避免竞争
        thread_local std::vector<tableint> select_neighbors_tls;
        auto &selectNeighbors = select_neighbors_tls;
        selectNeighbors.clear();
        if (selectNeighbors.capacity() < Mcurmax) selectNeighbors.reserve(Mcurmax);

        selectNeighborsHeuristic(top_candidates, M_);
        while(top_candidates.size()){
            selectNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }
        
        tableint next_close_ep = selectNeighbors.back();

        // 新节点写入本层的邻接
        // 使用锁保护，确保并发插入时的线程安全
        {
            std::lock_guard<std::mutex> new_lock(link_list_locks_[cur_c]);
            linklistsizeint *ll_new = get_linklist_at_level(cur_c, level);
            tableint *new_data = getNeighborsArray(ll_new);
            size_t write_cnt = std::min(selectNeighbors.size(), Mcurmax);
            for (size_t i = 0; i < write_cnt; ++i) new_data[i] = selectNeighbors[i];
            setListCount(ll_new, static_cast<unsigned short>(write_cnt));
        }

        

        for(size_t idx = 0; idx < selectNeighbors.size(); idx++){
            tableint neighbor_id = selectNeighbors[idx];
            std::lock_guard<std::mutex> other_lock(link_list_locks_[neighbor_id]);
            
            linklistsizeint *ll_other = get_linklist_at_level(neighbor_id, level);

            size_t sz_link_list_other = getListCount(ll_other);

            if(sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (neighbor_id == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[neighbor_id])
                throw std::runtime_error("Trying to make a link on a non-existent level");
            tableint *data = getNeighborsArray(ll_other);

            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            if(!is_cur_c_present){
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    dist_t d_max = L2Distance(getDataByInternalId(cur_c), getDataByInternalId(selectNeighbors[idx]));
                    // Heuristic:
                    priority_queue<distPair, vector<distPair>, CompareDist> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                                L2Distance(getDataByInternalId(data[j]), getDataByInternalId(selectNeighbors[idx])), data[j]);
                    }

                    selectNeighborsHeuristic(candidates, Mcurmax);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
                }
            }
        }
        return next_close_ep;
    }


    // 把数据插入到多层图中
    // 并发插入：通过原子 ID 分配 + per-node 邻接锁，确保多个线程可以安全地同时写入图结构。
    void insert(const void * vec, int level, labeltype label) {
        size_t cur_c = cur_element_count.fetch_add(1, std::memory_order_acq_rel);
        if (cur_c >= max_elements_)
            throw std::runtime_error("HNSW capacity exceeded");

        bool is_first = (cur_c == 0);

        // 随机初始化层数
        int curlevel = generateRandomLevel(mult_);
        if(level > 0) 
            curlevel = level;

        element_levels_[cur_c] = curlevel;
        tableint currObj = enterpoint_node_.load(std::memory_order_acquire);
        tableint enterpoint_copy = currObj;


        char* base_ptr = data_level0_memory_ + cur_c * size_data_per_element_;

        // 写入向量数据（不再对整块做 memset）
        memcpy(base_ptr + offsetData_, vec, data_size_);

        // 初始化 level0 链表头与槽
        linklistsizeint* ll_head = reinterpret_cast<linklistsizeint*>(base_ptr + offsetLevel0_);
        *reinterpret_cast<unsigned short*>(ll_head) = 0;
        tableint* ll_array = reinterpret_cast<tableint*>((char*)ll_head + sizeof(unsigned short));
        memset(ll_array, 0, maxM0_ * sizeof(tableint));

        // 高层链表（仅在 curlevel>0 分配并清零）
        if (curlevel > 0) {
            size_t bytes = size_links_per_element_ * curlevel;
            linkLists_[cur_c] = (char*)malloc(bytes);
            if (!linkLists_[cur_c]) throw std::runtime_error("malloc high level list failed");
            memset(linkLists_[cur_c], 0, bytes);/*
            for (int lvl = 1; lvl <= curlevel; ++lvl) {
                linklistsizeint* lvl_head = get_linklist_at_level(cur_c, lvl);
                setListCount(lvl_head, 0);
            }*/
        }
        // 初始化节点相关数据结构
        // memset(get_linklist0(static_cast<tableint>(cur_c)), 0, size_links_level0_);
        // memcpy(getDataByInternalId(cur_c), vec, data_size_);
        
        // memcpy(getExternalLabelPtr(cur_c), &label, sizeof(labeltype));

        // if(curlevel) {
        //     linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel);
        //     if (linkLists_[cur_c] == nullptr)
        //        throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
        //     memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel);
        // }

        if (is_first) {
            enterpoint_node_.store(static_cast<tableint>(cur_c), std::memory_order_release);
            maxlevel_.store(curlevel, std::memory_order_release);
            has_enterpoint_.store(true, std::memory_order_release);
            return;
        }

        while (!has_enterpoint_.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }

        currObj = enterpoint_node_.load(std::memory_order_acquire);
        enterpoint_copy = currObj;
        int maxLevelSnapshot = maxlevel_.load(std::memory_order_acquire);

        if (currObj != static_cast<tableint>(-1)) {
            if(curlevel < maxLevelSnapshot) {
                dist_t curdist = L2Distance(vec, getDataByInternalId(currObj));
                std::vector<tableint> neighbors;
                neighbors.reserve(maxM0_);
                for(int level = maxLevelSnapshot; level > curlevel; level--) {
                    bool changed = true;
                    while(changed) {
                        changed = false;
                        collectNeighbors(currObj, level, neighbors);
                        for(tableint cand : neighbors) {
                            if (cand >= cur_element_count.load(std::memory_order_acquire))
                                throw std::runtime_error("cand out of range");
                            dist_t d = L2Distance(vec, getDataByInternalId(cand));
                            if(d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }

                // 从 curlevel 往下，找到一定数量的邻居，这里设定是 M_
                for (int lvl = std::min(curlevel, maxLevelSnapshot); lvl >= 0; lvl--) {
                    auto top_candidates = searchLayer(currObj, vec, lvl, ef_construction_);
                    // 这里把 Algorithm 1 后面的部分 合并到连接邻居一起实现了
                    currObj = mutuallyConnectNewElement(vec, static_cast<tableint>(cur_c), top_candidates, lvl, false);
                }
            } else {
                tableint ep = enterpoint_copy;
                for (int lvl = maxLevelSnapshot; lvl >= 0; --lvl) {
                    auto top_candidates = searchLayer(ep, vec, lvl, ef_construction_);
                    ep = mutuallyConnectNewElement(vec, static_cast<tableint>(cur_c), top_candidates, lvl, false);
                }
                bool updated = false;
                int expected = maxLevelSnapshot;
                while (curlevel > expected) {
                    if (maxlevel_.compare_exchange_weak(expected, curlevel, std::memory_order_acq_rel)) {
                        updated = true;
                        break;
                    }
                }
                if (updated) {
                    enterpoint_node_.store(static_cast<tableint>(cur_c), std::memory_order_release);
                }
            }
        }
    }


    priority_queue<distPair, vector<distPair>, CompareDist>
    searchKnn(const void *query_data, size_t k ) const {
        priority_queue<distPair, vector<distPair>, CompareDist> result;

        if (cur_element_count.load(std::memory_order_acquire) == 0 ||
            enterpoint_node_.load(std::memory_order_acquire) == static_cast<tableint>(-1))
            return result;

        tableint enterpoint_snapshot = enterpoint_node_.load(std::memory_order_acquire);
        tableint currObj = enterpoint_snapshot;
        dist_t curdist = L2Distance(query_data, getDataByInternalId(enterpoint_snapshot));

        std::vector<tableint> neighbors;
        neighbors.reserve(maxM0_);

        int maxLevelSnapshot = maxlevel_.load(std::memory_order_acquire);
        for (int level = maxLevelSnapshot; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                collectNeighbors(currObj, level, neighbors);
                for (tableint cand : neighbors) {
                    if (cand >= cur_element_count.load(std::memory_order_acquire))
                        throw std::runtime_error("cand out of range");
                    dist_t d = L2Distance(query_data, getDataByInternalId(cand));

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

    priority_queue<distPair, vector<distPair>, CompareDist> top_candidates;
    // 查询阶段使用 ef_
    top_candidates = searchLayer(currObj, query_data, 0, ef_);

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        return top_candidates;
    }

    void build(int d, const vector<float>& base);

    void search(const vector<float>& query, int *res);


};


/*
// kmeans

typedef vector<float> V;
using dist_t = float;


class Point{
    public:
    V value;
    size_t id;
    Point(V value_, size_t id_):value(value_), id(id_){}
};
class Solution {
public:
    size_t dim_{0}; // 数据维数
    size_t data_size_{0}; // 数据向量大小
    size_t K_{0}; // 聚类数
    vector<Point> centroids_; // 聚类中心
    size_t n_points_{0};
    V base_; // size = n_points_ * dim_
    vector<vector<size_t>> cluster_members_;

    vector<V> center_distances_;
    int search_clusters_ = 5;


    dist_t L2Distance(const V& a, const V& b) const {
        dist_t dist = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            float diff = a[i] - b[i];
            dist += diff * diff;
        }
        return dist;
    }

    void initKmeans(const vector<Point>& points, size_t n, size_t K) {
        K_ = K;
        centroids_.clear();
        cluster_members_.clear();
        
        // Kmeans++
        mt19937 gen(random_device{}());
        centroids_.reserve(K_);

        if (!points.empty()) {
        // 第一个中心随机选择
        uniform_int_distribution<size_t> dist(0, n - 1);
        size_t first_idx = dist(gen);
        centroids_.emplace_back(points[first_idx].value, points[first_idx].id);
        
        // 选择剩余的中心
        for (size_t k = 1; k < K_; ++k) {
            vector<float> distances;
            distances.reserve(n);
            
            // 计算每个点到最近中心的距离
            for (const auto& point : points) {
                float min_dist = numeric_limits<float>::max();
                for (const auto& center : centroids_) {
                    float dist = L2Distance(point.value, center.value);
                    if (dist < min_dist) {
                        min_dist = dist;
                    }
                }
                distances.push_back(min_dist);
            }
            
            // 根据距离的平方作为概率选择下一个中心
            vector<float> probabilities;
            probabilities.reserve(n);
            float total = 0.0f;
            for (float dist : distances) {
                float prob = dist * dist;
                probabilities.push_back(prob);
                total += prob;
            }
            
            // 归一化
            for (float& prob : probabilities) {
                prob /= total;
            }
            
            // 轮盘赌选择
            uniform_real_distribution<float> real_dist(0.0f, 1.0f);
            float r = real_dist(gen);
            float cumulative = 0.0f;
            size_t selected_idx = 0;
            for (size_t i = 0; i < n; ++i) {
                cumulative += probabilities[i];
                if (r <= cumulative) {
                    selected_idx = i;
                    break;
                }
            }
            
            centroids_.emplace_back(points[selected_idx].value, points[selected_idx].id);
            }
        }

    const int max_iters = 20;
    const float convergence_threshold = 1e-6f;
    vector<vector<size_t>> current_clusters;
    
    for (int iter = 0; iter < max_iters; ++iter) {
        current_clusters.assign(K_, vector<size_t>());
        
        // 分配点到最近的聚类中心
        for (const auto& point : points) {
            float min_dist = L2Distance(point.value, centroids_[0].value);
            size_t best_cluster = 0;
            for (size_t c = 1; c < K_; ++c) {
                float dist = L2Distance(point.value, centroids_[c].value);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = c;
                }
            }
            current_clusters[best_cluster].push_back(point.id);
        }
        // 修复4：改进空簇处理
        bool has_empty_cluster = false;
        for (size_t c = 0; c < K_; ++c) {
            if (current_clusters[c].empty()) {
                has_empty_cluster = true;
                // 选择距离当前所有中心最远的点作为新中心
                float max_min_dist = -1.0f;
                size_t farthest_point_idx = 0;
                
                for (size_t i = 0; i < n; ++i) {
                    float min_dist_to_centers = numeric_limits<float>::max();
                    for (size_t other_c = 0; other_c < K_; ++other_c) {
                        if (other_c == c) continue;
                        float dist = L2Distance(points[i].value, centroids_[other_c].value);
                        if (dist < min_dist_to_centers) {
                            min_dist_to_centers = dist;
                        }
                    }
                    if (min_dist_to_centers > max_min_dist) {
                        max_min_dist = min_dist_to_centers;
                        farthest_point_idx = i;
                    }
                }
                
                centroids_[c].value = points[farthest_point_idx].value;
                centroids_[c].id = points[farthest_point_idx].id;
            }
        }
        
        // 如果有空簇，重新进行分配
        if (has_empty_cluster) {
            continue;
        }
        
        // 更新聚类中心并检查收敛
        bool converged = true;
        for (size_t c = 0; c < K_; ++c) {
            V new_centroid(dim_, 0.0f);
            for (const auto pid : current_clusters[c]) {
                // 注意：这里假设pid是points的索引
                for (size_t d = 0; d < dim_; ++d) {
                    new_centroid[d] += points[pid].value[d];
                }
            }
            for (size_t d = 0; d < dim_; ++d) {
                new_centroid[d] /= static_cast<float>(current_clusters[c].size());
            }
            
            // 检查是否收敛
            float center_move_dist = 0.0f;
            for (size_t d = 0; d < dim_; ++d) {
                float diff = new_centroid[d] - centroids_[c].value[d];
                center_move_dist += diff * diff;
            }
            
            if (center_move_dist > convergence_threshold) {
                converged = false;
            }
            
            centroids_[c].value = std::move(new_centroid);
        }
        
        // 如果收敛，提前结束
        if (converged && iter >= 5) {  // 至少迭代5次
            break;
        }
    }
    cluster_members_ = std::move(current_clusters);


    center_distances_.resize(K_, vector<float>(K_, 0.0));
    for (size_t i = 0; i < K_; ++i) {
        for (size_t j = i + 1; j < K_; ++j) {
            double dist = sqrt(L2Distance(centroids_[i].value, centroids_[j].value));
            center_distances_[i][j] = dist;
            center_distances_[j][i] = dist;
        }
    }
}

    // 找到查询点最近的簇
    int findNearestCluster(const V& query) const {
        float min_dist = numeric_limits<float>::max();
        int nearest_cluster = 0;
        
        for (size_t i = 0; i < centroids_.size(); ++i) {
            float dist = L2Distance(query, centroids_[i].value);
            if (dist < min_dist) {
                min_dist = dist;
                nearest_cluster = i;
            }
        }
        return nearest_cluster;
    }

     // 找到要搜索的簇集合
    vector<int> findSearchClusters(int nearest_cluster) const {
        vector<pair<double, int>> cluster_dists;
        
        for (size_t i = 0; i < centroids_.size(); ++i) {
            double dist = (i == nearest_cluster) ? 0.0 : center_distances_[nearest_cluster][i];
            cluster_dists.push_back({dist, i});
        }
        
        // 按距离排序
        sort(cluster_dists.begin(), cluster_dists.end());
        
        // 选择最近的search_clusters_个簇
        vector<int> result;
        int count = min(search_clusters_, static_cast<int>(cluster_dists.size()));
        for (int i = 0; i < count; ++i) {
            result.push_back(cluster_dists[i].second);
        }
        
        return result;
    }

    // 在选定的簇中搜索最近邻
    vector<pair<float, int>> searchInClusters(const vector<float>& query, 
                                            const vector<int>& cluster_ids, 
                                            size_t k) const {
        // 使用最大堆来维护最近的k个点
        priority_queue<pair<float, int>> max_heap;
        
        for (int cluster_id : cluster_ids) {
            if (cluster_id >= cluster_members_.size()) continue;
            
            const auto& members = cluster_members_[cluster_id];
            for (size_t point_id : members) {
                // 计算查询点与该点的距离
                float dist = 0.0f;
                for (size_t d = 0; d < dim_; ++d) {
                    float diff = query[d] - base_[point_id * dim_ + d];
                    dist += diff * diff;
                }
                
                // 使用最大堆维护最近的k个点
                if (max_heap.size() < k) {
                    max_heap.push({dist, point_id});
                } else if (dist < max_heap.top().first) {
                    max_heap.pop();
                    max_heap.push({dist, point_id});
                }
            }
        }
        
        // 将堆中的元素转换为有序列表
        vector<pair<float, int>> result;
        while (!max_heap.empty()) {
            result.push_back(max_heap.top());
            max_heap.pop();
        }
        
        // 由于是最大堆，需要反转以获得升序
        reverse(result.begin(), result.end());
        return result;
    }



    void build(int d, const vector<float>& base);

    void search(const vector<float>& query, int *res);

};
*/

#endif //CPP_SOLUTION_H
