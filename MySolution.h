#ifndef CPP_SOLUTION_H
#define CPP_SOLUTION_H

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <queue>
#include <climits>
#include <limits>
#include <random>
#include <cstring>
#include <utility>
#include <cstddef>
#include <cstdint>
#include <atomic>
#include <mutex>
#include <thread>
#include <memory>
#include <list>
#include <unordered_map>
#include <unordered_set>

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

class Solution;

struct DistanceStatContext {
    inline static thread_local const Solution* active_solution = nullptr;
    inline static thread_local uint64_t counter = 0;
};


// HNSW
typedef unsigned int tableint;   // unsigned 和 float 都是 4 字节 
typedef pair<float, tableint> distPair; // 距离-节点ID对类型
typedef unsigned int linklistsizeint; // 邻居数量类型
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
// 优化的标量版本：4路展开提升 ILP (指令级并行)
inline dist_t l2_distance_scalar(const float* a, const float* b, size_t dim) {
    dist_t acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    size_t i = 0;
    size_t bound = dim & ~static_cast<size_t>(3);
    for (; i < bound; i += 4) {
        dist_t d0 = a[i]   - b[i];
        dist_t d1 = a[i+1] - b[i+1];
        dist_t d2 = a[i+2] - b[i+2];
        dist_t d3 = a[i+3] - b[i+3];
        acc0 += d0 * d0;
        acc1 += d1 * d1;
        acc2 += d2 * d2;
        acc3 += d3 * d3;
    }
    for (; i < dim; ++i) {
        dist_t d = a[i] - b[i];
        acc0 += d * d;
    }
    return (acc0 + acc1) + (acc2 + acc3);
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
    // 4路展开累加器，减少依赖链，提升 ILP
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    size_t i = 0;
    size_t bound = dim & (~size_t(63));  // 64 元素对齐
    
    for (; i < bound; i += 64) {
        // 利用 FMA 指令：diff^2 可以用 FMA 优化
        __m512 va0 = _mm512_loadu_ps(a + i);
        __m512 vb0 = _mm512_loadu_ps(b + i);
        __m512 diff0 = _mm512_sub_ps(va0, vb0);
        acc0 = _mm512_fmadd_ps(diff0, diff0, acc0);  // acc0 += diff0 * diff0
        
        __m512 va1 = _mm512_loadu_ps(a + i + 16);
        __m512 vb1 = _mm512_loadu_ps(b + i + 16);
        __m512 diff1 = _mm512_sub_ps(va1, vb1);
        acc1 = _mm512_fmadd_ps(diff1, diff1, acc1);
        
        __m512 va2 = _mm512_loadu_ps(a + i + 32);
        __m512 vb2 = _mm512_loadu_ps(b + i + 32);
        __m512 diff2 = _mm512_sub_ps(va2, vb2);
        acc2 = _mm512_fmadd_ps(diff2, diff2, acc2);
        
        __m512 va3 = _mm512_loadu_ps(a + i + 48);
        __m512 vb3 = _mm512_loadu_ps(b + i + 48);
        __m512 diff3 = _mm512_sub_ps(va3, vb3);
        acc3 = _mm512_fmadd_ps(diff3, diff3, acc3);
    }
    
    // 处理剩余完整的 16 元素块
    size_t bound16 = dim & (~size_t(15));
    for (; i < bound16; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 diff = _mm512_sub_ps(va, vb);
        acc0 = _mm512_fmadd_ps(diff, diff, acc0);
    }
    
    // 合并累加器
    acc0 = _mm512_add_ps(acc0, acc1);
    acc2 = _mm512_add_ps(acc2, acc3);
    acc0 = _mm512_add_ps(acc0, acc2);
    dist_t sum = _mm512_reduce_add_ps(acc0);
    
    // 标量处理剩余元素
    for (; i < dim; ++i) {
        dist_t diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}
#endif

#if SOLUTION_COMPILE_AVX
SOLUTION_SIMD_TARGET_ATTR("avx,avx2,fma")  // 启用 FMA 支持
inline dist_t l2_distance_avx(const float* a, const float* b, size_t dim) {
    // 4路展开，避免依赖链
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    size_t i = 0;
    size_t bound = dim & (~size_t(31));  // 32 元素对齐
    
    for (; i < bound; i += 32) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 diff0 = _mm256_sub_ps(va0, vb0);
        acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);  // FMA: acc0 += diff0^2
        
        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        __m256 diff1 = _mm256_sub_ps(va1, vb1);
        acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);
        
        __m256 va2 = _mm256_loadu_ps(a + i + 16);
        __m256 vb2 = _mm256_loadu_ps(b + i + 16);
        __m256 diff2 = _mm256_sub_ps(va2, vb2);
        acc2 = _mm256_fmadd_ps(diff2, diff2, acc2);
        
        __m256 va3 = _mm256_loadu_ps(a + i + 24);
        __m256 vb3 = _mm256_loadu_ps(b + i + 24);
        __m256 diff3 = _mm256_sub_ps(va3, vb3);
        acc3 = _mm256_fmadd_ps(diff3, diff3, acc3);
    }
    
    // 处理剩余的 8 元素块
    size_t bound8 = dim & (~size_t(7));
    for (; i < bound8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        acc0 = _mm256_fmadd_ps(diff, diff, acc0);
    }
    
    // 合并累加器
    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);
    
    // 高效的向量规约（避免 hadd）
    __m128 low = _mm256_castps256_ps128(acc0);
    __m128 high = _mm256_extractf128_ps(acc0, 1);
    __m128 sum4 = _mm_add_ps(low, high);
    sum4 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));  // 避免 hadd
    sum4 = _mm_add_ss(sum4, _mm_movehdup_ps(sum4));
    dist_t sum = _mm_cvtss_f32(sum4);
    
    // 标量处理剩余
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
    // 4路展开
    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();
    __m128 acc2 = _mm_setzero_ps();
    __m128 acc3 = _mm_setzero_ps();
    size_t i = 0;
    size_t bound = dim & (~size_t(15));  // 16 元素对齐
    
    for (; i < bound; i += 16) {
        __m128 va0 = _mm_loadu_ps(a + i);
        __m128 vb0 = _mm_loadu_ps(b + i);
        __m128 diff0 = _mm_sub_ps(va0, vb0);
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(diff0, diff0));
        
        __m128 va1 = _mm_loadu_ps(a + i + 4);
        __m128 vb1 = _mm_loadu_ps(b + i + 4);
        __m128 diff1 = _mm_sub_ps(va1, vb1);
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(diff1, diff1));
        
        __m128 va2 = _mm_loadu_ps(a + i + 8);
        __m128 vb2 = _mm_loadu_ps(b + i + 8);
        __m128 diff2 = _mm_sub_ps(va2, vb2);
        acc2 = _mm_add_ps(acc2, _mm_mul_ps(diff2, diff2));
        
        __m128 va3 = _mm_loadu_ps(a + i + 12);
        __m128 vb3 = _mm_loadu_ps(b + i + 12);
        __m128 diff3 = _mm_sub_ps(va3, vb3);
        acc3 = _mm_add_ps(acc3, _mm_mul_ps(diff3, diff3));
    }
    
    // 处理剩余的 4 元素块
    size_t bound4 = dim & (~size_t(3));
    for (; i < bound4; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 diff = _mm_sub_ps(va, vb);
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(diff, diff));
    }
    
    // 合并累加器
    acc0 = _mm_add_ps(acc0, acc1);
    acc2 = _mm_add_ps(acc2, acc3);
    acc0 = _mm_add_ps(acc0, acc2);
    
    // 高效规约（避免 hadd）
    acc0 = _mm_add_ps(acc0, _mm_movehl_ps(acc0, acc0));
    acc0 = _mm_add_ss(acc0, _mm_movehdup_ps(acc0));
    dist_t sum = _mm_cvtss_f32(acc0);
    
    // 标量处理剩余
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
    // 4路展开
    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();
    __m128 acc2 = _mm_setzero_ps();
    __m128 acc3 = _mm_setzero_ps();
    size_t i = 0;
    size_t bound = dim & (~size_t(15));
    
    for (; i < bound; i += 16) {
        __m128 va0 = _mm_loadu_ps(a + i);
        __m128 vb0 = _mm_loadu_ps(b + i);
        __m128 diff0 = _mm_sub_ps(va0, vb0);
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(diff0, diff0));
        
        __m128 va1 = _mm_loadu_ps(a + i + 4);
        __m128 vb1 = _mm_loadu_ps(b + i + 4);
        __m128 diff1 = _mm_sub_ps(va1, vb1);
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(diff1, diff1));
        
        __m128 va2 = _mm_loadu_ps(a + i + 8);
        __m128 vb2 = _mm_loadu_ps(b + i + 8);
        __m128 diff2 = _mm_sub_ps(va2, vb2);
        acc2 = _mm_add_ps(acc2, _mm_mul_ps(diff2, diff2));
        
        __m128 va3 = _mm_loadu_ps(a + i + 12);
        __m128 vb3 = _mm_loadu_ps(b + i + 12);
        __m128 diff3 = _mm_sub_ps(va3, vb3);
        acc3 = _mm_add_ps(acc3, _mm_mul_ps(diff3, diff3));
    }
    
    // 处理剩余的 4 元素块
    size_t bound4 = dim & (~size_t(3));
    for (; i < bound4; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 diff = _mm_sub_ps(va, vb);
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(diff, diff));
    }
    
    // 合并累加器
    acc0 = _mm_add_ps(acc0, acc1);
    acc2 = _mm_add_ps(acc2, acc3);
    acc0 = _mm_add_ps(acc0, acc2);
    
    // 高效规约（SSE2，无 shuffle）
    __m128 shuf = _mm_shuffle_ps(acc0, acc0, _MM_SHUFFLE(2, 3, 0, 1));
    acc0 = _mm_add_ps(acc0, shuf);
    shuf = _mm_movehl_ps(shuf, acc0);
    acc0 = _mm_add_ss(acc0, shuf);
    dist_t sum = _mm_cvtss_f32(acc0);
    
    // 标量处理剩余
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


namespace onng{
    // 用于临时标记和排序
    struct EdgeInfo{
        tableint id;
        dist_t dist;
        
        EdgeInfo(tableint _id, dist_t _dist) : id(_id), dist(_dist) {}
    };
} // namespace onng

class Solution {
public:
    // ==================== ONNG 优化 =========================
    // maxReverseEdges 对应 reverseEdgeSize（入度限制）
    // 使用 onng ，首先在 HNSW 指定层添加反向边优化，对应原论文 Algorithm 3 的添加反向边并限制入度的实现
    // reconstructGraphWithConstraint
    void addReverseEdgesDirectly(int layer, size_t maxReverseEdges) {
        size_t node_count = cur_element_count.load(std::memory_order_acquire);
        size_t max_edges = (layer == 0) ? maxM0_ : maxM_;
        
        // Step 1: 统计每个节点的入度（被多少节点指向）
        std::vector<std::vector<onng::EdgeInfo>> reverseEdges(node_count);
        
        for (tableint src = 0; src < static_cast<tableint>(node_count); ++src) {
            if (layer > element_levels_[src]) continue;
            
            linklistsizeint* ll = get_linklist_at_level(src, layer);
            int size = getListCount(ll);
            const tableint* neighbors = getNeighborsArray(ll);
            const float* src_data = getDataByInternalId(src);
            
            for (int i = 0; i < size; ++i) {
                tableint dst = neighbors[i];
                if (dst >= node_count) continue;
                
                const float* dst_data = getDataByInternalId(dst);
                dist_t dist = L2Distance(src_data, dst_data);
                reverseEdges[dst].emplace_back(src, dist);
            }
        }
        
        // Step 2: 按入度排序，优先处理入度少的节点（提升连通性）
        std::vector<std::pair<size_t, tableint>> sortedByIndegree;
        sortedByIndegree.reserve(node_count);
        for (tableint id = 0; id < static_cast<tableint>(node_count); ++id) {
            if (layer > element_levels_[id]) continue;
            sortedByIndegree.emplace_back(reverseEdges[id].size(), id);
        }
        std::sort(sortedByIndegree.begin(), sortedByIndegree.end());
        
        // Step 3: 添加反向边
        std::vector<size_t> addedReverseCounts(node_count, 0);
        
        for (const auto& [indegree, dst] : sortedByIndegree) {
            if (reverseEdges[dst].empty()) continue;
            
            std::lock_guard<std::mutex> lock(link_list_locks_[dst]);
            
            linklistsizeint* ll = get_linklist_at_level(dst, layer);
            tableint* neighbors = getNeighborsArray(ll);
            int current_size = getListCount(ll);
            
            // 建立现有邻居的快速查找
            std::unordered_set<tableint> existing;
            for (int i = 0; i < current_size; ++i) {
                existing.insert(neighbors[i]);
            }
            
            // 按距离排序反向边候选
            auto& candidates = reverseEdges[dst];
            std::sort(candidates.begin(), candidates.end(), 
                [](const onng::EdgeInfo& a, const onng::EdgeInfo& b) {
                    return a.dist < b.dist;
                });
            
            // 添加新的反向边
            for (const auto& edge : candidates) {
                if (current_size >= static_cast<int>(max_edges)) break;
                if (addedReverseCounts[edge.id] >= maxReverseEdges) continue;
                if (existing.count(edge.id)) continue;
                
                neighbors[current_size++] = edge.id;
                existing.insert(edge.id);
                addedReverseCounts[edge.id]++;
            }
            
            setListCount(ll, static_cast<unsigned short>(current_size));
        }
    }
    
    // Algorithm 4
    // 调整路径以移除冗余边
    void pruneRedundantPathsDirectly(int layer, size_t minNoOfEdges) {
        size_t node_count = cur_element_count.load(std::memory_order_acquire);
        
        for (tableint src = 0; src < static_cast<tableint>(node_count); ++src) {
            if (layer > element_levels_[src]) continue;
            
            std::lock_guard<std::mutex> lock(link_list_locks_[src]);
            
            linklistsizeint* ll = get_linklist_at_level(src, layer);
            int size = getListCount(ll);
            
            if (size <= static_cast<int>(minNoOfEdges)) continue;
            
            tableint* neighbors = getNeighborsArray(ll);
            const float* src_data = getDataByInternalId(src);
            
            // 构建邻居信息
            std::vector<onng::EdgeInfo> edges;
            edges.reserve(size);
            for (int i = 0; i < size; ++i) {
                tableint nbr = neighbors[i];
                const float* nbr_data = getDataByInternalId(nbr);
                dist_t dist = L2Distance(src_data, nbr_data);
                edges.emplace_back(nbr, dist);
            }
            
            // 标记可删除的边
            std::vector<bool> toRemove(edges.size(), false);
            
            // 对每条边检查是否有更短的两跳路径
            for (size_t i = 0; i < edges.size(); ++i) {
                tableint dst = edges[i].id;
                dist_t direct_dist = edges[i].dist;
                
                // 遍历中间节点
                for (size_t j = 0; j < edges.size(); ++j) {
                    if (i == j) continue;
                    
                    tableint mid = edges[j].id;
                    dist_t dist_to_mid = edges[j].dist;
                    
                    if (dist_to_mid >= direct_dist) continue;
                    
                    // 检查 mid -> dst 是否存在
                    linklistsizeint* mid_ll = get_linklist_at_level(mid, layer);
                    int mid_size = getListCount(mid_ll);
                    const tableint* mid_neighbors = getNeighborsArray(mid_ll);
                    
                    bool found = false;
                    dist_t dist_mid_to_dst = 0.0f;
                    
                    for (int k = 0; k < mid_size; ++k) {
                        if (mid_neighbors[k] == dst) {
                            const float* mid_data = getDataByInternalId(mid);
                            const float* dst_data = getDataByInternalId(dst);
                            dist_mid_to_dst = L2Distance(mid_data, dst_data);
                            found = true;
                            break;
                        }
                    }
                    
                    // 如果两跳都比直接边短，标记删除
                    if (found && dist_mid_to_dst < direct_dist) {
                        toRemove[i] = true;
                        break;
                    }
                }
            }
            
            // 重建邻居列表，跳过标记删除的边
            int new_size = 0;
            for (size_t i = 0; i < edges.size(); ++i) {
                if (!toRemove[i] || new_size < static_cast<int>(minNoOfEdges)) {
                    neighbors[new_size++] = edges[i].id;
                }
            }
            
            setListCount(ll, static_cast<unsigned short>(new_size));
        }
    }
    
    // 统一优化接口
    void optimizeGraphDirectly(bool optimize_high_layers = false) {
        int max_level = maxlevel_.load(std::memory_order_acquire);
        
        // 第0层：添加反向边 + 剪枝
        addReverseEdgesDirectly(0, 160);
        pruneRedundantPathsDirectly(0, 24);
        
        if (optimize_high_layers && max_level > 0) {
            for (int level = 1; level <= max_level; ++level) {
                addReverseEdgesDirectly(level, static_cast<size_t>(M_));
                pruneRedundantPathsDirectly(level, static_cast<size_t>(M_));
            }
        }
    }
    
    
    // ==================== HNSW 结构 ====================

    // 图结构
    // 储存节点数据及第 0 层的邻居关系
    // 邻居数量，flag, 凑对齐，邻居节点 id，数据向量，标签
    // 对每一个 Node: size->->reserved->neighbors->data
    // size:2 bytes, reserved:2 byte

    char* data_level0_memory_{nullptr};
    
    // 二维数组，每一行代表一个节点从第 1 层到最高层的邻居关系
    // 邻居数量，凑对齐，邻居节点 id
    // 每一层存储格式：size->reserved->neighbors
    // size:2 bytes, reserved:2 bytes
    
    char** linkLists_{nullptr};

    std::vector<int> element_levels_; // 保持每个节点的层数
    mutable std::unique_ptr<std::mutex[]> link_list_locks_; // 每个节点的邻接锁，支持并发构建/查询

    size_t max_elements_{0}; // 最大节点数量
    size_t M_{0};               // 每个节点最大连接数
    size_t maxM_{0};           // 每个节点第 1 层及以上最大连接数
    size_t maxM0_{0};        // 每个节点第 0 层最大连接数
    size_t ef_construction_{0}; // 构建时候选列表大小
    size_t ef_{0};            // 查询时候选列表大小

    double mult_{0.0};          // 论文中的 m_l
    size_t dim_{0};

    size_t data_size_{0}; // 数据向量大小
    size_t size_links_level0_{0}; // 第 0 层每个节点链接大小
    size_t size_data_per_element_{0}; // 每个节点数据大小
    size_t size_links_per_element_{0}; // 每个节点第 1 层及以上链接大小
    size_t offsetData_{0}, offsetLevel0_{0}; // 偏移量

    std::atomic<tableint> enterpoint_node_{static_cast<tableint>(-1)};  // 并行构建时的入口点
    std::atomic<int> maxlevel_{-1}; // 当前最大层数
    std::atomic<bool> has_enterpoint_{false}; // 并行构建时是否已有入口点
    std::atomic<size_t> cur_element_count{0}; // 当前节点数量
    size_t worker_count_{0}; // 构建时的工作线程数

    std::default_random_engine level_generator_; // 用于生成随机层数

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
        offsetLevel0_ = 0;
        offsetData_ = offsetLevel0_ + size_links_level0_;
        size_data_per_element_ = size_links_level0_ + data_size_;

        data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");
        memset(data_level0_memory_, 0, max_elements_ * size_data_per_element_);
        

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
        
        std::memset(linkLists_, 0, sizeof(void*) * max_elements_);
        
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        
        cur_element_count.store(0, std::memory_order_relaxed);
        mult_ = 1 / log(1.0 * M_);
        // 层数记录
        element_levels_.assign(max_elements_, 0);

        link_list_locks_.reset(new std::mutex[max_elements_]);

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
    }


    // 获取节点数据指针
    inline const float *getDataByInternalId(tableint id) const {
        return reinterpret_cast<const float*>(data_level0_memory_ + id * size_data_per_element_ + offsetData_);
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
        if (!link_list_locks_ || id >= max_elements_) {
            out.clear();
            return;
        }
        std::lock_guard<std::mutex> guard(link_list_locks_[id]);
        if (level > element_levels_[id]) {
            out.clear();
            return;
        }
        linklistsizeint *header = get_linklist_at_level(id, level);
        int size = getListCount(header);
        tableint *data = getNeighborsArray(header);
        out.assign(data, data + size);
    }

    // 欧式距离
    dist_t L2Distance(const float * a, const float * b) const {
        /*
        if (DistanceStatContext::active_solution == this) {
            DistanceStatContext::counter++;
        }
        */
        return simd_detail::compute(a, b, dim_);
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
       // return 0;
    }

    // 在指定层搜索，返回至多 ef_limit 个近邻节点
    // 实现论文的 Algorithm 2，这里要求 query 是数据指针
    priority_queue<distPair, vector<distPair>, CompareDist> 
    searchLayer(tableint ep_id, const float *query, int layer, size_t ef_limit) const {
        priority_queue<distPair, vector<distPair>, CompareDist> top_candidates; // W
        priority_queue<distPair, vector<distPair>, CompareDist> candidate_set; // C

        dist_t lowerBound;
        dist_t dist = L2Distance(query, getDataByInternalId(ep_id));
        top_candidates.emplace(dist, ep_id);
        // 这里用负距离是为了让优先队列变成小顶堆
        candidate_set.emplace(-dist, ep_id);

        lowerBound = dist;

        // 多线程实现，使用 thread_local bitset + token，避免每次清空和大内存
        thread_local std::vector<uint64_t> visit_bitmap_tls;
        thread_local std::vector<uint32_t> visit_epoch_tls;
        thread_local uint32_t visit_token_tls = 1u;

        const size_t needed_words = (max_elements_ + 63ull) >> 6;
        if (visit_bitmap_tls.size() < needed_words) {
            visit_bitmap_tls.assign(needed_words, 0ull);
            visit_epoch_tls.assign(needed_words, 0u);
        } else if (visit_epoch_tls.size() < visit_bitmap_tls.size()) {
            visit_epoch_tls.resize(visit_bitmap_tls.size(), 0u);
        }

        visit_token_tls += 1u;
        if (visit_token_tls == 0u) {
            std::fill(visit_epoch_tls.begin(), visit_epoch_tls.end(), 0u);
            visit_token_tls = 1u;
        }

        auto touch_word = [&](size_t word_idx) {
            if (visit_epoch_tls[word_idx] != visit_token_tls) {
                visit_epoch_tls[word_idx] = visit_token_tls;
                visit_bitmap_tls[word_idx] = 0ull;
            }
        };
        auto mark_visited = [&](tableint id) {
            size_t word = static_cast<size_t>(id) >> 6;
            size_t bit = static_cast<size_t>(id) & 63ull;
            touch_word(word);
            visit_bitmap_tls[word] |= (1ull << bit);
        };
        auto is_visited = [&](tableint id) -> bool {
            size_t word = static_cast<size_t>(id) >> 6;
            size_t bit = static_cast<size_t>(id) & 63ull;
            touch_word(word);
            return (visit_bitmap_tls[word] >> bit) & 1ull;
        };

        mark_visited(ep_id);

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
            // 预取邻居数据
            linklistsizeint* ll = get_linklist_at_level(curID, layer);
            int size = getListCount(ll);
            const tableint* data = getNeighborsArray(ll);
            for (int i = 0; i < size; ++i) {
                _mm_prefetch(getDataByInternalId(data[i]), _MM_HINT_T0);
            }

            for(int i = 0; i < size; ++i){
                tableint candidateID = data[i];
                if (is_visited(candidateID)) continue;
                mark_visited(candidateID);

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
        tableint cur_c,
        priority_queue<distPair, vector<distPair>, CompareDist>& top_candidates,
        int level
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
        return next_close_ep;
    }


    // 把数据插入到多层图中
    // 并发插入：通过原子 ID 分配 + per-node 邻接锁，确保多个线程可以安全地同时写入图结构。
    void insert(const float *vec, int level) {
        tableint cur_c = cur_element_count.fetch_add(1, std::memory_order_acq_rel);
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
            memset(linkLists_[cur_c], 0, bytes);
        }

        if (is_first) {
            enterpoint_node_.store(cur_c, std::memory_order_release);
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
                    currObj = mutuallyConnectNewElement(cur_c, top_candidates, lvl);
                }
            } else {
                tableint ep = enterpoint_copy;
                for (int lvl = maxLevelSnapshot; lvl >= 0; --lvl) {
                    auto top_candidates = searchLayer(ep, vec, lvl, ef_construction_);
                    ep = mutuallyConnectNewElement(cur_c, top_candidates, lvl);
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
                    enterpoint_node_.store(cur_c, std::memory_order_release);
                }
            }
        }
    }


    priority_queue<distPair, vector<distPair>, CompareDist>
    searchKnn(const float *query, size_t k ) const {
        priority_queue<distPair, vector<distPair>, CompareDist> result;

        if (cur_element_count.load(std::memory_order_acquire) == 0 ||
            enterpoint_node_.load(std::memory_order_acquire) == static_cast<tableint>(-1))
            return result;

        tableint enterpoint_snapshot = enterpoint_node_.load(std::memory_order_acquire);
        tableint currObj = enterpoint_snapshot;
        dist_t curdist = L2Distance(query, getDataByInternalId(enterpoint_snapshot));

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
                    dist_t d = L2Distance(query, getDataByInternalId(cand));

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
    top_candidates = searchLayer(currObj, query, 0, ef_);

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        return top_candidates;
    }

    void build(int d, const vector<float>& base);

    void search(const vector<float>& query, int *res);

    double getAverageDistanceCalcsPerSearch() const {
        uint64_t searches = query_count_.load(std::memory_order_relaxed);
        if (searches == 0) {
            return 0.0;
        }
        uint64_t total = query_distance_calcs_.load(std::memory_order_relaxed);
        return static_cast<double>(total) / static_cast<double>(searches);
    }

private:
    class SearchDistanceScope {
    public:
        explicit SearchDistanceScope(Solution& sol)
            : sol_(sol),
              prev_solution_(DistanceStatContext::active_solution),
              prev_counter_(DistanceStatContext::counter) {
            DistanceStatContext::active_solution = &sol_;
            DistanceStatContext::counter = 0;
        }

        ~SearchDistanceScope() {
            const uint64_t count = DistanceStatContext::counter;
            DistanceStatContext::active_solution = prev_solution_;
            DistanceStatContext::counter = prev_counter_;
            sol_.record_search_distance(count);
        }

    private:
        Solution& sol_;
        const Solution* prev_solution_;
        uint64_t prev_counter_;
    };

    void record_search_distance(uint64_t count) const {
        query_distance_calcs_.fetch_add(count, std::memory_order_relaxed);
        query_count_.fetch_add(1, std::memory_order_relaxed);
    }

    mutable std::atomic<uint64_t> query_distance_calcs_{0};
    mutable std::atomic<uint64_t> query_count_{0};


};



#endif //CPP_SOLUTION_H
