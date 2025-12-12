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
#include <fstream>
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

#include "BinaryIO.h"

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

// ==================== Ablation Study Configuration ====================

// 搜索策略
enum class SearchMethod {
    DYNAMIC_GAMMA,   // 自适应 gamma（拥塞动态缩放）
    STATIC_GAMMA,    // 固定 gamma（不缩放）
    FIXED_EF         // 固定 ef 搜索
};

// Ablation 配置
struct SolutionConfig {
    // 构建参数
    size_t M = 96;                    // Level0 最大出度
    size_t ef_construction = 400;     // 构建 ef
    bool enable_onng = true;          // 开 ONNG
    size_t onng_out_degree = 96;      // ONNG e_o
    size_t onng_in_degree = 144;      // ONNG e_i
    size_t onng_min_edges = 64;       // ONNG min edges
    bool enable_bfs = true;           // 开 BFS 重排
    bool enable_simd = true;          // 开 SIMD，关则强制标量
    bool enable_multilayer = true;    // 开多层图（关则退化为单层）
    bool use_negative_inner_product = false; // 使用负内积（否则用 L2）

    // 搜索参数
    SearchMethod search_method = SearchMethod::DYNAMIC_GAMMA;
    float gamma = 0.19f;              // 自适应 gamma
    size_t ef_search = 704;           // 固定 ef
    size_t k = 10;                    // Top-K

    // 调试
    bool count_distance_computation = false;  // 计数距离计算

    // 入口点相关
    size_t ep_num = 100;              // 入口点数
    size_t random_seed = 114514;      // 随机种子
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

// 标量版本：4路展开提升 ILP
inline dist_t l2_distance_scalar(const float* a, const float* b, size_t dim) {
    dist_t acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    size_t i = 0, bound = dim & ~size_t(3);
    for (; i < bound; i += 4) {
        dist_t d0 = a[i] - b[i], d1 = a[i+1] - b[i+1], d2 = a[i+2] - b[i+2], d3 = a[i+3] - b[i+3];
        acc0 += d0*d0; acc1 += d1*d1; acc2 += d2*d2; acc3 += d3*d3;
    }
    for (; i < dim; ++i) { dist_t d = a[i] - b[i]; acc0 += d*d; }
    return (acc0 + acc1) + (acc2 + acc3);
}

// 标量负内积（返回 -sum(a*b)）
inline dist_t neg_inner_product_scalar(const float* a, const float* b, size_t dim) {
    dist_t acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    size_t i = 0, bound = dim & ~size_t(3);
    for (; i < bound; i += 4) {
        acc0 += a[i] * b[i];
        acc1 += a[i+1] * b[i+1];
        acc2 += a[i+2] * b[i+2];
        acc3 += a[i+3] * b[i+3];
    }
    for (; i < dim; ++i) acc0 += a[i] * b[i];
    return -(acc0 + acc1 + acc2 + acc3);
}

#if SOLUTION_PLATFORM_X86

using L2Kernel = dist_t (*)(const float*, const float*, size_t);
using DotKernel = dist_t (*)(const float*, const float*, size_t);

inline void cpuid_ex(int info[4], int func, int subfunc = 0) {
#if defined(_MSC_VER)
    __cpuidex(info, func, subfunc);
#elif defined(__GNUC__) || defined(__clang__)
    __asm__ volatile("cpuid" : "=a"(info[0]), "=b"(info[1]), "=c"(info[2]), "=d"(info[3]) 
                             : "a"(func), "c"(subfunc));
#else
    info[0] = info[1] = info[2] = info[3] = 0;
#endif
}

// 读取 XCR0 寄存器值
inline unsigned long long read_xcr0() {
#if defined(_MSC_VER)
    return _xgetbv(0);
#elif defined(__GNUC__) || defined(__clang__)
    unsigned eax, edx;
    __asm__ volatile(".byte 0x0f,0x01,0xd0" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<unsigned long long>(edx) << 32) | eax;
#else
    return 0;
#endif
}

#if SOLUTION_COMPILE_AVX512
SOLUTION_SIMD_TARGET_ATTR("avx512f")
inline dist_t l2_distance_avx512(const float* a, const float* b, size_t dim) {
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
    size_t i = 0;
    for (size_t bound = dim & ~size_t(63); i < bound; i += 64) {
        __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(a+i),    _mm512_loadu_ps(b+i));
        __m512 d1 = _mm512_sub_ps(_mm512_loadu_ps(a+i+16), _mm512_loadu_ps(b+i+16));
        __m512 d2 = _mm512_sub_ps(_mm512_loadu_ps(a+i+32), _mm512_loadu_ps(b+i+32));
        __m512 d3 = _mm512_sub_ps(_mm512_loadu_ps(a+i+48), _mm512_loadu_ps(b+i+48));
        acc0 = _mm512_fmadd_ps(d0, d0, acc0); acc1 = _mm512_fmadd_ps(d1, d1, acc1);
        acc2 = _mm512_fmadd_ps(d2, d2, acc2); acc3 = _mm512_fmadd_ps(d3, d3, acc3);
    }
    for (size_t bound16 = dim & ~size_t(15); i < bound16; i += 16) {
        __m512 d = _mm512_sub_ps(_mm512_loadu_ps(a+i), _mm512_loadu_ps(b+i));
        acc0 = _mm512_fmadd_ps(d, d, acc0);
    }
    dist_t sum = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3)));
    for (; i < dim; ++i) { dist_t d = a[i] - b[i]; sum += d*d; }
    return sum;
}

// 负内积 AVX512
SOLUTION_SIMD_TARGET_ATTR("avx512f")
inline dist_t neg_ip_avx512(const float* a, const float* b, size_t dim) {
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
    size_t i = 0;
    for (size_t bound = dim & ~size_t(63); i < bound; i += 64) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i),    _mm512_loadu_ps(b+i),    acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+16), _mm512_loadu_ps(b+i+16), acc1);
        acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+32), _mm512_loadu_ps(b+i+32), acc2);
        acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+48), _mm512_loadu_ps(b+i+48), acc3);
    }
    for (size_t bound16 = dim & ~size_t(15); i < bound16; i += 16) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i), _mm512_loadu_ps(b+i), acc0);
    }
    dist_t sum = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3)));
    for (; i < dim; ++i) sum += a[i] * b[i];
    return -sum;
}
#endif

#if SOLUTION_COMPILE_AVX
SOLUTION_SIMD_TARGET_ATTR("avx,avx2,fma")
inline dist_t l2_distance_avx(const float* a, const float* b, size_t dim) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    size_t i = 0;
    for (size_t bound = dim & ~size_t(31); i < bound; i += 32) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a+i),    _mm256_loadu_ps(b+i));
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a+i+8),  _mm256_loadu_ps(b+i+8));
        __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(a+i+16), _mm256_loadu_ps(b+i+16));
        __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(a+i+24), _mm256_loadu_ps(b+i+24));
        acc0 = _mm256_fmadd_ps(d0, d0, acc0); acc1 = _mm256_fmadd_ps(d1, d1, acc1);
        acc2 = _mm256_fmadd_ps(d2, d2, acc2); acc3 = _mm256_fmadd_ps(d3, d3, acc3);
    }
    for (size_t bound8 = dim & ~size_t(7); i < bound8; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i));
        acc0 = _mm256_fmadd_ps(d, d, acc0);
    }
    __m256 sum8 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    __m128 sum4 = _mm_add_ps(_mm256_castps256_ps128(sum8), _mm256_extractf128_ps(sum8, 1));
    sum4 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
    sum4 = _mm_add_ss(sum4, _mm_movehdup_ps(sum4));
    dist_t sum = _mm_cvtss_f32(sum4);
    for (; i < dim; ++i) { dist_t d = a[i] - b[i]; sum += d*d; }
    return sum;
}

// 负内积 AVX
SOLUTION_SIMD_TARGET_ATTR("avx,avx2,fma")
inline dist_t neg_ip_avx(const float* a, const float* b, size_t dim) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    size_t i = 0;
    for (size_t bound = dim & ~size_t(31); i < bound; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i),    _mm256_loadu_ps(b+i),    acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+8),  _mm256_loadu_ps(b+i+8),  acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+16), _mm256_loadu_ps(b+i+16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+24), _mm256_loadu_ps(b+i+24), acc3);
    }
    for (size_t bound8 = dim & ~size_t(7); i < bound8; i += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc0);
    }
    __m256 sum8 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    __m128 sum4 = _mm_add_ps(_mm256_castps256_ps128(sum8), _mm256_extractf128_ps(sum8, 1));
    sum4 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
    sum4 = _mm_add_ss(sum4, _mm_movehdup_ps(sum4));
    dist_t sum = _mm_cvtss_f32(sum4);
    for (; i < dim; ++i) sum += a[i] * b[i];
    return -sum;
}
#endif

#if SOLUTION_COMPILE_SSE
SOLUTION_SIMD_TARGET_ATTR("sse2")
inline dist_t l2_distance_sse(const float* a, const float* b, size_t dim) {
    __m128 acc0 = _mm_setzero_ps(), acc1 = _mm_setzero_ps();
    __m128 acc2 = _mm_setzero_ps(), acc3 = _mm_setzero_ps();
    size_t i = 0;
    for (size_t bound = dim & ~size_t(15); i < bound; i += 16) {
        __m128 d0 = _mm_sub_ps(_mm_loadu_ps(a+i),    _mm_loadu_ps(b+i));
        __m128 d1 = _mm_sub_ps(_mm_loadu_ps(a+i+4),  _mm_loadu_ps(b+i+4));
        __m128 d2 = _mm_sub_ps(_mm_loadu_ps(a+i+8),  _mm_loadu_ps(b+i+8));
        __m128 d3 = _mm_sub_ps(_mm_loadu_ps(a+i+12), _mm_loadu_ps(b+i+12));
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(d0, d0)); acc1 = _mm_add_ps(acc1, _mm_mul_ps(d1, d1));
        acc2 = _mm_add_ps(acc2, _mm_mul_ps(d2, d2)); acc3 = _mm_add_ps(acc3, _mm_mul_ps(d3, d3));
    }
    for (size_t bound4 = dim & ~size_t(3); i < bound4; i += 4) {
        __m128 d = _mm_sub_ps(_mm_loadu_ps(a+i), _mm_loadu_ps(b+i));
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(d, d));
    }
    __m128 sum4 = _mm_add_ps(_mm_add_ps(acc0, acc1), _mm_add_ps(acc2, acc3));
    __m128 shuf = _mm_shuffle_ps(sum4, sum4, _MM_SHUFFLE(2,3,0,1));
    sum4 = _mm_add_ps(sum4, shuf);
    sum4 = _mm_add_ss(sum4, _mm_movehl_ps(shuf, sum4));
    dist_t sum = _mm_cvtss_f32(sum4);
    for (; i < dim; ++i) { dist_t d = a[i] - b[i]; sum += d*d; }
    return sum;
}

// 负内积 SSE
SOLUTION_SIMD_TARGET_ATTR("sse2")
inline dist_t neg_ip_sse(const float* a, const float* b, size_t dim) {
    __m128 acc0 = _mm_setzero_ps(), acc1 = _mm_setzero_ps();
    __m128 acc2 = _mm_setzero_ps(), acc3 = _mm_setzero_ps();
    size_t i = 0;
    for (size_t bound = dim & ~size_t(15); i < bound; i += 16) {
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(a+i),    _mm_loadu_ps(b+i)));
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_loadu_ps(a+i+4),  _mm_loadu_ps(b+i+4)));
        acc2 = _mm_add_ps(acc2, _mm_mul_ps(_mm_loadu_ps(a+i+8),  _mm_loadu_ps(b+i+8)));
        acc3 = _mm_add_ps(acc3, _mm_mul_ps(_mm_loadu_ps(a+i+12), _mm_loadu_ps(b+i+12)));
    }
    for (size_t bound4 = dim & ~size_t(3); i < bound4; i += 4) {
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(a+i), _mm_loadu_ps(b+i)));
    }
    __m128 sum4 = _mm_add_ps(_mm_add_ps(acc0, acc1), _mm_add_ps(acc2, acc3));
    __m128 shuf = _mm_shuffle_ps(sum4, sum4, _MM_SHUFFLE(2,3,0,1));
    sum4 = _mm_add_ps(sum4, shuf);
    sum4 = _mm_add_ss(sum4, _mm_movehl_ps(shuf, sum4));
    dist_t sum = _mm_cvtss_f32(sum4);
    for (; i < dim; ++i) sum += a[i] * b[i];
    return -sum;
}
#endif

// 初始化 L2 距离计算内核，自动检测支持的指令集
inline L2Kernel init_kernel() {
    int info[4] = {};
    cpuid_ex(info, 0);
    int max_func = info[0];
    bool osxsave = false;
    unsigned long long xcr0 = 0;

    if (max_func >= 1) {
        cpuid_ex(info, 1);
        osxsave = (info[2] & (1 << 27)) != 0;
        if (osxsave) xcr0 = read_xcr0();
#if SOLUTION_COMPILE_AVX512
        if (max_func >= 7) {
            cpuid_ex(info, 7, 0);
            if ((info[1] & (1 << 16)) && (xcr0 & 0xE6) == 0xE6) return l2_distance_avx512;
        }
#endif
#if SOLUTION_COMPILE_AVX
        cpuid_ex(info, 1);
        if ((info[2] & (1 << 28)) && (xcr0 & 0x6) == 0x6) return l2_distance_avx;
#endif
#if SOLUTION_COMPILE_SSE
        if (info[3] & (1 << 25)) return l2_distance_sse;
#endif
    }
    return l2_distance_scalar;
}

// 初始化负内积计算内核
inline DotKernel init_dot_kernel() {
    int info[4] = {};
    cpuid_ex(info, 0);
    int max_func = info[0];
    bool osxsave = false;
    unsigned long long xcr0 = 0;

    if (max_func >= 1) {
        cpuid_ex(info, 1);
        osxsave = (info[2] & (1 << 27)) != 0;
        if (osxsave) xcr0 = read_xcr0();
#if SOLUTION_COMPILE_AVX512
        if (max_func >= 7) {
            cpuid_ex(info, 7, 0);
            if ((info[1] & (1 << 16)) && (xcr0 & 0xE6) == 0xE6) return neg_ip_avx512;
        }
#endif
#if SOLUTION_COMPILE_AVX
        cpuid_ex(info, 1);
        if ((info[2] & (1 << 28)) && (xcr0 & 0x6) == 0x6) return neg_ip_avx;
#endif
#if SOLUTION_COMPILE_SSE
        if (info[3] & (1 << 25)) return neg_ip_sse;
#endif
    }
    return neg_inner_product_scalar;
}

inline dist_t compute(const float* a, const float* b, size_t dim) {
    static const L2Kernel kernel = init_kernel();
    return kernel(a, b, dim);
}

// runtime 切换 SIMD/标量
inline dist_t compute_switchable(const float* a, const float* b, size_t dim, bool use_simd) {
    if (use_simd) {
        static const L2Kernel kernel = init_kernel();
        return kernel(a, b, dim);
    }
    return l2_distance_scalar(a, b, dim);
}

// runtime 选择 SIMD/标量 负内积
inline dist_t compute_neg_ip_switchable(const float* a, const float* b, size_t dim, bool use_simd) {
    if (use_simd) {
        static const DotKernel kernel = init_dot_kernel();
        return kernel(a, b, dim);
    }
    return neg_inner_product_scalar(a, b, dim);
}

#else // !SOLUTION_PLATFORM_X86

inline dist_t compute(const float* a, const float* b, size_t dim) {
    return l2_distance_scalar(a, b, dim);
}

inline dist_t compute_switchable(const float* a, const float* b, size_t dim, bool /*use_simd*/) {
    return l2_distance_scalar(a, b, dim);
}

inline dist_t compute_neg_ip_switchable(const float* a, const float* b, size_t dim, bool /*use_simd*/) {
    return neg_inner_product_scalar(a, b, dim);
}

#endif

} // namespace simd_detail


// ==================== 4-ary 最大堆 ====================
// 替换 priority_queue<distPair, vector<distPair>, CompareDist>
// 堆顶是最大距离
// 额外支持直接访问所有元素
class MaxHeap4 {
private:
    vector<distPair> data_;
    
    static inline size_t parent(size_t i) { return (i - 1) >> 2; }
    static inline size_t first_child(size_t i) { return (i << 2) + 1; }
    
    inline void sift_up(size_t i) {
        distPair val = data_[i];
        while (i > 0) {
            size_t p = parent(i);
            if (data_[p].first >= val.first) break;
            data_[i] = data_[p];
            i = p;
        }
        data_[i] = val;
    }
    
    inline void sift_down(size_t i) {
        const size_t n = data_.size();
        distPair val = data_[i];
        
        while (true) {
            size_t fc = first_child(i);
            if (fc >= n) break;
            
            size_t max_child = fc;
            dist_t max_dist = data_[fc].first;
            
            if (fc + 1 < n && data_[fc + 1].first > max_dist) {
                max_child = fc + 1;
                max_dist = data_[fc + 1].first;
            }
            if (fc + 2 < n && data_[fc + 2].first > max_dist) {
                max_child = fc + 2;
                max_dist = data_[fc + 2].first;
            }
            if (fc + 3 < n && data_[fc + 3].first > max_dist) {
                max_child = fc + 3;
                max_dist = data_[fc + 3].first;
            }
            
            if (val.first >= max_dist) break;
            
            data_[i] = data_[max_child];
            i = max_child;
        }
        data_[i] = val;
    }
    
public:
    MaxHeap4() { data_.reserve(64); }
    
    inline bool empty() const { return data_.empty(); }
    inline size_t size() const { return data_.size(); }
    inline const distPair& top() const { return data_[0]; }
    
    // 直接访问（用于遍历结果）
    inline const distPair* data() const { return data_.data(); }
    inline const distPair& operator[](size_t i) const { return data_[i]; }
    
    inline void emplace(dist_t dist, tableint id) {
        data_.emplace_back(dist, id);
        sift_up(data_.size() - 1);
    }
    
    inline void pop() {
        if (data_.size() <= 1) { data_.clear(); return; }
        data_[0] = data_.back();
        data_.pop_back();
        sift_down(0);
    }
    
    inline void push(const distPair& p) {
        data_.push_back(p);
        sift_up(data_.size() - 1);
    }
    
    inline void clear() { data_.clear(); }
    
    inline void reserve(size_t n) { data_.reserve(n); }
    
    // 保留最小的 k 个元素（按距离升序排列），比多次 pop 更高效
    // 返回实际保留的元素数量
    inline size_t shrink_to_k_smallest(size_t k) {
        if (data_.size() <= k) {
            // 元素不足 k 个，直接排序
            sort(data_.begin(), data_.end(), 
                [](const distPair& a, const distPair& b) { return a.first < b.first; });
            return data_.size();
        }
        // partial_sort: 把最小的 k 个放到前面，并排好序
        partial_sort(data_.begin(), data_.begin() + k, data_.end(),
            [](const distPair& a, const distPair& b) { return a.first < b.first; });
        data_.resize(k);
        return k;
    }
};


class Solution {
public:
    // ====================== Configuration =========================
    SolutionConfig config_;

    // 设置 / 获取配置（build 前调用）
    void setConfig(const SolutionConfig& cfg) {
        config_ = cfg;
    }
    const SolutionConfig& getConfig() const { return config_; }

    // 重置距离统计
    void resetDistanceStats() {
        query_distance_calcs_.store(0, memory_order_relaxed);
        query_count_.store(0, memory_order_relaxed);
    }

    // ====================== 随机多入口点 =========================
    void initRandomEpoints() {
        entry_points_.clear();
        size_t node_count = cur_element_count.load(std::memory_order_acquire);
        if (node_count == 0) return;
        size_t epNum = min(config_.ep_num, node_count);
        vector<tableint> ids(node_count);
        for (tableint i = 0; i < static_cast<tableint>(node_count); ++i) ids[i] = i;
        shuffle(ids.begin(), ids.end(), level_generator_);
        entry_points_.assign(ids.begin(), ids.begin() + epNum);
    }

    // ====================== ONNG 优化  =========================
    
    // Algorithm 4: AdjustPath - 剪枝冗余边
    // 返回剪枝后的邻接表
    vector<vector<tableint>> adjustPath(int layer, size_t minNoOfEdges) {
        size_t node_count = cur_element_count.load(memory_order_acquire);
        vector<vector<tableint>> G_p(node_count);  // 剪枝后的图
        
        // 每个节点建立邻居 HashSet
        vector<unordered_set<tableint>> neighborSets(node_count);
        for (tableint n = 0; n < static_cast<tableint>(node_count); ++n) {
            if (layer > element_levels_[n]) continue;
            linklistsizeint* ll = getLinkListAtLevel(n, layer);
            int size = getListCount(ll);
            const tableint* neighbors = getNeighborsArray(ll);
            neighborSets[n].reserve(size);
            for (int i = 0; i < size; ++i) {
                neighborSets[n].insert(neighbors[i]);
            }
        }
        
        // 遍历每个节点的邻居，检查替代路径
        for (tableint src = 0; src < static_cast<tableint>(node_count); ++src) {
            if (layer > element_levels_[src]) continue;
            
            linklistsizeint* ll = getLinkListAtLevel(src, layer);
            int size = getListCount(ll);
            const tableint* neighbors = getNeighborsArray(ll);
            const float* src_data = getDataByInternalId(src);
            
            if (size == 0) continue;
            
            // 构建邻居信息并按距离排序
            vector<distPair> edges;
            edges.reserve(size);
            for (int i = 0; i < size; ++i) {
                tableint nbr = neighbors[i];
                dist_t dist = L2Distance(src_data, getDataByInternalId(nbr));
                edges.emplace_back(dist, nbr);
            }
            sort(edges.begin(), edges.end());
            
            // 逐个检查，在正在构建的 G_p 上检查是否有替代路径
            for (const auto& [direct_dist, dst] : edges) {
                bool has_path = false;
                
                // 检查 G_p[src] 中已有的邻居能否两跳到达 dst
                for (tableint mid : G_p[src]) {
                    // mid 到 dst 的距离
                    dist_t dist_mid_to_dst = L2Distance(getDataByInternalId(mid), getDataByInternalId(dst));
                    if (dist_mid_to_dst >= direct_dist) continue;
                    
                    // 用 HashSet O(1) 检查 mid 的邻居是否包含 dst
                    if (neighborSets[mid].count(dst)) {
                        has_path = true;
                        break;
                    }
                }
                
                // 如果没有替代路径，或者边数不足下限，则保留这条边
                if (!has_path || G_p[src].size() < minNoOfEdges) {
                    G_p[src].push_back(dst);
                }
            }
        }
        
        return G_p;
    }
    
    // Algorithm 3: ConstructAdjustedGraphWithConstraint
    // 从剪枝后的图重构，同时添加反向边并控制入度
    // e_o: 出度上限, e_i: 入度上限, minEdges: adjustPath 最小边数
    void constructAdjustedGraph(int layer, size_t e_o, size_t e_i, size_t minEdges = 0) {
        size_t node_count = cur_element_count.load(memory_order_acquire);
        if (minEdges == 0) minEdges = e_o;
        
        // 调用 Algorithm 4 得到剪枝后的图 G_t
        vector<vector<tableint>> G_t = adjustPath(layer, minEdges);
        
        // 收集反向边
        vector<vector<distPair>> reverseEdges(node_count);
        for (tableint src = 0; src < static_cast<tableint>(node_count); ++src) {
            const float* src_data = getDataByInternalId(src);
            for (tableint dst : G_t[src]) {
                dist_t dist = L2Distance(src_data, getDataByInternalId(dst));
                reverseEdges[dst].emplace_back(dist, src);
            }
        }
        
        // 初始化输出图和入度计数
        vector<vector<tableint>> G_e(node_count);
        vector<size_t> indegree(node_count, 0);
        
        // 按出度排序，优先处理出度小的节点
        vector<pair<size_t, tableint>> sortedByOutdegree;
        sortedByOutdegree.reserve(node_count);
        for (tableint o = 0; o < static_cast<tableint>(node_count); ++o) {
            if (layer > element_levels_[o]) continue;
            sortedByOutdegree.emplace_back(G_t[o].size(), o);
        }
        sort(sortedByOutdegree.begin(), sortedByOutdegree.end());
        
        // 从 G_t 添加出边，控制入度
        for (const auto& [outdegree, o] : sortedByOutdegree) {
            const float* o_data = getDataByInternalId(o);
            
            // 按距离排序 G_t[o] 的邻居
            vector<distPair> candidates;
            for (tableint n : G_t[o]) {
                dist_t d = L2Distance(o_data, getDataByInternalId(n));
                candidates.emplace_back(d, n);
            }
            sort(candidates.begin(), candidates.end());
            
            // 添加边，检查入度限制
            for (const auto& [dist, n] : candidates) {
                if (G_e[o].size() >= e_o) break;
                if (indegree[n] == 0 || (indegree[n] < e_i && G_e[o].size() < e_o)) {
                    G_e[o].push_back(n);
                    indegree[n]++;
                }
            }
        }
        
        // 添加反向边
        for (const auto& [outdegree, o] : sortedByOutdegree) {
            if (G_e[o].size() >= e_o) continue;
            
            // 建立已有邻居的快速查找
            unordered_set<tableint> existing(G_e[o].begin(), G_e[o].end());
            
            // 按距离排序反向边
            auto& rev = reverseEdges[o];
            sort(rev.begin(), rev.end());
            
            // 添加反向边
            for (const auto& [dist, src] : rev) {
                if (G_e[o].size() >= e_o) break;
                if (existing.count(src)) continue;
                if (indegree[src] >= e_i) continue;
                
                G_e[o].push_back(src);
                existing.insert(src);
                indegree[src]++;
            }
        }
        
        // 如果不足，从原图补充
        for (tableint o = 0; o < static_cast<tableint>(node_count); ++o) {
            if (layer > element_levels_[o]) continue;
            if (G_e[o].size() >= e_o) continue;
            
            linklistsizeint* ll = getLinkListAtLevel(o, layer);
            int size = getListCount(ll);
            const tableint* neighbors = getNeighborsArray(ll);
            const float* o_data = getDataByInternalId(o);
            
            // 建立已有邻居查找
            unordered_set<tableint> existing(G_e[o].begin(), G_e[o].end());
            
            // 按距离排序原图邻居
            vector<distPair> orig;
            for (int i = 0; i < size; ++i) {
                if (!existing.count(neighbors[i])) {
                    orig.emplace_back(L2Distance(o_data, getDataByInternalId(neighbors[i])), neighbors[i]);
                }
            }
            sort(orig.begin(), orig.end());
            
            for (const auto& [dist, n] : orig) {
                if (G_e[o].size() >= e_o) break;
                G_e[o].push_back(n);
            }
        }
        
        // 写回 HNSW 图结构
        for (tableint o = 0; o < static_cast<tableint>(node_count); ++o) {
            if (layer > element_levels_[o]) continue;
            
            lock_guard<mutex> lock(link_list_locks_[o]);
            linklistsizeint* ll = getLinkListAtLevel(o, layer);
            tableint* arr = getNeighborsArray(ll);
            
            size_t cnt = min(G_e[o].size(), (layer == 0) ? maxM0_ : maxM_);
            for (size_t i = 0; i < cnt; ++i) {
                arr[i] = G_e[o][i];
            }
            setListCount(ll, static_cast<unsigned short>(cnt));
        }
    }
    
    // 统一优化接口
    // e_o: 出度上限
    // e_i: 入度上限
    // minEdges: adjustPath 的最小边数
    void optimizeGraphDirectly(bool optimize_high_layers = false, 
                                size_t e_o = 0,
                                size_t e_i = 0, 
                                size_t minEdges = 0,
                                int maxOnngLevel = 0) {

        int max_level = maxlevel_.load(memory_order_acquire);
        maxOnngLevel = (maxOnngLevel <= 0) ? max_level : maxOnngLevel;

        // ============ 强制中心化入口 ============
        // 计算所有点的几何质心，并选择离质心最近的点作为新的 enterpoint
        size_t node_count = cur_element_count.load(memory_order_acquire);
        if (node_count > 0 && dim_ > 0) {
            vector<float> centroid(dim_, 0.0f);
            for (tableint id = 0; id < static_cast<tableint>(node_count); ++id) {
                const float* v = getDataByInternalId(id);
                for (size_t d = 0; d < dim_; ++d) centroid[d] += v[d];
            }
            float inv_n = 1.0f / static_cast<float>(node_count);
            for (size_t d = 0; d < dim_; ++d) centroid[d] *= inv_n;

            tableint best_id = 0;
            dist_t best_dist = numeric_limits<dist_t>::max();
            for (tableint id = 0; id < static_cast<tableint>(node_count); ++id) {
                dist_t d = L2Distance(centroid.data(), getDataByInternalId(id));
                if (d < best_dist) {
                    best_dist = d;
                    best_id = id;
                }
            }
            enterpoint_node_.store(best_id, memory_order_release);
        }

        size_t e_o_l0 = e_o;
        size_t e_i_l0 = e_i;
        size_t min_l0 = minEdges / 10 * 9;
        
        // 第0层优化
        constructAdjustedGraph(0, e_o_l0, e_i_l0, min_l0);
        
        if (optimize_high_layers && max_level > 0) {
            // 高层优化
            size_t e_o_h = min(e_o, maxM_);
            size_t e_i_h = min(e_i, maxM_);
            size_t min_h = minEdges / 10 * 12;
            
            for (int level = 1; level <= min(max_level, maxOnngLevel); ++level) {
                constructAdjustedGraph(level, e_o_h, e_i_h, min_h);
            }
        }
    }
    
    
    // ==================== HNSW 结构 ====================

    // 图结构
    // 储存节点数据及第 0 层的邻居关系
    // 邻居数量，flag, 凑对齐，邻居节点 id，数据向量，标签
    // 对每一个 Node: size->reserved->neighbors
    // size:2 bytes, reserved:2 byte

    char* data_level0_memory_{nullptr};
    
    // 二维数组，每一行代表一个节点从第 1 层到最高层的邻居关系
    // 邻居数量，凑对齐，邻居节点 id
    // 每一层存储格式：size->reserved->neighbors
    // size:2 bytes, reserved:2 bytes
    
    char** linkLists_{nullptr};

    vector<int> element_levels_; // 记录每个节点的层数
    mutable unique_ptr<mutex[]> link_list_locks_; // 每个节点的邻接锁，支持并发构建

    size_t max_elements_{0};                // 最大节点数量
    size_t M_{0};                           // 每个节点最大连接数
    size_t maxM_{0};                        // 每个节点第 1 层及以上最大连接数
    size_t maxM0_{0};                       // 每个节点第 0 层最大连接数
    size_t ef_construction_{0};             // 构建时候选列表大小

    double mult_{0.0};                      // 论文中的 m_l
    size_t dim_{0};                         // 数据向量维度

    size_t data_size_{0};                   // 数据向量大小
    size_t size_links_level0_{0};           // 第 0 层每个节点链接大小
    size_t size_data_per_element_{0};       // 每个节点数据大小
    size_t size_links_per_element_{0};      // 每个节点第 1 层及以上链接大小
    size_t offsetData_{0}, offsetLevel0_{0};// 偏移量

    atomic<tableint> enterpoint_node_{static_cast<tableint>(-1)};  // 并行构建时的入口点
    atomic<int> maxlevel_{-1};              // 并行构建时当前最大层数
    atomic<bool> has_enterpoint_{false};    // 并行构建时是否已有入口点
    atomic<size_t> cur_element_count{0};    // 当前节点数量
    size_t worker_count_{0};                     // 构建时的工作线程数

    default_random_engine level_generator_; // 用于生成随机层数的随机数
    size_t ep_num_{100};                    // 多入口点数量（消融用）
    vector<tableint> entry_points_;         // 多入口点列表

    // 逻辑 ID (原始下标) 与物理 ID (内部存储下标) 映射
    // 构建时：logical_id = 插入顺序/原始下标；physical_id = 内部数组下标
    // 外部接口始终使用逻辑 ID，内部存图和搜索全部使用物理 ID
    vector<tableint> logical_to_physical_;
    vector<tableint> physical_to_logical_;

    void initHNSW(
        size_t max_elements,
        size_t M = 16,
        size_t random_seed = 100,
        size_t ef_construction = 200,
        size_t data_size = 0,
        size_t worker_count = 1
    ){
        // 设置参数
        max_elements_ = max_elements;
        M_ = M;
        maxM_ = M_;
        // 单层模式下 M 直接作为 maxM0；多层则沿用 2*M 的设置
        maxM0_ = config_.enable_multilayer ? (M_ * 2) : M_;
        data_size_ = data_size;
        ef_construction_ = max(ef_construction, M_);
        level_generator_.seed(random_seed);
        worker_count_ = worker_count;

        // 计算 mult
        mult_ = 1 / log(1.0 * M_);

        // 初始化当前节点数量
        cur_element_count.store(0, memory_order_relaxed);

        // 计算内存需求和偏移量
        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        offsetLevel0_ = 0;
        offsetData_ = offsetLevel0_ + size_links_level0_;
        size_data_per_element_ = size_links_level0_ + data_size_;
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        // 分配内存
        data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
        linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
        if (linkLists_) memset(linkLists_, 0, sizeof(void*) * max_elements_);
        
        // 层数记录
        element_levels_.assign(max_elements_, 0);

        // ID 映射表
        logical_to_physical_.assign(max_elements_, static_cast<tableint>(-1));
        physical_to_logical_.assign(max_elements_, static_cast<tableint>(-1));

        // 邻接锁
        link_list_locks_.reset(new mutex[max_elements_]);

        // 初始化入口点相关变量
        enterpoint_node_.store(static_cast<tableint>(-1), memory_order_relaxed);
        maxlevel_.store(-1, memory_order_relaxed);
        has_enterpoint_.store(false, memory_order_relaxed);
    }

    void clear() {
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
        tableint existing = static_cast<tableint>(cur_element_count.load(memory_order_relaxed));
        for (tableint i = 0; i < existing; i++) {
            if (element_levels_[i] > 0)
                free(linkLists_[i]);
        }
        free(linkLists_);
        linkLists_ = nullptr;
        cur_element_count.store(0, memory_order_relaxed);
        enterpoint_node_.store(static_cast<tableint>(-1), memory_order_relaxed);
        maxlevel_.store(-1, memory_order_relaxed);
        has_enterpoint_.store(false, memory_order_relaxed);
        link_list_locks_.reset();

        logical_to_physical_.clear();
        physical_to_logical_.clear();
    }


    // 获取节点数据指针
    inline const float *getDataByInternalId(tableint id) const {
        // 这里的 id 应为物理 ID
        return reinterpret_cast<const float*>(data_level0_memory_ + id * size_data_per_element_ + offsetData_);
    }

    // 获取第 0 层的邻居列表指针，不直接使用，需通过 getLinkListAtLevel 调用
    linklistsizeint *get_linklist0(tableint id) const {
    return (linklistsizeint *) (data_level0_memory_ + id * size_data_per_element_ + offsetLevel0_);
    }

    // 获取第 1 层及以上的邻居列表指针，不直接使用，需通过 getLinkListAtLevel 调用
    linklistsizeint *get_linklist(tableint id, int level) const {
        return (linklistsizeint *) (linkLists_[id] + (level - 1) * size_links_per_element_);
    }

    // 获取指定层的邻居列表指针
    linklistsizeint *getLinkListAtLevel(tableint id, int level) const {
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


    // 使用锁保护，确保并发插入时的线程安全（build 阶段使用）
    void collectNeighbors(tableint id, int level, vector<tableint>& out) const {
        if (!link_list_locks_ || id >= max_elements_) {
            out.clear();
            return;
        }
        lock_guard<mutex> guard(link_list_locks_[id]);
        if (level > element_levels_[id]) {
            out.clear();
            return;
        }
        linklistsizeint *header = getLinkListAtLevel(id, level);
        int size = getListCount(header);
        tableint *data = getNeighborsArray(header);
        out.assign(data, data + size);
    }

    // 无锁版本，用于 search 阶段（只读，build 完成后调用）
    inline void collectNeighborsLockFree(tableint id, int level, vector<tableint>& out) const {
        if (level > element_levels_[id]) {
            out.clear();
            return;
        }
        linklistsizeint *header = getLinkListAtLevel(id, level);
        int size = getListCount(header);
        tableint *data = getNeighborsArray(header);
        out.assign(data, data + size);
    }

    // 距离函数：根据配置选择 L2 或（单位向量下等价的）内积距离
    //
    // 重要：很多剪枝逻辑（尤其 gamma 搜索）默认“距离越小越好”且通常为非负。
    // 若直接用 -dot 作为距离，距离可能为负，剪枝条件会出现方向性问题，导致召回异常偏低。
    // 因此在 use_negative_inner_product 时，我们将 dot 转成非负距离：
    //   dist = 1 - dot  （当向量已归一化到单位范数时，与 L2^2/2 单调等价）
    dist_t L2Distance(const float * a, const float * b) const {
        if (config_.count_distance_computation) {
            query_distance_calcs_.fetch_add(1, memory_order_relaxed);
        }
        if (config_.use_negative_inner_product) {
            // compute_neg_ip_switchable 返回 -dot(a,b)
            dist_t dot = -simd_detail::compute_neg_ip_switchable(a, b, dim_, config_.enable_simd);
            // 数值稳定：限制到 [-1,1]，避免轻微溢出导致 dist 变负
            dot = std::max<dist_t>(-1.0f, std::min<dist_t>(1.0f, dot));
            return 1.0f - dot;
        }
        return simd_detail::compute_switchable(a, b, dim_, config_.enable_simd);
    }

    // 产生指数分布的层数
    int generateRandomLevel(double reverse_size){
        
        uniform_real_distribution<float> distribution(0.0f, 1.0f);
        float r = distribution(level_generator_);
        // 避免 r=0 导致 -log(0) 造成层数极大并引发巨额分配
        r = min(0.999999f, max(r, 1e-6f));
        double level = -log(r) * reverse_size;
        return (int)level;
       // return 0;
    }

    // 在指定层搜索，返回至多 ef_limit 个近邻节点
    // 实现论文的 Algorithm 2
    /*
    1. 初始化候选集和结果集，起点为入口点 ep_id
    2. 使用最大堆维护候选集和结果集，候选集存负距离模拟最小堆
    3. 迭代从候选集中取出距离查询点最近的节点，扩展其邻居
    4. 对每个邻居，计算距离，若满足条件加入候选集和结果集
    5. 终止条件：候选集最近的点比结果集最远的还远，且结果集已满
    */
    MaxHeap4 searchLayer(
        tableint ep_id,
        const float *query,
        int layer,
        size_t ef_limit) const {
        MaxHeap4 top_candidates;  // 最大堆，堆顶是最大距离
        MaxHeap4 candidate_set;   // 也用最大堆，存负距离模拟最小堆

        dist_t dist = L2Distance(query, getDataByInternalId(ep_id));
        top_candidates.emplace(dist, ep_id);
        candidate_set.emplace(-dist, ep_id);  // 负距离

        dist_t lowerBound = dist;       // 当前结果集中的最大距离

        thread_local vector<uint64_t> visit_bitmap_tls;
        thread_local vector<uint32_t> visit_epoch_tls;
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
            fill(visit_epoch_tls.begin(), visit_epoch_tls.end(), 0u);
            visit_token_tls = 1u;
        }
        auto touch_word = [&](size_t word_idx) {
            if (visit_epoch_tls[word_idx] != visit_token_tls) {
                visit_epoch_tls[word_idx] = visit_token_tls;
                visit_bitmap_tls[word_idx] = 0ull;
            }
        };
        auto is_visited = [&](tableint id) -> bool {
            size_t word = static_cast<size_t>(id) >> 6;
            size_t bit = static_cast<size_t>(id) & 63ull;
            touch_word(word);
            return (visit_bitmap_tls[word] >> bit) & 1ull;
        };
        auto mark_visited = [&](tableint id) {
            size_t word = static_cast<size_t>(id) >> 6;
            size_t bit = static_cast<size_t>(id) & 63ull;
            touch_word(word);
            visit_bitmap_tls[word] |= (1ull << bit);
        };

        mark_visited(ep_id);

        while (!candidate_set.empty()) {
            auto currPair = candidate_set.top();
            
            // 剪枝检查
            if ((-currPair.first) > lowerBound && top_candidates.size() == ef_limit) break;

            candidate_set.pop();
            tableint curID = currPair.second;

            linklistsizeint* ll = getLinkListAtLevel(curID, layer);
            int size = getListCount(ll);
            const tableint* data = getNeighborsArray(ll);
            for (int i = 0; i < size; ++i) {
                _mm_prefetch(getDataByInternalId(data[i]), _MM_HINT_T0);
            }

            for (int i = 0; i < size; ++i) {
                tableint candidateID = data[i];
                if (is_visited(candidateID)) continue;
                mark_visited(candidateID);
                dist_t d = L2Distance(query, getDataByInternalId(candidateID));
                if (top_candidates.size() < ef_limit || d < lowerBound) {
                    candidate_set.emplace(-d, candidateID);
                    top_candidates.emplace(d, candidateID);
                    if (top_candidates.size() > ef_limit) top_candidates.pop();
                    lowerBound = top_candidates.top().first;
                }
            }
        }
        return top_candidates;
    }

    // 启发式选邻居：按距离排序后，选择与已选邻居保持多样性的节点
    void selectNeighborsHeuristic(MaxHeap4& topCandidates, size_t M) {
        if (topCandidates.size() <= M) return;
        
        // 按距离升序排序
        topCandidates.shrink_to_k_smallest(topCandidates.size());
        
        thread_local vector<distPair> selected;
        selected.clear();
        selected.reserve(M);
        
        // 遍历排序后的候选，启发式选择
        for (size_t i = 0; i < topCandidates.size() && selected.size() < M; ++i) {
            const auto& cand = topCandidates[i];
            bool good = true;
            for (const auto& sel : selected) {
                if (L2Distance(getDataByInternalId(cand.second), getDataByInternalId(sel.second)) < cand.first) {
                    good = false;
                    break;
                }
            }
            if (good) selected.push_back(cand);
        }
        
        // 重建堆
        topCandidates.clear();
        for (const auto& p : selected) topCandidates.emplace(p.first, p.second);
    }

    // 将新节点和已有节点互相连接
    tableint mutuallyConnectNewElement(tableint cur_c, MaxHeap4& top_candidates, int level) {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        selectNeighborsHeuristic(top_candidates, M_);
        
        size_t neighbor_cnt = top_candidates.size();
        tableint next_close_ep = top_candidates[neighbor_cnt - 1].second;

        // 新节点写入本层的邻接
        {
            lock_guard<mutex> lock(link_list_locks_[cur_c]);
            linklistsizeint* ll = getLinkListAtLevel(cur_c, level);
            tableint* arr = getNeighborsArray(ll);
            size_t cnt = min(neighbor_cnt, Mcurmax);
            for (size_t i = 0; i < cnt; ++i) arr[i] = top_candidates[i].second;
            setListCount(ll, static_cast<unsigned short>(cnt));
        }

        // 反向连接：把 cur_c 加入每个邻居的邻接表
        for (size_t i = 0; i < neighbor_cnt; ++i) {
            tableint nbr = top_candidates[i].second;
            lock_guard<mutex> lock(link_list_locks_[nbr]);
            
            linklistsizeint* ll = getLinkListAtLevel(nbr, level);
            size_t sz = getListCount(ll);
            tableint* arr = getNeighborsArray(ll);

            if (sz < Mcurmax) {
                arr[sz] = cur_c;
                setListCount(ll, static_cast<unsigned short>(sz + 1));
            } else {
                // 邻居已满，需要启发式选择
                MaxHeap4 cands;
                cands.emplace(L2Distance(getDataByInternalId(cur_c), getDataByInternalId(nbr)), cur_c);
                for (size_t j = 0; j < sz; ++j)
                    cands.emplace(L2Distance(getDataByInternalId(arr[j]), getDataByInternalId(nbr)), arr[j]);
                
                selectNeighborsHeuristic(cands, Mcurmax);
                size_t new_sz = cands.size();
                for (size_t j = 0; j < new_sz; ++j) arr[j] = cands[j].second;
                setListCount(ll, static_cast<unsigned short>(new_sz));
            }
        }
        return next_close_ep;
    }


    // 把数据插入到多层图中
    // 并发插入：通过原子 ID 分配 + per-node 邻接锁，确保多个线程可以安全地同时写入图结构。
    /*
    1. 分配节点 ID cur_c
    2. 随机生成层数 curlevel
    3. 写入节点数据和初始化邻接列表
    4. 如果是第一个节点，设置为入口点并返回
    5. 贪心下降找到插入层 curlevel
    6. 从 curlevel 层开始，逐层向下插入节点并连接邻居 searchLayer + mutuallyConnect
    7. 如果 curlevel > 当前最大层数，更新入口点和最大层数
    */
    void insert(const float *vec, int level) {
        tableint cur_c = cur_element_count.fetch_add(1, memory_order_acq_rel);
        if (cur_c >= max_elements_)
            throw runtime_error("HNSW capacity exceeded");

        tableint logical_id = cur_c;
        tableint physical_id = cur_c;
        logical_to_physical_[logical_id] = physical_id;
        physical_to_logical_[physical_id] = logical_id;

        bool is_first = (cur_c == 0);

        // 随机初始化层数
        int curlevel = config_.enable_multilayer ? generateRandomLevel(mult_) : 0;
        if(level >= 0) curlevel = level;

        element_levels_[physical_id] = curlevel;
        tableint currObj = enterpoint_node_.load(memory_order_acquire);
        tableint enterpoint_copy = currObj;


        char* base_ptr = data_level0_memory_ + physical_id * size_data_per_element_;

        // 写入向量数据
        memcpy(base_ptr + offsetData_, vec, data_size_);

        // 初始化 level0 链表头与槽
        linklistsizeint* ll_head = reinterpret_cast<linklistsizeint*>(base_ptr + offsetLevel0_);
        *reinterpret_cast<unsigned short*>(ll_head) = 0;
        tableint* ll_array = reinterpret_cast<tableint*>((char*)ll_head + sizeof(unsigned short));
        memset(ll_array, 0, maxM0_ * sizeof(tableint));

        // 高层链表（仅在启用多层且 curlevel>0 分配并清零）
        if (config_.enable_multilayer && curlevel > 0) {
            size_t bytes = size_links_per_element_ * curlevel;
            linkLists_[cur_c] = (char*)malloc(bytes);
            memset(linkLists_[cur_c], 0, bytes);
        }

        if (is_first) {
            enterpoint_node_.store(physical_id, memory_order_release);
            if (config_.enable_multilayer) {
                maxlevel_.store(curlevel, memory_order_release);
                has_enterpoint_.store(true, memory_order_release);
            } else {
                maxlevel_.store(0, memory_order_release);
                has_enterpoint_.store(true, memory_order_release);
            }
            return;
        }

        while (!has_enterpoint_.load(memory_order_acquire)) {
            this_thread::yield();
        }

        currObj = enterpoint_node_.load(memory_order_acquire);
        enterpoint_copy = currObj;
        int maxLevelSnapshot = maxlevel_.load(memory_order_acquire);

        if (currObj != static_cast<tableint>(-1)) {
            if(config_.enable_multilayer && curlevel < maxLevelSnapshot) {
                dist_t curdist = L2Distance(vec, getDataByInternalId(currObj));
                vector<tableint> neighbors;
                neighbors.reserve(maxM0_);
                for(int level = maxLevelSnapshot; level > curlevel; level--) {
                    bool changed = true;
                    while(changed) {
                        changed = false;
                        collectNeighbors(currObj, level, neighbors);
                        for(tableint neighbor : neighbors) {
                            dist_t d = L2Distance(vec, getDataByInternalId(neighbor));
                            if(d < curdist) {
                                curdist = d;
                                currObj = neighbor;
                                changed = true;
                            }
                        }
                    }
                }

                // 从 curlevel 往下，找到一定数量的邻居，这里设定是 M_
                for (int lvl = min(curlevel, maxLevelSnapshot); lvl >= 0; lvl--) {
                    auto top_candidates = searchLayer(currObj, vec, lvl, ef_construction_);
                    // 这里把 Algorithm 1 后面的部分 合并到连接邻居一起实现了
                    currObj = mutuallyConnectNewElement(physical_id, top_candidates, lvl);
                }
            } else {
                tableint ep = enterpoint_copy;
                for (int lvl = maxLevelSnapshot; lvl >= 0; --lvl) {
                    auto top_candidates = searchLayer(ep, vec, lvl, ef_construction_);
                    ep = mutuallyConnectNewElement(physical_id, top_candidates, lvl);
                }
                if (config_.enable_multilayer) {
                    bool updated = false;
                    int expected = maxLevelSnapshot;
                    while (curlevel > expected) {
                        if (maxlevel_.compare_exchange_weak(expected, curlevel, memory_order_acq_rel)) {
                            updated = true;
                            break;
                        }
                    }
                    if (updated) {
                        enterpoint_node_.store(physical_id, memory_order_release);
                    }
                }
            }
        }
    }


    void build(int d, const vector<float>& base);
    // 显式配置版（消融）
    void buildWithConfig(int d, const vector<float>& base, const SolutionConfig& cfg);

    void search(const vector<float>& query, int *res);
    void searchWithK(const vector<float>& query, int *res, size_t k);

    // 对外暴露 BFS 重排
    void applyBFSReorder() { reorderNodesByBFS(); }

    // ====================== 图结构缓存 I/O（支持多层/单层区分） =========================
    bool saveGraph(const std::string &path) const {
        size_t node_count = cur_element_count.load(std::memory_order_acquire);
        if (node_count == 0 || data_level0_memory_ == nullptr) {
            std::cerr << "[saveGraph] No graph data to save.\n";
            return false;
        }

        std::ofstream ofs(path, std::ios::binary);
        if (!ofs.is_open()) return false;

        auto write_u32 = [&](uint32_t v) {
            uint8_t b[4];
            b[0] = static_cast<uint8_t>(v);
            b[1] = static_cast<uint8_t>(v >> 8);
            b[2] = static_cast<uint8_t>(v >> 16);
            b[3] = static_cast<uint8_t>(v >> 24);
            ofs.write(reinterpret_cast<const char*>(b), 4);
        };
        auto write_u64 = [&](uint64_t v) {
            uint8_t b[8];
            for (int i = 0; i < 8; ++i) b[i] = static_cast<uint8_t>(v >> (8 * i));
            ofs.write(reinterpret_cast<const char*>(b), 8);
        };

        // Magic (bump to version 3 to区分单层/多层)
        const char magic[8] = { 'G','R','A','P','H','C','3','\0' };
        ofs.write(magic, 8);

        // Header
        write_u32(static_cast<uint32_t>(dim_));
        write_u64(static_cast<uint64_t>(max_elements_));
        write_u64(static_cast<uint64_t>(node_count));
        write_u32(static_cast<uint32_t>(maxM0_));
        write_u32(static_cast<uint32_t>(M_));
        write_u32(static_cast<uint32_t>(maxM_));
        write_u32(static_cast<uint32_t>(ef_construction_));
        write_u64(static_cast<uint64_t>(size_data_per_element_));
        write_u64(static_cast<uint64_t>(size_links_level0_));
        write_u64(static_cast<uint64_t>(size_links_per_element_));
        write_u64(static_cast<uint64_t>(offsetData_));
        write_u64(static_cast<uint64_t>(offsetLevel0_));
        write_u64(static_cast<uint64_t>(data_size_));
        write_u32(static_cast<uint32_t>(entry_points_.size()));
        write_u32(static_cast<uint32_t>(maxlevel_.load(std::memory_order_acquire)));
        write_u32(config_.enable_multilayer ? 1u : 0u); // flag: multilayer or single layer

        // data_level0
        size_t data_bytes = node_count * size_data_per_element_;
        ofs.write(data_level0_memory_, static_cast<std::streamsize>(data_bytes));

        // entry points
        for (tableint ep : entry_points_) {
            write_u32(static_cast<uint32_t>(ep));
        }

        // element levels
        for (size_t i = 0; i < node_count; ++i) {
            write_u32(static_cast<uint32_t>(element_levels_[i]));
        }

        // logical_to_physical / physical_to_logical
        for (size_t i = 0; i < node_count; ++i) {
            write_u32(static_cast<uint32_t>(logical_to_physical_[i]));
        }
        for (size_t i = 0; i < node_count; ++i) {
            write_u32(static_cast<uint32_t>(physical_to_logical_[i]));
        }

        // linkLists_ per node
        for (size_t i = 0; i < node_count; ++i) {
            uint32_t bytes = 0;
            if (config_.enable_multilayer && element_levels_[i] > 0 && linkLists_[i]) {
                bytes = static_cast<uint32_t>(element_levels_[i] * size_links_per_element_);
            }
            write_u32(bytes);
            if (bytes > 0) {
                ofs.write(linkLists_[i], bytes);
            }
        }

        bool ok = ofs.good();
        if (ok) {
            std::cerr << "[saveGraph] Graph saved to: " << path
                      << " (nodes=" << node_count << ", dim=" << dim_ << ", max_level="
                      << maxlevel_.load(std::memory_order_acquire) << ")\n";
        } else {
            std::cerr << "[saveGraph] Failed to save graph to: " << path << "\n";
        }
        return ok;
    }

    bool loadGraph(const std::string &path, bool expected_multilayer) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.is_open()) return false;

        auto read_u32 = [&](uint32_t &v) -> bool {
            uint8_t b[4];
            ifs.read(reinterpret_cast<char*>(b), 4);
            if (!ifs) return false;
            v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
            return true;
        };
        auto read_u64 = [&](uint64_t &v) -> bool {
            uint8_t b[8];
            ifs.read(reinterpret_cast<char*>(b), 8);
            if (!ifs) return false;
            v = 0;
            for (int i = 0; i < 8; ++i) v |= ((uint64_t)b[i] << (8 * i));
            return true;
        };

        char magic[8];
        ifs.read(magic, 8);
        bool is_v3 = std::strncmp(magic, "GRAPHC3\0", 8) == 0;
        if (!is_v3) {
            // 老版本缓存不再兼容，强制重建
            return false;
        }

        uint32_t dim_rd, maxM0_rd, M_rd, maxM_rd, efc_rd, ep_cnt, maxlvl_rd, multilayer_flag;
        uint64_t max_elements_rd, node_count_rd, s_data_per_elem, s_links_l0, s_links_per, off_data, off_l0, data_sz;

        if (!read_u32(dim_rd)) return false;
        if (!read_u64(max_elements_rd)) return false;
        if (!read_u64(node_count_rd)) return false;
        if (!read_u32(maxM0_rd)) return false;
        if (!read_u32(M_rd)) return false;
        if (!read_u32(maxM_rd)) return false;
        if (!read_u32(efc_rd)) return false;
        if (!read_u64(s_data_per_elem)) return false;
        if (!read_u64(s_links_l0)) return false;
        if (!read_u64(s_links_per)) return false;
        if (!read_u64(off_data)) return false;
        if (!read_u64(off_l0)) return false;
        if (!read_u64(data_sz)) return false;
        if (!read_u32(ep_cnt)) return false;
        if (!read_u32(maxlvl_rd)) return false;
        if (!read_u32(multilayer_flag)) return false;

        if (static_cast<bool>(multilayer_flag) != expected_multilayer) {
            std::cerr << "[loadGraph] Cache multilayer flag mismatch, expect "
                      << expected_multilayer << " got " << static_cast<bool>(multilayer_flag) << "\n";
            return false;
        }

        // cleanup existing
        clear();

        dim_ = static_cast<size_t>(dim_rd);
        max_elements_ = static_cast<size_t>(max_elements_rd);
        maxM0_ = static_cast<size_t>(maxM0_rd);
        M_ = static_cast<size_t>(M_rd);
        maxM_ = static_cast<size_t>(maxM_rd);
        ef_construction_ = static_cast<size_t>(efc_rd);
        size_data_per_element_ = static_cast<size_t>(s_data_per_elem);
        size_links_level0_ = static_cast<size_t>(s_links_l0);
        size_links_per_element_ = static_cast<size_t>(s_links_per);
        offsetData_ = static_cast<size_t>(off_data);
        offsetLevel0_ = static_cast<size_t>(off_l0);
        data_size_ = static_cast<size_t>(data_sz);
        ep_num_ = static_cast<size_t>(ep_cnt);
        mult_ = 1 / std::log(1.0 * std::max<size_t>(M_, 1));

        size_t node_count = static_cast<size_t>(node_count_rd);

        data_level0_memory_ = (char*)malloc(max_elements_ * size_data_per_element_);
        if (!data_level0_memory_) return false;
        size_t data_bytes = node_count * size_data_per_element_;
        ifs.read(data_level0_memory_, static_cast<std::streamsize>(data_bytes));
        if (!ifs) return false;

        entry_points_.assign(ep_cnt, 0);
        for (uint32_t i = 0; i < ep_cnt; ++i) {
            uint32_t epv;
            if (!read_u32(epv)) return false;
            entry_points_[i] = static_cast<tableint>(epv);
        }

        element_levels_.assign(max_elements_, 0);
        for (size_t i = 0; i < node_count; ++i) {
            uint32_t lv;
            if (!read_u32(lv)) return false;
            element_levels_[i] = static_cast<int>(lv);
        }

        logical_to_physical_.assign(max_elements_, static_cast<tableint>(-1));
        physical_to_logical_.assign(max_elements_, static_cast<tableint>(-1));
        for (size_t i = 0; i < node_count; ++i) {
            uint32_t v;
            if (!read_u32(v)) return false;
            logical_to_physical_[i] = static_cast<tableint>(v);
        }
        for (size_t i = 0; i < node_count; ++i) {
            uint32_t v;
            if (!read_u32(v)) return false;
            physical_to_logical_[i] = static_cast<tableint>(v);
        }

        linkLists_ = (char**)malloc(sizeof(void*) * max_elements_);
        if (!linkLists_) return false;
        for (size_t i = 0; i < max_elements_; ++i) linkLists_[i] = nullptr;

        for (size_t i = 0; i < node_count; ++i) {
            uint32_t bytes;
            if (!read_u32(bytes)) return false;
            if (bytes > 0) {
                linkLists_[i] = (char*)malloc(bytes);
                if (!linkLists_[i]) return false;
                ifs.read(linkLists_[i], bytes);
                if (!ifs) return false;
            }
        }

        link_list_locks_.reset(new std::mutex[max_elements_]);
        cur_element_count.store(node_count, std::memory_order_release);
        maxlevel_.store(static_cast<int>(maxlvl_rd), std::memory_order_release);
        has_enterpoint_.store(node_count > 0, std::memory_order_release);
        if (node_count == 0 || entry_points_.empty()) {
            enterpoint_node_.store(static_cast<tableint>(-1), std::memory_order_release);
        } else {
            enterpoint_node_.store(entry_points_[0], std::memory_order_release);
        }

        std::cerr << "[loadGraph] Graph loaded from: " << path
                  << " (nodes=" << node_count << ", dim=" << dim_
                  << ", max_level=" << maxlvl_rd << ")\n";
        return true;
    }

    static bool isGraphCacheValid(const std::string &path, int expected_dim, size_t expected_count, bool expected_multilayer) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.is_open()) return false;

        auto read_u32 = [&](uint32_t &v) -> bool {
            uint8_t b[4];
            ifs.read(reinterpret_cast<char*>(b), 4);
            if (!ifs) return false;
            v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
            return true;
        };
        auto read_u64 = [&](uint64_t &v) -> bool {
            uint8_t b[8];
            ifs.read(reinterpret_cast<char*>(b), 8);
            if (!ifs) return false;
            v = 0;
            for (int i = 0; i < 8; ++i) v |= ((uint64_t)b[i] << (8 * i));
            return true;
        };

        char magic[8];
        ifs.read(magic, 8);
        bool is_v3 = std::strncmp(magic, "GRAPHC3\0", 8) == 0;
        if (!is_v3) return false;

        uint32_t dim_rd, maxM0_rd, M_rd, maxM_rd, efc_rd, ep_cnt, maxlvl_rd, multilayer_flag;
        uint64_t max_elements_rd, node_count_rd, s_data_per_elem, s_links_l0, s_links_per, off_data, off_l0, data_sz;

        if (!read_u32(dim_rd)) return false;
        if (!read_u64(max_elements_rd)) return false;
        if (!read_u64(node_count_rd)) return false;
        if (!read_u32(maxM0_rd)) return false;
        if (!read_u32(M_rd)) return false;
        if (!read_u32(maxM_rd)) return false;
        if (!read_u32(efc_rd)) return false;
        if (!read_u64(s_data_per_elem)) return false;
        if (!read_u64(s_links_l0)) return false;
        if (!read_u64(s_links_per)) return false;
        if (!read_u64(off_data)) return false;
        if (!read_u64(off_l0)) return false;
        if (!read_u64(data_sz)) return false;
        if (!read_u32(ep_cnt)) return false;
        if (!read_u32(maxlvl_rd)) return false;
        if (!read_u32(multilayer_flag)) return false;

        if (expected_dim > 0 && dim_rd != static_cast<uint32_t>(expected_dim)) return false;
        if (expected_count > 0 && node_count_rd != static_cast<uint64_t>(expected_count)) return false;
        if (static_cast<bool>(multilayer_flag) != expected_multilayer) return false;
        return true;
    }

    double getAverageDistanceCalcsPerSearch() const {
        uint64_t searches = query_count_.load(memory_order_relaxed);
        if (searches == 0) {
            return 0.0;
        }
        uint64_t total = query_distance_calcs_.load(memory_order_relaxed);
        return static_cast<double>(total) / static_cast<double>(searches);
    }
    uint64_t getTotalDistanceCalcs() const {
        return query_distance_calcs_.load(memory_order_relaxed);
    }
    uint64_t getQueryCount() const {
        return query_count_.load(memory_order_relaxed);
    }
    void incrementQueryCount() const {
        query_count_.fetch_add(1, memory_order_relaxed);
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
        query_distance_calcs_.fetch_add(count, memory_order_relaxed);
        query_count_.fetch_add(1, memory_order_relaxed);
    }

    mutable atomic<uint64_t> query_distance_calcs_{0};
    mutable atomic<uint64_t> query_count_{0};

    // 固定 ef 的搜索（标准 HNSW）
    void searchKNN_FixedEF(const float* query, int* res, size_t k, size_t ef) const {
        size_t node_count = cur_element_count.load(memory_order_acquire);
        if (node_count == 0) return;
        ef = max(ef, k);

        tableint currObj = enterpoint_node_.load(memory_order_acquire);
        if (currObj == static_cast<tableint>(-1) || currObj >= node_count) currObj = 0;

        dist_t curdist = L2Distance(query, getDataByInternalId(currObj));
        vector<tableint> neighbors;
        neighbors.reserve(maxM0_);

        int maxLevelSnapshot = maxlevel_.load(memory_order_acquire);
        for (int level = maxLevelSnapshot; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                collectNeighborsLockFree(currObj, level, neighbors);
                for (tableint cand : neighbors) {
                    dist_t d = L2Distance(query, getDataByInternalId(cand));
                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        MaxHeap4 top_candidates = searchLayer(currObj, query, 0, ef);

        // 输出排序后的前 k 个
        vector<distPair> buf;
        buf.reserve(top_candidates.size());
        while (!top_candidates.empty()) {
            buf.push_back(top_candidates.top());
            top_candidates.pop();
        }
        sort(buf.begin(), buf.end(), [](const distPair& a, const distPair& b) { return a.first < b.first; });

        size_t out = min(k, buf.size());
        for (size_t i = 0; i < out; ++i) {
            tableint physical_id = buf[i].second;
            tableint logical_id = (physical_id < physical_to_logical_.size()) ? physical_to_logical_[physical_id] : physical_id;
            res[i] = static_cast<int>(logical_id);
        }
        for (size_t i = out; i < k; ++i) res[i] = -1;
    }

    // 搜索 k 个最近邻，结果直接写入 res 数组
    void searchKNN(const float* query, int* res, float gamma, bool dynamic_gamma) const {
        size_t node_count = cur_element_count.load(memory_order_acquire);
        if (node_count == 0) {
            return;
        }

        const size_t k = 10;
        const float max_gamma = dynamic_gamma ? gamma * 1.1f : gamma;
        const float min_gamma = dynamic_gamma ? gamma * 0.995f : gamma;

        MaxHeap4 top_candidates;
        MaxHeap4 candidate_set;

        thread_local vector<uint64_t> visit_bitmap_tls;
        thread_local vector<uint32_t> visit_epoch_tls;
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
            fill(visit_epoch_tls.begin(), visit_epoch_tls.end(), 0u);
            visit_token_tls = 1u;
        }

        auto touch_word = [&](size_t word_idx) {
            if (visit_epoch_tls[word_idx] != visit_token_tls) {
                visit_epoch_tls[word_idx] = visit_token_tls;
                visit_bitmap_tls[word_idx] = 0ull;
            }
        };
        auto is_visited = [&](tableint id) -> bool {
            size_t word = static_cast<size_t>(id) >> 6;
            size_t bit = static_cast<size_t>(id) & 63ull;
            touch_word(word);
            return (visit_bitmap_tls[word] >> bit) & 1ull;
        };
        auto mark_visited = [&](tableint id) {
            size_t word = static_cast<size_t>(id) >> 6;
            size_t bit = static_cast<size_t>(id) & 63ull;
            touch_word(word);
            visit_bitmap_tls[word] |= (1ull << bit);
        };

        tableint enterpoint_snapshot = enterpoint_node_.load(memory_order_acquire);
        tableint currObj = enterpoint_snapshot;
        if (currObj == static_cast<tableint>(-1) || currObj >= node_count) {
            currObj = 0;
        }

        dist_t curdist = L2Distance(query, getDataByInternalId(currObj));

        vector<tableint> neighbors;
        neighbors.reserve(maxM0_);

        int maxLevelSnapshot = maxlevel_.load(memory_order_acquire);
        for (int level = maxLevelSnapshot; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                collectNeighborsLockFree(currObj, level, neighbors);
                for (tableint cand : neighbors) {
                    dist_t d = L2Distance(query, getDataByInternalId(cand));
                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        mark_visited(currObj);
        candidate_set.emplace(-curdist, currObj);
        top_candidates.emplace(curdist, currObj);

        const size_t congestion_threshold = k * 6;

        while (!candidate_set.empty()) {
            float current_gamma = max_gamma;
            size_t pending_size = candidate_set.size();
            if (dynamic_gamma && pending_size > congestion_threshold) {
                float ratio = (float)congestion_threshold / (float)pending_size;
                current_gamma = max_gamma * ratio;
                if (current_gamma < min_gamma) current_gamma = min_gamma;
            }

            auto curr = candidate_set.top();
            candidate_set.pop();

            dist_t curr_dist = -curr.first;

            if (!top_candidates.empty() && top_candidates.size() >= k) {
                dist_t worst_best = top_candidates.top().first;
                if (curr_dist > (1.0f + current_gamma) * worst_best) {
                    break;
                }
            }

            tableint cid = curr.second;
            linklistsizeint* ll = getLinkListAtLevel(cid, 0);
            int c_sz = getListCount(ll);
            tableint* c_ptr = getNeighborsArray(ll);

            if (c_sz > 0) {
                tableint first = c_ptr[0];
                _mm_prefetch(reinterpret_cast<const char*>(getDataByInternalId(first)), _MM_HINT_T0);
            }
            for (int i = 0; i < c_sz; ++i) {
                tableint e = c_ptr[i];
                if (i + 1 < c_sz) {
                    tableint next = c_ptr[i + 1];
                    _mm_prefetch(reinterpret_cast<const char*>(getDataByInternalId(next)), _MM_HINT_T0);
                }
                if (is_visited(e)) continue;
                mark_visited(e);

                dist_t d = L2Distance(query, getDataByInternalId(e));

                if (top_candidates.size() < k || d < top_candidates.top().first) {
                    top_candidates.emplace(d, e);
                    if (top_candidates.size() > k) {
                        top_candidates.pop();
                    }
                }
                if (top_candidates.size() < k || d < (1.0f + current_gamma) * top_candidates.top().first) {
                    candidate_set.emplace(-d, e);
                }
            }
        }

        int curid = static_cast<int>(k);
        while (!top_candidates.empty() && curid > 0) {
            tableint physical_id = top_candidates.top().second;
            tableint logical_id = (physical_id < physical_to_logical_.size()) ? physical_to_logical_[physical_id] : physical_id;
            res[--curid] = static_cast<int>(logical_id);
            top_candidates.pop();
        }
    }


    // ====================== 节点重排（BFS） =========================
    // 以提升访问局部性
    void reorderNodesByBFS() {
        size_t node_count = cur_element_count.load(memory_order_acquire);
        if (node_count == 0) return;

        vector<vector<tableint>> adj(node_count);
        adj.assign(node_count, {});

        for (tableint u = 0; u < static_cast<tableint>(node_count); ++u) {
            if (0 > element_levels_[u]) continue; // 理论上不会发生
            linklistsizeint* ll = getLinkListAtLevel(u, 0);
            int sz = getListCount(ll);
            const tableint* ng = getNeighborsArray(ll);
            for (int i = 0; i < sz; ++i) {
                tableint v = ng[i];
                if (v >= node_count) continue;
                adj[u].push_back(v);
                adj[v].push_back(u);
            }
        }

        vector<tableint> order;
        order.reserve(node_count);
        vector<char> visited(node_count, 0);

        auto bfs_from = [&](tableint start) {
            if (start >= node_count || visited[start]) return;
            queue<tableint> q;
            visited[start] = 1;
            q.push(start);
            while (!q.empty()) {
                tableint u = q.front();
                q.pop();
                order.push_back(u);
                const auto& nbrs = adj[u];
                for (tableint v : nbrs) {
                    if (v < node_count && !visited[v]) {
                        visited[v] = 1;
                        q.push(v);
                    }
                }
            }
        };

        tableint ep = enterpoint_node_.load(memory_order_acquire);
        if (ep < node_count) bfs_from(ep);
        for (tableint i = 0; i < static_cast<tableint>(node_count); ++i) {
            if (!visited[i]) bfs_from(i);
        }

        if (order.size() != node_count) return;

        vector<tableint> old_to_new(node_count);
        for (tableint new_id = 0; new_id < static_cast<tableint>(node_count); ++new_id) {
            tableint old_id = order[new_id];
            old_to_new[old_id] = new_id;
        }

        size_t bytes_total = max_elements_ * size_data_per_element_;
        char* new_data_level0 = (char*)malloc(bytes_total);
        if (!new_data_level0) return; 

        for (tableint new_id = 0; new_id < static_cast<tableint>(node_count); ++new_id) {
            tableint old_id = order[new_id];
            char* old_base = data_level0_memory_ + old_id * size_data_per_element_;
            char* new_base = new_data_level0 + new_id * size_data_per_element_;

            memcpy(new_base, old_base, size_data_per_element_);

            linklistsizeint* ll = reinterpret_cast<linklistsizeint*>(new_base + offsetLevel0_);
            int sz = getListCount(ll);
            tableint* arr = getNeighborsArray(ll);
            for (int i = 0; i < sz; ++i) {
                tableint old_nbr = arr[i];
                if (old_nbr < node_count) {
                    arr[i] = old_to_new[old_nbr];
                }
            }
        }

        if (node_count < max_elements_) {
            size_t remain = (max_elements_ - node_count) * size_data_per_element_;
            memcpy(new_data_level0 + node_count * size_data_per_element_,
                   data_level0_memory_ + node_count * size_data_per_element_,
                   remain);
        }

        char** new_linkLists = (char**)malloc(sizeof(void*) * max_elements_);
        if (!new_linkLists) {
            free(new_data_level0);
            return;
        }

        vector<int> new_levels(max_elements_);

        for (tableint new_id = 0; new_id < static_cast<tableint>(node_count); ++new_id) {
            tableint old_id = order[new_id];
            new_levels[new_id] = element_levels_[old_id];

            if (element_levels_[old_id] > 0 && linkLists_[old_id]) {
                size_t bytes = size_links_per_element_ * element_levels_[old_id];
                char* new_links = (char*)malloc(bytes);
                if (!new_links) {
                    new_linkLists[new_id] = nullptr;
                    continue;
                }
                memcpy(new_links, linkLists_[old_id], bytes);

                for (int lvl = 1; lvl <= element_levels_[old_id]; ++lvl) {
                    linklistsizeint* ll = get_linklist(old_id, lvl);
                    int sz = getListCount(ll);
                    tableint* old_arr = getNeighborsArray(ll);

                    linklistsizeint* new_ll = reinterpret_cast<linklistsizeint*>(new_links + (lvl - 1) * size_links_per_element_);
                    tableint* new_arr = getNeighborsArray(new_ll);

                    for (int i = 0; i < sz; ++i) {
                        tableint old_nbr = old_arr[i];
                        if (old_nbr < node_count) {
                            new_arr[i] = old_to_new[old_nbr];
                        }
                    }
                }
                new_linkLists[new_id] = new_links;
            } else {
                new_linkLists[new_id] = nullptr;
            }
        }

        // 其余未使用部分直接保持为空
        for (tableint id = static_cast<tableint>(node_count); id < static_cast<tableint>(max_elements_); ++id) {
            new_linkLists[id] = (id < max_elements_) ? linkLists_[id] : nullptr;
            new_levels[id] = element_levels_[id];
        }

        tableint old_ep = enterpoint_node_.load(memory_order_acquire);
        if (old_ep < node_count) {
            enterpoint_node_.store(old_to_new[old_ep], memory_order_release);
        }


        vector<tableint> new_physical_to_logical(max_elements_, static_cast<tableint>(-1));
        for (tableint old_phys = 0; old_phys < static_cast<tableint>(node_count); ++old_phys) {
            tableint new_phys = old_to_new[old_phys];
            tableint logical = physical_to_logical_[old_phys];
            if (logical != static_cast<tableint>(-1)) {
                new_physical_to_logical[new_phys] = logical;
                if (logical < logical_to_physical_.size()) {
                    logical_to_physical_[logical] = new_phys;
                }
            }
        }
        for (tableint id = 0; id < static_cast<tableint>(node_count); ++id) {
            if (element_levels_[id] > 0 && linkLists_[id]) {
                free(linkLists_[id]);
            }
        }

        free(linkLists_);
        free(data_level0_memory_);

        data_level0_memory_ = new_data_level0;
        linkLists_ = new_linkLists;
        element_levels_.swap(new_levels);
        physical_to_logical_.swap(new_physical_to_logical);
    }


};

#endif //CPP_SOLUTION_H
