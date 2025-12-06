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

#if SOLUTION_PLATFORM_X86

using L2Kernel = dist_t (*)(const float*, const float*, size_t);

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

inline dist_t compute(const float* a, const float* b, size_t dim) {
    static const L2Kernel kernel = init_kernel();
    return kernel(a, b, dim);
}

#else // !SOLUTION_PLATFORM_X86

inline dist_t compute(const float* a, const float* b, size_t dim) {
    return l2_distance_scalar(a, b, dim);
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
    size_t ef_{0};                          // 查询时候选列表大小

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

    void initHNSW(
        size_t max_elements,
        size_t M = 16,
        size_t random_seed = 100,
        size_t ef_construction = 200,
        size_t ef = 200,
        size_t data_size = 0,
        size_t worker_count = 1
    ){
        // 设置参数
        max_elements_ = max_elements;
        M_ = M;
        maxM_ = M_;
        maxM0_ = M_ * 2;
        data_size_ = data_size;
        ef_construction_ = max(ef_construction, M_);
        ef_ = ef;
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
        
        // 层数记录
        element_levels_.assign(max_elements_, 0);

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
    }


    // 获取节点数据指针
    inline const float *getDataByInternalId(tableint id) const {
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

    // 欧式距离
    dist_t L2Distance(const float * a, const float * b) const {
        /*
        if (DistanceStatContext::active_solution == this) {
            DistanceStatContext::counter++;
        }
        */
        return simd_detail::compute(a, b, dim_);
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

        // 多线程实现，使用 thread_local bitset + token，避免每次清空和大内存
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

        while (!candidate_set.empty()) {
            auto currPair = candidate_set.top();

            // 终止条件：候选集最近的点比结果集最远的还远，且结果集已满
            if ((-currPair.first) > lowerBound && top_candidates.size() == ef_limit) break;
            
            candidate_set.pop();

            tableint curID = currPair.second;
            // 预取邻居数据
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

                    if (top_candidates.size() > ef_limit) {
                        top_candidates.pop();
                    }
                    if (!top_candidates.empty()) {
                        lowerBound = top_candidates.top().first;
                    }
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

        bool is_first = (cur_c == 0);

        // 随机初始化层数
        int curlevel = generateRandomLevel(mult_);
        if(level > 0) curlevel = level;

        element_levels_[cur_c] = curlevel;
        tableint currObj = enterpoint_node_.load(memory_order_acquire);
        tableint enterpoint_copy = currObj;


        char* base_ptr = data_level0_memory_ + cur_c * size_data_per_element_;

        // 写入向量数据
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
            memset(linkLists_[cur_c], 0, bytes);
        }

        if (is_first) {
            enterpoint_node_.store(cur_c, memory_order_release);
            maxlevel_.store(curlevel, memory_order_release);
            has_enterpoint_.store(true, memory_order_release);
            return;
        }

        while (!has_enterpoint_.load(memory_order_acquire)) {
            this_thread::yield();
        }

        currObj = enterpoint_node_.load(memory_order_acquire);
        enterpoint_copy = currObj;
        int maxLevelSnapshot = maxlevel_.load(memory_order_acquire);

        if (currObj != static_cast<tableint>(-1)) {
            if(curlevel < maxLevelSnapshot) {
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
                    if (maxlevel_.compare_exchange_weak(expected, curlevel, memory_order_acq_rel)) {
                        updated = true;
                        break;
                    }
                }
                if (updated) {
                    enterpoint_node_.store(cur_c, memory_order_release);
                }
            }
        }
    }


    // 搜索 k 个最近邻，结果直接写入 res 数组
    void searchKnn(const float *query, size_t k, int* res) const {
        if (cur_element_count.load(memory_order_acquire) == 0 ||
            enterpoint_node_.load(memory_order_acquire) == static_cast<tableint>(-1)) {
            for (size_t i = 0; i < k; ++i) res[i] = -1;
            return;
        }

        tableint enterpoint_snapshot = enterpoint_node_.load(memory_order_acquire);
        tableint currObj = enterpoint_snapshot;
        dist_t curdist = L2Distance(query, getDataByInternalId(enterpoint_snapshot));

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

        // 查询阶段使用 ef_
        MaxHeap4 top_candidates = searchLayer(currObj, query, 0, ef_);

        // 使用 partial_sort 取出最小的 k 个
        size_t result_count = top_candidates.shrink_to_k_smallest(k);
        
        // 现在 data_ 已按距离升序排列，直接顺序写入
        for (size_t i = 0; i < result_count; ++i) {
            res[i] = static_cast<int>(top_candidates[i].second);
        }
        for (size_t i = result_count; i < k; ++i) {
            res[i] = -1;
        }
    }

    void build(int d, const vector<float>& base);

    void search(const vector<float>& query, int *res);

    double getAverageDistanceCalcsPerSearch() const {
        uint64_t searches = query_count_.load(memory_order_relaxed);
        if (searches == 0) {
            return 0.0;
        }
        uint64_t total = query_distance_calcs_.load(memory_order_relaxed);
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
        query_distance_calcs_.fetch_add(count, memory_order_relaxed);
        query_count_.fetch_add(1, memory_order_relaxed);
    }

    mutable atomic<uint64_t> query_distance_calcs_{0};
    mutable atomic<uint64_t> query_count_{0};


};

#endif //CPP_SOLUTION_H
