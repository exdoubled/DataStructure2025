#ifndef CPP_SOLUTION_H
#define CPP_SOLUTION_H

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <queue>
#include <climits>
#include <algorithm>
#include <unordered_set>
#include <atomic>
#include <random>
#include <cstring>

using namespace std;

// HNSW
typedef unsigned int tableint;   // unsigned 和 float 都是 4 字节 
typedef pair<float, tableint> distPair; // 距离-节点ID对类型
typedef unsigned int linklistsizeint; // 邻居数量类型
typedef size_t labeltype;  // 标签类型
typedef float dist_t;      // 距离类型


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

    size_t max_elements_{0}; // 最大节点数量
    size_t cur_element_count{0}; // 当前节点数量
    size_t M_{0};               // 每个节点最大连接数
    size_t maxM_{0};           // 每个节点第 1 层及以上最大连接数
    size_t maxM0_{0};        // 每个节点第 0 层最大连接数
    size_t ef_construction_{0}; // 构建时候选列表大小
    size_t ef_{0};            // 查询时候选列表大小
    int maxlevel_{0};           // 当前最高层数
    double reverse_size_{0.0};   // 论文中的 m_l
    double mult_{0.0};          // 计算随机层数的参数
    int dim_{0};

    size_t data_size_{0}; // 数据向量大小
    size_t size_links_level0_{0}; // 第 0 层每个节点链接大小
    size_t size_data_per_element_{0}; // 每个节点数据大小
    size_t size_links_per_element_{0}; // 每个节点第 1 层及以上链接大小
    size_t offsetData_{0}, offsetLevel0_{0}, offsetLable_{0}; // 偏移量
    tableint enterpoint_node_{0};

    std::default_random_engine level_generator_; // 用于生成随机层数
    // 访问标记池（避免每次搜索 O(N) 清零 visited）
    std::vector<uint32_t> visit_mark_;
    vector<float> qbuf_;
    uint32_t current_visit_token_{0};
    
    void initHNSW(
        size_t max_elements,
        size_t M = 16,
        size_t random_seed = 100,
        size_t ef_construction = 200,
        size_t ef = 200,
        size_t data_size = 0
    ){
        max_elements_ = max_elements;
        M_ = M;
        data_size_ = data_size;
        maxM_ = M_;
        maxM0_ = M_ * 2;
        ef_construction_ = std::max(ef_construction, M_);
        ef_ = ef;
        level_generator_.seed(random_seed);


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
        
        cur_element_count = 0;
        mult_ = 1 / log(1.0 * M_);
        reverse_size_ = 1.0 / mult_;

        // 修复#3：初始化/扩容层数记录
        element_levels_.assign(max_elements_, 0);

        // 修复#4：初始化入口点为“无”，最高层设为 -1
        enterpoint_node_ = static_cast<tableint>(-1);
        maxlevel_ = -1;

        // 初始化访问标记池
        visit_mark_.assign(max_elements_, 0u);
        current_visit_token_ = 1u;

    }

    void clear() {
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
        for (tableint i = 0; i < cur_element_count; i++) {
            if (element_levels_[i] > 0)
                free(linkLists_[i]);
        }
        free(linkLists_);
        linkLists_ = nullptr;
        cur_element_count = 0;
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

    // 获取目前 ListCount
    unsigned short int getListCount(linklistsizeint * ptr) const {
        return *reinterpret_cast<unsigned short*>(ptr);
    }

    // 
    void setListCount(linklistsizeint * ptr, unsigned short int size) const {
        *reinterpret_cast<unsigned short*>(ptr) = size;
    }

    // 统一通过头部后偏移得到邻居数组指针（头部固定 4 字节）
    inline tableint* getNeighborsArray(linklistsizeint* header) const {
        return reinterpret_cast<tableint*>(reinterpret_cast<char*>(header) + sizeof(linklistsizeint));
    }

    // 欧式距离
    dist_t L2Distance(const void * a, const void * b) const {
        dist_t dist = 0.0f;
        for(size_t i = 0; i < dim_; ++i){
            dist_t diff = ((float*)a)[i] - ((float*)b)[i];
            dist += diff * diff;
        }
        return dist;
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

        // 记录访问过的节点（使用访问标记池，避免 O(N) 清零）
        // 由于该方法是 const，需要将标记池声明为 mutable 或通过 const_cast 使用；这里用 const_cast 处理
        auto &mark = const_cast<std::vector<uint32_t>&>(visit_mark_);
        auto &token = const_cast<uint32_t&>(current_visit_token_);
        // 递增 token，必要时回绕并清空
        token += 1u;
        if (token == 0u) {
            std::fill(mark.begin(), mark.end(), 0u);
            token = 1u;
        }
        mark[ep_id] = token;

        while(!candidate_set.empty()){
            auto currPair = candidate_set.top();
            if((-currPair.first) > lowerBound && top_candidates.size() == ef_limit){
                break;
            }
            candidate_set.pop();

            tableint curID = currPair.second;
            linklistsizeint* header = get_linklist_at_level(curID, layer);
            int size = getListCount(header);
            tableint *datal = getNeighborsArray(header);

            for(int i = 0; i < size; i++){
                tableint candidateID = datal[i];

                if(mark[candidateID] == token) continue;
                mark[candidateID] = token;

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
        vector<tableint> selectNeighbors;

        selectNeighborsHeuristic(top_candidates, M_);
        while(top_candidates.size()){
            selectNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        // 若无可选邻居，直接返回自身以避免未定义行为
        if (selectNeighbors.empty()) {
            return cur_c;
        }

        // 新节点写入本层的邻接
        linklistsizeint *ll_new = get_linklist_at_level(cur_c, level);
        tableint *new_data = getNeighborsArray(ll_new);
        size_t write_cnt = std::min(selectNeighbors.size(), Mcurmax);
        for (size_t i = 0; i < write_cnt; ++i) new_data[i] = selectNeighbors[i];
        setListCount(ll_new, static_cast<unsigned short>(write_cnt));

        tableint next_close_ep = selectNeighbors.back();

        for(size_t idx = 0; idx < selectNeighbors.size(); idx++){
        
            linklistsizeint *ll_other;
            ll_other = get_linklist_at_level(selectNeighbors[idx], level);

            size_t sz_link_list_other = getListCount(ll_other);

            if(sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectNeighbors[idx] == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[selectNeighbors[idx]])
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
    void insert(const void * vec, int level, labeltype label) {
        size_t cur_c = cur_element_count;
        cur_element_count++;

        // 随机初始化层数
        int curlevel = generateRandomLevel(mult_);
        if(level > 0) 
            curlevel = level;

        element_levels_[cur_c] = curlevel;
        tableint currObj = enterpoint_node_;
        tableint enterpoint_copy = enterpoint_node_;


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
        // 初始化节点相关数据结构：仅清零第0层链表头与邻接区，避免不必要的大块清零
        // memset(get_linklist0(static_cast<tableint>(cur_c)), 0, size_links_level0_);
        // memcpy(getDataByInternalId(cur_c), vec, data_size_);
        
        // memcpy(getExternalLabelPtr(cur_c), &label, sizeof(labeltype));

        // if(curlevel) {
        //     linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel);
        //     if (linkLists_[cur_c] == nullptr)
        //        throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
        //     memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel);
        // }

        //  空图场景，设置入口点并返回
        if (enterpoint_node_ == static_cast<tableint>(-1)) {
            enterpoint_node_ = static_cast<tableint>(cur_c);
            maxlevel_ = curlevel;
            return;
        }

        if((signed)currObj != -1) {
            if(curlevel < maxlevel_) {
                dist_t curdist = L2Distance(vec, getDataByInternalId(currObj));
                for(int level = maxlevel_; level > curlevel; level--) { // 逐层往下寻找直到 curlevel+1，找到最近的节点
                    bool changed = true;
                    while(changed) {
                        changed = false;
                        linklistsizeint *header = get_linklist(currObj, level);
                        int size = getListCount(header);
                        tableint *datal = getNeighborsArray(header);
                        for(int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if (cand >= cur_element_count)
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
                for (int level = std::min(curlevel, maxlevel_); level >= 0; level--) {
                    priority_queue<distPair, vector<distPair>, CompareDist> top_candidates
                     = searchLayer(currObj, vec, level, ef_construction_);
                     // 这里把 Algorithm 1 后面的部分 合并到连接邻居一起实现了
                    currObj = mutuallyConnectNewElement(vec, cur_c, top_candidates, level, false);
                }
            } else{
                // 新节点层数 >= 当前最高层：仍需在现有层(0..maxlevel_)建立连边
                tableint ep = enterpoint_copy; // 旧入口点
                for (int lvl = maxlevel_; lvl >= 0; --lvl) {
                    auto top_candidates = searchLayer(ep, vec, lvl, ef_construction_);
                    ep = mutuallyConnectNewElement(vec, static_cast<tableint>(cur_c), top_candidates, lvl, false);
                }
                if (curlevel > maxlevel_) {
                    enterpoint_node_ = static_cast<tableint>(cur_c);
                    maxlevel_ = curlevel;
                }
            }
        }
    }


    priority_queue<distPair, vector<distPair>, CompareDist>
    searchKnn(const void *query_data, size_t k ) const {
        priority_queue<distPair, vector<distPair>, CompareDist> result;

        if (cur_element_count == 0 || enterpoint_node_ == static_cast<tableint>(-1))
            return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = L2Distance(query_data, getDataByInternalId(enterpoint_node_));

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                linklistsizeint *header = get_linklist(currObj, level);
                int size = getListCount(header);
                tableint *datal = getNeighborsArray(header);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand >= cur_element_count)
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
