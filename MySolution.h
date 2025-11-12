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


typedef unsigned int tableint;   // unsigned 和 float 都是 4 字节 
typedef pair<float, tableint> distPair; // 距离-节点ID对类型
typedef unsigned int linklistsizeint; // 邻居数量类型
typedef size_t labeltype;  // 标签类型
typedef float dist_t;      // 距离类型

// HNSW
class Solution {
public:
    // 图结构
    /*
    储存节点数据及第 0 层的邻居关系
    邻居数量，flag, 凑对齐，邻居节点 id，数据向量，标签
    对每一个 Node: size->flag->reserved->neighbors->data->label
    size:2 bytes, flag:1 byte, reserved:1 byte
    */
    char* data_level0_memory_{nullptr};
    /*
    二维数组，每一行代表一个节点从第 1 层到最高层的邻居关系
    邻居数量，凑对齐，邻居节点 id
    每一层存储格式：size->reserved->neighbors
    size:2 bytes, reserved:2 bytes
    */
    char** linkLists_{nullptr};

    std::vector<int> element_levels_; // 保持每个节点的层数

    size_t max_elements_{0}; // 最大节点数量
    size_t cur_element_count{0}; // 当前节点数量
    size_t M_{0};               // 每个节点最大连接数
    size_t maxM_{0};           // 每个节点第 0 层最大连接数
    size_t maxM0_{0};        // 每个节点第 1 层及以上最大连接数
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
    
    void initHNSW(
        size_t max_elements,
        size_t M = 16,
        size_t random_seed = 100,
        size_t ef_construction = 200,
        size_t ef = 10,
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
        
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        
        cur_element_count = 0;
        mult_ = 1 / log(1.0 * M_);
        reverse_size_ = 1.0 / mult_;

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
        return *((unsigned short int *)ptr);
    }

    // 
    void setListCount(linklistsizeint * ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
    }

    // 欧式距离
    dist_t L2Distance(const void * a, const void * b) const {
        dist_t dist = 0.0f;
        for(int i = 0; i < dim_; ++i){
            dist_t diff = ((tableint*)a)[i] - ((tableint*)b)[i];
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
        double level = -log(distribution(level_generator_)) * reverse_size;
        return (int)level;
    }

    // 在指定层搜索，返回 efConstruction_ 个近邻节点
    // 实现论文的 Algorithm 2，这里要求 query 是数据指针
    priority_queue<distPair, vector<distPair>, CompareDist> 
    searchLayer(tableint ep_id, const void * query, int layer) const {
        priority_queue<distPair, vector<distPair>, CompareDist> top_candidates; // W
        priority_queue<distPair, vector<distPair>, CompareDist> candidate_set; // C

        dist_t lowerBound;
        dist_t dist = L2Distance(query, getDataByInternalId(ep_id));
        top_candidates.emplace(dist, ep_id);
        // 这里用负距离是为了让优先队列变成小顶堆
        candidate_set.emplace(-dist, ep_id);

        lowerBound = dist;

        // 记录访问过的节点
        std::vector<bool> visited(max_elements_, false); // v
        visited[ep_id] = true;

        while(!candidate_set.empty()){
            auto currPair = candidate_set.top();
            if((-currPair.first) > lowerBound && top_candidates.size() == ef_construction_){
                break;
            }
            candidate_set.pop();

            tableint curID = currPair.second;
            int *data;  // = (int *)(linkList0_ + curID * size_links_per_element0

            data = (int*)get_linklist_at_level(curID, layer);

            int size = getListCount((linklistsizeint *)data);
            tableint *datal = (tableint *)(data + 1);

            for(int i = 0; i < size; i++){
                tableint candidateID = datal[i];

                if(visited[candidateID]) continue;
                visited[candidateID] = true;

                dist_t d = L2Distance(query, getDataByInternalId(candidateID));

                if(top_candidates.size() < ef_construction_ || d < lowerBound){
                    candidate_set.emplace(-d, candidateID);

                    top_candidates.emplace(d, candidateID);

                    if(top_candidates.size() > ef_construction_){
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
    // 注意这里和论文的 algorithm 4 实现不一样
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
        }

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
                tableint *data = (tableint *) (ll_other + 1);

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
                    // Nearest K:
                    /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
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


        // 初始化节点相关数据结构
        memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);
        memcpy(getDataByInternalId(cur_c), vec, data_size_);
        
        memcpy(getExternalLabelPtr(cur_c), &label, sizeof(labeltype));

        if(curlevel) {
            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        }

        if((signed)currObj != -1) {
            if(curlevel < maxlevel_) {
                dist_t curdist = L2Distance(vec, getDataByInternalId(currObj));
                for(int level = maxlevel_; level > curlevel; level--) { // 逐层往下寻找直到 curlevel+1，找到最近的节点
                    bool changed = true;
                    while(changed) {
                        changed = false;
                        unsigned int *data;
                        data = (unsigned int *) get_linklist(currObj, level);
                        int size = getListCount(data);

                        tableint *datal = (tableint *)(data + 1);
                        for(int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if(cand < 0 || cand > max_elements_)
                                throw std::runtime_error("cand error");
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
                     = searchLayer(currObj, vec, level);
                     // 这里把 Algorithm 1 后面的部分 合并到连接邻居一起实现了
                    currObj = mutuallyConnectNewElement(vec, cur_c, top_candidates, level, false);
                }
            } else{
                enterpoint_node_ = 0;
                maxlevel_ = curlevel;
            }
        }
    }


    priority_queue<distPair, vector<distPair>, CompareDist>
    searchKnn(const void *query_data, size_t k ) const {
        priority_queue<distPair, vector<distPair>, CompareDist> result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = L2Distance(query_data, getDataByInternalId(enterpoint_node_));

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
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
        top_candidates = searchLayer(currObj, query_data, 0);

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        return top_candidates;
    }

    void build(int d, const vector<float>& base);

    void search(const vector<float>& query, int *res);


};
#endif //CPP_SOLUTION_H
