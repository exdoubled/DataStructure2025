#include "MySolution.h"


float Solution::L2Distance(const T& a, const T& b) {
    float dist = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}


float Solution::computeDistance(int id, const T& query) {
    return L2Distance(dataPoints[id], query);
}

int Solution::generateRandomLevel() {
    int level = 0;
    while (level + 1 < maxLayer && (static_cast<float>(rand()) / RAND_MAX) < 0.5f) {
        ++level;
    }
    return level;
}

void Solution::searchLayer(int layer, const vector<float>& query, int& entryPoint, priority_queue<SearchNode>& pq) {
    unordered_set<int> visited;
    priority_queue<SearchNode> candidate;

    if (hnsw.empty() || hnsw[layer].empty()) return;
    // 如果 entryPoint 越界，重置为 0
    if (entryPoint < 0 || static_cast<size_t>(entryPoint) >= hnsw[layer].size()) entryPoint = 0;

    visited.insert(entryPoint);
    float entryDist = computeDistance(entryPoint, query);
    candidate.push({entryPoint, entryDist});

    int iter = 0;
    while (!candidate.empty() && iter < efSearch) {
        SearchNode node = candidate.top();
        candidate.pop();

        pq.push(node);
        if (node.id < static_cast<int>(hnsw[layer].size())) {
            for (int neighbor : hnsw[layer][node.id].neighbors) {
                if (visited.insert(neighbor).second) {
                    float dist = computeDistance(neighbor, query);
                    candidate.push({neighbor, dist});
                }
            }
        }
        ++iter;
    }
}



void Solution::insert(const T& vec) {
    // 给每一个新数据点分配一个 id
    int id = dataPoints.size();
    dataPoints.push_back(vec);

    // 确保层结构已分配
    if (hnsw.size() < static_cast<size_t>(maxLayer)) hnsw.resize(maxLayer);

    int curLevel = generateRandomLevel();

    // 确保每一层都有对应的节点槽，保持索引一致
    for (int level = 0; level <= curLevel; ++level) {
        while (hnsw[level].size() <= static_cast<size_t>(id)) {
            hnsw[level].push_back(Node());
        }
    }

    // 如果这是第一个点，直接返回（无邻居）
    if (id == 0) return;

    // 在每一层找到近邻并建立双向连接
    for (int level = curLevel; level >= 0; --level) {
        priority_queue<SearchNode> pq; // 每一层的候选邻居
        int entryPoint = 0; // 使用 0 作为接入点
    
        searchLayer(level, vec, entryPoint, pq);

        // 选择最接近的 M 个邻居
        vector<int> neighbors;
        while (!pq.empty() && static_cast<int>(neighbors.size()) < M) {
            neighbors.push_back(pq.top().id);
            pq.pop();
        }

        // 连接图
        for (int neighbor : neighbors) {
            auto &v1 = hnsw[level][id].neighbors;
            auto &v2 = hnsw[level][neighbor].neighbors;
            if (find(v1.begin(), v1.end(), neighbor) == v1.end()) v1.push_back(neighbor);
            if (find(v2.begin(), v2.end(), id) == v2.end()) v2.push_back(id);
        }
    }
}

void Solution::build(int d, const vector<float>& base) {
    dim = d;
    dataPoints.clear();
    hnsw.clear();
    hnsw.resize(maxLayer);

    for (size_t i = 0; i < base.size(); i += dim) {
        vector<float> vec(base.begin() + i, base.begin() + i + dim);
        insert(vec);  
    }

    
}

void Solution::search(const vector<float>& query, int* res) {
    priority_queue<SearchNode> pq;

    int entryPoint = 0;
    for (int layer = maxLayer - 1; layer >= 1; --layer) {
        priority_queue<SearchNode> tmp;
        searchLayer(layer, query, entryPoint, tmp);
        if (!tmp.empty()) entryPoint = tmp.top().id;
    }

    searchLayer(0, query, entryPoint, pq);

    int k = 10;
    int idx = 0;
    while (!pq.empty() && idx < k) {
        res[idx++] = pq.top().id;
        pq.pop();
    }
}
