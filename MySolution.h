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

using namespace std;
using T = vector<float>;

// HNSW
class Solution {
public:
    // 图节点
    struct Node {
        vector<int> neighbors;
    };

    // 搜索节点（用于优先队列）
    struct SearchNode {
        int id;
        float distance;
        bool operator<(const SearchNode& other) const {
            return distance > other.distance;
        }
    };

    // 向量欧式距离
    float L2Distance(const T& a, const T& b);


    // 计算数据点 id 与查询向量 query 的距离
    float computeDistance(int id, const T& query);

    // 产生几何分布的层数，范围 [0, maxLayer-1]
    int generateRandomLevel();

    // q 为查询向量， entryPoint 为入口点， pq 为结果优先队列
    void searchLayer(int layer, const vector<float>& query, int& entryPoint, priority_queue<SearchNode>& pq);

    //  把数据插入到多层图中
    void insert(const T& vec);

    void build(int d, const vector<float>& base);

    void search(const vector<float>& query, int *res);

private:
    int dim;                   // 向量维度
    int maxLayer = 5;          // 最大层数
    vector<vector<Node>> hnsw; // 多层图
    vector<T> dataPoints;      // 存储所有数据点
    int efConstruction = 100;  // 构建时候选列表大小
    int M = 48;                // 每个节点最大连接数
    int efSearch = 500;         // 查询时候选列表大小
};
#endif //CPP_SOLUTION_H
