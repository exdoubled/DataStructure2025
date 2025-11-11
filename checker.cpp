// 编译：& "C:/mingw64/bin/g++.exe" -std=c++17 -O2 -Wall -Wextra -o ".\checker.exe" ".\checker.cpp" ".\MySolution.cpp" ".\Brute.cpp"

/*
主要作用是使用暴力解对拍 MySolution 和 Brute 的结果，确保两者一致
需要先用 main.exe 生成 ans.txt 文件，再用 checker.exe 进行对拍
在终端中运行：
./checker.exe 0
可以进行对拍，0 代表使用 GloVe 数据集，1 代表使用 SIFT 数据集
运行选项说明：
--ans=<path>       指定答案文件路径，默认为 Config 中 "./ans.txt"
--algo=<name>     指定使用的算法，solution 或 brute，默认为 solution
--first=<N>       只对拍前 N 条查询，默认为全部
选项无顺序要求
*/
/*
常用命令示例：
./checker.exe 2 --ans "./anstest.txt" --algo=solution --first=10
./checker.exe 0 --ans="./ans.txt" --algo=solution --first=100
./checker.exe 1 --ans="./ans.txt" --algo=solution --first=100
./checker.exe 0 --ans="./ans.txt" --algo=solution
./checker.exe 1 --ans="./ans.txt" --algo=solution
*/


#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "MySolution.h"
#include "Config.h"
#include "Brute.h"

// 见 Config.h

static std::string filePath;
static size_t pointsnum = 0;
static int dimensions_ = 0;

// 检查文件是否存在
static bool file_exists(const std::string &path) {
    if (path.empty()) return false;
    std::ifstream ifs(path);
    return ifs.good();
}

// 读取数据集信息
static void read_data(int &dataset_case) {
    if (dataset_case == 0) {
        filePath = GLOVEPATH;
        pointsnum = 1183514;
        dimensions_ = 100;
    } else if (dataset_case == 1) {
        filePath = SIFTPATH;
        pointsnum = 1000000;
        dimensions_ = 128;
    } else if (dataset_case == 2) {
        filePath = TESTPATH; // small test dataset
        pointsnum = 10;
        dimensions_ = 10;
    } else {
        std::cerr << "Wrong number of cases! Using default small dataset." << std::endl;
        filePath = "";
        pointsnum = 1000;
        dimensions_ = 16;
    }
}

// 读取数据集文件
static size_t load_flat_vectors_from_txt(const std::string &path, int dim, size_t max_count, std::vector<float> &out_flat) {
    out_flat.clear();
    if (dim <= 0) return 0;
    std::ifstream ifs(path);
    if (!ifs.is_open()) return 0;

    std::string line;
    size_t read = 0;
    while (read < max_count && std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        float val;
        std::vector<float> row;
        while (row.size() < (size_t)dim && ss >> val) row.push_back(val);
        if (row.size() < (size_t)dim) continue; // skip incomplete rows
        out_flat.insert(out_flat.end(), row.begin(), row.end());
        ++read;
    }
    return read;
}

// 读取查询向量文件
static size_t load_queries_from_txt(const std::string &base_path, int dim, size_t max_queries, std::vector<std::vector<float>> &out_queries) {
    out_queries.clear();

    // 1) try parent_dir/query.txt
    if (!base_path.empty()) {
        size_t pos_sep = base_path.find_last_of("\\/");
        if (pos_sep != std::string::npos) {
            char sep = (base_path.find('\\') != std::string::npos) ? '\\' : '/';
            std::string cand = base_path.substr(0, pos_sep) + sep + QUERYFILEPATH;
            if (file_exists(cand)) {
                std::ifstream ifs(cand);
                std::string line;
                while (out_queries.size() < max_queries && std::getline(ifs, line)) {
                    if (line.empty()) continue;
                    std::istringstream ss(line);
                    std::vector<float> row;
                    float v;
                    while (row.size() < (size_t)dim && ss >> v) row.push_back(v);
                    if (row.size() == (size_t)dim) out_queries.push_back(std::move(row));
                }
                return out_queries.size();
            }
        }

        // 2) replace base.txt -> query.txt
        size_t pos = base_path.find("base.txt");
        if (pos != std::string::npos) {
            std::string cand = base_path.substr(0, pos) + QUERYFILEPATH;
            if (file_exists(cand)) {
                std::ifstream ifs(cand);
                std::string line;
                while (out_queries.size() < max_queries && std::getline(ifs, line)) {
                    if (line.empty()) continue;
                    std::istringstream ss(line);
                    std::vector<float> row;
                    float v;
                    while (row.size() < (size_t)dim && ss >> v) row.push_back(v);
                    if (row.size() == (size_t)dim) out_queries.push_back(std::move(row));
                }
                return out_queries.size();
            }
        }
    }

    // 3) current working directory
    {
        std::vector<std::string> cands;
        cands.emplace_back(QUERYFILEPATH);
        cands.emplace_back(std::string(".\\") + QUERYFILEPATH);
        cands.emplace_back(std::string("./") + QUERYFILEPATH);
        for (const auto &cand : cands) {
            if (file_exists(cand)) {
                std::ifstream ifs(cand);
                std::string line;
                while (out_queries.size() < max_queries && std::getline(ifs, line)) {
                    if (line.empty()) continue;
                    std::istringstream ss(line);
                    std::vector<float> row;
                    float v;
                    while (row.size() < (size_t)dim && ss >> v) row.push_back(v);
                    if (row.size() == (size_t)dim) out_queries.push_back(std::move(row));
                }
                return out_queries.size();
            }
        }
    }

    return 0;
}

// 真实答案
struct GroundTruth {
    int k = 0;
    size_t num = 0;
    std::vector<std::vector<int>> indices; // size=num, each length>=k
    int dataset_case_in_ans = -1;          // parsed from header if present
};

// 解析答案文件
static bool parse_ans(const std::string &path, GroundTruth &gt) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return false;
    std::string line;
    bool header_ok = false;
    while (std::getline(ifs, line)) {
        if (line.rfind("Dataset case:", 0) == 0) {
            // format: Dataset case: <int>
            std::istringstream ds(line.substr(13));
            int dc; if (ds >> dc) gt.dataset_case_in_ans = dc;
        }
        if (line.rfind("Results (top-", 0) == 0) {
            size_t p1 = line.find("top-");
            size_t p2 = line.find(" ", p1 + 4);
            if (p1 != std::string::npos) {
                std::string ks = line.substr(p1 + 4, p2 - (p1 + 4));
                gt.k = std::max(0, std::atoi(ks.c_str()));
                header_ok = true;
            }
            break;
        }
        if (ifs.tellg() == std::streampos(-1)) break; // EOF
    }
    if (!header_ok) return false;

    // read lines of indices
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::vector<int> row;
        int x;
        while (ss >> x) row.push_back(x);
        if (!row.empty()) {
            gt.indices.push_back(std::move(row));
        }
    }
    gt.num = gt.indices.size();
    return gt.num > 0 && gt.k > 0;
}

int main(int argc, char** argv) {
    int dataset_case = 0;
    std::string ans_path = ANSFILEPATH;
    size_t firstN = 0; // 0 = evaluate all
    std::string algo = "solution"; // solution | brute
    std::string query_override;     // optional custom query file path

    // Parse args
    auto is_unsigned_integer = [](const std::string &s) -> bool {
        if (s.empty()) return false;
        for (char c : s) if (!std::isdigit((unsigned char)c)) return false;
        return true;
    };

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (is_unsigned_integer(a)) {
            dataset_case = std::atoi(a.c_str());
        } else if (a == "--ans" && i + 1 < argc) {
            ans_path = argv[++i];
        } else if (a.rfind("--ans=", 0) == 0) {
            ans_path = a.substr(6);
        } else if (a == "--first" && i + 1 < argc) {
            firstN = (size_t)std::max(0, std::atoi(argv[++i]));
        } else if (a.rfind("--first=", 0) == 0) {
            firstN = (size_t)std::max(0, std::atoi(a.substr(8).c_str()));
        } else if (a.rfind("--first", 0) == 0 && a.size() > 7) {
            // 支持 --first10 简写
            const std::string num = a.substr(7);
            bool all_digits = !num.empty() && std::all_of(num.begin(), num.end(), [](char c){ return std::isdigit((unsigned char)c); });
            if (all_digits) firstN = (size_t)std::max(0, std::atoi(num.c_str()));
        } else if (a == "--algo" && i + 1 < argc) {
            algo = argv[++i];
        } else if (a.rfind("--algo=", 0) == 0) {
            algo = a.substr(7);
        } else if (a == "--query" && i + 1 < argc) {
            query_override = argv[++i];
        } else if (a.rfind("--query=", 0) == 0) {
            query_override = a.substr(8);
        }
    }

    // Load base data
    GroundTruth gt;
    if (!parse_ans(ans_path, gt)) {
        std::cerr << "Failed to parse ground-truth from: " << ans_path << std::endl;
        return 1;
    }
    std::cerr << "Loaded ground-truth: num=" << gt.num << " k=" << gt.k << " from " << ans_path;
    if (gt.dataset_case_in_ans != -1) std::cerr << " (dataset_case=" << gt.dataset_case_in_ans << ")";
    std::cerr << std::endl;
    if (gt.dataset_case_in_ans != -1 && gt.dataset_case_in_ans != dataset_case) {
        std::cerr << "[ERROR] Dataset case mismatch: ans file has dataset_case=" << gt.dataset_case_in_ans
                  << ", but you specified " << dataset_case << "." << std::endl;
        std::cerr << "        Please run checker with dataset_case=" << gt.dataset_case_in_ans
                  << " or regenerate the ans file for dataset_case=" << dataset_case << "." << std::endl;
        return 1;
    }

    read_data(dataset_case);
    int dim = dimensions_;
    if (dim <= 0) { std::cerr << "Invalid dim" << std::endl; return 1; }

    std::vector<float> base_data;
    size_t loaded = 0;
    const size_t max_load = pointsnum;
    if (!filePath.empty() && file_exists(filePath)) {
        loaded = load_flat_vectors_from_txt(filePath, dim, max_load, base_data);
    }
    if (loaded == 0) {
        size_t N = std::min<size_t>(pointsnum ? pointsnum : 1000, 1000);
        base_data.assign(N * dim, 0.0f);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < N; ++i)
            for (int j = 0; j < dim; ++j) base_data[i * dim + j] = dist(rng);
        std::cerr << "Using synthetic base with N=" << N << " dim=" << dim << std::endl;
    } else {
        std::cerr << "Loaded base: N=" << loaded << " dim=" << dim << std::endl;
    }

    // Load queries
    std::vector<std::vector<float>> queries;
    size_t qloaded = 0;
    if (!query_override.empty()) {
        std::ifstream qf(query_override);
        if (!qf.is_open()) {
            std::cerr << "[ERROR] Cannot open query file: " << query_override << std::endl;
            return 1;
        }
        std::string qline;
        while (queries.size() < gt.num && std::getline(qf, qline)) {
            if (qline.empty()) continue;
            std::istringstream ss(qline);
            std::vector<float> row; float v;
            while (row.size() < (size_t)dim && ss >> v) row.push_back(v);
            if (row.size() == (size_t)dim) queries.push_back(std::move(row));
        }
        qloaded = queries.size();
        std::cerr << "Loaded queries from override file: " << query_override << ", count=" << qloaded << std::endl;
    } else {
        qloaded = load_queries_from_txt(filePath, dim, gt.num, queries);
    }
    if (qloaded < gt.num) {
        std::cerr << "Warning: queries loaded (" << qloaded << ") < ground-truth num (" << gt.num << ")" << std::endl;
    }
    size_t num_to_eval = std::min<size_t>(gt.num, queries.size());
    if (firstN > 0) num_to_eval = std::min(num_to_eval, firstN);
    if (num_to_eval == 0) { std::cerr << "No queries to evaluate" << std::endl; return 1; }

    // Build and search with timing; allow algo selection
    const int pred_k = 10;
    std::vector<std::vector<int>> pred(num_to_eval, std::vector<int>(pred_k, -1));
    double build_ms = 0.0, search_ms = 0.0, avg_ms = 0.0;

    if (algo == "solution") {
        Solution sol;
        auto t_build_start = std::chrono::high_resolution_clock::now();
        sol.build(dim, base_data);
        auto t_build_end = std::chrono::high_resolution_clock::now();

        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_to_eval; ++i) {
            sol.search(queries[i], pred[i].data());
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        build_ms = std::chrono::duration<double>(t_build_end - t_build_start).count() * 1000.0;
        search_ms = std::chrono::duration<double>(t1 - t0).count() * 1000.0;
    } else { // algo == "brute"
        BSolution sol;
        auto t_build_start = std::chrono::high_resolution_clock::now();
        sol.build(dim, base_data);
        auto t_build_end = std::chrono::high_resolution_clock::now();

        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_to_eval; ++i) {
            sol.search(queries[i], pred[i].data());
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        build_ms = std::chrono::duration<double>(t_build_end - t_build_start).count() * 1000.0;
        search_ms = std::chrono::duration<double>(t1 - t0).count() * 1000.0;
    }
    avg_ms = (num_to_eval > 0) ? (search_ms / (double)num_to_eval) : 0.0;

    // Compute metrics
    size_t k_gt = (size_t)gt.k; // 真实答案的K值
    size_t correct_top1 = 0;    // top-1 正确数
    double recall_sum = 0.0;    // recall@k 累计
    size_t total_hits = 0;      // 总命中数
    size_t total_possible = num_to_eval * k_gt; // 总可能命中数

    for (size_t i = 0; i < num_to_eval; ++i) {
        const auto &gt_row = gt.indices[i];
        std::unordered_set<int> gt_set(gt_row.begin(), gt_row.end());

        // top-1
        if (!gt_row.empty() && !pred[i].empty()) {
            if (pred[i][0] == gt_row[0]) ++correct_top1;
        }
        // recall@k (vs gt)
        size_t hits = 0;
        for (int id : pred[i]) {
            if (gt_set.find(id) != gt_set.end()) ++hits;
        }
        recall_sum += (double)hits / (double)k_gt;
        total_hits += hits;
    }

    double top1_acc = (double)correct_top1 / (double)num_to_eval;
    double mean_recall_at_k = recall_sum / (double)num_to_eval;
    double overall_match_rate = (total_possible > 0) ? ((double)total_hits / (double)total_possible) : 0.0;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Compared queries: " << num_to_eval << "\n";
    std::cout << "Ground-truth K: " << gt.k << "\n";
    std::cout << "Top-1 accuracy: " << (top1_acc * 100.0) << "%\n";
    std::cout << "Recall@K (mean): " << (mean_recall_at_k * 100.0) << "%\n";
    std::cout << "Overall match rate: " << (overall_match_rate * 100.0) << "%\n";
    std::cout << "Algo: " << algo << "\n";
    std::cout << "Build latency: " << build_ms << " ms\n";
    std::cout << "Search latency (total): " << search_ms << " ms\n";
    std::cout << "Average search latency: " << avg_ms << " ms/query\n";

    return 0;
}
