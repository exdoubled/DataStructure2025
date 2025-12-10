// 编译：& 'C:/TDM-GCC-64/bin/g++.exe' -std=c++17 -O2 -Wall -Wextra -pthread -o './checker.exe' './checker.cpp' './MySolution.cpp' './Brute.cpp'

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
#include <mutex>
#include <thread>
#include <cstdio>

#include "MySolution.h"
#include "Config.h"
#include "Brute.h"
#include "BinaryIO.h"

// 见 Config.h
// ---- Runtime knobs (easy to tweak) ----
// 是否打印进度（1 打印 / 0 不打印）
#define PRINT_PROGRESS 1
// 进度打印的步长（每处理多少条查询打印一次）
#define PROGRESS_EVERY 200
// 只处理前 N 条查询（0 表示不限制，处理全部）
#define RUN_FIRST_N 0

// ---- Unified progress bar macros ----
#define ENABLE_PROGRESS 1
#define PROGRESS_BAR_WIDTH 30
#define PROGRESS_STEP_LOAD 200000
#define PROGRESS_STEP_SEARCH 200
#define PROGRESS_STEP_WRITE 1000
#define PROGRESS_STEP_EVAL 500
static void progress_bar(const char* stage, size_t current, size_t total,
                         std::chrono::high_resolution_clock::time_point start_tp);

// 前置声明：进度条函数
static void print_progress_bar(const char* stage, size_t current, size_t total,
                               std::chrono::high_resolution_clock::time_point start_tp);

static std::string filePath;
static size_t pointsnum = 0;
static int dimensions_ = 0;

// 检查文件是否存在
static bool file_exists(const std::string &path) {
    if (path.empty()) return false;
    std::ifstream ifs(path);
    return ifs.good();
}

// 前向声明：供自动加载函数调用的文本读取实现
static size_t load_flat_vectors_from_txt(const std::string &path, int dim, size_t max_count, std::vector<float> &out_flat);
static size_t load_queries_from_txt(const std::string &base_path, int dim, size_t max_queries, std::vector<std::vector<float>> &out_queries, std::string* used_txt_path=nullptr);

// util: derive sibling path in same directory as txt_path
static std::string sibling_bin_path(const std::string &txt_path, const std::string &bin_name) {
    if (!txt_path.empty()) {
        size_t pos_sep = txt_path.find_last_of("\\/");
        if (pos_sep != std::string::npos) {
            char sep = (txt_path.find('\\') != std::string::npos) ? '\\' : '/';
            (void)sep; // unused, but kept for symmetry
            return txt_path.substr(0, pos_sep + 1) + bin_name;
        }
    }
    return bin_name;
}

static std::string replace_base_txt_with_bin(const std::string &base_txt, const std::string &bin_name) {
    size_t pos = base_txt.find("base.txt");
    if (pos != std::string::npos) return base_txt.substr(0, pos) + bin_name;
    return sibling_bin_path(base_txt, bin_name);
}

// 从 txt 路径派生同目录 .bin 路径：.txt -> .bin，否则追加 .bin
static std::string derive_bin_path_from_txt(const std::string &txt_path) {
    size_t pos_sep = txt_path.find_last_of("\\/");
    std::string dir = (pos_sep != std::string::npos) ? txt_path.substr(0, pos_sep + 1) : std::string();
    std::string file = (pos_sep != std::string::npos) ? txt_path.substr(pos_sep + 1) : txt_path;
    size_t dot = file.find_last_of('.');
    if (dot != std::string::npos && dot > 0) {
        std::string ext = file.substr(dot);
        if (ext == ".txt" || ext == ".TXT") {
            return dir + file.substr(0, dot) + ".bin";
        }
    }
    return dir + file + ".bin";
}

// auto-load base: try .bin first then txt
static size_t load_base_vectors_auto(const std::string &base_txt_path, int expected_dim, size_t max_count,
                                     std::vector<float> &out_flat, int &out_dim_actual) {
    out_dim_actual = expected_dim;
    std::vector<std::string> cands;
    cands.emplace_back(replace_base_txt_with_bin(base_txt_path, "base.bin"));
    cands.emplace_back("base.bin");
    for (const auto &cand : cands) {
        if (file_exists(cand)) {
            int dim_rd = 0; size_t cnt = 0; std::vector<float> flat;
            if (binio::read_vecbin(cand, dim_rd, cnt, flat)) {
                if (expected_dim > 0 && dim_rd != expected_dim) {
                    std::cerr << "[WARN] Dimension mismatch in bin (" << cand << "): " << dim_rd
                              << " != expected " << expected_dim << ". Ignored.\n";
                } else {
                    if (max_count > 0 && cnt > max_count) cnt = max_count; // clipping not applied to data here
                    out_flat.swap(flat);
                    out_dim_actual = dim_rd;
                    std::cerr << "Loaded base from bin: " << cand << ", N=" << cnt << ", dim=" << dim_rd << "\n";
                    return cnt;
                }
            }
        }
    }
    size_t read = load_flat_vectors_from_txt(base_txt_path, expected_dim, max_count, out_flat);
    out_dim_actual = expected_dim;
    if (read > 0 && !base_txt_path.empty()) {
        std::string bin_path = replace_base_txt_with_bin(base_txt_path, "base.bin");
        if (binio::write_vecbin(bin_path, out_dim_actual, out_flat)) {
            std::cerr << "Saved base to bin: " << bin_path << " (N=" << read << ")\n";
        } else {
            std::cerr << "[WARN] Failed to save base bin: " << bin_path << "\n";
        }
    }
    return read;
}

// auto-load queries .bin -> vector<vector<float>>, fallback to txt
static size_t load_queries_auto(const std::string &base_txt_path, int expected_dim, size_t max_queries,
                                std::vector<std::vector<float>> &out_queries,
                                const std::string &override_path = std::string()) {
    if (!override_path.empty()) {
        // caller handles override text path loading separately
        return 0;
    }
    // 先查找与 QUERYFILEPATH 对应的 .bin（如 query0.bin）
    std::vector<std::string> cands_qbin;
    const std::string qbin_name = derive_bin_path_from_txt(QUERYFILEPATH);
    if (!base_txt_path.empty()) {
        size_t pos_sep = base_txt_path.find_last_of("\\/");
        if (pos_sep != std::string::npos) {
            const std::string dir = base_txt_path.substr(0, pos_sep + 1);
            cands_qbin.emplace_back(dir + qbin_name);
        }
        size_t pos = base_txt_path.find("base.txt");
        if (pos != std::string::npos) {
            const std::string dir2 = base_txt_path.substr(0, pos);
            cands_qbin.emplace_back(dir2 + qbin_name);
        }
    }
    cands_qbin.emplace_back(qbin_name);
    for (const auto &cand : cands_qbin) {
        if (file_exists(cand)) {
            std::vector<std::vector<float>> qs;
            if (binio::read_queries_vecbin(cand, expected_dim, max_queries, qs)) {
                out_queries.swap(qs);
                std::cerr << "Loaded queries from bin: " << cand << ", count=" << out_queries.size() << ", dim=" << expected_dim << "\n";
                return out_queries.size();
            }
        }
    }
    // 若没有匹配到 query0.bin，则优先从 QUERYFILEPATH 文本读取并写出同名 .bin
    std::string used_txt;
    size_t read = load_queries_from_txt(base_txt_path, expected_dim, max_queries, out_queries, &used_txt);
    if (read > 0 && !used_txt.empty()) {
        std::string qbin = derive_bin_path_from_txt(used_txt);
        binio::write_queries_vecbin(qbin, expected_dim, out_queries);
        return read;
    }
    // 最后才兼容性回退到 "query.bin"
    std::vector<std::string> cands_generic;
    if (!base_txt_path.empty()) {
        size_t pos_sep = base_txt_path.find_last_of("\\/");
        if (pos_sep != std::string::npos) {
            const std::string dir = base_txt_path.substr(0, pos_sep + 1);
            cands_generic.emplace_back(dir + std::string("query.bin"));
        }
        size_t pos = base_txt_path.find("base.txt");
        if (pos != std::string::npos) {
            const std::string dir2 = base_txt_path.substr(0, pos);
            cands_generic.emplace_back(dir2 + std::string("query.bin"));
        }
    }
    cands_generic.emplace_back("query.bin");
    for (const auto &cand : cands_generic) {
        if (file_exists(cand)) {
            std::vector<std::vector<float>> qs;
            if (binio::read_queries_vecbin(cand, expected_dim, max_queries, qs)) {
                out_queries.swap(qs);
                std::cerr << "Loaded queries from bin: " << cand << ", count=" << out_queries.size() << ", dim=" << expected_dim << "\n";
                return out_queries.size();
            }
        }
    }
    return 0;
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
    auto start_tp = std::chrono::high_resolution_clock::now();
    while (read < max_count && std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        float val;
        std::vector<float> row;
        while (row.size() < (size_t)dim && ss >> val) row.push_back(val);
        if (row.size() < (size_t)dim) continue; // skip incomplete rows
        out_flat.insert(out_flat.end(), row.begin(), row.end());
        ++read;
        if (ENABLE_PROGRESS && (read % PROGRESS_STEP_LOAD == 0 || read == max_count)) {
            progress_bar("Loading base", read, max_count, start_tp);
        }
    }
    if (ENABLE_PROGRESS) progress_bar("Loading base", read, read, start_tp);
    return read;
}

// 读取查询向量文件
static size_t load_queries_from_txt(const std::string &base_path, int dim, size_t max_queries, std::vector<std::vector<float>> &out_queries, std::string* used_txt_path) {
    if (used_txt_path) *used_txt_path = "";
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
                auto start_tp = std::chrono::high_resolution_clock::now();
                while (out_queries.size() < max_queries && std::getline(ifs, line)) {
                    if (line.empty()) continue;
                    std::istringstream ss(line);
                    std::vector<float> row;
                    float v;
                    while (row.size() < (size_t)dim && ss >> v) row.push_back(v);
                    if (row.size() == (size_t)dim) out_queries.push_back(std::move(row));
                    if (ENABLE_PROGRESS) {
                        size_t cur = out_queries.size();
                        if (cur % PROGRESS_STEP_LOAD == 0 || cur == max_queries)
                            progress_bar("Loading queries", cur, max_queries, start_tp);
                    }
                }
                if (ENABLE_PROGRESS) progress_bar("Loading queries", out_queries.size(), out_queries.size(), start_tp);
                if (used_txt_path) *used_txt_path = cand;
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
                auto start_tp = std::chrono::high_resolution_clock::now();
                while (out_queries.size() < max_queries && std::getline(ifs, line)) {
                    if (line.empty()) continue;
                    std::istringstream ss(line);
                    std::vector<float> row;
                    float v;
                    while (row.size() < (size_t)dim && ss >> v) row.push_back(v);
                    if (row.size() == (size_t)dim) out_queries.push_back(std::move(row));
                }
                if (ENABLE_PROGRESS) progress_bar("Loading queries", out_queries.size(), out_queries.size(), start_tp);
                if (used_txt_path) *used_txt_path = cand;
                return out_queries.size();
            }
        }
    }

    // 3) current working directory
    {
        std::vector<std::string> cands;
        // default
        cands.emplace_back(QUERYFILEPATH);
        cands.emplace_back(std::string(".\\") + QUERYFILEPATH);
        cands.emplace_back(std::string("./") + QUERYFILEPATH);
        // dataset-case friendly fallbacks in cwd
        cands.emplace_back("query0.txt");
        cands.emplace_back(".\\query0.txt");
        cands.emplace_back("./query0.txt");
        cands.emplace_back("query1.txt");
        cands.emplace_back(".\\query1.txt");
        cands.emplace_back("./query1.txt");
        for (const auto &cand : cands) {
            if (file_exists(cand)) {
                std::ifstream ifs(cand);
                std::string line;
                auto start_tp = std::chrono::high_resolution_clock::now();
                while (out_queries.size() < max_queries && std::getline(ifs, line)) {
                    if (line.empty()) continue;
                    std::istringstream ss(line);
                    std::vector<float> row;
                    float v;
                    while (row.size() < (size_t)dim && ss >> v) row.push_back(v);
                    if (row.size() == (size_t)dim) out_queries.push_back(std::move(row));
                }
                if (ENABLE_PROGRESS) progress_bar("Loading queries", out_queries.size(), out_queries.size(), start_tp);
                if (used_txt_path) *used_txt_path = cand;
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
    if (ENABLE_PROGRESS) {
        std::cerr << "[Progress] Parsed ground-truth queries: " << gt.num << "\n";
    }
    return gt.num > 0 && gt.k > 0;
}

static void progress_bar(const char* stage, size_t current, size_t total,
                         std::chrono::high_resolution_clock::time_point start_tp) {
#if ENABLE_PROGRESS
    if (total == 0) return;
    double ratio = (double)current / (double)total;
    int filled = (int)std::round(ratio * PROGRESS_BAR_WIDTH);
    double elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_tp).count();
    double per = elapsed / std::max<size_t>(1, current);
    double eta = per * (current < total ? (total - current) : 0);
    std::ostringstream bar;
    bar << '['; for (int i = 0; i < PROGRESS_BAR_WIDTH; ++i) bar << (i < filled ? '#' : '.'); bar << ']';
    std::cout << "\r" << stage << ' ' << bar.str()
              << ' ' << current << '/' << total
              << " (" << std::fixed << std::setprecision(1) << ratio * 100.0 << "%)"
              << " elapsed:" << std::setprecision(2) << elapsed << "s"
              << " ETA:" << std::setprecision(2) << eta << "s" << std::flush;
    if (current == total) std::cout << "\n";
#endif
}

int main(int argc, char** argv) {
    int dataset_case = 0;
    std::string ans_path = ANSFILEPATH;
    size_t firstN = 0; // 0 = evaluate all
    std::string algo = "solution"; // solution | brute
    std::string query_override;     // optional custom query file path
    bool save_bin_base = false;
    
    bool force_bin = false;         // 仅从 .bin 读取

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
        } else if (a == "--save-bin-base") {
            save_bin_base = true;
        } else if (a == "--bin") {
            force_bin = true;
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
        if (force_bin) {
            std::vector<std::string> cands;
            cands.emplace_back(replace_base_txt_with_bin(filePath, "base.bin"));
            cands.emplace_back("base.bin");
            bool ok = false;
            for (const auto &cand : cands) {
                if (file_exists(cand)) {
                    int dim_rd = 0; size_t cnt = 0; std::vector<float> flat;
                    if (binio::read_vecbin(cand, dim_rd, cnt, flat)) {
                        base_data.swap(flat);
                        dim = dim_rd; loaded = cnt; ok = true;
                        std::cerr << "Loaded base (bin forced): " << cand << ", N=" << loaded << ", dim=" << dim << "\n";
                        break;
                    }
                }
            }
            if (!ok) {
                std::cerr << "[ERROR] --bin specified, but base bin not found or invalid." << std::endl;
                return 1;
            }
        } else {
            int actual_dim = dim;
            loaded = load_base_vectors_auto(filePath, dim, max_load, base_data, actual_dim);
            dim = actual_dim;
        }
    }
    if (loaded == 0) {
        size_t N = std::min<size_t>(pointsnum ? pointsnum : 1000, 1000);
        base_data.assign(N * dim, 0.0f);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < N; ++i)
            for (int j = 0; j < dim; ++j) base_data[i * dim + j] = dist(rng);
        loaded = N; // ensure subsequent logic (e.g., HNSW max_elements & addPoint loop) uses correct count
        std::cerr << "Using synthetic base with N=" << N << " dim=" << dim << std::endl;
    } else {
        std::cerr << "Loaded base: N=" << loaded << " dim=" << dim << std::endl;
        if (save_bin_base) {
            std::string bbin = replace_base_txt_with_bin(filePath, "base.bin");
            if (binio::write_vecbin(bbin, dim, base_data)) {
                std::cerr << "Saved base to bin: " << bbin << " (N=" << (base_data.size()/(size_t)dim) << ")\n";
            }
        }
    }

    // Load queries
    std::vector<std::vector<float>> queries;
    size_t qloaded = 0;
    if (!query_override.empty()) {
        if (force_bin) {
            // 强制从 override 路径按 .bin 读取
            if (!binio::read_queries_vecbin(query_override, dim, gt.num, queries)) {
                std::cerr << "[ERROR] --bin specified, but override query bin invalid: " << query_override << std::endl;
                return 1;
            }
            qloaded = queries.size();
            std::cerr << "Loaded queries (bin forced override): " << query_override << ", count=" << qloaded << std::endl;
        } else {
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
            if (qloaded > 0) {
                std::string qbin = derive_bin_path_from_txt(query_override);
                binio::write_queries_vecbin(qbin, dim, queries);
            }
        }
    } else {
        if (force_bin) {
            std::vector<std::string> qcands;
            const std::string qbin_name = derive_bin_path_from_txt(QUERYFILEPATH);
            if (!filePath.empty()) {
                size_t pos_sep = filePath.find_last_of("\\/");
                if (pos_sep != std::string::npos) {
                    const std::string dir = filePath.substr(0, pos_sep + 1);
                    qcands.emplace_back(dir + qbin_name);
                    qcands.emplace_back(dir + std::string("query.bin"));
                }
                size_t pos = filePath.find("base.txt");
                if (pos != std::string::npos) {
                    const std::string dir2 = filePath.substr(0, pos);
                    qcands.emplace_back(dir2 + qbin_name);
                    qcands.emplace_back(dir2 + std::string("query.bin"));
                }
            }
            qcands.emplace_back(qbin_name);
            qcands.emplace_back("query.bin");
            bool ok = false;
            for (const auto &cand : qcands) {
                if (file_exists(cand)) {
                    std::vector<std::vector<float>> qs;
                    if (binio::read_queries_vecbin(cand, dim, gt.num, qs)) {
                        queries.swap(qs);
                        qloaded = queries.size(); ok = true;
                        std::cerr << "Loaded queries (bin forced): " << cand << ", count=" << qloaded << std::endl;
                        break;
                    }
                }
            }
            if (!ok) {
                std::cerr << "[ERROR] --bin specified, but query bin not found or invalid." << std::endl;
                return 1;
            }
        } else {
            qloaded = load_queries_auto(filePath, dim, gt.num, queries);
        }
    }
    if (qloaded < gt.num) {
        std::cerr << "Warning: queries loaded (" << qloaded << ") < ground-truth num (" << gt.num << ")" << std::endl;
    }
    // 不再在此处额外生成 query.bin；若从文本读取会在读取阶段自动生成一次
    size_t num_to_eval = std::min<size_t>(gt.num, queries.size());
    if (firstN > 0) num_to_eval = std::min(num_to_eval, firstN);
    if (num_to_eval == 0) { std::cerr << "No queries to evaluate" << std::endl; return 1; }

    // Build and search with timing; allow algo selection
    const int pred_k = 10;
    std::vector<std::vector<int>> pred(num_to_eval, std::vector<int>(pred_k, -1));
    double build_ms = 0.0, search_ms = 0.0, avg_ms = 0.0;
    double avg_distance_calcs = 0.0;
    bool has_distance_stats = false;

    auto search_start_tp = std::chrono::high_resolution_clock::now();
    if (algo == "solution") {
        Solution sol;
        auto t_build_start = std::chrono::high_resolution_clock::now();
        sol.build(dim, base_data);
    //#ifdef CPP_SOLUTION_FAKE_STD_SYNC
    //    std::fprintf(stderr, "[checker] threading disabled (fallback)\n");
    //#else
    //    std::fprintf(stderr, "[checker] threading enabled, worker_count=%zu (hc=%u)\n",
    //             sol.worker_count_, std::thread::hardware_concurrency());
    //#endif
        if(ENABLE_PROGRESS) {
            progress_bar("Building index", 1, 1, t_build_start);
        }
        auto t_build_end = std::chrono::high_resolution_clock::now();

        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_to_eval; ++i) {
            sol.search(queries[i], pred[i].data());
            if (ENABLE_PROGRESS && ((i + 1) % PROGRESS_STEP_SEARCH == 0 || i + 1 == num_to_eval))
                progress_bar("Searching", i + 1, num_to_eval, t0);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        build_ms = std::chrono::duration<double>(t_build_end - t_build_start).count() * 1000.0;
        search_ms = std::chrono::duration<double>(t1 - t0).count() * 1000.0;
        avg_distance_calcs = sol.getAverageDistanceCalcsPerSearch();
        has_distance_stats = true;
    } else { // brute
        BSolution sol;
        auto t_build_start = std::chrono::high_resolution_clock::now();
        sol.build(dim, base_data);
        auto t_build_end = std::chrono::high_resolution_clock::now();

        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_to_eval; ++i) {
            sol.search(queries[i], pred[i].data());
            if (ENABLE_PROGRESS && ((i + 1) % PROGRESS_STEP_SEARCH == 0 || i + 1 == num_to_eval))
                progress_bar("Searching", i + 1, num_to_eval, t0);
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

    auto eval_start_tp = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_to_eval; ++i) {
        const auto &gt_row = gt.indices[i];
        // Only use the top-k_gt ground truth items for recall calculation
        // If the ground truth file contains more items (e.g. 100) but header says top-10,
        // we should only consider the first 10 for strict Recall@10.
        size_t limit = std::min(gt_row.size(), k_gt);
        std::unordered_set<int> gt_set(gt_row.begin(), gt_row.begin() + limit);

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
        if (ENABLE_PROGRESS && ((i + 1) % PROGRESS_STEP_EVAL == 0 || i + 1 == num_to_eval))
            progress_bar("Evaluating", i + 1, num_to_eval, eval_start_tp);
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
    if (has_distance_stats) {
        std::cout << "Average distance computations: " << avg_distance_calcs << " per query\n";
    }

    return 0;
}