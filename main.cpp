// 编译：& "C:/mingw64/bin/g++.exe" -std=c++17 -O2 -Wall -Wextra -o ".\main.exe" ".\main.cpp" ".\MySolution.cpp" ".\Brute.cpp"
// 用法：./main.exe 0/1 --gen-queries  生成 query
// 用法：./main.exe 0/1 [--algo=solution|brute] 运行，默认 solution（使用 MySolution）

/*
主要作用是生成 query 文件和运行出一份暴力文件供 checker.cpp 使用对拍
在终端中运行：
./main.exe 0 --gen-queries
可以生成 query.txt 文件，0 代表使用 GloVe 数据集，1 代表使用 SIFT 数据集
运行时可指定算法：--algo=solution 或 --algo=brute 分别代表使用 MySolution 和 Brute 进行搜索
*/

#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <sstream>
#include <cctype>
#include <cmath>
#include <iomanip>

#include "MySolution.h"
#include "Brute.h"
#include "Config.h"

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

std::string filePath;
size_t pointsnum = 0;
int dimensions_ = 0;

Solution solution;
BSolution bsolution;

int dim = 0;
std::vector<float> base_data;                 // flat base data: N * dim
std::vector<std::vector<float>> query_data;   // queries, each a vector<float>
std::vector<std::vector<int>> result;         // results: store single int per query for now

// 检查文件是否存在
static bool file_exists(const std::string &path) {
    if (path.empty()) return false;
    std::ifstream ifs(path);
    return ifs.good();
}

// 读取数据集信息
void read_data(int &dataset_case) {
    if (dataset_case == 0) {
        filePath = GLOVEPATH;
        pointsnum = 1183514;
        dimensions_ = 100;
    }
    else if (dataset_case == 1) {
        filePath = SIFTPATH;
        pointsnum = 1000000;
        dimensions_ = 128;
    }
    else if (dataset_case == 2) {
        filePath = TESTPATH; // small test dataset
        pointsnum = 10;      // N
        dimensions_ = 10;    // dim
    } else {
        std::cerr << "Wrong number of cases! Using default small dataset." << std::endl;
        // fallback small dataset
        filePath = "";
        pointsnum = 1000;
        dimensions_ = 16;
    }
}

// 读取数据集文件
// Returns the number of vectors read and fills `out_flat` as a flattened row-major array (count * dim elements).
size_t load_flat_vectors_from_txt(const std::string &path, int dim, size_t max_count, std::vector<float> &out_flat) {
    out_flat.clear();
    if (dim <= 0) return 0;
    std::ifstream ifs(path);
    if (!ifs.is_open()) return 0;

    out_flat.reserve(std::min<size_t>(max_count, 1024) * dim);
    std::string line;
    size_t read = 0;
    auto start_tp = std::chrono::high_resolution_clock::now();
    while (read < max_count && std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        float val;
        std::vector<float> row;
        while (row.size() < (size_t)dim && ss >> val) {
            row.push_back(val);
        }
        if (row.size() < (size_t)dim) {
            // incomplete line; skip
            continue;
        }
        out_flat.insert(out_flat.end(), row.begin(), row.end());
        ++read;
        if (ENABLE_PROGRESS && (read % PROGRESS_STEP_LOAD == 0 || read == max_count))
            progress_bar("Loading base", read, max_count, start_tp);
    }
    if (ENABLE_PROGRESS) progress_bar("Loading base", read, read, start_tp);
    return read;
}

// 读取查询向量文件
size_t load_queries_from_txt(const std::string &base_path, int dim, size_t max_queries, std::vector<std::vector<float>> &out_queries) {
    if (base_path.empty()) return 0;
    out_queries.clear();

    // 1) try parent_dir/query.txt
    size_t pos_sep = base_path.find_last_of("\\/");
    if (pos_sep != std::string::npos) {
        // preserve original separator style if present
        char sep = (base_path.find('\\') != std::string::npos) ? '\\' : '/';
        std::string candidate = base_path.substr(0, pos_sep) + sep + QUERYFILEPATH;
        if (file_exists(candidate)) {
            std::ifstream ifs(candidate);
            if (!ifs.is_open()) return 0;
            std::string line;
            auto start_tp = std::chrono::high_resolution_clock::now();
            while ((int)out_queries.size() < (int)max_queries && std::getline(ifs, line)) {
                if (line.empty()) continue;
                std::istringstream ss(line);
                std::vector<float> row;
                float v;
                while ((int)row.size() < dim && ss >> v) row.push_back(v);
                if ((int)row.size() == dim) out_queries.push_back(std::move(row));

                if (ENABLE_PROGRESS) {
                    size_t cur = out_queries.size();
                    if (cur % PROGRESS_STEP_LOAD == 0 || cur == max_queries)
                        progress_bar("Loading queries", cur, max_queries, start_tp);
                }
            }
            if (ENABLE_PROGRESS) progress_bar("Loading queries", out_queries.size(), out_queries.size(), start_tp);
            return out_queries.size();
        }
    }

    // 2) try replacing "base.txt" -> "query.txt"
    size_t pos = base_path.find("base.txt");
    if (pos != std::string::npos) {
        std::string candidate = base_path.substr(0, pos) + QUERYFILEPATH;
        if (file_exists(candidate)) {
            std::ifstream ifs(candidate);
            if (!ifs.is_open()) return 0;
            std::string line;
            auto start_tp = std::chrono::high_resolution_clock::now();
            while ((int)out_queries.size() < (int)max_queries && std::getline(ifs, line)) {
                if (line.empty()) continue;
                std::istringstream ss(line);
                std::vector<float> row;
                float v;
                while ((int)row.size() < dim && ss >> v) row.push_back(v);
                if ((int)row.size() == dim) out_queries.push_back(std::move(row));
            }
            if (ENABLE_PROGRESS) progress_bar("Loading queries", out_queries.size(), out_queries.size(), start_tp);
            return out_queries.size();
        }
    }

    // 3) try current working directory: ".\\query.txt" or "./query.txt"
    {
        std::vector<std::string> candidates;
        candidates.emplace_back(QUERYFILEPATH);
        candidates.emplace_back(std::string(".\\") + QUERYFILEPATH);
        candidates.emplace_back(std::string("./") + QUERYFILEPATH);
        for (const auto &cand : candidates) {
            if (file_exists(cand)) {
                std::ifstream ifs(cand);
                if (!ifs.is_open()) return 0;
                std::string line;
                auto start_tp = std::chrono::high_resolution_clock::now();
                while ((int)out_queries.size() < (int)max_queries && std::getline(ifs, line)) {
                    if (line.empty()) continue;
                    std::istringstream ss(line);
                    std::vector<float> row;
                    float v;
                    while ((int)row.size() < dim && ss >> v) row.push_back(v);
                    if ((int)row.size() == dim) out_queries.push_back(std::move(row));
                }
                if (ENABLE_PROGRESS) progress_bar("Loading queries", out_queries.size(), out_queries.size(), start_tp);
                return out_queries.size();
            }
        }
    }

    return 0; // not found
}

// 生成极端数据文件（query 用）
static void generate_extreme_queries_to_file(const std::string &outfile, int dim, size_t total_count) {
    auto write_vec = [](std::ofstream &ofs, const std::vector<float> &v) {
        for (int i = 0; i < (int)v.size(); ++i) {
            if (i) ofs << ' ';
            ofs << v[i];
        }
        ofs << '\n';
    };

    std::ofstream ofs(outfile);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open for writing: " << outfile << std::endl;
        return;
    }

    std::mt19937 rng(20251105);
    std::uniform_real_distribution<float> uni_m1_1(-1.f, 1.f);
    std::uniform_real_distribution<float> uni_small(-1e-6f, 1e-6f);
    std::uniform_real_distribution<float> uni_wide(-1e3f, 1e3f);
    std::bernoulli_distribution coin(0.5);

    const float BIG = 1e6f;
    const float EPS = 1e-9f;

    // 新增：进度起始时间
    auto start_tp = std::chrono::high_resolution_clock::now();

    size_t written = 0;
    std::vector<float> v(dim, 0.0f);

    auto ensure_cap = [&](size_t need = 1) { return written + need <= total_count; };

    // 1) fixed patterns if room
    if (ensure_cap()) { std::fill(v.begin(), v.end(), 0.0f); write_vec(ofs, v); ++written;
        if (ENABLE_PROGRESS && (written % PROGRESS_STEP_WRITE == 0 || written == total_count))
            progress_bar("Gen queries", written, total_count, start_tp);
    }
    if (ensure_cap()) { std::fill(v.begin(), v.end(), 1.0f); write_vec(ofs, v); ++written;
        if (ENABLE_PROGRESS && (written % PROGRESS_STEP_WRITE == 0 || written == total_count))
            progress_bar("Gen queries", written, total_count, start_tp);
    }
    if (ensure_cap()) { std::fill(v.begin(), v.end(), -1.0f); write_vec(ofs, v); ++written;
        if (ENABLE_PROGRESS && (written % PROGRESS_STEP_WRITE == 0 || written == total_count))
            progress_bar("Gen queries", written, total_count, start_tp);
    }
    if (ensure_cap()) { std::fill(v.begin(), v.end(), BIG); write_vec(ofs, v); ++written;
        if (ENABLE_PROGRESS && (written % PROGRESS_STEP_WRITE == 0 || written == total_count))
            progress_bar("Gen queries", written, total_count, start_tp);
    }
    if (ensure_cap()) { std::fill(v.begin(), v.end(), -BIG); write_vec(ofs, v); ++written;
        if (ENABLE_PROGRESS && (written % PROGRESS_STEP_WRITE == 0 || written == total_count))
            progress_bar("Gen queries", written, total_count, start_tp);
    }
    if (ensure_cap()) { for (int i = 0; i < dim; ++i) v[i] = (i % 2 == 0) ? BIG : -BIG; write_vec(ofs, v); ++written;
        if (ENABLE_PROGRESS && (written % PROGRESS_STEP_WRITE == 0 || written == total_count))
            progress_bar("Gen queries", written, total_count, start_tp);
    }
    if (ensure_cap()) { std::fill(v.begin(), v.end(), EPS); write_vec(ofs, v); ++written;
        if (ENABLE_PROGRESS && (written % PROGRESS_STEP_WRITE == 0 || written == total_count))
            progress_bar("Gen queries", written, total_count, start_tp);
    }

    // 2) one-hot and negative one-hot
    for (int i = 0; i < dim && ensure_cap(); ++i) {
        std::fill(v.begin(), v.end(), 0.0f); v[i] = 1.0f; write_vec(ofs, v); ++written;
        if (ENABLE_PROGRESS && (written % PROGRESS_STEP_WRITE == 0 || written == total_count))
            progress_bar("Gen queries", written, total_count, start_tp);
    }
    for (int i = 0; i < dim && ensure_cap(); ++i) {
        std::fill(v.begin(), v.end(), 0.0f); v[i] = -1.0f; write_vec(ofs, v); ++written;
        if (ENABLE_PROGRESS && (written % PROGRESS_STEP_WRITE == 0 || written == total_count))
            progress_bar("Gen queries", written, total_count, start_tp);
    }

    // 3) ramps
    if (ensure_cap()) { for (int i = 0; i < dim; ++i) v[i] = (float)i / std::max(1, dim - 1); write_vec(ofs, v); ++written;
        if (ENABLE_PROGRESS && (written % PROGRESS_STEP_WRITE == 0 || written == total_count))
            progress_bar("Gen queries", written, total_count, start_tp);
    }
    if (ensure_cap()) { for (int i = 0; i < dim; ++i) v[i] = (float)(dim - 1 - i) / std::max(1, dim - 1); write_vec(ofs, v); ++written;
        if (ENABLE_PROGRESS && (written % PROGRESS_STEP_WRITE == 0 || written == total_count))
            progress_bar("Gen queries", written, total_count, start_tp);
    }

    // 4) sinusoid and checkerboard
    if (ensure_cap()) { for (int i = 0; i < dim; ++i) v[i] = std::sin(2 * 3.1415926535 * i / std::max(1, dim)); write_vec(ofs, v); ++written;
        if (ENABLE_PROGRESS && (written % PROGRESS_STEP_WRITE == 0 || written == total_count))
            progress_bar("Gen queries", written, total_count, start_tp);
    }
    if (ensure_cap()) { for (int i = 0; i < dim; ++i) v[i] = (i % 2 == 0) ? 1.0f : -1.0f; write_vec(ofs, v); ++written;
        if (ENABLE_PROGRESS && (written % PROGRESS_STEP_WRITE == 0 || written == total_count))
            progress_bar("Gen queries", written, total_count, start_tp);
    }
    if (ensure_cap()) { for (int i = 0; i < dim; ++i) v[i] = (i % 2 == 0) ? -1.0f : 1.0f; write_vec(ofs, v); ++written;
        if (ENABLE_PROGRESS && (written % PROGRESS_STEP_WRITE == 0 || written == total_count))
            progress_bar("Gen queries", written, total_count, start_tp);
    }

    auto gen_sparse_spikes = [&](int count, int spikes) {
        std::uniform_int_distribution<int> uid(0, std::max(0, dim - 1));
        for (int c = 0; c < count && ensure_cap(); ++c) {
            std::fill(v.begin(), v.end(), 0.0f);
            std::vector<int> idx; idx.reserve(spikes);
            while ((int)idx.size() < spikes) {
                int id = uid(rng);
                bool seen = false; for (int t : idx) if (t == id) { seen = true; break; }
                if (!seen) idx.push_back(id);
            }
            for (int id : idx) v[id] = coin(rng) ? BIG : -BIG;
            write_vec(ofs, v); ++written;
            if (ENABLE_PROGRESS && (written % PROGRESS_STEP_WRITE == 0 || written == total_count))
                progress_bar("Gen queries", written, total_count, start_tp);
        }
    };

    auto gen_random = [&](int count, std::uniform_real_distribution<float> &dist) {
        for (int c = 0; c < count && ensure_cap(); ++c) {
            for (int i = 0; i < dim; ++i) v[i] = dist(rng);
            write_vec(ofs, v); ++written;
            if (ENABLE_PROGRESS && (written % PROGRESS_STEP_WRITE == 0 || written == total_count))
                progress_bar("Gen queries", written, total_count, start_tp);
        }
    };

    auto gen_dense_extremes = [&](int count) {
        for (int c = 0; c < count && ensure_cap(); ++c) {
            for (int i = 0; i < dim; ++i) v[i] = coin(rng) ? BIG : -BIG;
            write_vec(ofs, v); ++written;
            if (ENABLE_PROGRESS && (written % PROGRESS_STEP_WRITE == 0 || written == total_count))
                progress_bar("Gen queries", written, total_count, start_tp);
        }
    };

    auto gen_alternating_shifts = [&](int blocks) {
        for (int b = 0; b < blocks && ensure_cap(); ++b) {
            for (int s = 0; s < dim && ensure_cap(); ++s) {
                for (int i = 0; i < dim; ++i) v[(i + s) % dim] = (i % 2 == 0) ? 1.0f : -1.0f;
                write_vec(ofs, v); ++written;
                if (ENABLE_PROGRESS && (written % PROGRESS_STEP_WRITE == 0 || written == total_count))
                    progress_bar("Gen queries", written, total_count, start_tp);
            }
        }
    };

    // 5) compose to reach total_count
    gen_sparse_spikes(1000, 5);
    gen_random(1000, uni_small);
    gen_random(2000, uni_m1_1);
    gen_random(2000, uni_wide);
    gen_dense_extremes(2000);
    gen_alternating_shifts(10); // 10 * dim lines

    // 6) duplicates to fill remainder
    while (written < total_count) {
        std::fill(v.begin(), v.end(), 0.1f);
        write_vec(ofs, v); ++written;
        if (ENABLE_PROGRESS && (written % PROGRESS_STEP_WRITE == 0 || written == total_count))
            progress_bar("Gen queries", written, total_count, start_tp);
    }

    ofs.close();
    std::cerr << "Generated " << written << " extreme queries to " << outfile << std::endl;
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
    bool gen_only = false;
    std::string algo = "solution"; // solution | brute
    // 新增：生成小体量 base 测试集到 test.txt
    bool gen_test_base = false;
    size_t gen_test_base_N = 10; // 默认 10 行，与 case 2 对齐
        auto is_unsigned_integer = [](const std::string &s) -> bool {
            if (s.empty()) return false;
            for (char c : s) if (!std::isdigit((unsigned char)c)) return false;
            return true;
        };
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--gen-queries") gen_only = true;
            else if (arg == "--algo" && i + 1 < argc) { algo = argv[++i]; }
            else if (arg.rfind("--algo=", 0) == 0) { algo = arg.substr(7); }
            else if (arg == "--gen-test-base") { gen_test_base = true; }
            else if (arg.rfind("--gen-test-base=", 0) == 0) { gen_test_base = true; gen_test_base_N = (size_t)std::max(1, std::atoi(arg.substr(17).c_str())); }
            else if (is_unsigned_integer(arg)) {
                dataset_case = std::atoi(arg.c_str());
            }
    }

    read_data(dataset_case);

    dim = dimensions_;
    if (dim <= 0) {
        std::cerr << "Invalid dimension: " << dim << std::endl;
        return 1;
    }

    // 若仅生成小体量 base 测试集，则直接输出 test.txt 并退出
    if (gen_test_base) {
        const std::string outfile = TESTPATH;
        std::mt19937 rng(20251110);
        std::uniform_real_distribution<float> uni(-1.f, 1.f);
        std::ofstream ofs(outfile);
        if (!ofs.is_open()) {
            std::cerr << "Failed to open for writing: " << outfile << std::endl;
            return 1;
        }
        // dim 取当前 dimensions_，若不是 10 也允许自定义 N 行 dim 列
        for (size_t i = 0; i < gen_test_base_N; ++i) {
            for (int j = 0; j < dim; ++j) {
                float v = uni(rng);
                if (j) ofs << ' ';
                ofs << std::fixed << std::setprecision(6) << v;
            }
            ofs << '\n';
        }
        ofs.close();
        std::cout << "Generated test dataset to " << outfile << ": N=" << gen_test_base_N << ", dim=" << dim << "\n";
        return 0;
    }

    size_t N = pointsnum;
    // Try to load base vectors from disk. If fails, fallback to synthetic small dataset (like before).
    size_t loaded = 0;
    const size_t max_load = pointsnum; // try to read up to pointsnum
    if (!filePath.empty() && file_exists(filePath)) {
        std::cerr << "Loading base vectors from: " << filePath << std::endl;
        loaded = load_flat_vectors_from_txt(filePath, dim, max_load, base_data);
        if (loaded == 0) {
            std::cerr << "Failed to read base vectors from file or file empty; falling back to synthetic subset." << std::endl;
        } else {
            N = loaded;
            std::cerr << "Loaded " << N << " vectors from disk (dim=" << dim << ")" << std::endl;
        }
    }

    if (loaded == 0) {
        // fallback: small synthetic dataset to allow testing
        const size_t demo_cap = std::min<size_t>(pointsnum ? pointsnum : 1000, 1000);
        N = demo_cap;
        base_data.assign(N * dim, 0.0f);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < N; ++i) {
            for (int j = 0; j < dim; ++j) base_data[i * dim + j] = dist(rng);
        }
        std::cerr << "Using synthetic base with N=" << N << " dim=" << dim << std::endl;
    }

    // 可以选择生成极端数据并退出
    if (gen_only) {
        generate_extreme_queries_to_file(QUERYFILEPATH, dim, 10000);
        return 0;
    }

    // 尝试从 query.txt 中读取数据，否则生成随机 query
    int num_queries = 10; // 默认生成 query 的个数
    size_t qloaded = load_queries_from_txt(filePath, dim, 1000000, query_data);
    if (qloaded == 0) {
        // fallback: generate random queries
        std::mt19937 rng(123);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        query_data.assign(num_queries, std::vector<float>(dim));
        for (int q = 0; q < num_queries; ++q)
            for (int j = 0; j < dim; ++j) query_data[q][j] = dist(rng);
        std::cerr << "Using synthetic " << num_queries << " queries." << std::endl;
    } else {
        num_queries = (int)qloaded;
        std::cerr << "Loaded " << num_queries << " queries from disk." << std::endl;
    }

    // 为每一个 TopK 分配空间
    const int k = 10;
    result.assign(num_queries, std::vector<int>(k, -1));

    // Build/Search
    std::chrono::duration<double> build_duration{0};
    std::chrono::duration<double> search_duration{0};
    int processed_queries = num_queries;
    if (RUN_FIRST_N > 0) processed_queries = std::min(num_queries, RUN_FIRST_N);
    
    if (algo == "solution") {
        // Build
        auto start1 = std::chrono::high_resolution_clock::now();
        solution.build(dim, base_data);
        auto end1 = std::chrono::high_resolution_clock::now();
        build_duration = end1 - start1;
        if (ENABLE_PROGRESS) std::cerr << "[Progress] Build done in " 
                                       << std::chrono::duration<double>(build_duration).count() << "s\n";
        // Search
        auto start2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < processed_queries; ++i) {
            solution.search(query_data[i], &result[i][0]);
            if (ENABLE_PROGRESS && ((i + 1) % PROGRESS_STEP_SEARCH == 0 || i + 1 == processed_queries))
                progress_bar("Searching", i + 1, processed_queries, start2);
        }
        auto end2 = std::chrono::high_resolution_clock::now();
        search_duration = end2 - start2;
    } else { // brute
        // Build
        auto start1 = std::chrono::high_resolution_clock::now();
        bsolution.build(dim, base_data);
        auto end1 = std::chrono::high_resolution_clock::now();
        build_duration = end1 - start1;
        if (ENABLE_PROGRESS) std::cerr << "[Progress] Build done in " 
                                       << std::chrono::duration<double>(build_duration).count() << "s\n";
        // Search
        auto start2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < processed_queries; ++i) {
            bsolution.search(query_data[i], &result[i][0]);
            if (ENABLE_PROGRESS && ((i + 1) % PROGRESS_STEP_SEARCH == 0 || i + 1 == processed_queries))
                progress_bar("Searching", i + 1, processed_queries, start2);
        }
        auto end2 = std::chrono::high_resolution_clock::now();
        search_duration = end2 - start2;
    }

    // 打印到终端（单位统一为 ms）
    double build_ms = build_duration.count() * 1000.0;
    double search_ms = search_duration.count() * 1000.0;
    double avg_query_ms = (processed_queries > 0) ? (search_ms / processed_queries) : 0.0;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Algo: " << algo << "\n";
    std::cout << "Build latency: " << build_ms << " ms\n";
    std::cout << "Search latency (total): " << search_ms << " ms\n";
    std::cout << "Average search latency: " << avg_query_ms << " ms/query\n";

    // 准备写入文件
    std::string outname = ANSFILEPATH; // fixed output filename as required
    std::ofstream ofs(outname);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open output file: " << outname << std::endl;
        return 1;
    }

    ofs << "Dataset case: " << dataset_case << "\n";
    ofs << "Base vectors loaded: " << N << "\n";
    ofs << "Dimension: " << dim << "\n";
    ofs << std::fixed << std::setprecision(2);
    ofs << "Algorithm: " << algo << "\n";
    ofs << "Build latency ms: " << build_ms << "\n";
    ofs << "Search latency ms: " << search_ms << "\n";
    ofs << "Average search latency ms_per_query: " << avg_query_ms << "\n";
    ofs << "Num queries: " << processed_queries << "\n";
    ofs << "Results (top-" << k << " indices per query, space-separated):\n";
    auto write_start_tp = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < processed_queries; ++i) {
        for (int j = 0; j < k; ++j) {
            ofs << result[i][j];
            if (j + 1 < k) ofs << ' ';
        }
        ofs << '\n';

        if (ENABLE_PROGRESS && ((i + 1) % PROGRESS_STEP_WRITE == 0 || i + 1 == processed_queries))
            progress_bar("Writing ans", i + 1, processed_queries, write_start_tp);
    }
    ofs.close();

    std::cout << "Wrote results to " << outname << std::endl;

    return 0;
}