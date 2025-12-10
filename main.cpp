// 编译：& 'C:/TDM-GCC-64/bin/g++.exe' -std=c++17 -O2 -Wall -Wextra -pthread -o './main.exe' './main.cpp' './MySolution.cpp' './Brute.cpp'
// 用法：./main.exe 0/1 --gen-queries  生成 query
// 用法：./main.exe 0/1 [--algo=solution|brute] 运行，默认 solution（使用 MySolution）

/*
主要作用是生成 query 文件和运行出一份暴力文件供 checker.cpp 使用对拍
在终端中运行：
./main.exe 0 --gen-queries
--save-bin-base 实现如果不能自动从 base.bin 读取基数据，则从 base.txt 读取后保存为 base.bin
可以生成 query.txt 文件，0 代表使用 GloVe 数据集，1 代表使用 SIFT 数据集
运行时可指定算法：--algo=solution 或 --algo=brute 分别代表使用 MySolution 和 Brute 进行搜索
*/

#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <mutex>
#include <thread>
#include <cstdio>

#include "MySolution.h"
#include "Brute.h"
#include "Config.h"
#include "BinaryIO.h"

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

// 文本读取实现（在本文件后面定义），供下方 auto 加载函数调用
size_t load_flat_vectors_from_txt(const std::string &path, int dim, size_t max_count, std::vector<float> &out_flat);
size_t load_queries_from_txt(const std::string &base_path, int dim, size_t max_queries, std::vector<std::vector<float>> &out_queries, std::string* used_txt_path=nullptr);

// 返回 base.txt 同目录的 "base.bin" 或 "query.bin" 路径；若无法解析目录，则返回工作目录下的目标文件名
static std::string sibling_bin_path(const std::string &txt_path, const std::string &bin_name) {
    if (!txt_path.empty()) {
        size_t pos_sep = txt_path.find_last_of("\\/");
        if (pos_sep != std::string::npos) {
            char sep = (txt_path.find('\\') != std::string::npos) ? '\\' : '/';
            return txt_path.substr(0, pos_sep + 1) + bin_name;
        }
    }
    return bin_name; // cwd fallback
}

// 从给定的 txt 路径派生对应的 .bin 路径：同目录，若后缀为 .txt 则替换为 .bin，否则追加 .bin
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

// 将 ".../base.txt" 替换为 ".../base.bin"；若不包含 base.txt，退化到同目录 + bin_name
static std::string replace_base_txt_with_bin(const std::string &base_txt, const std::string &bin_name) {
    size_t pos = base_txt.find("base.txt");
    if (pos != std::string::npos) {
        return base_txt.substr(0, pos) + bin_name;
    }
    return sibling_bin_path(base_txt, bin_name);
}

// 优先从二进制 .bin 读取 base，成功则返回向量数；否则回退到 txt
static size_t load_base_vectors_auto(const std::string &base_txt_path, int expected_dim, std::vector<float> &out_flat, int &out_dim_actual) {
    out_dim_actual = expected_dim;
    std::vector<std::string> candidates;
    // 尝试与 base.txt 同目录的 base.bin
    candidates.push_back(replace_base_txt_with_bin(base_txt_path, "base.bin"));
    // 也尝试工作目录下的 base.bin
    candidates.emplace_back("base.bin");

    for (const auto &cand : candidates) {
        if (file_exists(cand)) {
            int dim_rd = 0; size_t cnt = 0; std::vector<float> flat;
            if (binio::read_vecbin(cand, dim_rd, cnt, flat)) {
                if (expected_dim > 0 && dim_rd != expected_dim) {
                    std::cerr << "[WARN] Dimension mismatch in bin (" << cand << "): " << dim_rd
                              << " != expected " << expected_dim << ". Ignored.\n";
                } else {
                    out_flat.swap(flat);
                    out_dim_actual = dim_rd;
                    std::cerr << "Loaded base from bin: " << cand << ", N=" << cnt << ", dim=" << dim_rd << "\n";
                    return cnt;
                }
            }
        }
    }
    // 回退到文本读取
    size_t read = load_flat_vectors_from_txt(base_txt_path, expected_dim, pointsnum, out_flat);
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

// 优先从二进制 .bin 读取 queries，成功则返回数量；否则回退到 txt
static size_t load_queries_auto(const std::string &base_txt_path, int expected_dim, size_t max_queries, std::vector<std::vector<float>> &out_queries) {
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
    cands_qbin.emplace_back(qbin_name); // 当前目录

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

    // 若没有匹配到 query0.bin，则优先从指定的 QUERYFILEPATH 文本读取并写出同名 .bin
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
size_t load_queries_from_txt(const std::string &base_path, int dim, size_t max_queries, std::vector<std::vector<float>> &out_queries, std::string* used_txt_path) {
    if (used_txt_path) *used_txt_path = "";
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
            if (used_txt_path) *used_txt_path = candidate;
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
            if (used_txt_path) *used_txt_path = candidate;
            return out_queries.size();
        }
    }

    // 3) try current working directory: ".\\query.txt" or "./query.txt"
    {
        std::vector<std::string> candidates;
        candidates.emplace_back(QUERYFILEPATH);
        candidates.emplace_back(std::string(".\\") + QUERYFILEPATH);
        candidates.emplace_back(std::string("./") + QUERYFILEPATH);
        // dataset-case friendly fallbacks
        candidates.emplace_back("query0.txt");
        candidates.emplace_back(".\\query0.txt");
        candidates.emplace_back("./query0.txt");
        candidates.emplace_back("query1.txt");
        candidates.emplace_back(".\\query1.txt");
        candidates.emplace_back("./query1.txt");
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
                if (used_txt_path) *used_txt_path = cand;
                return out_queries.size();
            }
        }
    }

    return 0; // not found
}

// 生成极端数据文件（query 用）
static void generate_extreme_queries_to_file(const std::string &outfile, int dim, size_t total_count, int dataset_case) {
    // 改为：生成完全随机的 query（均匀分布 [-1, 1]），不再包含任何极端/特例模式
    std::ofstream ofs(outfile);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open for writing: " << outfile << std::endl;
        return;
    }

    std::mt19937 rng(20251113);
    std::uniform_real_distribution<float> uni01(0.f, 1.f);
    std::uniform_real_distribution<float> uni11(-1.f, 1.f);
    std::uniform_int_distribution<int> uni255(0, 255);
    auto start_tp = std::chrono::high_resolution_clock::now();

    std::vector<float> v(dim);
    for (size_t i = 0; i < total_count; ++i) {
        for (int j = 0; j < dim; ++j) {
            if (dataset_case == 1) {
                int iv = uni255(rng);
                if (j) ofs << ' ';
                ofs << iv; // SIFT: 整型 0..255
            } else if (dataset_case == 0) {
                float x = uni01(rng);
                if (j) ofs << ' ';
                ofs << std::fixed << std::setprecision(6) << x; // GloVe: (0,1) 小数
            } else {
                float x = uni11(rng);
                if (j) ofs << ' ';
                ofs << std::fixed << std::setprecision(6) << x; // 其它：[-1,1]
            }
        }
        ofs << '\n';
        if (ENABLE_PROGRESS && ((i + 1) % PROGRESS_STEP_WRITE == 0 || i + 1 == total_count))
            progress_bar("Gen queries", i + 1, total_count, start_tp);
    }

    ofs.close();
    std::cerr << "Generated " << total_count << " random queries to " << outfile << std::endl;
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
    // 生成小体量 base 测试集到 test.txt
    bool gen_test_base = false;
    size_t gen_test_base_N = 10; // 默认 10 行，与 case 2 对齐
    // 保存为二进制文件选项
    bool save_bin_base = false;
    // 强制从二进制读取
    bool force_bin = false;
    // 查询文件路径覆盖（txt 或 bin，取决于 --bin）
    std::string query_override;
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
            else if (arg == "--save-bin-base") { save_bin_base = true; }
            else if (arg == "--bin") { force_bin = true; }
            else if (arg == "--query" && i + 1 < argc) { query_override = argv[++i]; }
            else if (arg.rfind("--query=", 0) == 0) { query_override = arg.substr(8); }
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
    // 加载 base：若 --bin 强制二进制，则仅尝试 .bin；否则自动（bin 优先，失败回退 txt）
    size_t loaded = 0;
    if (!filePath.empty() && file_exists(filePath)) {
        if (force_bin) {
            std::vector<std::string> cands;
            cands.push_back(replace_base_txt_with_bin(filePath, "base.bin"));
            cands.emplace_back("base.bin");
            bool ok = false;
            for (const auto &cand : cands) {
                if (file_exists(cand)) {
                    int dim_rd = 0; size_t cnt = 0; std::vector<float> flat;
                    if (binio::read_vecbin(cand, dim_rd, cnt, flat)) {
                        base_data.swap(flat);
                        dim = dim_rd; N = loaded = cnt; ok = true;
                        std::cerr << "Loaded base (bin forced): " << cand << ", N=" << N << ", dim=" << dim << "\n";
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
            loaded = load_base_vectors_auto(filePath, dim, base_data, actual_dim);
            dim = actual_dim; // 若 bin 中记录了维度，采用之
            if (loaded > 0) {
                N = loaded;
            } else {
                std::cerr << "Failed to read base vectors from bin/txt; falling back to synthetic subset." << std::endl;
            }
        }
    }

    if (loaded == 0) {
        // fallback: small synthetic dataset to allow testing (dataset-aware)
        const size_t demo_cap = std::min<size_t>(pointsnum ? pointsnum : 1000, 1000);
        N = demo_cap;
        base_data.assign(N * dim, 0.0f);
        std::mt19937 rng(42);
        if (dataset_case == 1) {
            // SIFT: 整型 0..255（以 float 存储）
            std::uniform_int_distribution<int> di(0, 255);
            for (size_t i = 0; i < N; ++i)
                for (int j = 0; j < dim; ++j)
                    base_data[i * dim + j] = static_cast<float>(di(rng));
        } else if (dataset_case == 0) {
            // GloVe: (0,1) 小数
            std::uniform_real_distribution<float> df(0.f, 1.f);
            for (size_t i = 0; i < N; ++i)
                for (int j = 0; j < dim; ++j)
                    base_data[i * dim + j] = df(rng);
        } else {
            // 其它：[-1,1]
            std::uniform_real_distribution<float> df(-1.f, 1.f);
            for (size_t i = 0; i < N; ++i)
                for (int j = 0; j < dim; ++j)
                    base_data[i * dim + j] = df(rng);
        }
        std::cerr << "Using synthetic base with N=" << N << " dim=" << dim << std::endl;
    }

    // 可以选择生成极端数据并退出
    if (gen_only) {
        generate_extreme_queries_to_file(QUERYFILEPATH, dim, 10000, dataset_case);
        return 0;
    }

    // 尝试从 query（优先 bin 或文本）读取数据，否则生成随机 query
    int num_queries = 10; // 默认生成 query 的个数
    size_t qloaded = 0;
    if (force_bin) {
        if (!query_override.empty()) {
            // 强制从 override 路径按 .bin 读取
            std::vector<std::vector<float>> qs;
            if (!binio::read_queries_vecbin(query_override, dim, 1000000, qs)) {
                std::cerr << "[ERROR] --bin specified, but override query bin invalid: " << query_override << std::endl;
                return 1;
            }
            query_data.swap(qs);
            qloaded = query_data.size();
            std::cerr << "Loaded queries (bin forced override): " << query_override << ", count=" << qloaded << ", dim=" << dim << "\n";
        } else {
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
                if (binio::read_queries_vecbin(cand, dim, 1000000, qs)) {
                    query_data.swap(qs);
                    qloaded = query_data.size(); ok = true;
                    std::cerr << "Loaded queries (bin forced): " << cand << ", count=" << qloaded << ", dim=" << dim << "\n";
                    break;
                }
            }
        }
        if (!ok) {
            std::cerr << "[ERROR] --bin specified, but query bin not found or invalid." << std::endl;
            return 1;
        }
        }
    } else {
        if (!query_override.empty()) {
            // 明确指定了 txt 路径：按文本读取并派生写出 .bin
            std::ifstream qf(query_override);
            if (!qf.is_open()) {
                std::cerr << "[ERROR] Cannot open query file: " << query_override << std::endl;
                return 1;
            }
            std::string line;
            std::vector<std::vector<float>> qs;
            while (qs.size() < 1000000 && std::getline(qf, line)) {
                if (line.empty()) continue;
                std::istringstream ss(line);
                std::vector<float> row; float v;
                while ((int)row.size() < dim && ss >> v) row.push_back(v);
                if ((int)row.size() == dim) qs.push_back(std::move(row));
            }
            if (!qs.empty()) {
                query_data.swap(qs);
                qloaded = query_data.size();
                std::string qbin = derive_bin_path_from_txt(query_override);
                binio::write_queries_vecbin(qbin, dim, query_data);
                std::cerr << "Loaded queries from override txt: " << query_override << ", wrote bin: " << qbin << ", count=" << qloaded << "\n";
            } else {
                qloaded = 0;
            }
        } else {
            qloaded = load_queries_auto(filePath, dim, 1000000, query_data);
        }
    }
    if (qloaded == 0) {
        // fallback: generate dataset-aware random queries
        std::mt19937 rng(123);
        query_data.assign(num_queries, std::vector<float>(dim));
        if (dataset_case == 1) {
            std::uniform_int_distribution<int> di(0, 255);
            for (int q = 0; q < num_queries; ++q)
                for (int j = 0; j < dim; ++j)
                    query_data[q][j] = static_cast<float>(di(rng));
        } else if (dataset_case == 0) {
            std::uniform_real_distribution<float> df(0.f, 1.f);
            for (int q = 0; q < num_queries; ++q)
                for (int j = 0; j < dim; ++j)
                    query_data[q][j] = df(rng);
        } else {
            std::uniform_real_distribution<float> df(-1.f, 1.f);
            for (int q = 0; q < num_queries; ++q)
                for (int j = 0; j < dim; ++j)
                    query_data[q][j] = df(rng);
        }
        std::cerr << "Using synthetic " << num_queries << " queries (dataset-aware)." << std::endl;
    } else {
        num_queries = (int)qloaded;
        std::cerr << "Loaded " << num_queries << " queries from disk." << std::endl;
    }

    // 如需转换保存为 .bin，则此时写出 base/queries 的 .bin 文件
    if (save_bin_base && !base_data.empty()) {
        std::string bbin = replace_base_txt_with_bin(filePath, "base.bin");
        if (binio::write_vecbin(bbin, dim, base_data)) {
            std::cerr << "Saved base to bin: " << bbin << " (N=" << (base_data.size()/ (size_t)dim) << ")\n";
        } else {
            std::cerr << "[WARN] Failed to save base bin: " << bbin << "\n";
        }
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
#ifdef CPP_SOLUTION_FAKE_STD_SYNC
        std::fprintf(stderr, "[main] threading disabled (fallback)\n");
#else
        std::fprintf(stderr, "[main] threading enabled, worker_count=%zu (hc=%u)\n",
                     solution.worker_count_, std::thread::hardware_concurrency());
#endif
        if(ENABLE_PROGRESS) {
            progress_bar("Building", N, N, start1);
        }
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