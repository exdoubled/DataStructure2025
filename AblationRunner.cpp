// AblationRunner.cpp - CLI tool for HNSW ablation study
// Compile: g++ -std=c++17 -O2 -Wall -Wextra -pthread -o AblationRunner.exe AblationRunner.cpp MySolution.cpp Brute.cpp

/*
Usage:
  AblationRunner.exe [options]

I/O Options:
  --data=<path>           Path to base vectors file (required)
  --query=<path>          Path to query vectors file (required)
  --groundtruth=<path>    Path to ground truth file (required)
  --cache-path=<path>     Path for graph cache (optional, enables smart caching)

Build Options:
  --M=<int>               Max connections per node (default: 96)
  --efC=<int>             Construction ef parameter (default: 400)
  --onng-out=<int>        ONNG output degree (default: 96)
  --onng-in=<int>         ONNG input degree (default: 144)
  --onng-min=<int>        ONNG minimum edges (default: 64)
  --no-onng               Disable ONNG optimization
  --no-bfs                Disable BFS memory reordering
  --no-simd               Disable SIMD (force scalar distance)

Search Options:
  --k=<int>               Top-K results (default: 10)
  --strategy=<gamma|fixed> Search strategy (default: gamma)
  --gamma=<float>         Gamma for adaptive search (default: 0.19)
  --ef=<int>              EF for fixed ef search (default: 100)

Debug Options:
  --count-dist            Enable distance computation counting (slower but provides stats)

Output:
  Prints a single JSON line with all parameters and metrics.
*/

#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <fstream>

#include "MySolution.h"
#include "BinaryIO.h"
#include "Config.h"

// ==================== Argument Parsing ====================

struct AblationArgs {
    // I/O
    std::string data_path;
    std::string query_path;
    std::string groundtruth_path;
    std::string cache_path;
    
    // Build params
    size_t M = 96;
    size_t ef_construction = 400;
    size_t onng_out = 96;
    size_t onng_in = 144;
    size_t onng_min = 64;
    bool enable_onng = true;
    bool enable_bfs = true;
    bool enable_simd = true;
    bool enable_multilayer = true;
    bool use_negative_inner_product = false;
    bool opt_only = false;
    std::string cache_out;
    std::string opt_order = "onng-first"; // or "bfs-first"
    bool log_to_file = false;
    
    // Search params
    size_t k = 10;
    std::string strategy = "gamma";  // "gamma" (dynamic), "gamma-static", "fixed"
    float gamma = 0.19f;
    size_t ef_search = 704;
    
    // Debug params
    bool count_dist = false;  // If true, count distance computations (slower)
    
    // Visualization params
    std::string vis_graph_path;   // 导出图结构JSON路径（可选）
    std::string vis_paths_path;   // 导出搜索路径JSON路径（可选）
    size_t vis_samples = 100;     // 采样多少个查询的搜索路径（0=全部）
    size_t vis_max_nodes = 0;     // 导出图结构时最多导出多少节点（0=全部）
    
    // Derived
    int dim = 0;
    size_t base_count = 0;
    size_t query_count = 0;
};

static bool file_exists(const std::string &path) {
    if (path.empty()) return false;
    std::ifstream ifs(path);
    return ifs.good();
}

static void print_usage() {
    std::cerr << R"(
AblationRunner - HNSW Ablation Study Tool

Usage: AblationRunner.exe [options]

Required:
  --data=<path>        Base vectors file (.bin or .txt)
  --query=<path>       Query vectors file
  --groundtruth=<path> Ground truth file (line format: idx1 idx2 ... idxK)

Optional I/O:
  --cache-path=<path>  Graph cache file (load if exists, save after build)

Build Options:
  --M=<int>            Max connections (default: 96)
  --efC=<int>          Construction ef (default: 400)
  --onng-out=<int>     ONNG out degree (default: 96)
  --onng-in=<int>      ONNG in degree (default: 144)
  --onng-min=<int>     ONNG min edges (default: 64)
  --no-onng            Disable ONNG
  --no-bfs             Disable BFS reorder
  --no-simd            Force scalar distance
  --single-layer       Disable multi-layer graph (use single-layer)
  --use-neg-ip         Use negative inner product instead of L2
  --opt-only           Load cache, apply ONNG/BFS, save, and exit (no search)
  --cache-out=<path>   Output cache path (after search, or when --opt-only is used)
  --opt-order=<s>      "onng-first" (default) or "bfs-first" when both are enabled
  --log-to-file        Write stderr/logs to a file next to cache (cache-path + ".log")

Search Options:
  --k=<int>            Top-K (default: 10)
  --strategy=<s>       "gamma" (dynamic, default) | "gamma-static" | "fixed"
  --gamma=<float>      Gamma value (default: 0.19)
  --ef=<int>           Fixed ef (default: 100)

Debug Options:
  --count-dist         Count distance computations (slower)

Visualization Options:
  --vis-graph=<path>   Export graph structure to JSON file (for visualize_search.py)
  --vis-paths=<path>   Export search paths to JSON file (for visualize_search.py)
  --vis-samples=<int>  Number of queries to record paths for (default: 100, 0=all)
  --vis-max-nodes=<n>  Max nodes to export in graph JSON (default: 0=all)
)";
}

static bool parse_args(int argc, char** argv, AblationArgs& args) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        // I/O
        if (arg.rfind("--data=", 0) == 0) {
            args.data_path = arg.substr(7);
        } else if (arg.rfind("--query=", 0) == 0) {
            args.query_path = arg.substr(8);
        } else if (arg.rfind("--groundtruth=", 0) == 0) {
            args.groundtruth_path = arg.substr(14);
        } else if (arg.rfind("--cache-path=", 0) == 0) {
            args.cache_path = arg.substr(13);
        }
        // Build
        else if (arg.rfind("--M=", 0) == 0) {
            args.M = std::max(1, std::atoi(arg.substr(4).c_str()));
        } else if (arg.rfind("--efC=", 0) == 0) {
            args.ef_construction = std::max(1, std::atoi(arg.substr(6).c_str()));
        } else if (arg.rfind("--onng-out=", 0) == 0) {
            args.onng_out = std::max(1, std::atoi(arg.substr(11).c_str()));
        } else if (arg.rfind("--onng-in=", 0) == 0) {
            args.onng_in = std::max(1, std::atoi(arg.substr(10).c_str()));
        } else if (arg.rfind("--onng-min=", 0) == 0) {
            args.onng_min = std::max(1, std::atoi(arg.substr(11).c_str()));
        } else if (arg == "--no-onng") {
            args.enable_onng = false;
        } else if (arg == "--no-bfs") {
            args.enable_bfs = false;
        } else if (arg == "--no-simd") {
            args.enable_simd = false;
        } else if (arg == "--single-layer") {
            args.enable_multilayer = false;
        } else if (arg == "--use-neg-ip") {
            args.use_negative_inner_product = true;
        } else if (arg == "--opt-only") {
            args.opt_only = true;
        } else if (arg.rfind("--cache-out=", 0) == 0) {
            args.cache_out = arg.substr(12);
        } else if (arg.rfind("--opt-order=", 0) == 0) {
            args.opt_order = arg.substr(12);
        } else if (arg == "--log-to-file") {
            args.log_to_file = true;
        }
        // Search
        else if (arg.rfind("--k=", 0) == 0) {
            args.k = std::max(1, std::atoi(arg.substr(4).c_str()));
        } else if (arg.rfind("--strategy=", 0) == 0) {
            args.strategy = arg.substr(11);
        } else if (arg.rfind("--gamma=", 0) == 0) {
            args.gamma = std::stof(arg.substr(8));
        } else if (arg.rfind("--ef=", 0) == 0) {
            args.ef_search = std::max(1, std::atoi(arg.substr(5).c_str()));
        }
        // Debug
        else if (arg == "--count-dist") {
            args.count_dist = true;
        }
        // Visualization
        else if (arg.rfind("--vis-graph=", 0) == 0) {
            args.vis_graph_path = arg.substr(12);
        } else if (arg.rfind("--vis-paths=", 0) == 0) {
            args.vis_paths_path = arg.substr(12);
        } else if (arg.rfind("--vis-samples=", 0) == 0) {
            args.vis_samples = std::max(0, std::atoi(arg.substr(14).c_str()));
        } else if (arg.rfind("--vis-max-nodes=", 0) == 0) {
            args.vis_max_nodes = std::max(0, std::atoi(arg.substr(16).c_str()));
        }
        else if (arg == "--help" || arg == "-h") {
            print_usage();
            return false;
        }
    }
    
    // Validate required args
    if (!args.opt_only) {
        if (args.data_path.empty()) {
            std::cerr << "Error: --data is required\n";
            return false;
        }
        if (args.query_path.empty()) {
            std::cerr << "Error: --query is required\n";
            return false;
        }
        if (args.groundtruth_path.empty()) {
            std::cerr << "Error: --groundtruth is required\n";
            return false;
        }
    } else {
        if (args.cache_path.empty()) {
            std::cerr << "Error: --cache-path is required for --opt-only\n";
            return false;
        }
    }
    
    return true;
}

// ==================== Data Loading ====================

static bool load_base_data(const std::string& path, int& dim, std::vector<float>& data) {
    // Try binary first
    std::string bin_path = path;
    if (path.size() > 4 && path.substr(path.size() - 4) == ".txt") {
        bin_path = path.substr(0, path.size() - 4) + ".bin";
    }
    
    if (file_exists(bin_path)) {
        size_t count = 0;
        if (binio::read_vecbin(bin_path, dim, count, data)) {
            std::cerr << "[AblationRunner] Loaded base from: " << bin_path 
                      << " (N=" << count << ", dim=" << dim << ")\n";
            return true;
        }
    }
    
    // Try text
    if (file_exists(path)) {
        std::ifstream ifs(path);
        if (!ifs.is_open()) return false;
        
        std::string line;
        data.clear();
        dim = 0;
        
        while (std::getline(ifs, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            std::vector<float> row;
            float v;
            while (ss >> v) row.push_back(v);
            
            if (dim == 0) dim = (int)row.size();
            if ((int)row.size() == dim) {
                data.insert(data.end(), row.begin(), row.end());
            }
        }
        
        std::cerr << "[AblationRunner] Loaded base from: " << path 
                  << " (N=" << (data.size() / dim) << ", dim=" << dim << ")\n";
        return !data.empty();
    }
    
    return false;
}

static bool load_queries(const std::string& path, int dim, std::vector<std::vector<float>>& queries) {
    // Try binary first
    std::string bin_path = path;
    if (path.size() > 4 && path.substr(path.size() - 4) == ".txt") {
        bin_path = path.substr(0, path.size() - 4) + ".bin";
    }
    
    if (file_exists(bin_path)) {
        if (binio::read_queries_vecbin(bin_path, dim, 1000000, queries)) {
            std::cerr << "[AblationRunner] Loaded queries from: " << bin_path 
                      << " (count=" << queries.size() << ")\n";
            return true;
        }
    }
    
    // Try text
    if (file_exists(path)) {
        std::ifstream ifs(path);
        if (!ifs.is_open()) return false;
        
        std::string line;
        queries.clear();
        
        while (std::getline(ifs, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            std::vector<float> row;
            float v;
            while ((int)row.size() < dim && ss >> v) row.push_back(v);
            if ((int)row.size() == dim) {
                queries.push_back(std::move(row));
            }
        }
        
        std::cerr << "[AblationRunner] Loaded queries from: " << path 
                  << " (count=" << queries.size() << ")\n";
        return !queries.empty();
    }
    
    return false;
}

static bool load_groundtruth(const std::string& path, size_t k, std::vector<std::vector<int>>& gt) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return false;
    
    std::string line;
    gt.clear();
    
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::vector<int> row;
        int v;
        while (row.size() < k && ss >> v) row.push_back(v);
        if (!row.empty()) {
            gt.push_back(std::move(row));
        }
    }
    
    std::cerr << "[AblationRunner] Loaded groundtruth from: " << path 
              << " (count=" << gt.size() << ", k=" << k << ")\n";
    return !gt.empty();
}

// ==================== Metrics ====================

static double compute_recall(const std::vector<std::vector<int>>& predictions,
                             const std::vector<std::vector<int>>& groundtruth,
                             size_t k) {
    if (predictions.size() != groundtruth.size()) return 0.0;
    
    double total_recall = 0.0;
    size_t count = predictions.size();
    
    for (size_t i = 0; i < count; ++i) {
        const auto& pred = predictions[i];
        const auto& gt = groundtruth[i];
        
        size_t gt_k = std::min(gt.size(), k);
        std::unordered_set<int> gt_set(gt.begin(), gt.begin() + gt_k);
        
        size_t hits = 0;
        for (size_t j = 0; j < std::min(pred.size(), k); ++j) {
            if (gt_set.count(pred[j])) ++hits;
        }
        
        total_recall += (gt_k > 0) ? ((double)hits / (double)gt_k) : 0.0;
    }
    
    return (count > 0) ? (total_recall / (double)count) : 0.0;
}

// ==================== JSON Output ====================

static std::string escape_json_string(const std::string& s) {
    std::string result;
    for (char c : s) {
        switch (c) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default: result += c; break;
        }
    }
    return result;
}

static void output_json(const AblationArgs& args,
                        double build_ms,
                        double search_ms,
                        double qps,
                        double recall,
                        double avg_dist_calcs) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "{";
    
    // Input args
    std::cout << "\"data\":\"" << escape_json_string(args.data_path) << "\",";
    std::cout << "\"query\":\"" << escape_json_string(args.query_path) << "\",";
    std::cout << "\"groundtruth\":\"" << escape_json_string(args.groundtruth_path) << "\",";
    std::cout << "\"cache_path\":\"" << escape_json_string(args.cache_path) << "\",";
    
    // Build params
    std::cout << "\"M\":" << args.M << ",";
    std::cout << "\"ef_construction\":" << args.ef_construction << ",";
    std::cout << "\"onng_out\":" << args.onng_out << ",";
    std::cout << "\"onng_in\":" << args.onng_in << ",";
    std::cout << "\"onng_min\":" << args.onng_min << ",";
    std::cout << "\"enable_onng\":" << (args.enable_onng ? "true" : "false") << ",";
    std::cout << "\"enable_bfs\":" << (args.enable_bfs ? "true" : "false") << ",";
    std::cout << "\"enable_simd\":" << (args.enable_simd ? "true" : "false") << ",";
    
    // Search params
    std::cout << "\"k\":" << args.k << ",";
    std::cout << "\"strategy\":\"" << args.strategy << "\",";
    std::cout << "\"gamma\":" << args.gamma << ",";
    std::cout << "\"ef_search\":" << args.ef_search << ",";
    
    // Data info
    std::cout << "\"dim\":" << args.dim << ",";
    std::cout << "\"base_count\":" << args.base_count << ",";
    std::cout << "\"query_count\":" << args.query_count << ",";
    
    // Metrics
    std::cout << "\"build_ms\":" << build_ms << ",";
    std::cout << "\"search_ms\":" << search_ms << ",";
    std::cout << "\"qps\":" << qps << ",";
    std::cout << "\"recall\":" << recall << ",";
    std::cout << "\"avg_dist_calcs\":" << avg_dist_calcs;
    
    std::cout << "}" << std::endl;
}

// ==================== Main ====================

int main(int argc, char** argv) {
    AblationArgs args;
    if (!parse_args(argc, argv, args)) {
        return 1;
    }

    // 准备日志输出
    std::unique_ptr<std::ofstream> log_file;
    std::streambuf* cerr_buf = std::cerr.rdbuf();
    if (args.log_to_file && !args.cache_path.empty()) {
        std::string log_path = args.cache_path + ".log";
        log_file = std::make_unique<std::ofstream>(log_path, std::ios::app);
        if (log_file->is_open()) {
            std::cerr.rdbuf(log_file->rdbuf());
            std::cerr << "[LOG] Redirecting stderr to " << log_path << "\n";
        } else {
            std::cerr << "[WARN] Failed to open log file: " << log_path << ", fallback to stderr.\n";
            log_file.reset();
        }
    }
    
    // 如果 opt-only，只需要缓存，不加载数据/查询/真值
    std::vector<float> base_data;
    std::vector<std::vector<float>> queries;
    std::vector<std::vector<int>> groundtruth;

    if (!args.opt_only) {
        // Load data
        if (!load_base_data(args.data_path, args.dim, base_data)) {
            std::cerr << "Error: Failed to load base data from: " << args.data_path << "\n";
            return 1;
        }
        args.base_count = base_data.size() / args.dim;
        
        if (!load_queries(args.query_path, args.dim, queries)) {
            std::cerr << "Error: Failed to load queries from: " << args.query_path << "\n";
            return 1;
        }
        args.query_count = queries.size();
        
        if (!load_groundtruth(args.groundtruth_path, args.k, groundtruth)) {
            std::cerr << "Error: Failed to load groundtruth from: " << args.groundtruth_path << "\n";
            return 1;
        }
        
        if (groundtruth.size() < queries.size()) {
            std::cerr << "Warning: groundtruth count (" << groundtruth.size() 
                      << ") < query count (" << queries.size() << ")\n";
            queries.resize(groundtruth.size());
            args.query_count = queries.size();
        }
    }
    
    // Build configuration
    SolutionConfig config;
    config.M = args.M;
    config.ef_construction = args.ef_construction;
    config.enable_onng = args.enable_onng;
    config.onng_out_degree = args.onng_out;
    config.onng_in_degree = args.onng_in;
    config.onng_min_edges = args.onng_min;
    config.enable_bfs = args.enable_bfs;
    config.enable_simd = args.enable_simd;
    config.enable_multilayer = args.enable_multilayer;
    config.use_negative_inner_product = args.use_negative_inner_product;
    config.k = args.k;
    config.gamma = args.gamma;
    config.ef_search = args.ef_search;
    if (args.strategy == "fixed") {
        config.search_method = SearchMethod::FIXED_EF;
    } else if (args.strategy == "gamma-static") {
        config.search_method = SearchMethod::STATIC_GAMMA;
    } else { // "gamma" 或默认
        config.search_method = SearchMethod::DYNAMIC_GAMMA;
    }
    // Set distance counting based on command line flag
    config.count_distance_computation = args.count_dist;

    auto log_build_cfg = [&]() {
        std::cerr << "[CONFIG] "
                  << "multilayer=" << (config.enable_multilayer ? 1 : 0)
                  << " onng=" << (config.enable_onng ? 1 : 0)
                  << " bfs=" << (config.enable_bfs ? 1 : 0)
                  << " simd=" << (config.enable_simd ? 1 : 0)
                  << " neg_ip=" << (config.use_negative_inner_product ? 1 : 0)
                  << " M=" << config.M
                  << " efC=" << config.ef_construction
                  << " oo=" << config.onng_out_degree
                  << " oi=" << config.onng_in_degree
                  << " om=" << config.onng_min_edges
                  << " cache=" << args.cache_path
                  << "\n";
    };
    auto log_search_cfg = [&]() {
        std::cerr << "[SEARCH] strategy=" << args.strategy
                  << " gamma=" << args.gamma
                  << " ef=" << args.ef_search
                  << " k=" << args.k
                  << " count_dist=" << (args.count_dist ? 1 : 0)
                  << "\n";
    };
    log_build_cfg();
    if (!args.opt_only) log_search_cfg();
    
    Solution solution;
    double build_ms = 0.0;
    
    // Smart caching logic
    bool loaded_from_cache = false;
    if (!args.cache_path.empty() && file_exists(args.cache_path)) {
        // Try to load from cache
        auto load_start = std::chrono::high_resolution_clock::now();
        if (Solution::isGraphCacheValid(args.cache_path, args.dim, args.base_count, config.enable_multilayer)) {
            if (solution.loadGraph(args.cache_path, config.enable_multilayer)) {
                loaded_from_cache = true;
                solution.setConfig(config);  // Apply search config
                auto load_end = std::chrono::high_resolution_clock::now();
                build_ms = std::chrono::duration<double>(load_end - load_start).count() * 1000.0;
                std::cerr << "[AblationRunner] Loaded graph from cache in " << build_ms << " ms\n";
            }
        }
        
        if (!loaded_from_cache) {
            std::cerr << "[AblationRunner] Cache invalid or mismatch, will rebuild\n";
        }
    }
    
    // Build if not loaded from cache
    if (!loaded_from_cache) {
        if (args.opt_only) {
            std::cerr << "[AblationRunner] --opt-only requires a valid input cache. Rebuild is not performed.\n";
            return 1;
        }
        auto build_start = std::chrono::high_resolution_clock::now();
        solution.buildWithConfig(args.dim, base_data, config);
        auto build_end = std::chrono::high_resolution_clock::now();
        build_ms = std::chrono::duration<double>(build_end - build_start).count() * 1000.0;
        std::cerr << "[AblationRunner] Built graph in " << build_ms << " ms\n";
        
        // Save to cache if path provided
        if (!args.cache_path.empty()) {
            if (solution.saveGraph(args.cache_path)) {
                std::cerr << "[AblationRunner] Saved graph to cache: " << args.cache_path << "\n";
            }
        }
    }

    // 如果仅做优化并退出
    if (args.opt_only) {
        // 按顺序执行 ONNG / BFS
        if (args.opt_order == "bfs-first") {
            if (config.enable_bfs) solution.applyBFSReorder();
            if (config.enable_onng) solution.optimizeGraphDirectly(false, config.onng_out_degree, config.onng_in_degree, config.onng_min_edges);
        } else { // 默认 onng-first
            if (config.enable_onng) solution.optimizeGraphDirectly(false, config.onng_out_degree, config.onng_in_degree, config.onng_min_edges);
            if (config.enable_bfs) solution.applyBFSReorder();
        }

        std::string out_path = args.cache_out.empty() ? args.cache_path : args.cache_out;
        if (out_path.empty()) {
            std::cerr << "[AblationRunner] No output cache path specified.\n";
            return 1;
        }
        if (!solution.saveGraph(out_path)) {
            std::cerr << "[AblationRunner] Failed to save optimized cache.\n";
            return 1;
        }
        std::cerr << "[AblationRunner] Optimized graph saved to: " << out_path << "\n";
        if (log_file) {
            std::cerr.rdbuf(cerr_buf);
        }
        return 0;
    }
    
    // Prepare result storage
    std::vector<std::vector<int>> predictions(args.query_count, std::vector<int>(args.k, -1));
    
    // Reset distance stats before search
    solution.resetDistanceStats();
    
    // ==================== Visualization Setup ====================
    bool enable_vis_recording = !args.vis_paths_path.empty();
    size_t vis_sample_count = (args.vis_samples == 0 || args.vis_samples > args.query_count) 
                              ? args.query_count : args.vis_samples;
    
    if (enable_vis_recording) {
        solution.clearRecordedPaths();  // 清空之前的路径记录
        solution.enablePathRecording(true);
        std::cerr << "[VIS] Path recording enabled for " << vis_sample_count << " queries\n";
        std::cerr << "[VIS] Strategy: " << args.strategy << ", Output: " << args.vis_paths_path << "\n";
    }
    
    // ==================== Run Search ====================
    auto search_start = std::chrono::high_resolution_clock::now();
    std::cerr << "[SEARCH] start strategy=" << args.strategy
              << " gamma=" << args.gamma
              << " ef=" << args.ef_search
              << " k=" << args.k
              << " queries=" << args.query_count
              << " cache=" << args.cache_path
              << "\n";
    for (size_t i = 0; i < args.query_count; ++i) {
        // 只对采样的查询记录路径
        if (enable_vis_recording && i >= vis_sample_count) {
            solution.enablePathRecording(false);
            enable_vis_recording = false;  // 不再重复禁用
        }
        solution.searchWithK(queries[i], predictions[i].data(), args.k);
    }
    auto search_end = std::chrono::high_resolution_clock::now();
    
    // 禁用路径记录
    solution.enablePathRecording(false);
    
    double search_ms = std::chrono::duration<double>(search_end - search_start).count() * 1000.0;
    double qps = (search_ms > 0) ? ((double)args.query_count / (search_ms / 1000.0)) : 0.0;
    
    // Compute recall
    double recall = compute_recall(predictions, groundtruth, args.k);
    
    // Get distance stats (only meaningful if --count-dist was used)
    double avg_dist_calcs = args.count_dist ? solution.getAverageDistanceCalcsPerSearch() : 0.0;
    
    std::cerr << "[AblationRunner] Search: " << search_ms << " ms, QPS=" << qps 
              << ", Recall@" << args.k << "=" << (recall * 100.0) << "%";
    if (args.count_dist) {
        std::cerr << ", Avg dist calcs=" << avg_dist_calcs;
    }
    std::cerr << "\n";
    
    // ==================== Export Visualization Files ====================
    if (!args.vis_graph_path.empty()) {
        std::cerr << "[VIS] Exporting graph structure to: " << args.vis_graph_path << "\n";
        if (!solution.exportGraphForVisualization(args.vis_graph_path, args.vis_max_nodes)) {
            std::cerr << "[VIS] Warning: Failed to export graph structure\n";
        }
    }
    
    if (!args.vis_paths_path.empty()) {
        std::cerr << "[VIS] Exporting " << solution.getRecordedPathsCount() 
                  << " search paths to: " << args.vis_paths_path << "\n";
        if (!solution.exportSearchPaths(args.vis_paths_path)) {
            std::cerr << "[VIS] Warning: Failed to export search paths\n";
        }
    }
    
    // ==================== Save Graph Cache (if cache-out specified) ====================
    if (!args.cache_out.empty()) {
        std::cerr << "[AblationRunner] Saving graph to cache: " << args.cache_out << "\n";
        if (!solution.saveGraph(args.cache_out)) {
            std::cerr << "[AblationRunner] Warning: Failed to save graph to cache\n";
        } else {
            std::cerr << "[AblationRunner] Graph saved to cache: " << args.cache_out << "\n";
        }
    }
    
    // ==================== Output JSON ====================
    output_json(args, build_ms, search_ms, qps, recall, avg_dist_calcs);
    
    if (log_file) {
        std::cerr.rdbuf(cerr_buf);
    }
    return 0;
}

