# Build & Run Guide

## 编译器要求

- 支持 C++17 标准，且完整支持 `<thread>` / `<mutex>` 的现代编译器（推荐：GCC ≥ 9，Clang ≥ 10，MSVC ≥ 19.28）。
- Windows 上若使用 MinGW-w64，请选择 **POSIX 线程变体**（如 MSYS2 UCRT64、x86_64-posix-seh 等），并确保 `g++` 支持并实际传入 `-pthread` 选项；若使用 MSVC，则按默认多线程运行时即可。




## 项目结构概览

主要文件和目录说明：

| 文件 | 说明 |
|------|------|
| `main.cpp` | 主程序入口，负责加载数据集、构建索引、搜索并生成 `ans.txt` |
| `checker.cpp` | 对拍程序，使用暴力解与 `MySolution` 的结果进行比对 |
| `AblationRunner.cpp` | **消融实验 CLI 工具**，支持参数化构建和搜索，输出 JSON 结果 |
| `run_experiments.py` | **Python 自动化脚本**，批量运行消融实验并输出 CSV |
| `MySolution.h` / `.cpp` | 近邻搜索实现，基于 HNSW + ONNG + 多线程 + SIMD |
| `Brute.h` / `.cpp` | 朴素暴力搜索实现，用于生成标准答案和对拍 |
| `BinaryIO.h` | 二进制读写工具，负责向量数据和**图缓存**的读写 |
| `Config.h` | 配置文件，包含数据集路径和图缓存路径常量 |


## 数据集与文件路径

相关路径在 `Config.h` 中配置：

```cpp
// 数据集路径
#define GLOVEPATH "./data_o/glove/base.txt"   // dataset_case = 0, dim=100
#define SIFTPATH  "./data_o/sift/base.txt"    // dataset_case = 1, dim=128
#define TESTPATH  "./test.txt"                // dataset_case = 2, 小规模测试

// 图结构缓存路径
#define GRAPH_CACHE_GLOVE "./data_o/glove/graph.bin"
#define GRAPH_CACHE_SIFT  "./data_o/sift/graph.bin"
#define GRAPH_CACHE_TEST  "./graph_test.bin"
```

程序会优先尝试从与 `base.txt` 同目录的 `base.bin` 读取二进制数据；若不存在则从 `base.txt` 读取并自动写出 `base.bin` 以加速后续运行


## 主程序用法（`main.cpp`）

### 编译 main

**Windows（MinGW-w64 POSIX）**

```pwsh
g++ -std=c++17 -O2 -Wall -Wextra -pthread main.cpp MySolution.cpp Brute.cpp -o main.exe
```

**Linux / macOS**

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pthread main.cpp MySolution.cpp Brute.cpp -o main
```

### 基本参数

| 参数 | 说明 |
|------|------|
| `0` / `1` / `2` | 数据集选择：GloVe / SIFT / Test |
| `--algo=solution\|brute` | 算法选择，默认 `solution` |
| `--gen-queries` | 生成随机查询文件 |
| `--save-graph[=path]` | **构建后保存图缓存** |
| `--load-graph[=path]` | **从缓存加载图（跳过构建）** |
| `--bin` | 强制仅使用二进制文件 |
| `--query=<path>` | 指定查询文件路径 |

### 常用命令示例

```bash
# 构建索引并搜索
./main.exe 0 --algo=solution

# 构建并保存图缓存（首次运行）
./main.exe 0 --algo=solution --save-graph

# 从缓存加载图（后续运行，跳过构建）
./main.exe 0 --algo=solution --load-graph

# 使用暴力算法生成标准答案
./main.exe 0 --algo=brute

# 生成随机查询
./main.exe 0 --gen-queries
```


## 对拍程序用法（`checker.cpp`）

### 编译 checker

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pthread checker.cpp MySolution.cpp Brute.cpp -o checker.exe
```

### 基本参数

| 参数 | 说明 |
|------|------|
| `0` / `1` / `2` | 数据集选择 |
| `--ans=<path>` | 指定答案文件，默认 `ANSFILEPATH` |
| `--algo=solution\|brute` | 被测算法，默认 `solution` |
| `--first=<N>` | 只检查前 N 条查询 |
| `--save-graph[=path]` | 构建后保存图缓存 |
| `--load-graph[=path]` | 从缓存加载图 |

### 常用对拍流程

```bash
# 1. 生成答案
./main.exe 0 --algo=solution

# 2. 对拍（从缓存加载图加速）
./checker.exe 0 --load-graph --ans="./ans.txt" --algo=solution

# 3. 只检查前 100 条
./checker.exe 0 --load-graph --first=100
```


## 图结构缓存

图缓存功能可以保存构建好的 HNSW 图结构，避免重复构建：

```bash
# 首次运行：构建并保存
./main.exe 0 --save-graph

# 后续运行：直接加载（跳过构建）
./main.exe 0 --load-graph

# 指定自定义缓存路径
./main.exe 0 --save-graph=my_graph.bin
./main.exe 0 --load-graph=my_graph.bin
```

默认缓存路径在 `Config.h` 中定义：
- GloVe: `./data_o/glove/graph.bin`
- SIFT: `./data_o/sift/graph.bin`


## 消融实验工具（`AblationRunner.cpp`）

用于系统性地评估各组件（ONNG、BFS、SIMD、搜索策略）对性能的影响。

### 编译

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pthread -o runner.exe AblationRunner.cpp MySolution.cpp Brute.cpp
```

### 参数说明

| 类别 | 参数 | 说明 |
|------|------|------|
| **I/O** | `--data=<path>` | 基向量文件（必需） |
| | `--query=<path>` | 查询文件（必需） |
| | `--groundtruth=<path>` | Ground truth 文件（必需） |
| | `--cache-path=<path>` | 图缓存路径 |
| **Build** | `--M=<int>` | 最大连接数，默认 96 |
| | `--efC=<int>` | 构建 ef，默认 400 |
| | `--onng-out/in/min=<int>` | ONNG 参数 |
| | `--no-onng` | 禁用 ONNG 优化 |
| | `--no-bfs` | 禁用 BFS 内存重排 |
| **Search** | `--k=<int>` | Top-K，默认 10 |
| | `--strategy=<gamma\|fixed>` | 搜索策略 |
| | `--gamma=<float>` | Adaptive Gamma 值 |
| | `--ef=<int>` | Fixed EF 值 |
| | `--no-simd` | 禁用 SIMD 加速 |
| **Debug** | `--count-dist` | 统计距离计算次数（影响 QPS） |

### 使用示例

```bash
# 基准测试
./runner.exe --data=data_o/glove/base.bin --query=query0.bin --groundtruth=ansglove.txt --strategy=gamma --gamma=0.19

# 禁用 ONNG 的消融测试
./runner.exe --data=data_o/glove/base.bin --query=query0.bin --groundtruth=ansglove.txt --no-onng --strategy=gamma

# 使用图缓存加速
./runner.exe --data=data_o/glove/base.bin --query=query0.bin --groundtruth=ansglove.txt --cache-path=graph_cache/my_graph.bin
```

### 输出格式

输出单行 JSON，包含所有参数和指标：

```json
{
  "M": 96, "ef_construction": 400,
  "enable_onng": true, "enable_bfs": true, "enable_simd": true,
  "strategy": "gamma", "gamma": 0.19, "k": 10,
  "build_ms": 30000.0, "search_ms": 2000.0,
  "qps": 5000.0, "recall": 0.95, "avg_dist_calcs": 500.0
}
```


## Python 自动化实验（`run_experiments.py`）

批量运行消融实验，自动管理图缓存，输出 CSV 结果。

### 依赖

Python 3.7+（仅使用标准库）

### 配置

编辑脚本顶部的常量：

```python
DATA_PATH = "data_o/glove/base.bin"
QUERY_PATH = "query0.bin"
GROUNDTRUTH_PATH = "ansglove.txt"
RUNNER_PATH = "runner"  # 或 runner.exe
```

### 实验组设计

| 组 | 名称 | 配置 | 目标 |
|---|---|---|---|
| **A** | Baseline | M=96, ONNG✓, BFS✓, SIMD✓ | 基准 Pareto 曲线 |
| **B** | No ONNG | M=96, **ONNG✗**, BFS✓ | 评估 ONNG 效果 |
| **C** | No BFS | M=96, ONNG✓, **BFS✗** | 评估内存局部性效果 |
| **D** | No SIMD | M=96, ONNG✓, BFS✓, **SIMD✗** | 评估 SIMD 加速效果 |
| **E** | M Sweep | M∈[24,48,96,128,200,256] | 评估图密度影响 |

### 运行

```bash
# 1. 确保 runner 已编译
g++ -std=c++17 -O2 -pthread -o runner.exe AblationRunner.cpp MySolution.cpp Brute.cpp

# 2. 运行实验
python run_experiments.py

# 3. 结果保存在 experiment_results.csv
```

### 输出 CSV 格式

| 列 | 说明 |
|---|---|
| `group_name` | 实验组名称 |
| `M`, `efC`, `onng`, `bfs`, `simd` | 构建参数 |
| `strategy`, `param_val` | 搜索参数（gamma 值或 ef 值） |
| `recall10` | Recall@10 |
| `qps` | 每秒查询数 |
| `ms_per_query` | 单次查询延迟 |
| `dist_calcs` | 平均距离计算次数 |
| `build_time_ms` | 构建/加载时间 |


## SolutionConfig 配置系统

`MySolution` 支持通过 `SolutionConfig` 结构体进行运行时配置：

```cpp
struct SolutionConfig {
    // Build 参数
    size_t M = 96;                    // 最大连接数
    size_t ef_construction = 400;     // 构建时 ef
    bool enable_onng = true;          // ONNG 图优化
    size_t onng_out_degree = 96;      // ONNG 出度
    size_t onng_in_degree = 144;      // ONNG 入度
    size_t onng_min_edges = 64;       // ONNG 最小边数
    bool enable_bfs = true;           // BFS 内存重排
    bool enable_simd = true;          // SIMD 加速
    
    // Search 参数
    SearchMethod search_method = SearchMethod::ADAPTIVE_GAMMA;
    float gamma = 0.19f;              // Adaptive Gamma 值
    size_t ef_search = 704;           // Fixed EF 值
    size_t k = 10;                    // Top-K
    
    // Debug 参数
    bool count_distance_computation = false;  // 距离计算统计
};
```

### 搜索策略

| 策略 | 说明 |
|------|------|
| `ADAPTIVE_GAMMA` | 自适应 gamma 搜索，根据候选队列拥塞动态调整终止条件 |
| `FIXED_EF` | 标准 HNSW 固定 ef 搜索 |

### 使用示例

```cpp
Solution solution;
SolutionConfig config;
config.M = 96;
config.enable_onng = true;
config.search_method = SearchMethod::ADAPTIVE_GAMMA;
config.gamma = 0.19f;

solution.buildWithConfig(dim, base_data, config);
solution.searchWithK(query, result, 10);
```


## 多线程相关

- 项目完全依赖 C++ 标准库（`<thread>` / `<mutex>` 等），不需要额外第三方线程库
- 对于 GCC/Clang，请务必添加 `-pthread`
- Windows MinGW-w64 请选择 **POSIX 线程变体**


## SIMD 优化相关

SIMD 加速仅在 x86/x64 平台上启用；其他架构会自动回退到纯标量实现。

`MySolution` 的 L2 距离计算实现了 **运行时指令集分发**：

- 支持：Scalar / SSE / AVX / AVX-512
- 启动后通过 CPUID + XCR0 自动检测并选择最优实现
- 可通过 `config.enable_simd = false` 运行时强制使用标量

编译期禁用 SIMD：

```bash
# 强制纯标量
g++ -std=c++17 -O2 -pthread -DSOLUTION_PLATFORM_X86=0 main.cpp MySolution.cpp Brute.cpp -o main.exe
```


## 快速开始

```bash
# 1. 编译所有工具
g++ -std=c++17 -O2 -pthread -o main.exe main.cpp MySolution.cpp Brute.cpp
g++ -std=c++17 -O2 -pthread -o checker.exe checker.cpp MySolution.cpp Brute.cpp
g++ -std=c++17 -O2 -pthread -o runner.exe AblationRunner.cpp MySolution.cpp Brute.cpp

# 2. 生成查询和暴力答案
./main.exe 0 --gen-queries
./main.exe 0 --algo=brute

# 3. 构建索引并保存缓存
./main.exe 0 --algo=solution --save-graph

# 4. 对拍验证
./checker.exe 0 --load-graph --algo=solution

# 5. 运行消融实验
python run_experiments.py
```
