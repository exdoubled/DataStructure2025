# PJVS：HNSW/ONNG/BFS/SIMD 近邻搜索 + 消融实验工具链

本仓库是一个 ANN（Approximate Nearest Neighbor）检索项目，包含：

- `MySolution`：HNSW 图索引 + ONNG 图优化 + BFS 内存重排 + SIMD（均可开关）
- 三个 C++ 可执行入口：`main` / `checker` / `runner(AblationRunner)`
- 两个 Python 脚本：批量实验 `run_experiments.py` 与结果清理 `clean_experiments.py`

---

## 环境要求

- **C++**：C++17；`<thread>/<mutex>` 可用
  - GCC ≥ 9 / Clang ≥ 10 / MSVC ≥ 19.28
  - **Windows + MinGW-w64**：建议使用 **POSIX 线程变体**（如 MSYS2 UCRT64 / x86_64-posix-seh），并确保命令带 `-pthread`
- **Python**：3.7+（脚本仅使用标准库）

---

## 项目结构

| 路径 | 说明 |
|---|---|
| `MySolution.h/.cpp` | HNSW + ONNG + BFS 重排 + SIMD 的实现（含 `SolutionConfig`） |
| `Brute.h/.cpp` | 暴力解（对拍/生成答案等） |
| `BinaryIO.h` | `.bin` 向量文件、图缓存的二进制读写（header-only） |
| `Config.h` | 数据集路径、默认 query/ans 路径、`main/checker` 的图缓存路径 |
| `main.cpp` | 主程序：生成 query、跑 solution/brute、保存/加载图缓存 |
| `checker.cpp` | 对拍：用 brute 校验 solution（可配合图缓存） |
| `AblationRunner.cpp` | 消融实验 CLI：参数化构建/搜索，输出单行 JSON |
| `run_experiments.py` | 批量跑消融实验（断点续跑），输出 `experiment_results.pkl/csv` |
| `clean_experiments.py` | 按条件删除 pkl 中部分结果并重导出 csv（用于“只重跑某几组”） |

---

## 数据与路径

### `main/checker` 的数据集与默认路径（`Config.h`）

`Config.h` 中有默认设置：

- 数据集：
  - GloVe：`./data_o/glove/base.txt`（case=0，dim=100）
  - SIFT：`./data_o/sift/base.txt`（case=1，dim=128）
  - Test：`./test.txt`（case=2，小规模）
- 默认 query/ans：
  - `QUERYFILEPATH`（默认 `query0.txt`）
  - `ANSFILEPATH`（默认 `ansglove.txt`）
- `main/checker` 的**单文件**图缓存：
  - `GRAPH_CACHE_GLOVE` / `GRAPH_CACHE_SIFT` / `GRAPH_CACHE_TEST`

### `.txt` 与 `.bin`（自动生成/优先读取）

`main.cpp` 与 `checker.cpp` 的加载逻辑是：

- 优先读取与 `base.txt` 同目录的 `base.bin`
- 优先读取与 `QUERYFILEPATH` 同名的 `.bin`（如 `query0.bin`）
- 如果 `.bin` 不存在，会从 `.txt` 读取并自动写回 `.bin`（格式由 `BinaryIO.h` 定义，magic 为 `VECBIN1\0`）

---

## 图缓存（两套体系）

### 1）`main/checker`：单文件图缓存（由 `Config.h` 控制）

- 用途：主程序/对拍时避免重复构图
- 典型路径：`./data_o/glove/graph.bin`
- 控制开关：
  - `--save-graph[=path]`：构建后保存
  - `--load-graph[=path]`：从缓存加载（跳过构建）

### 2）`run_experiments.py`：参数化缓存目录（大量文件）

`run_experiments.py` 使用独立缓存目录（默认）：

- `CACHE_DIR = r"I:\graph_cache"`

它会按 build 配置（含哈希后缀）生成大量 cache 文件用于实验复用。

> 如果你修改了 C++ 内部构建逻辑，且怀疑旧缓存图污染结果：除了清理 `experiment_results.pkl` 外，可能还需要清理 `I:\graph_cache`（实验缓存）中对应文件。

---

## C++：主程序 `main`

### 编译

**Windows（PowerShell / MinGW-w64）**

```pwsh
g++ -std=c++17 -O2 -Wall -Wextra -pthread main.cpp MySolution.cpp Brute.cpp -o main.exe
```

**Linux / macOS**

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pthread main.cpp MySolution.cpp Brute.cpp -o main
```

### 用法示例

```bash
# 选择数据集：0=GloVe, 1=SIFT, 2=Test
./main.exe 0

# 生成 query（输出到 QUERYFILEPATH，并可能生成 query0.bin）
./main.exe 0 --gen-queries

# 跑算法：solution（MySolution）或 brute（暴力）
./main.exe 0 --algo=solution
./main.exe 0 --algo=brute

# 保存/加载 main/checker 的单文件图缓存
./main.exe 0 --save-graph
./main.exe 0 --load-graph
```

---

## C++：对拍 `checker`

### 编译

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pthread checker.cpp MySolution.cpp Brute.cpp -o checker.exe
```

### 用法示例

```bash
# 对拍（建议配合 --load-graph 加速）
./checker.exe 0 --load-graph --algo=solution

# 指定答案文件 & 只对拍前 N 条
./checker.exe 0 --ans=ansglove.txt --first=100 --algo=solution
```

---

## C++：消融实验 CLI `runner`（`AblationRunner.cpp`）

### 编译（建议输出名为 runner.exe）

`run_experiments.py` 默认 `RUNNER_PATH="runner"`，会自动尝试补 `.exe`，因此推荐命名为 `runner.exe`：

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pthread -o runner.exe AblationRunner.cpp MySolution.cpp Brute.cpp
```

### 关键参数（以 `AblationRunner.cpp` 的 `--help` 为准）

必需：

- `--data=<path>`
- `--query=<path>`
- `--groundtruth=<path>`

常用可选项：

- I/O：`--cache-path=<path>`
- Build：`--M=`, `--efC=`, `--no-onng`, `--no-bfs`, `--no-simd`, `--single-layer`, `--use-neg-ip`
- Search：`--strategy=gamma|gamma-static|fixed`，配合 `--gamma=` / `--ef=`
- Debug：`--count-dist`
- Opt-only：`--opt-only` + `--cache-out=` + `--opt-order=onng-first|bfs-first`

### 示例

```bash
# 动态 gamma
./runner.exe --data=data_o/glove/base.bin --query=query0.bin --groundtruth=ansglove.txt --strategy=gamma --gamma=0.19

# 固定 ef
./runner.exe --data=data_o/glove/base.bin --query=query0.bin --groundtruth=ansglove.txt --strategy=fixed --ef=200
```

输出：单行 JSON（最后一行）。

---

## Python：批量实验 `run_experiments.py`

### 配置位置（脚本顶部常量）

你通常需要检查/修改：

- `DATA_PATH`（默认 `data_o/glove/base.bin`）
- `QUERY_PATH`（默认 `query0.bin`）
- `GROUNDTRUTH_PATH`（默认 `ansglove.txt`）
- `RUNNER_PATH`（默认 `runner`）
- `CACHE_DIR`（默认 `I:\graph_cache`，建议放 SSD）

### 运行

```bash
python run_experiments.py
```

### 结果文件与断点续跑机制（非常关键）

- `experiment_results.pkl`：**断点续跑与去重唯一依据**（脚本启动会读取它重建 done_keys）
- `experiment_results.csv`：仅“最终导出表”（脚本结束时重写生成），删它不影响续跑状态

---

## 只重跑某几组且不影响其他组：`clean_experiments.py`

当你发现某些组跑错了，应当清理 `experiment_results.pkl`（而不是手改 CSV）：

```bash
# 预览：不写文件
python clean_experiments.py --code-bits 11110 --dry-run --show-groups

# 真删：只删 code_bits=11110 的结果（会自动备份 pkl/csv）
python clean_experiments.py --code-bits 11110

# 更细：只删某策略/参数范围
python clean_experiments.py --code-bits 11110 --strategy fixed --param-min 200 --param-max 600

# 自定义条件（变量 r 是一条结果；可用 re）
python clean_experiments.py --expr "r.group_name.startswith('11110_') and r.recall10 < 0.8"
```

清理后重新编译 C++（如 `runner.exe`），再跑 `run_experiments.py` 即会自动补跑被删掉的部分。

---

## MySolution 的运行时配置（`SolutionConfig`）

`MySolution.h` 中 `SolutionConfig` 支持运行时开关/调参（消融实验 CLI 也是基于它）：

- Build：`M / ef_construction / enable_onng / enable_bfs / enable_simd / enable_multilayer / use_negative_inner_product`
- Search：`search_method`（`DYNAMIC_GAMMA / STATIC_GAMMA / FIXED_EF`）、`gamma / ef_search / k`
- Debug：`count_distance_computation`

---

## 常见问题

- **Windows + MinGW-w64**：务必使用 POSIX 线程变体并带 `-pthread`，否则多线程/链接容易踩坑
- **路径**：Windows 下 Python 建议用 raw string（如 `r"I:\graph_cache"`），或使用正斜杠
- **缓存污染排查顺序**：
  1. 结果逻辑错：先用 `clean_experiments.py` 清理 `experiment_results.pkl`
  2. 构图逻辑改动或怀疑复用旧图：清理实验缓存 `I:\graph_cache` 或主程序缓存 `data_o/*/graph.bin`

