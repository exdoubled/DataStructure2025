# DataStructure（experiment）：HNSW/ONNG/BFS/SIMD 近邻搜索 + 消融实验

experiment 分支聚焦于**消融实验/批量实验**，用于系统性评估 ONNG/BFS/SIMD/单层/负内积与搜索策略（gamma / gamma-static / fixed-ef）对性能与召回的影响。

当前入口：

- **`runner.exe`（`AblationRunner.cpp`）**：消融实验 CLI，输出单行 JSON（供脚本解析）
- **`run_experiments.py`**：批量运行实验（支持断点续跑），输出 `experiment_results.pkl/csv`
- **`clean_experiments.py`**：按条件删除 pkl 中部分结果，实现“只重跑某几组且不影响其他结果”

> 只想跑主程序/对拍工具请参考 main 分支的文档

---

## 编译器要求

- **C++17**，且 `<thread>/<mutex>` 可用（GCC/Clang/MSVC 均可）
- **Windows + MinGW-w64**：建议使用 **POSIX 线程变体**，并确保命令带 `-pthread`
- **Python**：3.7+（脚本仅使用标准库）

---

## 项目结构（experiment 分支）

| 文件 | 说明 |
|---|---|
| `AblationRunner.cpp` | 消融实验 CLI（参数化 build/search，输出 JSON） |
| `run_experiments.py` | 批量实验脚本（断点续跑、参数 sweep、导出 CSV） |
| `clean_experiments.py` | 按条件清理 `experiment_results.pkl` 并重导出 CSV |
| `MySolution.h/.cpp` | HNSW + ONNG + BFS 重排 + SIMD 实现（含 `SolutionConfig`） |
| `BinaryIO.h` | `.bin` 向量文件读写（header-only，magic 为 `VECBIN1\0`） |
| `Config.h` | 数据集路径、默认 query/ans 路径（供 main/checker 使用） |
| `main.cpp` / `checker.cpp` | 主程序/对拍工具（实验前可用于生成 ground-truth 或 sanity check） |

---

## 数据与路径

### 实验脚本（`run_experiments.py`）的配置常量

脚本顶部常量（你通常需要按机器修改）：

- `DATA_PATH`（默认 `data_o/glove/base.bin`）
- `QUERY_PATH`（默认 `query0.bin`）
- `GROUNDTRUTH_PATH`（默认 `ansglove.txt`）
- `RUNNER_PATH`（默认 `runner`，Windows 会自动尝试 `runner.exe`）
- `CACHE_DIR`（默认 `I:\graph_cache`，建议放 SSD 且空间足够）

### `.txt` 与 `.bin`

- `main/checker/runner` 相关工具都会优先使用 `.bin`（不存在则可能从 `.txt` 读取并写回 `.bin`）
- `BinaryIO.h` 定义的向量二进制格式 magic 为 `VECBIN1\0`

---

## 实验分组：5 位二进制码（`code_bits`）含义

`run_experiments.py` 用一个 **5 位二进制字符串**（例如 `11110`）来表示一组“构图/搜索开关组合”，并据此命名实验组（`group_name`）。

### 位序定义（从左到右）

| 位序 | 含义 | `0` | `1` |
|---|---|---|---|
| bit0 | `single_layer` | 多层图（multilayer） | **单层图**（等价于 runner 的 `--single-layer`） |
| bit1 | `onng` | 关闭 ONNG（等价于 `--no-onng`） | **开启 ONNG** |
| bit2 | `bfs` | 关闭 BFS 重排（等价于 `--no-bfs`） | **开启 BFS 重排** |
| bit3 | `simd` | 关闭 SIMD（等价于 `--no-simd`） | **开启 SIMD** |
| bit4 | `neg_ip` | 使用 L2 距离 | **使用负内积**（等价于 `--use-neg-ip`） |

### `group_name` 命名规则

`group_name` 由 `code_bits` 前缀 + 策略后缀组成：

- `xxxxx_gamma_dyn`：动态 gamma（runner `--strategy=gamma`）
- `xxxxx_gamma_static`：静态 gamma（runner `--strategy=gamma-static`）
- `xxxxx_fixed`：固定 ef（runner `--strategy=fixed`）

例如：

- `11110_fixed`：单层+ONNG+BFS+SIMD，L2 距离，固定 ef
- `00000_gamma_dyn`：多层、无 ONNG、无 BFS、无 SIMD、L2，动态 gamma

---

## 图缓存（两套体系）

### 1）`main/checker`：单文件图缓存（由 `Config.h` 控制）

这套缓存用于“主程序/对拍工具”复用图结构：

- `--save-graph[=path]`：构建后保存
- `--load-graph[=path]`：从缓存加载（跳过构建）

### 2）`run_experiments.py`：参数化缓存目录（大量文件）

实验脚本会为不同 build 配置生成不同 cache 文件（含参数+哈希），存放在：

- `CACHE_DIR = r"I:\graph_cache"`

> 如果你修改了构图逻辑且怀疑旧缓存污染实验结果：除了清理 `experiment_results.pkl` 外，可能还需要清理 `I:\graph_cache` 中对应缓存文件。

---

## C++：消融实验 CLI（`AblationRunner.cpp`）

### 编译（推荐输出名为 `runner.exe`）

`run_experiments.py` 默认 `RUNNER_PATH="runner"`，因此推荐：

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pthread -o runner.exe AblationRunner.cpp MySolution.cpp Brute.cpp
```

### 关键参数（以 `--help` 为准）

必需：

- `--data=<path>`
- `--query=<path>`
- `--groundtruth=<path>`

常用可选项（部分）：

- `--cache-path=<path>`：图缓存文件（存在则加载；构建后写回）
- build：`--M=`, `--efC=`, `--no-onng`, `--no-bfs`, `--no-simd`, `--single-layer`, `--use-neg-ip`
- search：`--strategy=gamma|gamma-static|fixed` + `--gamma=` / `--ef=`
- debug：`--count-dist`
- opt-only：`--opt-only` + `--cache-out=` + `--opt-order=onng-first|bfs-first`

---

## 参数含义速查（实验最常用）

### Build 参数（构图）

- **`M`**（`--M=<int>`）：图中每个点的最大连接数（越大通常召回更高但构建更慢/内存更大）
- **`efC`**（`--efC=<int>`）：构建时的候选队列大小（越大通常图质量更好但构建更慢）
- **ONNG 参数**（`--onng-out/in/min`）：ONNG 优化相关的出度/入度/最小边数  
  在 `run_experiments.py` 中，这三个值会根据 `M` 自动派生（避免参数全笛卡尔爆炸）。

### Search 参数（搜索）

- **`k`**（`--k=<int>`）：Top-K（脚本默认 `DEFAULT_K=10`）
- **`strategy`**（`--strategy=<gamma|gamma-static|fixed>`）：
  - `gamma`：动态 gamma（脚本里叫 `gamma_dyn`）
  - `gamma-static`：静态 gamma（脚本里叫 `gamma_static`）
  - `fixed`：固定 ef（脚本里叫 `fixed`）
- **`gamma`**（`--gamma=<float>`）：gamma 策略的参数（脚本扫描 `GAMMA_RANGE`）
- **`ef`**（`--ef=<int>`）：fixed 策略的参数（脚本扫描 `EF_RANGE`）

### Debug/统计

- **`--count-dist`**：统计距离计算次数（更慢，但会在结果里给出 `avg_dist_calcs` 等统计）
  - `run_experiments.py` 对每个参数点会跑两次：一次正常计时（不带 `--count-dist`），一次统计（带 `--count-dist`），再合并为一条结果记录。

### 缓存与 `opt-only`（脚本如何复用图）

- **`--cache-path=<path>`**：runner 的缓存文件路径（存在就加载；否则构建后写回）
- **`--opt-only`**：只对已有 cache 做 ONNG/BFS 优化并保存，不执行搜索  
  `run_experiments.py` 用它把“基础图（无 ONNG/BFS）”逐步变换成目标图（先 ONNG 后 BFS）。

---

## Python：批量实验（`run_experiments.py`）

### 运行

```bash
python run_experiments.py
```

### 结果文件与断点续跑机制（非常关键）

- `experiment_results.pkl`：**断点续跑与去重唯一依据**（脚本启动时读取它重建 done_keys）
- `experiment_results.csv`：仅“最终导出表”（脚本结束时重写生成），删它不影响续跑状态

---

## 只重跑某几组且不影响其他结果：`clean_experiments.py`

当你发现某几组结果因为内部逻辑错误跑坏了，不要手改 CSV，应当清理 pkl：

```bash
# 预览（不写文件）
python clean_experiments.py --code-bits 11110 --dry-run --show-groups

# 真删：只删 code_bits=11110 的结果（默认会备份 pkl/csv）
python clean_experiments.py --code-bits 11110

# 更细：只删某策略/参数范围
python clean_experiments.py --code-bits 11110 --strategy fixed --param-min 200 --param-max 600

# 自定义条件（变量 r 是一条结果；可用 re）
python clean_experiments.py --expr "r.group_name.startswith('11110_') and r.recall10 < 0.8"
```

清理后重新编译 `runner.exe`（或你改动的 C++ 文件），再跑 `run_experiments.py` 即会自动补跑被删掉的部分。

---

## 常见问题

- **Windows + MinGW-w64**：务必使用 POSIX 线程变体并带 `-pthread`，否则多线程/链接容易踩坑
- **路径**：Windows 下 Python 建议用 raw string（如 `r"I:\graph_cache"`），或使用正斜杠
- **结果不更新**：删了 `experiment_results.csv` 但没删 `experiment_results.pkl` → 脚本仍会跳过已完成任务
- **缓存污染**：清理顺序建议：
  1. 结果逻辑错：先用 `clean_experiments.py` 清理 `experiment_results.pkl`
  2. 构图逻辑改动或怀疑复用旧图：清理 `I:\graph_cache`（实验缓存）

---

## 快速开始（实验链路）

```bash
# 1) 编译 runner
g++ -std=c++17 -O2 -pthread -o runner.exe AblationRunner.cpp MySolution.cpp Brute.cpp

# 2) 修改 run_experiments.py 顶部常量（DATA_PATH/QUERY_PATH/GROUNDTRUTH_PATH/CACHE_DIR）

# 3) 跑实验（断点续跑）
python run_experiments.py
```


