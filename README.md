# DataStructure（experiment）：HNSW/ONNG/BFS/SIMD 近邻搜索 + 消融实验

## 项目概述

本项目实现了基于图的近似最近邻（ANN）搜索算法，用于在高维向量空间中快速查找 K 个最近邻。核心实现包括 HNSW 分层图、ONNG 边优化、BFS 内存重排和 SIMD 向量化加速，支持 GloVe/SIFT 等标准数据集的评测。

experiment 分支聚焦于**消融实验/批量实验**，用于系统性评估 ONNG/BFS/SIMD/单层/负内积与搜索策略（gamma / gamma-static / fixed-ef）对性能与召回的影响。

当前入口：

- **`runner.exe`（`AblationRunner.cpp`）**：消融实验 CLI，输出单行 JSON（供脚本解析）
- **`run_experiments.py`**：批量运行实验（支持断点续跑），输出 `experiment_results.pkl/csv`
- **`clean_experiments.py`**：按条件删除 pkl 中部分结果，实现"只重跑某几组且不影响其他结果"
- **`plot_analysis.py`**：实验结果分析可视化，生成多维度图表到 `analysis_plots/`
- **`visualize_search.py`**：搜索路径可视化工具，生成图结构与路径轨迹图

> 只想跑主程序/对拍工具请参考 main 分支的文档

---

## 算法实现

| 算法                     | 说明                                                       | 位置             |
| ------------------------ | ---------------------------------------------------------- | ---------------- |
| **HNSW**           | 分层可导航小世界图（Hierarchical Navigable Small World）   | `MySolution.h` |
| **ONNG**           | 优化导航近邻图（Optimized Navigable Neighbor Graph）边裁剪 | `MySolution.h` |
| **BFS 重排**       | 基于 BFS 的节点内存布局优化，提升缓存命中率                | `MySolution.h` |
| **SIMD**           | 向量化距离计算（SSE/AVX/AVX512 自动检测）                  | `MySolution.h` |
| **Adaptive Gamma** | 自适应搜索策略，动态调整终止阈值                           | `MySolution.h` |
| **Brute Force**    | 暴力精确搜索基线（O(n)）                                   | `Brute.h/.cpp` |

### 实现说明

- **HNSW**：多层跳表结构，上层稀疏用于快速定位，底层稠密保证召回
- **ONNG**：在单层图基础上进行边裁剪，减少冗余边以提升搜索效率
- **BFS 重排**：将图节点按 BFS 顺序重新排列，使相邻访问的节点在内存中连续
- **SIMD**：支持 SSE、AVX、AVX-512 指令集，x86/x64 平台自动启用

---

## 数据集

| 数据集          | 向量数量  | 维度 | 查询数量 | 说明                  |
| --------------- | --------- | ---- | -------- | --------------------- |
| **GloVe** | 1,183,514 | 100  | 10,000   | 词向量数据集，L2 距离 |
| **SIFT**  | 1,000,000 | 128  | 10,000   | 图像特征数据集        |
| **Test**  | 10        | 10   | -        | 调试用小数据集        |

> **注意**：仓库中仅包含 `.bin` 二进制格式文件。`.txt` 文本格式因体积较大（约 800MB）未包含。

---

## 性能结果

### GloVe 数据集（1.18M 向量，100 维，10K 查询）

| 方法                     | 搜索延迟 (ms/q) | Recall@10 | 状态 |
| ------------------------ | --------------- | --------- | ---- |
| Brute Force              | -               | 100.00%   | 基线 |
| HNSW + ONNG + BFS + SIMD | 0.25            | ≥99%     | ✓   |

### SIFT 数据集（1M 向量，128 维，10K 查询）

| 方法                     | 搜索延迟 (ms/q) | Recall@10 | 状态 |
| ------------------------ | --------------- | --------- | ---- |
| Brute Force              | -               | 100.00%   | 基线 |
| HNSW + ONNG + BFS + SIMD | 1.51            | ≥99%     | ✓   |

---

## 测试环境

| 项目               | 配置                                           |
| ------------------ | ---------------------------------------------- |
| **CPU**      | AMD Ryzen 9 7945HX with Radeon Graphics 十六核 |
| **内存**     | 三星 16GB DDR5 5200MHz                         |
| **操作系统** | Windows                                        |
| **编译器**   | GCC（C++17）                                   |

---

## 编译器要求

- **C++17**，且 `<thread>/<mutex>` 可用（GCC/Clang/MSVC 均可）
- **Windows + MinGW-w64**：建议使用 **POSIX 线程变体**，并确保命令带 `-pthread`
- **Python**：3.7+
  - 实验脚本（`run_experiments.py`、`clean_experiments.py`）：仅使用标准库
  - 可视化脚本：需安装 `pip install numpy matplotlib pandas scikit-learn`（可选 `ijson` 用于大文件流式解析）

---

## 项目结构（experiment 分支）

```
.
├── AblationRunner.cpp    # 消融实验 CLI
├── run_experiments.py    # 批量实验脚本
├── clean_experiments.py  # 结果清理工具
├── plot_analysis.py      # 实验结果分析可视化
├── compare_strategies.py # 策略分歧点对比
├── visualize_search.py   # 搜索路径可视化
├── MySolution.h/.cpp     # HNSW + ONNG + BFS + SIMD 实现
├── Brute.h/.cpp          # 暴力解基线
├── BinaryIO.h            # 二进制向量文件读写
├── Config.h              # 配置文件
├── main.cpp / checker.cpp # 主程序/对拍工具
├── analysis_plots/       # 分析图表输出目录
└── output/               # 可视化输出目录
```

### 核心文件

| 文件                           | 说明                                                             |
| ------------------------------ | ---------------------------------------------------------------- |
| `AblationRunner.cpp`         | 消融实验 CLI（参数化 build/search，输出 JSON）                   |
| `run_experiments.py`         | 批量实验脚本（断点续跑、参数 sweep、导出 CSV）                   |
| `clean_experiments.py`       | 按条件清理 `experiment_results.pkl` 并重导出 CSV               |
| `MySolution.h/.cpp`          | HNSW + ONNG + BFS 重排 + SIMD 实现（含 `SolutionConfig`）      |
| `BinaryIO.h`                 | `.bin` 向量文件读写（header-only，magic 为 `VECBIN1\0`）     |
| `Config.h`                   | 数据集路径、默认 query/ans 路径（供 main/checker 使用）          |
| `main.cpp` / `checker.cpp` | 主程序/对拍工具（实验前可用于生成 ground-truth 或 sanity check） |

### 数据分析与可视化脚本

| 文件                      | 说明                                                              |
| ------------------------- | ----------------------------------------------------------------- |
| `plot_analysis.py`      | 实验结果分析可视化（生成 Recall/QPS、M/efC 影响、模块消融等图表） |
| `compare_strategies.py` | 搜索策略分歧点对比分析（比较不同策略访问节点顺序的差异）          |
| `visualize_search.py`   | HNSW 图结构和搜索路径可视化工具（支持完整模式与轻量模式）         |

### 输出目录

| 目录                | 说明                                               |
| ------------------- | -------------------------------------------------- |
| `analysis_plots/` | `plot_analysis.py` 生成的分析图表（fig01~fig13） |
| `output/`         | `visualize_search.py` 生成的搜索路径可视化图     |
| `vis_paths*.json` | runner 导出的搜索路径记录（供可视化脚本使用）      |

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

`run_experiments.py` 用一个 **5 位二进制字符串**（例如 `11110`）来表示一组"构图/搜索开关组合"，并据此命名实验组（`group_name`）。

### 位序定义（从左到右）

| 位序 | 含义             | `0`                                | `1`                                                   |
| ---- | ---------------- | ------------------------------------ | ------------------------------------------------------- |
| bit0 | `single_layer` | 多层图（multilayer）                 | **单层图**（等价于 runner 的 `--single-layer`） |
| bit1 | `onng`         | 关闭 ONNG（等价于 `--no-onng`）    | **开启 ONNG**                                     |
| bit2 | `bfs`          | 关闭 BFS 重排（等价于 `--no-bfs`） | **开启 BFS 重排**                                 |
| bit3 | `simd`         | 关闭 SIMD（等价于 `--no-simd`）    | **开启 SIMD**                                     |
| bit4 | `neg_ip`       | 使用 L2 距离                         | **使用负内积**（等价于 `--use-neg-ip`）         |

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

这套缓存用于"主程序/对拍工具"复用图结构：

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
  `run_experiments.py` 用它把"基础图（无 ONNG/BFS）"逐步变换成目标图（先 ONNG 后 BFS）。

---

## Python：批量实验（`run_experiments.py`）

### 运行

```bash
python run_experiments.py
```

### 结果文件与断点续跑机制（非常关键）

- `experiment_results.pkl`：**断点续跑与去重唯一依据**（脚本启动时读取它重建 done_keys）
- `experiment_results.csv`：仅"最终导出表"（脚本结束时重写生成），删它不影响续跑状态

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

## 数据分析与可视化

### 1）实验结果分析：`plot_analysis.py`

对 `experiment_results.csv` 进行多维度分析并生成图表：

```bash
python plot_analysis.py
```

**生成的图表**（保存到 `analysis_plots/`）：

| 图表                                      | 内容                           |
| ----------------------------------------- | ------------------------------ |
| `fig01_strategy_recall_qps.png`         | 不同搜索策略的 Recall-QPS 曲线 |
| `fig02_module_ablation_recall_qps.png`  | 模块消融对 Recall-QPS 的影响   |
| `fig03_M_impact_recall_qps.png`         | M 参数对 Recall-QPS 的影响     |
| `fig04_efC_impact_recall_qps.png`       | efC 参数对 Recall-QPS 的影响   |
| `fig05_M_efC_heatmap.png`               | M 与 efC 组合的热力图          |
| `fig06_dist_calcs_vs_qps.png`           | 距离计算次数 vs QPS            |
| `fig07_dist_calcs_vs_recall.png`        | 距离计算次数 vs Recall         |
| `fig08_module_qps_improvement.png`      | 各模块的 QPS 提升率            |
| `fig09_strategy_qps_comparison.png`     | 策略间 QPS 对比                |
| `fig10_strategy_improvement_matrix.png` | 策略提升矩阵                   |
| `fig11_params_for_target_recall.png`    | 达成目标召回率所需参数         |
| `fig12_params_by_module.png`            | 按模块分析参数影响             |
| `fig13_param_recall_summary_table.png`  | 参数与召回率汇总表             |

### 2）策略分歧点对比：`compare_strategies.py`

比较两个搜索策略在访问节点顺序上的分歧点：

```bash
# 对比两个策略的搜索路径（位置参数）
python compare_strategies.py vis_paths.json vis_paths_fix.json

# 限制分析的查询数量
python compare_strategies.py vis_paths.json vis_paths_fix.json --max-queries 100

# 指定输出路径和策略名称
python compare_strategies.py vis_paths.json vis_paths_fix.json --name1 Gamma --name2 FixedEF -o output/divergence.png
```

**输出**：分歧点统计和可视化图（默认保存到当前目录 `strategy_divergence.png`）

> 依赖：推荐安装 `ijson`（`pip install ijson`）用于流式解析大 JSON 文件，避免内存问题。

### 3）搜索路径可视化：`visualize_search.py`

可视化 HNSW 图结构和搜索路径，支持两种模式：

**完整模式**（需要 graph.json，内存占用大）：

```bash
python visualize_search.py --graph graph.json --paths vis_paths.json --samples 30 -o output/hnsw
```

**轻量模式**（只需 paths.json，内存占用小，适合大图）：

```bash
python visualize_search.py --paths vis_paths.json --light --samples 100 -o output/hnsw_light
```

**输出内容**：

- 图结构的 2D 降维可视化
- 搜索路径轨迹（显示距离下降过程）
- 路径长度分布直方图
- 改进点标注

**依赖**：

```bash
pip install numpy matplotlib scikit-learn
```

### 4）生成搜索路径文件

runner 可通过 `--vis-paths` 参数导出搜索路径供可视化使用：

```bash
./runner.exe --data=data_o/glove/base.bin --query=query0.bin --groundtruth=ansglove.txt \
    --strategy=gamma --gamma=1.2 --vis-paths=vis_paths.json
```

---

## 常见问题

- **Windows + MinGW-w64**：务必使用 POSIX 线程变体并带 `-pthread`，否则多线程/链接容易踩坑
- **路径**：Windows 下 Python 建议用 raw string（如 `r"I:\graph_cache"`），或使用正斜杠
- **结果不更新**：删了 `experiment_results.csv` 但没删 `experiment_results.pkl` → 脚本仍会跳过已完成任务
- **缓存污染**：清理顺序建议：
  1. 结果逻辑错：先用 `clean_experiments.py` 清理 `experiment_results.pkl`
  2. 构图逻辑改动或怀疑复用旧图：清理实验缓存

---

## 快速开始（实验链路）

```bash
# 1) 编译 runner
g++ -std=c++17 -O2 -pthread -o runner.exe AblationRunner.cpp MySolution.cpp Brute.cpp

# 2) 修改 run_experiments.py 顶部常量（DATA_PATH/QUERY_PATH/GROUNDTRUTH_PATH/CACHE_DIR）

# 3) 跑实验（断点续跑）
python run_experiments.py

# 4) 分析实验结果（生成图表到 analysis_plots/）
python plot_analysis.py

# 5) [可选] 可视化搜索路径
./runner.exe --data=data_o/glove/base.bin --query=query0.bin --groundtruth=ansglove.txt \
    --strategy=gamma --gamma=1.2 --vis-paths=vis_paths.json
python visualize_search.py --paths vis_paths.json --light --samples 50 -o output/search_vis
```

---

## 依赖说明

### C++ 依赖

本项目 C++ 部分仅使用标准库，无需额外第三方依赖：

| 依赖          | 用途              | 说明                                                      |
| ------------- | ----------------- | --------------------------------------------------------- |
| C++ 标准库    | 容器、多线程、I/O | `<vector>`, `<thread>`, `<mutex>`, `<fstream>` 等 |
| SIMD 内建函数 | 向量化距离计算    | `<immintrin.h>`（x86/x64 平台自动可用）                 |

### Python 依赖

| 依赖          | 用途              | 安装命令                     |
| ------------- | ----------------- | ---------------------------- |
| numpy         | 数值计算          | `pip install numpy`        |
| matplotlib    | 图表绑制          | `pip install matplotlib`   |
| pandas        | 数据分析          | `pip install pandas`       |
| scikit-learn  | 降维（PCA/t-SNE） | `pip install scikit-learn` |
| ijson（可选） | 大 JSON 流式解析  | `pip install ijson`        |

---

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 数据集链接

- **GloVe 数据集**：[Stanford NLP](https://nlp.stanford.edu/projects/glove/)
- **SIFT 数据集**：[Corpus-Texmex](http://corpus-texmex.irisa.fr/)[an - NGT](https://github.com/yahoojapan/NGT)
