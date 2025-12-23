# DataStructure（main）：HNSW/ONNG/BFS/SIMD 近邻搜索

## 项目概述

本项目实现了基于图的近似最近邻（ANN）搜索算法，用于在高维向量空间中快速查找 K 个最近邻。核心实现包括 HNSW 分层图、ONNG 边优化、BFS 内存重排和 SIMD 向量化加速，支持 GloVe/SIFT 等标准数据集的评测。

主分支聚焦于**主程序**与**对拍/评估工具**：

- **`main.cpp`**：加载 base/query，构建索引并检索，写出答案文件（默认 `ANSFILEPATH`）
- **`checker.cpp`**：读取 ground-truth 答案文件，评估/对拍 `solution` 或 `brute`

> 批量实验/断点续跑等内容请参考 experiment 分支文档

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

- C++17，且 `<thread>/<mutex>` 可用（GCC/Clang/MSVC 均可）
- Windows 使用 MinGW-w64 时，建议使用 **POSIX 线程变体**，并确保编译命令带 `-pthread`

---

## 项目结构（主分支）

```
.
├── main.cpp            # 主程序：构建索引、执行搜索、输出结果
├── checker.cpp         # 对拍/评估工具
├── MySolution.h/.cpp   # HNSW + ONNG + BFS + SIMD 实现
├── Brute.h/.cpp        # 暴力解基线
├── BinaryIO.h          # 二进制向量文件读写
├── Config.h            # 配置文件（需自行创建）
└── ConfigExample.h     # 配置示例

```

| 文件                  | 说明                                                                                |
| --------------------- | ----------------------------------------------------------------------------------- |
| `main.cpp`          | 主程序：生成 query、跑 `solution/brute`、写出答案文件                             |
| `checker.cpp`       | 对拍/评估：读取答案文件（带 header 的文本格式），计算 top1/recall/耗时等            |
| `MySolution.h/.cpp` | HNSW + ONNG + BFS 重排 + SIMD 的实现（含 `SolutionConfig`）                       |
| `Brute.h/.cpp`      | 暴力解└──                                                                        |
| `BinaryIO.h`        | `.bin` 向量文件读写（header-only，magic 为 `VECBIN1\0`）                        |
| `Config.h`          | 数据集路径、默认 query/ans 文件名（**需自行创建**，参考 `ConfigExample.h`） |
| `ConfigExample.h`   | 配置文件示例，复制为 `Config.h` 后按需修改路径                                    |

---

## 数据与路径（`Config.h`）

`Config.h` 定义了：

- **数据集路径**：`GLOVEPATH / SIFTPATH / TESTPATH`
- **默认 query 文件名**：`QUERYFILEPATH`
- **默认答案文件名**：`ANSFILEPATH`

### `.txt` 与 `.bin`

- 程序会**优先读取**派生的二进制文件（如 `base.bin`、`query0.bin`）
- 若 `.bin` 不存在，会从 `.txt` 读取，并可能自动写出 `.bin` 加速后续运行
- `--bin` 会**强制只从 `.bin` 读取**（找不到/格式不对会直接报错退出）

> 若 base/query 文件缺失，程序会回退到"合成数据（synthetic）"以便调试；这不适合做真实性能/准确率对比。

---

## C++：主程序（`main.cpp`）

### 编译

**Windows（PowerShell / MinGW-w64）**

```pwsh
g++ -std=c++17 -O2 -Wall -Wextra -pthread main.cpp MySolution.cpp Brute.cpp -o main.exe
```

**Linux / macOS**

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pthread main.cpp MySolution.cpp Brute.cpp -o main
```

### 参数（主分支实际支持）

- **dataset_case**：位置参数 `0/1/2`（GloVe/SIFT/Test）
- **`--algo=solution|brute`** 或 `--algo solution|brute`
- **`--gen-queries`**：生成"极端/覆盖性"query 到 `QUERYFILEPATH`（默认 10000 条）并退出
- **`--gen-test-base[=N]`**：生成小测试集到 `TESTPATH` 并退出（默认 N=10）
- **`--query=<path>`** 或 `--query <path>`：覆盖 query 输入路径
  - 未指定 `--bin`：按 txt 读取，并派生写出同名 `.bin`
  - 指定 `--bin`：该路径按 `.bin` 读取
- **`--bin`**：强制仅从 `.bin` 读取 base/query
- **`--save-bin-base`**：将 base 数据写出为 `base.bin`（优先写在与 `base.txt` 同目录）

### 输出文件格式（给 `checker` 用）

`main` 会写出 `ANSFILEPATH`（例如 `ansglove.txt`），包含 header 与结果区块：

- header：`Dataset case:` / `Dimension:` / `Num queries:` 等
- 结果区块开始行：`Results (top-10 indices per query, space-separated):`
- 随后每行是一个 query 的 top-10 id（空格分隔）

`checker` 的 `--ans` 需要这种格式才能解析（它会从 `Results (top-` 这一行解析出 K）。

### 常用命令

```bash
# 1) 生成 query（可选）
./main.exe 0 --gen-queries

# 2) 用 brute 生成 ground-truth（写到 ANSFILEPATH）
./main.exe 0 --algo=brute

# 3) 跑 solution（注意：也会写到 ANSFILEPATH；如果不想覆盖 gt，请先备份文件或改 Config.h）
./main.exe 0 --algo=solution

# 4) 强制只从 bin 读取 + 指定自定义 query.bin
./main.exe 0 --bin --query=query0.bin --algo=solution
```

---

## C++：对拍/评估（`checker.cpp`）

### 编译

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pthread checker.cpp MySolution.cpp Brute.cpp -o checker.exe
```

### 参数（主分支实际支持）

- **dataset_case**：位置参数 `0/1/2`
- **`--ans=<path>`** 或 `--ans <path>`：ground-truth 文件（默认 `ANSFILEPATH`）
- **`--first=<N>`** / `--first <N>` / `--firstN`：只评估前 N 条 query（0 表示全部）支持简写形式如 `--first10`、`--first100`、`--first1000` 等
- **`--algo=solution|brute`**：选择要评估的算法（默认 solution）
- **`--query=<path>`**：覆盖 query 输入路径（逻辑与 main 类似）
- **`--bin`**：强制仅从 `.bin` 读取 base/query
- **`--save-bin-base`**：将 base 写出为 `base.bin`

### 常用命令

```bash
# 使用 brute 结果作为 gt，对拍 solution
./checker.exe 0 --ans=ansglove.txt --algo=solution

# 只检查前 100 条（支持 --first100 / --first=100 / --first 100 三种写法）
./checker.exe 0 --ans=ansglove.txt --first100 --algo=solution

# 强制从 bin 读 query，并用自定义 query0.bin
./checker.exe 0 --bin --query=query0.bin --ans=ansglove.txt --algo=solution
```

---

## 多线程与 SIMD 说明

- 多线程依赖标准库；GCC/Clang 记得带 `-pthread`
- SIMD 默认在 x86/x64 平台可用；如需强制关闭，参考 `MySolution.h` 的 `SolutionConfig`

---

## 快速开始（推荐流程）

```bash
# 1) 编译
g++ -std=c++17 -O2 -pthread -o main.exe main.cpp MySolution.cpp Brute.cpp
g++ -std=c++17 -O2 -pthread -o checker.exe checker.cpp MySolution.cpp Brute.cpp

# 2) （可选）生成 query
./main.exe 0 --gen-queries

# 3) 生成 ground-truth（brute）
./main.exe 0 --algo=brute

# 4) 对拍 solution（读取上一步的 ans 文件）
./checker.exe 0 --ans=ansglove.txt --algo=solution
```

---

## 依赖说明

本项目仅使用 C++ 标准库，无需额外第三方依赖：

| 依赖          | 用途              | 说明                                                      |
| ------------- | ----------------- | --------------------------------------------------------- |
| C++ 标准库    | 容器、多线程、I/O | `<vector>`, `<thread>`, `<mutex>`, `<fstream>` 等 |
| SIMD 内建函数 | 向量化距离计算    | `<immintrin.h>`（x86/x64 平台自动可用）                 |

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
- **SIFT 数据集**：[Corpus-Texmex](http://corpus-texmex.irisa.fr/)
