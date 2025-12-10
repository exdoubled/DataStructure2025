# Build & Run Guide

## 编译器要求

- 支持 C++17 标准，且完整支持 `<thread>` / `<mutex>` 的现代编译器（推荐：GCC ≥ 9，Clang ≥ 10，MSVC ≥ 19.28）。
- Windows 上若使用 MinGW-w64，请选择 **POSIX 线程变体**（如 MSYS2 UCRT64、x86_64-posix-seh 等），并确保 `g++` 支持并实际传入 `-pthread` 选项；若使用 MSVC，则按默认多线程运行时即可。




## 项目结构概览

主要文件和目录说明：

| 文件                    | 说明                                                        |
| ----------------------- | ----------------------------------------------------------- |
| `main.cpp`              | 主程序入口，负责加载数据集、构建索引、搜索并生成 `ans.txt`  |
| `checker.cpp`           | 对拍程序，使用暴力解与 `MySolution` 的结果进行比对          |
| `AblationRunner.cpp`    | **消融实验 CLI 工具**，支持参数化构建和搜索，输出 JSON 结果 |
| `run_experiments.py`    | **Python 自动化脚本**，批量运行消融实验并输出 CSV           |
| `MySolution.h` / `.cpp` | 近邻搜索实现，基于 HNSW + ONNG + 多线程 + SIMD              |
| `Brute.h` / `.cpp`      | 朴素暴力搜索实现，用于生成标准答案和对拍                    |
| `BinaryIO.h`            | 二进制读写工具，负责向量数据和**图缓存**的读写              |
| `Config.h`              | 配置文件，包含数据集路径和图缓存路径常量                    |

更好的包装性以及相关实验请查看 **experiment** 分支

## 数据集与文件路径

相关路径在 `Config.h` 中配置：

```cpp
// 数据集路径
#define GLOVEPATH "./data_o/glove/base.txt"   // dataset_case = 0, dim=100
#define SIFTPATH  "./data_o/sift/base.txt"    // dataset_case = 1, dim=128
#define TESTPATH  "./test.txt"                // dataset_case = 2, 小规模测试
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

| 参数                     | 说明                            |
| ------------------------ | ------------------------------- |
| `0` / `1` / `2`          | 数据集选择：GloVe / SIFT / Test |
| `--algo=solution\|brute` | 算法选择，默认 `solution`       |
| `--gen-queries`          | 生成随机查询文件                |
| `--bin`                  | 强制仅使用二进制文件            |
| `--query=<path>`         | 指定查询文件路径                |

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

| 参数                     | 说明                             |
| ------------------------ | -------------------------------- |
| `0` / `1` / `2`          | 数据集选择                       |
| `--ans=<path>`           | 指定答案文件，默认 `ANSFILEPATH` |
| `--algo=solution\|brute` | 被测算法，默认 `solution`        |
| `--first=<N>`            | 只检查前 N 条查询                |

### 常用对拍流程

```bash
# 1. 生成答案
./main.exe 0 --algo=solution

# 2. 对拍（从缓存加载图加速）
./checker.exe 0 --load-graph --ans="./ans.txt" --algo=solution

# 3. 只检查前 100 条
./checker.exe 0 --load-graph --first=100
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
# 1. 配置相关参数

# 2. 编译所有工具
g++ -std=c++17 -O2 -pthread -o main.exe main.cpp MySolution.cpp Brute.cpp
g++ -std=c++17 -O2 -pthread -o checker.exe checker.cpp MySolution.cpp Brute.cpp

# 3. 生成查询和暴力答案
./main.exe 0 --gen-queries
./main.exe 0 --algo=brute

# 4. 构建索引并保存缓存
./main.exe 0 --algo=solution

# 5. 对拍验证
./checker.exe 0 --load-graph --algo=solution

```