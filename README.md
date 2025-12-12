# DataStructure（main）：HNSW/ONNG/BFS/SIMD 近邻搜索

主分支聚焦于**主程序**与**对拍/评估工具**
当前入口：

- **`main.cpp`**：加载 base/query，构建索引并检索，写出答案文件（默认 `ANSFILEPATH`）
- **`checker.cpp`**：读取 ground-truth 答案文件，评估/对拍 `solution` 或 `brute`

> 批量实验/断点续跑等内容请参考 experiment 分支文档

---

## 编译器要求

- C++17，且 `<thread>/<mutex>` 可用（GCC/Clang/MSVC 均可）
- Windows 使用 MinGW-w64 时，建议使用 **POSIX 线程变体**，并确保编译命令带 `-pthread`

---

## 项目结构（主分支）

| 文件 | 说明 |
|---|---|
| `main.cpp` | 主程序：生成 query、跑 `solution/brute`、写出答案文件 |
| `checker.cpp` | 对拍/评估：读取答案文件（带 header 的文本格式），计算 top1/recall/耗时等 |
| `MySolution.h/.cpp` | HNSW + ONNG + BFS 重排 + SIMD 的实现（含 `SolutionConfig`） |
| `Brute.h/.cpp` | 暴力解 |
| `BinaryIO.h` | `.bin` 向量文件读写（header-only，magic 为 `VECBIN1\0`） |
| `Config.h` | 数据集路径、默认 query/ans 文件名 |

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

> 若 base/query 文件缺失，程序会回退到“合成数据（synthetic）”以便调试；这不适合做真实性能/准确率对比。

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
- **`--gen-queries`**：生成“极端/覆盖性”query 到 `QUERYFILEPATH`（默认 10000 条）并退出
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
- **`--first=<N>`** / `--first <N>` / `--first10`：只评估前 N 条 query（0 表示全部）
- **`--algo=solution|brute`**：选择要评估的算法（默认 solution）
- **`--query=<path>`**：覆盖 query 输入路径（逻辑与 main 类似）
- **`--bin`**：强制仅从 `.bin` 读取 base/query
- **`--save-bin-base`**：将 base 写出为 `base.bin`

### 常用命令

```bash
# 使用 brute 结果作为 gt，对拍 solution
./checker.exe 0 --ans=ansglove.txt --algo=solution

# 只检查前 100 条（也支持 --first100 简写）
./checker.exe 0 --ans=ansglove.txt --first=100 --algo=solution

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


