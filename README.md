# Build & Run Guide

## 编译器要求

- 支持 C++17 标准，且完整支持 `<thread>` / `<mutex>` 的现代编译器（推荐：GCC ≥ 9，Clang ≥ 10，MSVC ≥ 19.28）。
- Windows 上若使用 MinGW-w64，请选择 **POSIX 线程变体**（如 MSYS2 UCRT64、x86_64-posix-seh 等），并确保 `g++` 支持并实际传入 `-pthread` 选项；若使用 MSVC，则按默认多线程运行时即可。




## 项目结构概览

主要文件和目录说明：

- `.gitignore`：Git 忽略规则，排除中间产物、数据集等不需要提交的文件
- `main.cpp`：主程序入口，负责加载数据集、构建索引（`MySolution` / `Brute`），并生成 `ans.txt`
- `checker.cpp`：对拍程序，使用暴力解与 `MySolution` 的结果进行比对，统计准确率与延迟
- `MySolution.h` / `MySolution.cpp`：近邻搜索实现，基于 HNSW + ONNG + 多线程 + SIMD
- `Brute.h` / `Brute.cpp`：朴素暴力搜索实现，用于生成标准答案和对拍
- `BinaryIO.h`：二进制读写工具，负责 `base.bin` / `query*.bin` 的读写（小端格式）
- `ConfigExample.h`：配置示例文件，包含数据集路径常量；实际使用时通常复制为 `Config.h` 并按本机路径修改


## 数据集与文件路径

相关路径在 `Config.h` 中配置：

- `GLOVEPATH`：`./data_o/glove/base.txt`，对应 `dataset_case = 0`，维度 100，约 1183514 条向量
- `SIFTPATH`：`./data_o/sift/base.txt`，对应 `dataset_case = 1`，维度 128，约 1000000 条向量
- `TESTPATH`：`./test.txt`，对应 `dataset_case = 2`，用于小规模自测
- `QUERYFILEPATH`：查询文件路径，默认 `query.txt`
- `ANSFILEPATH`：结果输出文件，默认 `ans.txt`

程序会优先尝试从与 `base.txt` 同目录的 `base.bin` 读取二进制数据；若不存在则从 `base.txt` 读取并自动写出 `base.bin` 以加速后续运行
查询数据同理，优先尝试 `.bin` 文件（如 `query0.bin` / `query1.bin` 等），失败时回退到文本文件并自动写出 `.bin`


## 主程序用法（`main.cpp`）

可执行文件统一记为 `main`（Windows 下为 `main.exe`）：

### 编译 main

链接 `MySolution.cpp` / `Brute.cpp`：

**Windows（MinGW-w64 POSIX）**

```pwsh
& "C:/mingw64/bin/g++.exe" -std=c++17 -O2 -Wall -Wextra -pthread main.cpp MySolution.cpp Brute.cpp -o main.exe
```

**Linux（GCC）**

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pthread main.cpp MySolution.cpp Brute.cpp -o main
```

**macOS（Apple clang）**

```bash
clang++ -std=c++17 -O2 -Wall -Wextra -pthread main.cpp MySolution.cpp Brute.cpp -o main
```

### 基本参数

- 位置参数：`dataset_case`
  - `0`：使用 GloVe 数据集（`GLOVEPATH`）
  - `1`：使用 SIFT 数据集（`SIFTPATH`）
  - `2`：使用测试数据集（`TESTPATH`）
- `--algo=solution|brute`
	- `solution`：使用当前作业实现（`MySolution`，基于 HNSW+ONNG，多线程+SIMD）
	- `brute`：使用暴力搜索（`Brute`），用于对拍/校验

### 常用命令示例

1. 仅生成查询（按当前数据集分布随机生成 10000 条查询到 `QUERYFILEPATH`）：

	```pwsh
	# GloVe
	./main.exe 0 --gen-queries

	# SIFT
	./main.exe 1 --gen-queries
	```

2. 在 GloVe 上构建索引并用 `MySolution` 搜索，结果写入 `ans.txt`：

	```pwsh
	./main.exe 0 --algo=solution
	```

3. 使用暴力算法生成标准答案（用于 `checker.cpp` 对拍）：

	```pwsh
	./main.exe 0 --algo=brute
	```

4. 强制从二进制文件读取 base/query（仅使用 `.bin`，若不存在则报错）：

	```pwsh
	./main.exe 0 --bin --algo=solution
	```

5. 自定义查询文件（文本），程序会按该路径读取并自动写出对应 `.bin`：

	```pwsh
	./main.exe 0 --algo=solution --query=./query0.txt
	```

6. 生成一个小规模 base 测试集到 `TESTPATH`（默认 N=10 行）：

	```pwsh
	./main.exe 2 --gen-test-base
	# 或指定行数
	./main.exe 2 --gen-test-base=100
	```


## 对拍程序用法（`checker.cpp`）

### 编译 checker

与 `main` 保持一致，链接同一份 `MySolution.cpp` / `Brute.cpp`：

**Windows（MinGW-w64 POSIX）**

```pwsh
& "C:/mingw64/bin/g++.exe" -std=c++17 -O2 -Wall -Wextra -pthread checker.cpp MySolution.cpp Brute.cpp -o checker.exe
```

**Linux（GCC）**

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pthread checker.cpp MySolution.cpp Brute.cpp -o checker
```

**macOS（Apple clang）**

```bash
clang++ -std=c++17 -O2 -Wall -Wextra -pthread checker.cpp MySolution.cpp Brute.cpp -o checker
```

### 基本参数

- 位置参数：`dataset_case`（与 `main` 相同）
  - `0`：GloVe（`GLOVEPATH`）
  - `1`：SIFT（`SIFTPATH`）
  - `2`：测试集（`TESTPATH`）
- `--ans=<path>`：指定待检查的答案文件，默认为 `ANSFILEPATH`（例如 `./ans.txt`）
- `--algo=solution|brute`：指定被对拍的算法实现，默认 `solution`（`MySolution`）
- `--first=<N>`：只检查前 N 条查询（默认 0 = 全部）

checker 会自己加载 base / query（同样优先 `.bin`）、重建暴力解和被测解，对比 Top-K 结果的一致性，并打印命中率等统计信息。

### 常用对拍流程

1. 用 `main` 生成答案文件（例如使用 `MySolution`）：

	```pwsh
	./main.exe 0 --algo=solution
	```

2. 用 `checker` 对拍（默认读取 `./ans.txt`）：

	```pwsh
	./checker.exe 0 --algo=solution
	```

3. 只检查前 100 条查询、并显式指定答案文件：

	```pwsh
	./checker.exe 0 --ans="./ans.txt" --algo=solution --first=100
	```


## 文本 / 二进制转换

### base：`base.txt` → `base.bin`

1. **自动转换（推荐）**  
	直接按正常方式运行程序，例如：

	```pwsh
	./main.exe 0 --algo=solution
	```

	- 程序会先尝试读取同目录的 `base.bin`
	- 若不存在，则从 `base.txt` 读取，并在同目录自动写出 `base.bin`

2. **显式保存二进制**  
	若你只想从 `base.txt` 读一次并固定写出 `base.bin`，可以加上 `--save-bin-base`：

	```pwsh
	./main.exe 0 --algo=solution --save-bin-base
	```

	- 读取来源：优先 `GLOVEPATH` 指向的 `base.txt`
	- 输出路径：与 `base.txt` 同目录的 `base.bin`

3. **强制仅使用二进制**  
	当你已经有 `base.bin`，可以用 `--bin` 强制只用二进制文件（没有就报错，不再回退到 txt）：

	```pwsh
	./main.exe 0 --algo=solution --bin
	```


### query：`query*.txt` ↔ `query*.bin`

1. **从文本读取并写出 `.bin`**  
	使用 `--query=...` 指定一个查询文本文件：

	```pwsh
	./main.exe 0 --algo=solution --query=./query0.txt
	```

	- 程序会从 `query0.txt` 读取查询
	- 同目录写出 `query0.bin`（名字由 txt 路径自动派生）

2. **自动查找并使用 `.bin`**  
	不指定 `--query` 时，程序会：

	- 先在数据集目录及当前目录查找与 `QUERYFILEPATH`/`query0.txt`/`query1.txt` 对应的 `.bin`
	- 若找到（例如 `query0.bin`），则直接从其读取查询
	- 若没有 `.bin`，会回退到文本（如 `query.txt` / `query0.txt`）并自动写出相应 `.bin`

3. **强制仅使用查询二进制**  
	若使用 `--bin`，程序会只从 `.bin` 中加载查询（包括你指定的 `--query=foo.bin`）：

	```pwsh
	# 强制从覆盖路径 foo.bin 读取
	./main.exe 0 --bin --query=./foo.bin --algo=solution
	```

	- 若 `foo.bin` / 自动推断的 `.bin` 不存在或维度不匹配，将直接报错退出


## 多线程相关

- 项目完全依赖 C++ 标准库（`<thread>` / `<mutex>` 等），不需要额外第三方线程库；如果 `<thread>` 不可用，请升级编译器/标准库
- 对于 GCC/Clang，请务必添加 `-pthread`，以启用线程支持并正确链接；在 Windows 上使用 MinGW-w64 时，请选择 **POSIX 线程变体**（如 MSYS2 UCRT64 或 x86_64-posix-seh）

## SIMD 优化相关

SIMD 加速仅在 x86/x64 平台上启用；其他架构（如 ARM）会自动回退到纯标量实现，不需要额外配置

`MySolution` 的 L2 距离计算在 x86/x64 平台上实现了 **运行时指令集分发**，支持 Scalar / SSE / AVX / AVX-512 ：

- 启动后通过 CPUID + XCR0 检测当前 CPU/OS 支持的指令集，只选择一套最优实现并以函数指针缓存（一次检测，多次复用）
- 上层代码只调用统一的 `L2Distance` 接口，自动获得 SIMD 加速或回退到纯标量实现，对调用者完全透明

如果需要在编译期禁用部分或全部 SIMD：

- `SOLUTION_PLATFORM_X86=0`：彻底关闭 x86 SIMD 路径，强制使用标量实现。
- `SOLUTION_COMPILE_SSE` / `SOLUTION_COMPILE_SSE3` / `SOLUTION_COMPILE_AVX` / `SOLUTION_COMPILE_AVX512`：设为 `0` 可剔除对应 ISA 的实现（例如 `-DSOLUTION_COMPILE_AVX=0`）

示例（强制纯标量）：

```pwsh
& "C:/mingw64/bin/g++.exe" -std=c++17 -O2 -Wall -Wextra -pthread -DSOLUTION_PLATFORM_X86=0 main.cpp MySolution.cpp Brute.cpp -o main.exe
```







