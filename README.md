# Build & Run Guide

## 编译器要求

- 支持C++17标准的编译器，且完整支持`<thread>`和`<mutex>`库（GCC ≥ 9，Clang ≥ 10，MSVC ≥ 19.28）
- 在Windows上使用MinGW-w64时，请选择POSIX线程变体（例如MSYS2 UCRT64或x86_64-posix-seh），并确保支持`-pthread`选项



## 构建与运行

### Windows (MinGW-w64 POSIX)
```pwsh
& "C:/mingw64/bin/g++.exe" -std=c++17 -O2 -Wall -Wextra -pthread main.cpp MySolution.cpp Brute.cpp -o main.exe
```

### Linux
```bash
g++ -std=c++17 -O2 -Wall -Wextra -pthread main.cpp MySolution.cpp Brute.cpp -o main
```

### macOS (Apple clang)
```bash
clang++ -std=c++17 -O2 -Wall -Wextra -pthread main.cpp MySolution.cpp Brute.cpp -o main
```



## Notes

- 这个项目完全依赖于C++标准线程库；如果`<thread>`不可用，请升级工具链

- `-pthread` 选项对于基于 GCC/Clang 的工具链是强制的，为了保证正确运行，启用线程支持并链接正确
- 在 Windows 上使用 MinGW-w64 时，确保选择了 POSIX 线程变体，以避免与 Windows 线程模型的冲突
- SIMD 优化仅在 x86/x64 平台上启用；在其他平台（如 ARM）上编译时，将自动回退到纯标量实现



## 多线程相关

本项目使用C++标准线程库实现多线程功能，无需额外依赖。请确保所用编译器和标准库版本支持 `<thread>` 和 `<mutex>`



## SIMD 优化相关

本项目在 x86/x64 平台上实现了基于运行时分发的 SIMD 优化，支持 SSE2、SSE3、AVX 和 AVX-512 指令集，请确保所用编译器支持相应的指令集扩展

- 在同一个可执行文件中同时编译多套实现（Scalar / SSE / SSE3 / AVX / AVX‑512）
- 启动时通过 CPUID + XGETBV 检测当前 CPU/OS 对指令集的支持情况，选择最优可用实现，并以函数指针缓存（一次检测，多次复用）
- 上层代码只调用统一入口，完全透明地获得 SIMD 加速或回退到纯标量实现

如果需要禁用 SIMD 优化

- `SOLUTION_PLATFORM_X86`：设为 `0` 可彻底禁用 x86 SIMD 路径，强制使用纯标量
- `SOLUTION_COMPILE_SSE` / `SOLUTION_COMPILE_SSE3` / `SOLUTION_COMPILE_AVX` / `SOLUTION_COMPILE_AVX512`：设为 `0` 可在编译期剔除对应 ISA 的实现（例如 `-DSOLUTION_COMPILE_AVX=0`）

示例（强制纯标量）：

```pwsh
& "C:/mingw64/bin/g++.exe" -std=c++17 -O2 -Wall -Wextra -pthread -DSOLUTION_PLATFORM_X86=0 main.cpp MySolution.cpp -o main.exe
```

### 编译器支持与说明

- GCC/Clang：支持 `__attribute__((target("...")))`，可为每个函数单独指定 ISA，无需全局添加 `-mavx`/`-mavx512f` 等开关
- MSVC：不支持上述 target attribute，是否编译某个 SIMD 变体取决于是否定义了相应的宏（例如 `__AVX__`）

- Windows/TDM-GCC 兼容性：某些 MinGW 发行版对 `_xgetbv`/`<cpuid.h>` 支持不完整，可能导致编译错误。实现中改用内联汇编版本，避免相关问题







