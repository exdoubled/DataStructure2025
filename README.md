# Build & Run Guide

## 编译器要求

- 支持C++17标准的编译器，且完整支持`<thread>`和`<mutex>`库（GCC ≥ 9，Clang ≥ 10，MSVC ≥ 19.28）
- 在Windows上使用MinGW-w64时，请选择POSIX线程变体（例如MSYS2 UCRT64或x86_64-posix-seh），并确保支持`-pthread`选项

## 构建与运行
### Windows (MinGW-w64 POSIX)
```pwsh
& "C:/mingw64/bin/g++.exe" -std=c++17 -O2 -Wall -Wextra -pthread main.cpp MySolution.cpp -o main.exe
./main.exe 0
```

### Linux
```bash
g++ -std=c++17 -O2 -Wall -Wextra -pthread main.cpp MySolution.cpp -o main
./main 0
```

### macOS (Apple clang)
```bash
clang++ -std=c++17 -O2 -Wall -Wextra -pthread main.cpp MySolution.cpp -o main
./main 0
```

## Notes

- 这个项目完全依赖于C++标准线程库；如果`<thread>`不可用，请升级工具链

- `-pthread`选项对于基于 GCC/Clang 的工具链是强制的，为了保证正确运行，启用线程支持并链接正确
- 在 Windows 上使用 MinGW-w64 时，确保选择了 POSIX 线程变体，以避免与 Windows 线程模型的冲突
