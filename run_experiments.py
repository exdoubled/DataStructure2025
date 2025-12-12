#!/usr/bin/env python3
"""
run_experiments.py - Comprehensive Ablation Study and Parameter Tuning for ANN Search

This script automates experiments using the AblationRunner C++ tool to:
1. Compare search strategies (Adaptive Gamma vs Fixed EF)
2. Ablate graph optimization components (ONNG, BFS, SIMD)
3. Tune graph density parameter (M)

Each experiment generates Pareto frontier data (Recall vs QPS tradeoff).
"""

import subprocess
import json
import csv
import hashlib
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path
import pickle

# ==================== Configuration Constants ====================

# Dataset paths - MODIFY THESE FOR YOUR SETUP
# Note: Use forward slashes or raw strings on Windows
DATA_PATH = "data_o/glove/base.bin"
QUERY_PATH = "query0.bin"  
GROUNDTRUTH_PATH = "ansglove.txt"  # Ground truth file with true top-K indices

# Runner executable (without extension - will auto-detect .exe on Windows)
RUNNER_PATH = "runner"

# Cache directory for graph files (place on SSD)
CACHE_DIR = r"I:\graph_cache"

# Output file
OUTPUT_CSV = "experiment_results.csv"
RESUME_PKL = "experiment_results.pkl"

# Search parameter sweep ranges
EF_RANGE = [20, 40, 60, 80, 100, 150, 200, 300, 400, 600, 700, 800, 1000]
GAMMA_RANGE = [0.0, 0.1, 0.14, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.3]

# Build parameter sweeps
M_RANGE = [96, 48, 24, 128]
EFC_RANGE = [200, 300, 400, 500]

# 默认 top-K
DEFAULT_K = 10

# ==================== Data Classes ====================

@dataclass
class BuildConfig:
    """Configuration for graph building"""
    M: int = 96
    efC: int = 400
    onng: bool = True
    onng_out: int = 96
    onng_in: int = 144
    onng_min: int = 64
    bfs: bool = True
    single_layer: bool = True
    neg_ip: bool = True
    simd: bool = True

    def get_cache_filename(self) -> str:
        """Generate unique cache filename based on build parameters"""
        param_str = (
            f"L{int(self.single_layer)}"
            f"_neg{int(self.neg_ip)}"
            f"_simd{int(self.simd)}"
            f"_M{self.M}_efC{self.efC}"
            f"_onng{int(self.onng)}"
            f"_bfs{int(self.bfs)}"
        )
        if self.onng:
            param_str += f"_oo{self.onng_out}_oi{self.onng_in}_om{self.onng_min}"
        # Add hash for extra uniqueness
        hash_suffix = hashlib.md5(param_str.encode()).hexdigest()[:8]
        return f"graph_{param_str}_{hash_suffix}.bin"


@dataclass
class SearchConfig:
    """Configuration for search"""
    strategy: str = "fixed"  # "fixed", "gamma", "gamma-static"
    ef: int = 100
    gamma: float = 0.19
    k: int = DEFAULT_K
    count_dist: bool = False


@dataclass
class ExperimentResult:
    """Single experiment result"""
    group_name: str
    M: int
    efC: int
    onng: bool
    bfs: bool
    simd: bool
    single_layer: bool
    neg_ip: bool
    strategy: str
    param_val: float  # ef or gamma value
    top1: float = 0.0
    recall10: float = 0.0
    qps: float = 0.0
    ms_per_query: float = 0.0
    latency_ms: float = 0.0
    dist_calcs: float = 0.0
    build_time_ms: float = 0.0


# ==================== Runner Interface ====================

def build_runner_command(
    build_cfg: BuildConfig,
    search_cfg: SearchConfig,
    cache_path: str,
    count_dist: bool = False,
    opt_only: bool = False,
    cache_out: str = "",
    opt_order: str = "onng-first"
) -> List[str]:
    """Build the command line arguments for the runner"""
    cmd = [RUNNER_PATH]
    
    # IO args
    cmd.extend(["--data=" + DATA_PATH])
    cmd.extend(["--query=" + QUERY_PATH])
    cmd.extend(["--groundtruth=" + GROUNDTRUTH_PATH])
    cmd.extend(["--cache-path=" + cache_path])
    
    # Build args
    cmd.extend([f"--M={build_cfg.M}"])
    cmd.extend([f"--efC={build_cfg.efC}"])
    
    if build_cfg.onng:
        cmd.extend([f"--onng-out={build_cfg.onng_out}"])
        cmd.extend([f"--onng-in={build_cfg.onng_in}"])
        cmd.extend([f"--onng-min={build_cfg.onng_min}"])
    else:
        cmd.append("--no-onng")
    
    if not build_cfg.bfs:
        cmd.append("--no-bfs")

    if build_cfg.single_layer:
        cmd.append("--single-layer")

    if build_cfg.neg_ip:
        cmd.append("--use-neg-ip")

    if not build_cfg.simd:
        cmd.append("--no-simd")
    
    # Search args
    cmd.extend([f"--k={search_cfg.k}"])
    cmd.extend([f"--strategy={search_cfg.strategy}"])
    
    if search_cfg.strategy.startswith("gamma"):
        cmd.extend([f"--gamma={search_cfg.gamma}"])
    else:
        cmd.extend([f"--ef={search_cfg.ef}"])
    
    # Debug / opt args
    if count_dist:
        cmd.append("--count-dist")
    if opt_only:
        cmd.append("--opt-only")
        if cache_out:
            cmd.append(f"--cache-out={cache_out}")
        if opt_order:
            cmd.append(f"--opt-order={opt_order}")
    
    return cmd


def run_single_experiment(
    build_cfg: BuildConfig,
    search_cfg: SearchConfig,
    cache_path: str,
    count_dist: bool = False
) -> Optional[Dict[str, Any]]:
    """Run a single experiment and return parsed JSON result"""
    cmd = build_runner_command(build_cfg, search_cfg, cache_path, count_dist)
    
    try:
        print(f"[RUN] {' '.join(cmd)}", flush=True)
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            msg = result.stdout[-500:] if result.stdout else ""
            print(f"  [ERROR] Runner failed: {msg}", file=sys.stderr)
            return None
        
        # Parse JSON from stdout (last non-empty line)
        lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
        if not lines:
            print(f"  [ERROR] No output from runner", file=sys.stderr)
            return None
        
        json_line = lines[-1]
        try:
            return json.loads(json_line)
        except json.JSONDecodeError as e:
            print(f"  [ERROR] JSON parse error: {e}", file=sys.stderr)
            print(f"  Output was: {json_line[:200]}", file=sys.stderr)
            return None
            
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Runner timed out", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  [ERROR] Exception: {e}", file=sys.stderr)
        return None


def save_resume(results: List[ExperimentResult], idx: int, total: int):
    """Save resume checkpoint"""
    try:
        with open(RESUME_PKL, "wb") as f:
            # 兼容：idx/total 仅用于展示；真正的续跑靠 results 去重
            pickle.dump({"results": results, "idx": idx, "total": total}, f)
    except Exception as e:
        print(f"[WARN] Failed to save resume file: {e}", file=sys.stderr)


def load_resume():
    """Load resume checkpoint"""
    if not os.path.exists(RESUME_PKL):
        return None
    try:
        with open(RESUME_PKL, "rb") as f:
            data = pickle.load(f)
            return data
    except Exception as e:
        print(f"[WARN] Failed to load resume file: {e}", file=sys.stderr)
        return None


def exp_key(group_name: str, build_cfg: BuildConfig, search_strategy: str, param_val: float) -> str:
    """
    为“一个 experiment_result（两次搜索合并后）”生成稳定 key。
    注意：param_val 对 gamma 用 6 位小数，对 ef 用整数表示。
    """
    if search_strategy.startswith("gamma"):
        p = f"g={param_val:.6f}"
    else:
        p = f"ef={int(param_val)}"
    return (
        f"{group_name}"
        f"|M={build_cfg.M}|efC={build_cfg.efC}"
        f"|onng={int(build_cfg.onng)}|bfs={int(build_cfg.bfs)}"
        f"|simd={int(build_cfg.simd)}|single_layer={int(build_cfg.single_layer)}|neg_ip={int(build_cfg.neg_ip)}"
        f"|strategy={search_strategy}|{p}"
    )


def run_opt_only(
    build_cfg: BuildConfig,
    cache_in: str,
    cache_out: str,
    enable_onng: bool,
    enable_bfs: bool,
    opt_order: str = "onng-first"
) -> bool:
    """Run runner in opt-only mode to derive new cache"""
    tmp_build = BuildConfig(**asdict(build_cfg))
    tmp_build.onng = enable_onng
    tmp_build.bfs = enable_bfs
    # onng params already in build_cfg
    dummy_search = SearchConfig(strategy="fixed", ef=EF_RANGE[0], gamma=GAMMA_RANGE[0])
    cmd = build_runner_command(tmp_build, dummy_search, cache_in, count_dist=False,
                               opt_only=True, cache_out=cache_out, opt_order=opt_order)
    try:
        print(f"[RUN OPT] {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1800)
        if result.returncode != 0:
            msg = result.stdout[-500:] if result.stdout else ""
            print(f"[ERROR] opt-only failed: {msg}", file=sys.stderr)
            return False
        return True
    except subprocess.TimeoutExpired:
        print("[ERROR] opt-only timed out", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[ERROR] opt-only exception: {e}", file=sys.stderr)
        return False


def run_two_pass_experiment(
    group_name: str,
    build_cfg: BuildConfig,
    search_cfg: SearchConfig,
    cache_path: str,
    param_val: float
) -> Optional[ExperimentResult]:
    """
    Run two-pass experiment:
    Pass 1: Performance measurement (no --count-dist)
    Pass 2: Statistics measurement (with --count-dist)
    """
    # Pass 1: Performance
    perf_result = run_single_experiment(build_cfg, search_cfg, cache_path, count_dist=False)
    if perf_result is None:
        return None
    
    # Pass 2: Stats
    stats_result = run_single_experiment(build_cfg, search_cfg, cache_path, count_dist=True)
    
    # Merge results
    result = ExperimentResult(
        group_name=group_name,
        M=build_cfg.M,
        efC=build_cfg.efC,
        onng=build_cfg.onng,
        bfs=build_cfg.bfs,
        simd=build_cfg.simd,
        single_layer=build_cfg.single_layer,
        neg_ip=build_cfg.neg_ip,
        strategy=search_cfg.strategy,
        param_val=param_val,
        recall10=perf_result.get("recall", 0.0),
        qps=perf_result.get("qps", 0.0),
        latency_ms=perf_result.get("search_ms", 0.0),
        build_time_ms=perf_result.get("build_ms", 0.0)
    )
    
    # Calculate ms per query
    query_count = perf_result.get("query_count", 1)
    if query_count > 0:
        result.ms_per_query = result.latency_ms / query_count
    
    # Add dist_calcs from pass 2 if available
    if stats_result is not None:
        result.dist_calcs = stats_result.get("avg_dist_calcs", 0.0)
    
    # Top-1 accuracy (if available in result)
    result.top1 = perf_result.get("top1", result.recall10)  # Use recall as fallback
    
    return result


"""
实验分组：按 5 位编码 (单/多层, onng, bfs, simd, neg_ip) 以及 3 档策略 (gamma 动态 / gamma 静态 / fixed ef) 扫描。
为减少冗余：
 - onng=0 时不扫 onng 参数，使用基于 M 的默认派生值
 - onng=1 时 onng_out/onng_in/onng_min 随 M 自动派生（避免全笛卡尔爆炸）
 - 两次搜索：一次正常，一次 --count-dist，用同一缓存
"""


def ensure_cache_dir():
    """Ensure cache directory exists"""
    os.makedirs(CACHE_DIR, exist_ok=True)


def get_cache_path(build_cfg: BuildConfig) -> str:
    """Get full cache path for a build configuration"""
    return os.path.join(CACHE_DIR, build_cfg.get_cache_filename())


def derive_onng_params(M: int):
    """派生 ONNG 参数：按默认最优 (M=96, oo=96, oi=144, om=64) 等比例缩放"""
    onng_out = M                                  # 96 -> 96
    onng_in = max(M, int(M * (144.0 / 96.0)))     # 1.5x
    onng_min = max(16, int(M * (64.0 / 96.0)))    # ~0.67x
    return onng_out, onng_in, onng_min


def run_code_group(code_bits: str, results: List[ExperimentResult], done_keys: set, progress_cb=None):
    """按 5 位编码跑三种策略"""
    assert len(code_bits) == 5 and set(code_bits) <= {"0", "1"}

    single_layer = (code_bits[0] == "1")
    use_onng = (code_bits[1] == "1")
    use_bfs = (code_bits[2] == "1")
    use_simd = (code_bits[3] == "1")
    use_neg_ip = (code_bits[4] == "1")

    group_prefix = code_bits

    for M in M_RANGE:
        for efC in EFC_RANGE:
            # ONNG 参数派生
            onng_out, onng_in, onng_min = derive_onng_params(M)

            # 基础图（无 ONNG/BFS）缓存
            base_cfg = BuildConfig(
                M=M,
                efC=efC,
                onng=False,
                onng_out=onng_out,
                onng_in=onng_in,
                onng_min=onng_min,
                bfs=False,
                single_layer=single_layer,
                neg_ip=use_neg_ip,
                simd=use_simd,
            )
            base_cache = get_cache_path(base_cfg)

            if not os.path.exists(base_cache):
                # 构建基础图（最小化搜索，目的是生成缓存）
                print(f"  [BUILD base] code={code_bits} M={M} efC={efC} -> {base_cache}")
                build_search_cfg = SearchConfig(strategy="fixed", ef=EF_RANGE[0], gamma=GAMMA_RANGE[0])
                ok = run_single_experiment(base_cfg, build_search_cfg, base_cache, count_dist=False)
                if ok is None:
                    print("    [ERROR] build base failed")
                    continue

            # 目标图（按 onng/bfs）
            target_cfg = BuildConfig(
                M=M,
                efC=efC,
                onng=use_onng,
                onng_out=onng_out,
                onng_in=onng_in,
                onng_min=onng_min,
                bfs=use_bfs,
                single_layer=single_layer,
                neg_ip=use_neg_ip,
                simd=use_simd,
            )
            final_cache = get_cache_path(target_cfg)

            if not os.path.exists(final_cache):
                # 生成目标缓存：onng -> bfs（如需）
                current_cache = base_cache
                if use_onng:
                    inter_cache = final_cache if not use_bfs else final_cache + ".onngtmp"
                    if not os.path.exists(inter_cache):
                        ok_onng = run_opt_only(target_cfg, current_cache, inter_cache,
                                               enable_onng=True, enable_bfs=False, opt_order="onng-first")
                        if not ok_onng:
                            print("    [ERROR] opt-only onng failed")
                            continue
                    current_cache = inter_cache
                if use_bfs:
                    if not os.path.exists(final_cache):
                        ok_bfs = run_opt_only(target_cfg, current_cache, final_cache,
                                              enable_onng=False, enable_bfs=True, opt_order="onng-first")
                        if not ok_bfs:
                            print("    [ERROR] opt-only bfs failed")
                            continue
                else:
                    # 如果只做 ONNG，且 final_cache 还未生成（inter_cache==final_cache 已处理）
                    if use_onng and current_cache != final_cache and not os.path.exists(final_cache):
                        os.replace(current_cache, final_cache)

            cache_path = final_cache

            # 策略 1: gamma 动态
            for gamma in GAMMA_RANGE:
                search_cfg = SearchConfig(strategy="gamma", gamma=gamma, ef=EF_RANGE[0])  # ef占位
                gname = f"{group_prefix}_gamma_dyn"
                k = exp_key(gname, target_cfg, search_cfg.strategy, gamma)
                if k in done_keys:
                    continue
                result = run_two_pass_experiment(gname, target_cfg, search_cfg, cache_path, gamma)
                if result:
                    results.append(result)
                    done_keys.add(k)
                    if progress_cb:
                        progress_cb(1)

            # 策略 2: gamma 静态
            for gamma in GAMMA_RANGE:
                search_cfg = SearchConfig(strategy="gamma-static", gamma=gamma, ef=EF_RANGE[0])  # ef占位
                gname = f"{group_prefix}_gamma_static"
                k = exp_key(gname, target_cfg, search_cfg.strategy, gamma)
                if k in done_keys:
                    continue
                result = run_two_pass_experiment(gname, target_cfg, search_cfg, cache_path, gamma)
                if result:
                    results.append(result)
                    done_keys.add(k)
                    if progress_cb:
                        progress_cb(1)

            # 策略 3: 固定 EF
            for ef in EF_RANGE:
                search_cfg = SearchConfig(strategy="fixed", ef=ef, gamma=GAMMA_RANGE[0])
                gname = f"{group_prefix}_fixed"
                k = exp_key(gname, target_cfg, search_cfg.strategy, float(ef))
                if k in done_keys:
                    continue
                result = run_two_pass_experiment(gname, target_cfg, search_cfg, cache_path, float(ef))
                if result:
                    results.append(result)
                    done_keys.add(k)
                    if progress_cb:
                        progress_cb(1)


def save_results_to_csv(results: List[ExperimentResult], filename: str):
    """Save all results to CSV file"""
    if not results:
        print("No results to save!")
        return
    
    fieldnames = [
        "group_name", "M", "efC", "onng", "bfs", "simd", "single_layer", "neg_ip",
        "strategy", "param_val", "top1", "recall10", 
        "qps", "ms_per_query", "latency_ms", "dist_calcs", "build_time_ms"
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in results:
            row = {
                "group_name": r.group_name,
                "M": r.M,
                "efC": r.efC,
                "onng": r.onng,
                "bfs": r.bfs,
                "simd": r.simd,
                "single_layer": r.single_layer,
                "neg_ip": r.neg_ip,
                "strategy": r.strategy,
                "param_val": r.param_val,
                "top1": f"{r.top1:.6f}",
                "recall10": f"{r.recall10:.6f}",
                "qps": f"{r.qps:.2f}",
                "ms_per_query": f"{r.ms_per_query:.6f}",
                "latency_ms": f"{r.latency_ms:.2f}",
                "dist_calcs": f"{r.dist_calcs:.2f}",
                "build_time_ms": f"{r.build_time_ms:.2f}"
            }
            writer.writerow(row)
    
    print(f"\nResults saved to: {filename}")


def print_summary(results: List[ExperimentResult]):
    """Print summary statistics"""
    if not results:
        return
    
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    # Group results by group_name
    groups: Dict[str, List[ExperimentResult]] = {}
    for r in results:
        if r.group_name not in groups:
            groups[r.group_name] = []
        groups[r.group_name].append(r)
    
    for group_name, group_results in groups.items():
        print(f"\n{group_name}:")
        
        # Find best recall and best QPS
        best_recall = max(group_results, key=lambda x: x.recall10)
        best_qps = max(group_results, key=lambda x: x.qps)
        
        # Find result with recall >= 0.95 and highest QPS
        high_recall_results = [r for r in group_results if r.recall10 >= 0.95]
        best_qps_at_95 = max(high_recall_results, key=lambda x: x.qps) if high_recall_results else None
        
        print(f"  Best Recall: {best_recall.recall10:.4f} (param={best_recall.param_val}, QPS={best_recall.qps:.1f})")
        print(f"  Best QPS: {best_qps.qps:.1f} (param={best_qps.param_val}, Recall={best_qps.recall10:.4f})")
        if best_qps_at_95:
            print(f"  Best QPS@95%: {best_qps_at_95.qps:.1f} (param={best_qps_at_95.param_val}, Recall={best_qps_at_95.recall10:.4f})")


# ==================== Main ====================

def main():
    """Main entry point"""
    global RUNNER_PATH  # Declare global at the start of function
    
    print("="*60)
    print("ANN Ablation Study - Experiment Runner (Code-driven)")
    print("="*60)
    
    # Check runner exists (handle Windows/Linux differences)
    runner_candidates = [
        RUNNER_PATH,
        RUNNER_PATH + ".exe",
        "./" + RUNNER_PATH,
        "./" + RUNNER_PATH + ".exe",
        os.path.join(".", RUNNER_PATH),
        os.path.join(".", RUNNER_PATH + ".exe"),
    ]
    
    found = False
    for candidate in runner_candidates:
        if os.path.exists(candidate):
            RUNNER_PATH = candidate
            found = True
            print(f"Found runner at: {RUNNER_PATH}")
            break
    
    if not found:
        print(f"ERROR: Runner not found. Tried: {runner_candidates}")
        print("Please compile AblationRunner.cpp first:")
        print("  Windows: g++ -std=c++17 -O2 -pthread -o runner.exe AblationRunner.cpp MySolution.cpp Brute.cpp")
        print("  Linux:   g++ -std=c++17 -O2 -pthread -o runner AblationRunner.cpp MySolution.cpp Brute.cpp")
        sys.exit(1)
    
    # Check data files
    for path, name in [(DATA_PATH, "data"), (QUERY_PATH, "query"), (GROUNDTRUTH_PATH, "groundtruth")]:
        # Normalize path for current OS
        normalized_path = os.path.normpath(path)
        if not os.path.exists(normalized_path):
            print(f"WARNING: {name} file not found at {normalized_path}")
        else:
            print(f"Found {name} file: {normalized_path}")
    
    # Ensure cache directory exists
    ensure_cache_dir()
    
    # Collect all results (resume support)
    all_results: List[ExperimentResult] = []
    resume_state = load_resume()
    selected_codes = ["00000", "11111", "11110", "11101", "11011", "10111", "01111"]
    total_tasks = len(selected_codes) * len(M_RANGE) * len(EFC_RANGE) * (len(GAMMA_RANGE)*2 + len(EF_RANGE))
    completed = 0
    done_keys: set = set()
    if resume_state:
        all_results = resume_state.get("results", [])
        # 基于 results 重建完成集合，避免修改 code/total 后 idx 失真
        for r in all_results:
            # 这里用 result 自身字段生成 key（param_val 已是数值）
            bc = BuildConfig(
                M=r.M, efC=r.efC, onng=r.onng,
                onng_out=0, onng_in=0, onng_min=0,  # 不影响 key（exp_key 用 build_cfg 中的这些布尔/核心参数）
                bfs=r.bfs, single_layer=r.single_layer, neg_ip=r.neg_ip, simd=r.simd
            )
            done_keys.add(exp_key(r.group_name, bc, r.strategy, float(r.param_val)))
        completed = len(done_keys)
        print(f"[RESUME] Loaded checkpoint: completed={completed}/{total_tasks}, results={len(all_results)}")
    else:
        print(f"[INFO] Total tasks to run: {total_tasks}")
    
    start_time = time.time()
    
    try:
        task_idx = 0
        def progress_cb(step):
            nonlocal task_idx
            task_idx += step
            if task_idx % 50 == 0:
                cur = completed + task_idx
                print(f"[PROGRESS] {cur}/{total_tasks} tasks done in current run segment", flush=True)
                save_resume(all_results, cur, total_tasks)

        for code_bits in selected_codes:
            print(f"\n{'='*40}\nRunning code {code_bits}\n{'='*40}")
            run_code_group(code_bits, all_results, done_keys=done_keys, progress_cb=progress_cb)
            completed += task_idx
            task_idx = 0
            print(f"[INFO] Finished code {code_bits}, progress {completed}/{total_tasks}", flush=True)
            save_resume(all_results, completed, total_tasks)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving partial results...")
        save_resume(all_results, completed, total_tasks)
    
    elapsed = time.time() - start_time
    
    # Save results
    save_results_to_csv(all_results, OUTPUT_CSV)
    
    # Print summary
    print_summary(all_results)
    
    print(f"\nTotal experiments: {len(all_results)}")
    print(f"Total time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()

