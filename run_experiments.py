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

# ==================== Configuration Constants ====================

# Dataset paths - MODIFY THESE FOR YOUR SETUP
# Note: Use forward slashes or raw strings on Windows
DATA_PATH = "data_o/glove/base.bin"
QUERY_PATH = "query0.bin"  
GROUNDTRUTH_PATH = "ansglove.txt"  # Ground truth file with true top-K indices

# Runner executable (without extension - will auto-detect .exe on Windows)
RUNNER_PATH = "runner"

# Cache directory for graph files
CACHE_DIR = "graph_cache"

# Output file
OUTPUT_CSV = "experiment_results.csv"

# Search parameter sweep ranges
EF_RANGE = [20, 40, 60, 80, 100, 150, 200, 300, 400, 600, 700, 800, 1000]
GAMMA_RANGE = [0.0, 0.1, 0.14, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22]

# For Group E: M sweep
M_RANGE = [24, 48, 96, 128, 200, 256]
EF_RANGE_FOR_M_SWEEP = [40, 100, 200, 400, 500, 600, 700, 800, 900, 1000]

# Default build parameters
DEFAULT_M = 96  # Groups A, B, C, D use M=96
DEFAULT_EFC = 400
DEFAULT_K = 10

# ==================== Data Classes ====================

@dataclass
class BuildConfig:
    """Configuration for graph building"""
    M: int = DEFAULT_M
    efC: int = DEFAULT_EFC
    onng: bool = True
    onng_out: int = 96
    onng_in: int = 144
    onng_min: int = 64
    bfs: bool = True
    
    def get_cache_filename(self) -> str:
        """Generate unique cache filename based on build parameters"""
        param_str = f"M{self.M}_efC{self.efC}_onng{int(self.onng)}_bfs{int(self.bfs)}"
        if self.onng:
            param_str += f"_oo{self.onng_out}_oi{self.onng_in}_om{self.onng_min}"
        # Add hash for extra uniqueness
        hash_suffix = hashlib.md5(param_str.encode()).hexdigest()[:8]
        return f"graph_{param_str}_{hash_suffix}.bin"


@dataclass
class SearchConfig:
    """Configuration for search"""
    strategy: str = "fixed"  # "fixed" or "gamma"
    ef: int = 100
    gamma: float = 0.19
    k: int = DEFAULT_K
    simd: bool = True


@dataclass
class ExperimentResult:
    """Single experiment result"""
    group_name: str
    M: int
    efC: int
    onng: bool
    bfs: bool
    simd: bool
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
    count_dist: bool = False
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
    
    # Search args
    cmd.extend([f"--k={search_cfg.k}"])
    cmd.extend([f"--strategy={search_cfg.strategy}"])
    
    if search_cfg.strategy == "gamma":
        cmd.extend([f"--gamma={search_cfg.gamma}"])
    else:
        cmd.extend([f"--ef={search_cfg.ef}"])
    
    if not search_cfg.simd:
        cmd.append("--no-simd")
    
    # Debug args
    if count_dist:
        cmd.append("--count-dist")
    
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
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            print(f"  [ERROR] Runner failed: {result.stderr[:200]}", file=sys.stderr)
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
        simd=search_cfg.simd,
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


# ==================== Experiment Groups ====================

def ensure_cache_dir():
    """Ensure cache directory exists"""
    os.makedirs(CACHE_DIR, exist_ok=True)


def get_cache_path(build_cfg: BuildConfig) -> str:
    """Get full cache path for a build configuration"""
    return os.path.join(CACHE_DIR, build_cfg.get_cache_filename())


def run_group_a_baseline(results: List[ExperimentResult]):
    """
    Group A: Search Strategy Comparison (Baseline)
    Build: M=96, efC=400, ONNG=True, BFS=True
    Run 1: Strategy=Fixed, sweep ef
    Run 2: Strategy=Gamma, sweep gamma
    """
    print("\n" + "="*60)
    print("GROUP A: Search Strategy Comparison (Baseline)")
    print("="*60)
    
    build_cfg = BuildConfig(M=DEFAULT_M, efC=DEFAULT_EFC, onng=True, bfs=True)
    cache_path = get_cache_path(build_cfg)
    
    # Run 1: Fixed EF sweep
    print("\n[A.1] Strategy=Fixed, sweeping EF...")
    for ef in EF_RANGE:
        print(f"  Running: ef={ef}...", end=" ", flush=True)
        search_cfg = SearchConfig(strategy="fixed", ef=ef, simd=True)
        result = run_two_pass_experiment("A_Baseline_Fixed", build_cfg, search_cfg, cache_path, float(ef))
        if result:
            results.append(result)
            print(f"Recall={result.recall10:.4f}, QPS={result.qps:.1f}")
        else:
            print("FAILED")
    
    # Run 2: Gamma sweep
    print("\n[A.2] Strategy=Gamma, sweeping gamma...")
    for gamma in GAMMA_RANGE:
        print(f"  Running: gamma={gamma}...", end=" ", flush=True)
        search_cfg = SearchConfig(strategy="gamma", gamma=gamma, simd=True)
        result = run_two_pass_experiment("A_Baseline_Gamma", build_cfg, search_cfg, cache_path, gamma)
        if result:
            results.append(result)
            print(f"Recall={result.recall10:.4f}, QPS={result.qps:.1f}")
        else:
            print("FAILED")


def run_group_b_no_onng(results: List[ExperimentResult]):
    """
    Group B: Ablation - Graph Optimization (No ONNG)
    Build: M=96, efC=400, ONNG=False, BFS=True
    Run: Strategy=Gamma, sweep gamma
    """
    print("\n" + "="*60)
    print("GROUP B: Ablation - No ONNG")
    print("="*60)
    
    build_cfg = BuildConfig(M=DEFAULT_M, efC=DEFAULT_EFC, onng=False, bfs=True)
    cache_path = get_cache_path(build_cfg)
    
    print("\n[B] Strategy=Gamma, sweeping gamma (No ONNG)...")
    for gamma in GAMMA_RANGE:
        print(f"  Running: gamma={gamma}...", end=" ", flush=True)
        search_cfg = SearchConfig(strategy="gamma", gamma=gamma, simd=True)
        result = run_two_pass_experiment("B_NoONNG", build_cfg, search_cfg, cache_path, gamma)
        if result:
            results.append(result)
            print(f"Recall={result.recall10:.4f}, QPS={result.qps:.1f}")
        else:
            print("FAILED")


def run_group_c_no_bfs(results: List[ExperimentResult]):
    """
    Group C: Ablation - Memory Layout (No BFS)
    Build: M=96, efC=400, ONNG=True, BFS=False
    Run: Strategy=Gamma, sweep gamma
    """
    print("\n" + "="*60)
    print("GROUP C: Ablation - No BFS Reordering")
    print("="*60)
    
    build_cfg = BuildConfig(M=DEFAULT_M, efC=DEFAULT_EFC, onng=True, bfs=False)
    cache_path = get_cache_path(build_cfg)
    
    print("\n[C] Strategy=Gamma, sweeping gamma (No BFS)...")
    for gamma in GAMMA_RANGE:
        print(f"  Running: gamma={gamma}...", end=" ", flush=True)
        search_cfg = SearchConfig(strategy="gamma", gamma=gamma, simd=True)
        result = run_two_pass_experiment("C_NoBFS", build_cfg, search_cfg, cache_path, gamma)
        if result:
            results.append(result)
            print(f"Recall={result.recall10:.4f}, QPS={result.qps:.1f}")
        else:
            print("FAILED")


def run_group_d_no_simd(results: List[ExperimentResult]):
    """
    Group D: Ablation - Instruction Set (No SIMD)
    Build: M=96, efC=400, ONNG=True, BFS=True
    Run: Strategy=Gamma, sweep gamma (with --no-simd flag)
    """
    print("\n" + "="*60)
    print("GROUP D: Ablation - No SIMD (Scalar Distance)")
    print("="*60)
    
    build_cfg = BuildConfig(M=DEFAULT_M, efC=DEFAULT_EFC, onng=True, bfs=True)
    cache_path = get_cache_path(build_cfg)
    
    print("\n[D] Strategy=Gamma, sweeping gamma (No SIMD)...")
    for gamma in GAMMA_RANGE:
        print(f"  Running: gamma={gamma}...", end=" ", flush=True)
        search_cfg = SearchConfig(strategy="gamma", gamma=gamma, simd=False)
        result = run_two_pass_experiment("D_NoSIMD", build_cfg, search_cfg, cache_path, gamma)
        if result:
            results.append(result)
            print(f"Recall={result.recall10:.4f}, QPS={result.qps:.1f}")
        else:
            print("FAILED")


def run_group_e_m_sweep(results: List[ExperimentResult]):
    """
    Group E: Impact of Graph Density (M)
    Build: Sweep M in [24, 48, 96, 128, 200, 256], ONNG=True, BFS=True
    Run: Strategy=Gamma, sweep gamma
    """
    print("\n" + "="*60)
    print("GROUP E: Graph Density (M) Impact")
    print("="*60)
    
    for M in M_RANGE:
        print(f"\n[E] M={M}, sweeping gamma...")
        
        # Adjust ONNG params based on M
        onng_out = M
        onng_in = int(M * 1.5)
        onng_min = max(16, M // 2)
        
        build_cfg = BuildConfig(
            M=M, efC=DEFAULT_EFC, 
            onng=True, onng_out=onng_out, onng_in=onng_in, onng_min=onng_min,
            bfs=True
        )
        cache_path = get_cache_path(build_cfg)
        
        for gamma in GAMMA_RANGE:
            print(f"  Running: M={M}, gamma={gamma}...", end=" ", flush=True)
            search_cfg = SearchConfig(strategy="gamma", gamma=gamma, simd=True)
            result = run_two_pass_experiment(f"E_M{M}", build_cfg, search_cfg, cache_path, gamma)
            if result:
                results.append(result)
                print(f"Recall={result.recall10:.4f}, QPS={result.qps:.1f}")
            else:
                print("FAILED")


# ==================== Output ====================

def save_results_to_csv(results: List[ExperimentResult], filename: str):
    """Save all results to CSV file"""
    if not results:
        print("No results to save!")
        return
    
    fieldnames = [
        "group_name", "M", "efC", "onng", "bfs", "simd",
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
    print("ANN Ablation Study - Experiment Runner")
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
    
    # Collect all results
    all_results: List[ExperimentResult] = []
    
    start_time = time.time()
    
    # Run all experiment groups
    try:
        run_group_a_baseline(all_results)
        run_group_b_no_onng(all_results)
        run_group_c_no_bfs(all_results)
        run_group_d_no_simd(all_results)
        run_group_e_m_sweep(all_results)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving partial results...")
    
    elapsed = time.time() - start_time
    
    # Save results
    save_results_to_csv(all_results, OUTPUT_CSV)
    
    # Print summary
    print_summary(all_results)
    
    print(f"\nTotal experiments: {len(all_results)}")
    print(f"Total time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()

