#!/usr/bin/env python3
"""Auto-tuning script for Solution parameters.

This script explores combinations of HNSW construction/search parameters and
ONNG post-optimization knobs, compiles the project, runs the checker to gather
latency and accuracy metrics, and records combinations meeting recall criteria.

It also builds an instrumented variant (based on 1/MySolutionWithCount.cpp) to
collect average distance computations for qualifying configurations.
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

RECALL_THRESHOLD = 99.0

INIT_PATTERN = re.compile(
    r"initHNSW\(\s*n\s*,\s*\d+\s*,\s*114514\s*,\s*\d+\s*,\s*\d+\s*,\s*data_size_\s*,\s*worker_count\s*\);"
)
ADD_PATTERN = re.compile(r"addReverseEdgesDirectly\(\s*0\s*,\s*\d+\s*\);")
PRUNE_PATTERN = re.compile(r"pruneRedundantPathsDirectly\(\s*0\s*,\s*\d+\s*\);")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-tune Solution parameters")
    parser.add_argument(
        "--gcc",
        default=os.environ.get("GCC", "C:/TDM-GCC-64/bin/g++.exe"),
        help="Path to g++ executable (default: %(default)s or GCC env var)",
    )
    parser.add_argument(
        "--dataset",
        default="1",
        help="Dataset case passed to checker (default: %(default)s)",
    )
    parser.add_argument(
        "--ans",
        default="./ans.txt",
        help="Path to answer file used by checker (default: %(default)s)",
    )
    parser.add_argument(
        "--algo",
        default="solution",
        help="Algo argument forwarded to checker (default: %(default)s)",
    )
    parser.add_argument(
        "--results",
        default="results.txt",
        help="Output file collecting successful configurations (default: %(default)s)",
    )
    return parser.parse_args()


def update_init_call(source: str, m_val: int, efc: int, efs: int) -> str:
    replacement = f"initHNSW(n, {m_val}, 114514, {efc}, {efs}, data_size_, worker_count);"
    return INIT_PATTERN.sub(replacement, source, count=1)


def update_onng_knobs(header: str, a_val: int, b_val: int) -> str:
    header = ADD_PATTERN.sub(f"addReverseEdgesDirectly(0, {a_val});", header, count=1)
    header = PRUNE_PATTERN.sub(f"pruneRedundantPathsDirectly(0, {b_val});", header, count=1)
    return header


def compile_checker(gcc_path: Path, sources: list[str], output: Path, cwd: Path) -> None:
    cmd = [str(gcc_path), "-std=c++17", "-O2", "-Wall", "-Wextra", "-pthread"] + sources + ["-o", str(output)]
    subprocess.run(cmd, cwd=cwd, check=True)


def run_checker(executable: Path, dataset: str, ans_path: str, algo: str, cwd: Path) -> subprocess.CompletedProcess:
    cmd = [str(executable), dataset, f"--ans={ans_path}", f"--algo={algo}"]
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def parse_metrics(output: str) -> Optional[Dict[str, float]]:
    def grab(pattern: str) -> Optional[float]:
        match = re.search(pattern, output)
        if not match:
            return None
        return float(match.group(1))

    metrics = {
        "build": grab(r"Build latency:\s*([0-9]+(?:\.[0-9]+)?)"),
        "search_total": grab(r"Search latency \(total\):\s*([0-9]+(?:\.[0-9]+)?)"),
        "avg_search": grab(r"Average search latency:\s*([0-9]+(?:\.[0-9]+)?)"),
        "top1": grab(r"Top-1 accuracy:\s*([0-9]+(?:\.[0-9]+)?)"),
        "recall": grab(r"Recall@K \(mean\):\s*([0-9]+(?:\.[0-9]+)?)"),
    }
    if any(v is None for v in metrics.values()):
        return None
    return metrics


def parse_distance(output: str) -> Optional[float]:
    match = re.search(r"Average distance computations:\s*([0-9]+(?:\.[0-9]+)?)", output)
    return float(match.group(1)) if match else None


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent
    gcc_path = Path(args.gcc)
    if not gcc_path.exists():
        print(f"[ERROR] g++ not found at {gcc_path}", file=sys.stderr)
        return 2

    my_solution_cpp = repo_root / "MySolution.cpp"
    my_solution_with_count_cpp = repo_root / "MySolutionWithCount.cpp"
    my_solution_h = repo_root / "MySolution.h"

    if not (my_solution_cpp.exists() and my_solution_with_count_cpp.exists() and my_solution_h.exists()):
        print("[ERROR] Required source files are missing.", file=sys.stderr)
        return 2

    original_ms_cpp = my_solution_cpp.read_text(encoding="utf-8")
    original_ms_with_count = my_solution_with_count_cpp.read_text(encoding="utf-8")
    original_ms_h = my_solution_h.read_text(encoding="utf-8")

    results_path = repo_root / args.results
    results_path.write_text("", encoding="utf-8")

    combos_iter = itertools.product(
        [16, 32],
        range(400, 1001, 100),
        [70, 100, 150, 200],
        range(50, 101, 10),
        [20, 25, 30],
    )

    checker_exe = repo_root / "checker.exe"
    checker_count_exe = repo_root / "checker_with_count.exe"

    try:
        with results_path.open("w", encoding="utf-8") as results_file:
            for (m_val, efc, efs, a_val, b_val) in combos_iter:
                combo_key = f"M={m_val}, efConstruction={efc}, efSearch={efs}, a={a_val}, b={b_val}"

                ms_cpp_content = update_init_call(original_ms_cpp, m_val, efc, efs)
                ms_with_count_content = update_init_call(original_ms_with_count, m_val, efc, efs)
                ms_h_content = update_onng_knobs(original_ms_h, a_val, b_val)

                try:
                    my_solution_cpp.write_text(ms_cpp_content, encoding="utf-8")
                    my_solution_with_count_cpp.write_text(ms_with_count_content, encoding="utf-8")
                    my_solution_h.write_text(ms_h_content, encoding="utf-8")
                except OSError as exc:
                    print(f"[ERROR] Failed to write sources: {exc}")
                    return 1

                print(f"[BUILD] {combo_key}")

                try:
                    compile_checker(
                        gcc_path,
                        ["checker.cpp", "MySolution.cpp", "Brute.cpp"],
                        checker_exe,
                        repo_root,
                    )
                except subprocess.CalledProcessError as exc:
                    print(f"[ERROR] Compile failed for main variant: {exc}")
                    continue

                try:
                    result = run_checker(checker_exe, args.dataset, args.ans, args.algo, repo_root)
                except subprocess.CalledProcessError as exc:
                    print(f"[ERROR] Checker run failed (main) for {combo_key}: {exc}")
                    continue

                metrics = parse_metrics(result.stdout)
                if metrics is None:
                    print(f"[ERROR] Unable to parse metrics for {combo_key}")
                    continue

                if metrics["recall"] < RECALL_THRESHOLD:
                    print(f"[INFO] Recall {metrics['recall']:.2f}% below threshold for {combo_key}")
                    continue

                try:
                    compile_checker(
                        gcc_path,
                        ["checker.cpp", "1/MySolutionWithCount.cpp", "Brute.cpp"],
                        checker_count_exe,
                        repo_root,
                    )
                except subprocess.CalledProcessError as exc:
                    print(f"[ERROR] Compile failed for instrumented variant: {exc}")
                    continue

                try:
                    count_result = run_checker(checker_count_exe, args.dataset, args.ans, args.algo, repo_root)
                except subprocess.CalledProcessError as exc:
                    print(f"[ERROR] Checker run failed (instrumented) for {combo_key}: {exc}")
                    continue

                distance_avg = parse_distance(count_result.stdout)
                if distance_avg is None:
                    print(f"[ERROR] Unable to parse distance computations for {combo_key}")
                    continue

                line = (
                    f"{combo_key} : "
                    f"Build latency(ms)={metrics['build']:.2f}, "
                    f"Search latency(total)(ms)={metrics['search_total']:.2f}, "
                    f"Average search latency={metrics['avg_search']:.4f}, "
                    f"Top-1 accuracy (%)={metrics['top1']:.2f}, "
                    f"Recall@K (mean) (%)={metrics['recall']:.2f}, "
                    f"Average distance computations={distance_avg:.2f}"
                )

                results_file.write(line + "\n")
                results_file.flush()
                print(f"[RECORDED] {combo_key}")
    finally:
        my_solution_cpp.write_text(original_ms_cpp, encoding="utf-8")
        my_solution_with_count_cpp.write_text(original_ms_with_count, encoding="utf-8")
        my_solution_h.write_text(original_ms_h, encoding="utf-8")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
