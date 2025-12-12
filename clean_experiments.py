#!/usr/bin/env python3
"""
clean_experiments.py

用于在不影响其他实验结果的情况下，按条件从 run_experiments.py 的断点续跑文件
`experiment_results.pkl` 中删除部分结果，并（可选）重导出 `experiment_results.csv`。

典型场景：
- 某些 code_bits 组跑错了：删掉对应结果 -> 重新编译 C++ -> 重新跑 run_experiments.py
- 只想重跑某个 strategy / 某个 M/efC / 某个 group_name 子集

注意：
- run_experiments.py 的“是否跳过已完成”完全依赖 pkl（而不是 csv）。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import shutil
import sys
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _import_run_experiments():
    # 让 unpickle 能找到 run_experiments.ExperimentResult 类
    import run_experiments  # noqa: F401
    return run_experiments


def _load_resume(pkl_path: str) -> Dict[str, Any]:
    run_experiments = _import_run_experiments()
    # 兼容性：如果 pkl 是通过 `python run_experiments.py` 运行生成的，
    # dataclass 的 qualname 很可能被写成 "__main__.ExperimentResult"。
    # 当我们从 clean_experiments.py 里反序列化时，__main__ 变成 clean_experiments，
    # 会导致 AttributeError: Can't get attribute 'ExperimentResult' on <module '__main__'>。
    # 这里把 run_experiments 里的同名类临时挂到当前 __main__ 模块上，保证旧 pkl 可读。
    try:
        import __main__ as _main_mod
        for _name in ("ExperimentResult", "BuildConfig", "SearchConfig"):
            if hasattr(run_experiments, _name):
                setattr(_main_mod, _name, getattr(run_experiments, _name))
    except Exception:
        # 仅为兼容旧 pkl；失败也不应阻塞（新 pkl 通常不会依赖该 hack）
        pass
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Resume pkl not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = run_experiments.pickle.load(f)  # type: ignore[attr-defined]
    if not isinstance(data, dict) or "results" not in data:
        raise ValueError(f"Invalid resume pkl format: {pkl_path}")
    if not isinstance(data["results"], list):
        raise ValueError(f"Invalid resume pkl: results is not a list: {pkl_path}")
    return data


def _save_resume(pkl_path: str, data: Dict[str, Any]) -> None:
    run_experiments = _import_run_experiments()
    with open(pkl_path, "wb") as f:
        run_experiments.pickle.dump(data, f)  # type: ignore[attr-defined]


def _maybe_backup(path: str, enable_backup: bool) -> Optional[str]:
    if not enable_backup:
        return None
    if not os.path.exists(path):
        return None
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{path}.bak_{ts}"
    shutil.copy2(path, backup_path)
    return backup_path


def _as_namespace(r: Any) -> SimpleNamespace:
    # r 是 run_experiments.ExperimentResult（dataclass），也可能是 dict（兼容）
    if hasattr(r, "__dict__"):
        return SimpleNamespace(**r.__dict__)
    if isinstance(r, dict):
        return SimpleNamespace(**r)
    # 最后兜底：尝试按常见字段 getattr
    fields = [
        "group_name",
        "M",
        "efC",
        "onng",
        "bfs",
        "simd",
        "single_layer",
        "neg_ip",
        "strategy",
        "param_val",
        "top1",
        "recall10",
        "qps",
        "ms_per_query",
        "latency_ms",
        "dist_calcs",
        "build_time_ms",
    ]
    return SimpleNamespace(**{k: getattr(r, k, None) for k in fields})


def _match_code_bits(group_name: str, code_bits: str) -> bool:
    # run_experiments.py 的 group_name 形如："{code_bits}_fixed" / "{code_bits}_gamma_static" ...
    # 这里用 prefix 匹配最稳。
    return group_name.startswith(code_bits + "_") or group_name == code_bits


def _compile_regexes(pats: Optional[List[str]]) -> Optional[List[re.Pattern[str]]]:
    if not pats:
        return None
    out: List[re.Pattern[str]] = []
    for p in pats:
        out.append(re.compile(p))
    return out


def _expr_match(expr: str, r_ns: SimpleNamespace) -> bool:
    """
    支持一个“自定义删除条件”的 Python 表达式（布尔表达式）。
    示例：
      - "r.group_name.startswith('11110_') and r.strategy == 'fixed'"
      - "r.M == 96 and r.efC == 400 and r.recall10 < 0.8"
      - "re.search(r'gamma', r.group_name) is not None"

    安全性：
    - 禁用 builtins，只提供 r / re / abs / min / max / float / int / str / round
    """
    safe_globals = {"__builtins__": {}}
    safe_locals = {
        "r": r_ns,
        "re": re,
        "abs": abs,
        "min": min,
        "max": max,
        "float": float,
        "int": int,
        "str": str,
        "round": round,
    }
    return bool(eval(expr, safe_globals, safe_locals))


def _build_matcher(args: argparse.Namespace):
    code_bits_list: List[str] = args.code_bits or []
    strategies: List[str] = args.strategy or []
    group_contains: List[str] = args.group_contains or []
    group_regexes = _compile_regexes(args.group_regex)

    Ms: List[int] = args.M or []
    efCs: List[int] = args.efC or []

    def match(r_any: Any) -> bool:
        r = _as_namespace(r_any)

        # 若传了任意条件，则必须同时满足所有“已指定”的过滤类别；
        # 同一类别内（如多 code_bits / 多 strategy）使用 OR。
        if code_bits_list:
            if not any(_match_code_bits(str(getattr(r, "group_name", "")), cb) for cb in code_bits_list):
                return False

        if strategies:
            if str(getattr(r, "strategy", "")) not in strategies:
                return False

        if group_contains:
            gn = str(getattr(r, "group_name", ""))
            if not all(s in gn for s in group_contains):
                return False

        if group_regexes:
            gn = str(getattr(r, "group_name", ""))
            if not any(p.search(gn) for p in group_regexes):
                return False

        if Ms:
            if int(getattr(r, "M", -1)) not in Ms:
                return False

        if efCs:
            if int(getattr(r, "efC", -1)) not in efCs:
                return False

        if args.param_eq is not None:
            if float(getattr(r, "param_val", float("nan"))) != float(args.param_eq):
                return False

        if args.param_min is not None:
            if float(getattr(r, "param_val", float("inf"))) < float(args.param_min):
                return False

        if args.param_max is not None:
            if float(getattr(r, "param_val", float("-inf"))) > float(args.param_max):
                return False

        if args.min_recall10 is not None:
            if float(getattr(r, "recall10", float("inf"))) >= float(args.min_recall10):
                return False

        if args.max_recall10 is not None:
            if float(getattr(r, "recall10", float("-inf"))) <= float(args.max_recall10):
                return False

        if args.min_qps is not None:
            if float(getattr(r, "qps", float("inf"))) >= float(args.min_qps):
                return False

        if args.max_qps is not None:
            if float(getattr(r, "qps", float("-inf"))) <= float(args.max_qps):
                return False

        if args.expr:
            if not _expr_match(args.expr, r):
                return False

        return True

    return match


def _summarize_by_group(results: Iterable[Any]) -> List[Tuple[str, int]]:
    counter: Dict[str, int] = {}
    for r_any in results:
        r = _as_namespace(r_any)
        gn = str(getattr(r, "group_name", ""))
        counter[gn] = counter.get(gn, 0) + 1
    return sorted(counter.items(), key=lambda x: (-x[1], x[0]))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="按条件删除 experiment_results.pkl 中的部分实验结果，并可重导出 csv。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--pkl", default=None, help="resume pkl 路径（默认使用 run_experiments.RESUME_PKL）")
    parser.add_argument("--csv", default=None, help="csv 路径（默认使用 run_experiments.OUTPUT_CSV）")
    parser.add_argument("--no-export-csv", action="store_true", help="不重导出 CSV")

    parser.add_argument("--backup", action="store_true", default=True, help="操作前自动备份 pkl/csv")
    parser.add_argument("--no-backup", action="store_false", dest="backup", help="禁用自动备份")

    parser.add_argument("--dry-run", action="store_true", help="只打印将删除的数量，不写文件")
    parser.add_argument("--show-groups", action="store_true", help="打印删除/保留的 group_name 计数摘要")

    # 条件（可组合）
    parser.add_argument("--code-bits", action="append", help="按 code_bits 删除（可重复）")
    parser.add_argument("--strategy", action="append", help="按 strategy 删除（fixed/gamma-static/gamma-dynamic，可重复）")
    parser.add_argument("--group-contains", action="append", help="group_name 必须包含该子串（可重复；全部都要包含）")
    parser.add_argument("--group-regex", action="append", help="group_name 正则匹配（可重复；满足任意一个即可）")

    parser.add_argument("--M", type=int, action="append", help="按 M 删除（可重复）")
    parser.add_argument("--efC", type=int, action="append", help="按 efC 删除（可重复）")

    parser.add_argument("--param-eq", type=float, default=None, help="param_val 等于该值（ef 或 gamma）")
    parser.add_argument("--param-min", type=float, default=None, help="param_val 最小值（含）")
    parser.add_argument("--param-max", type=float, default=None, help="param_val 最大值（含）")

    parser.add_argument("--min-recall10", type=float, default=None, help="删除 recall10 < 该值 的结果（阈值）")
    parser.add_argument("--max-recall10", type=float, default=None, help="删除 recall10 > 该值 的结果（阈值）")
    parser.add_argument("--min-qps", type=float, default=None, help="删除 qps < 该值 的结果（阈值）")
    parser.add_argument("--max-qps", type=float, default=None, help="删除 qps > 该值 的结果（阈值）")

    parser.add_argument(
        "--expr",
        type=str,
        default=None,
        help="自定义 Python 布尔表达式（变量 r 表示一条结果；可用 re）。",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="删除全部结果（危险：会清空 pkl 里的 results）",
    )

    args = parser.parse_args()

    run_experiments = _import_run_experiments()
    pkl_path = args.pkl or getattr(run_experiments, "RESUME_PKL", "experiment_results.pkl")
    csv_path = args.csv or getattr(run_experiments, "OUTPUT_CSV", "experiment_results.csv")

    data = _load_resume(pkl_path)
    results = data.get("results", [])

    # 防止误删：没有指定任何条件且没 --all -> 直接退出
    any_filter = any(
        [
            args.all,
            args.code_bits,
            args.strategy,
            args.group_contains,
            args.group_regex,
            args.M,
            args.efC,
            args.param_eq is not None,
            args.param_min is not None,
            args.param_max is not None,
            args.min_recall10 is not None,
            args.max_recall10 is not None,
            args.min_qps is not None,
            args.max_qps is not None,
            args.expr is not None,
        ]
    )
    if not any_filter:
        print("[ABORT] 你没有指定任何删除条件。请加上 --code-bits/--strategy/--group-regex/... 或者明确使用 --all。", file=sys.stderr)
        return 2

    if args.all:
        to_delete = list(results)
        to_keep: List[Any] = []
    else:
        matcher = _build_matcher(args)
        to_delete = [r for r in results if matcher(r)]
        to_keep = [r for r in results if not matcher(r)]

    print(f"[INFO] pkl: {pkl_path}")
    print(f"[INFO] csv: {csv_path}")
    print(f"[INFO] total results: {len(results)}")
    print(f"[INFO] will delete: {len(to_delete)}")
    print(f"[INFO] will keep:   {len(to_keep)}")

    if args.show_groups:
        print("\n[DELETE] group_name counts:")
        for gn, c in _summarize_by_group(to_delete)[:50]:
            print(f"  {c:6d}  {gn}")
        if len(_summarize_by_group(to_delete)) > 50:
            print("  ... (truncated)")

        print("\n[KEEP] group_name counts:")
        for gn, c in _summarize_by_group(to_keep)[:50]:
            print(f"  {c:6d}  {gn}")
        if len(_summarize_by_group(to_keep)) > 50:
            print("  ... (truncated)")

    if args.dry_run:
        print("\n[DRY-RUN] 未写入任何文件。")
        return 0

    # 备份
    pkl_bak = _maybe_backup(pkl_path, args.backup)
    csv_bak = _maybe_backup(csv_path, args.backup)
    if pkl_bak:
        print(f"[BACKUP] pkl -> {pkl_bak}")
    if csv_bak:
        print(f"[BACKUP] csv -> {csv_bak}")

    # 写回 pkl（保持其他字段）
    data["results"] = to_keep
    # idx/total 仅用于展示；这里更新 idx 以免误导
    if isinstance(data.get("idx", None), int):
        data["idx"] = len(to_keep)
    _save_resume(pkl_path, data)
    print(f"[WRITE] updated pkl: {pkl_path}")

    # 重导出 csv
    if not args.no_export_csv:
        if to_keep:
            run_experiments.save_results_to_csv(to_keep, csv_path)
        else:
            # 如果删空了，为避免 run_experiments.save_results_to_csv 打印 “No results”，
            # 我们直接写一个只有 header 的 csv
            fieldnames = [
                "group_name", "M", "efC", "onng", "bfs", "simd", "single_layer", "neg_ip",
                "strategy", "param_val", "top1", "recall10",
                "qps", "ms_per_query", "latency_ms", "dist_calcs", "build_time_ms",
            ]
            import csv as _csv

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = _csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
            print(f"[WRITE] exported empty csv (header only): {csv_path}")

    print("[DONE] 现在重新编译 C++ 后，跑 run_experiments.py 会自动重跑被删掉的那部分。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


