#!/usr/bin/env python3
"""
对比两个搜索策略的分歧点分析脚本

比较两个搜索策略（如 Gamma vs FixedEF）在访问节点顺序上的分歧点，
并生成可视化图表。
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 尝试导入 ijson 用于流式解析大文件
try:
    import ijson
    HAS_IJSON = True
except ImportError:
    HAS_IJSON = False
    print("[Warning] ijson not installed. Large files may cause memory issues.")

# 设置绘图风格
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#16213e'
plt.rcParams['axes.edgecolor'] = '#e94560'
plt.rcParams['axes.labelcolor'] = '#eaeaea'
plt.rcParams['text.color'] = '#eaeaea'
plt.rcParams['xtick.color'] = '#eaeaea'
plt.rcParams['ytick.color'] = '#eaeaea'
plt.rcParams['grid.color'] = '#4a5a7a'
plt.rcParams['grid.alpha'] = 0.3


def load_paths_streaming(path):
    """流式加载 JSON 文件，返回生成器"""
    if HAS_IJSON:
        with open(path, 'rb') as f:
            for item in ijson.items(f, 'item'):
                yield item
    else:
        # 回退到完整加载
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                yield item


def count_paths(path):
    """计算 JSON 文件中的查询数量"""
    count = 0
    if HAS_IJSON:
        with open(path, 'rb') as f:
            for _ in ijson.items(f, 'item'):
                count += 1
    else:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            count = len(data)
    return count


def find_divergence_point(steps1, steps2):
    """
    找到两个步骤序列的第一个分歧点
    
    返回: (divergence_step, min_steps, max_steps)
        - divergence_step: 第一个分歧的步数，-1 表示完全相同（或一方为子集）
        - min_steps: 较短序列的长度
        - max_steps: 较长序列的长度
    """
    min_len = min(len(steps1), len(steps2))
    max_len = max(len(steps1), len(steps2))
    
    for i in range(min_len):
        to1 = steps1[i].get('to')
        to2 = steps2[i].get('to')
        if to1 != to2:
            return i, min_len, max_len
    
    # 前 min_len 步完全相同
    if len(steps1) == len(steps2):
        return -1, min_len, max_len  # 完全相同
    else:
        return min_len, min_len, max_len  # 一方是另一方的前缀


def analyze_divergence(path1, path2, max_queries=None):
    """
    分析两个策略文件的分歧点
    
    返回: list of dict, 每个元素包含:
        - query_idx: 查询索引
        - divergence_step: 分歧步数 (-1 表示无分歧)
        - steps1: 策略1的总步数
        - steps2: 策略2的总步数
        - divergence_pct1: 分歧点占策略1总步数的百分比
        - divergence_pct2: 分歧点占策略2总步数的百分比
    """
    print(f"[1/3] Counting queries in {Path(path1).name}...")
    count1 = count_paths(path1)
    print(f"       Found {count1} queries")
    
    print(f"[2/3] Counting queries in {Path(path2).name}...")
    count2 = count_paths(path2)
    print(f"       Found {count2} queries")
    
    if count1 != count2:
        print(f"[Warning] Query count mismatch: {count1} vs {count2}")
    
    num_queries = min(count1, count2)
    if max_queries and max_queries < num_queries:
        num_queries = max_queries
        print(f"[Info] Limiting to {num_queries} queries")
    
    print(f"[3/3] Analyzing divergence points...")
    
    results = []
    gen1 = load_paths_streaming(path1)
    gen2 = load_paths_streaming(path2)
    
    for idx in range(num_queries):
        try:
            p1 = next(gen1)
            p2 = next(gen2)
        except StopIteration:
            break
        
        steps1 = p1.get('steps', [])
        steps2 = p2.get('steps', [])
        
        div_step, min_steps, max_steps = find_divergence_point(steps1, steps2)
        
        # 计算分歧点占各自总步数的百分比
        len1 = len(steps1)
        len2 = len(steps2)
        if div_step < 0:  # 无分歧
            pct1 = 100.0
            pct2 = 100.0
        elif div_step == 0:
            pct1 = 0.0
            pct2 = 0.0
        else:
            pct1 = (div_step / len1 * 100) if len1 > 0 else 0.0
            pct2 = (div_step / len2 * 100) if len2 > 0 else 0.0
        
        results.append({
            'query_idx': idx,
            'divergence_step': div_step,
            'steps1': len1,
            'steps2': len2,
            'min_steps': min_steps,
            'divergence_pct1': pct1,
            'divergence_pct2': pct2
        })
        
        if (idx + 1) % 100 == 0:
            print(f"       Processed {idx + 1}/{num_queries} queries...")
    
    print(f"       Done! Analyzed {len(results)} queries")
    return results


def plot_divergence_analysis(results, name1, name2, output_path=None):
    """
    绘制分歧点分析图表
    """
    # 提取数据
    query_indices = [r['query_idx'] for r in results]
    divergence_steps = [r['divergence_step'] for r in results]
    divergence_pcts1 = [r['divergence_pct1'] for r in results]
    divergence_pcts2 = [r['divergence_pct2'] for r in results]
    steps1 = [r['steps1'] for r in results]
    steps2 = [r['steps2'] for r in results]
    
    # 分离有分歧和无分歧的查询
    has_divergence_data = [(i, d, p1, p2) for i, d, p1, p2 in 
                           zip(query_indices, divergence_steps, divergence_pcts1, divergence_pcts2) if d >= 0]
    no_divergence = [i for i, d in zip(query_indices, divergence_steps) if d < 0]
    
    if has_divergence_data:
        div_indices, div_steps, div_pcts1, div_pcts2 = zip(*has_divergence_data)
    else:
        div_indices, div_steps, div_pcts1, div_pcts2 = [], [], [], []
    
    # 统计信息
    total_queries = len(results)
    num_with_div = len(has_divergence_data)
    num_no_div = len(no_divergence)
    
    if div_steps:
        avg_div_step = np.mean(div_steps)
        median_div_step = np.median(div_steps)
        avg_div_pct1 = np.mean(div_pcts1)
        avg_div_pct2 = np.mean(div_pcts2)
    else:
        avg_div_step = median_div_step = avg_div_pct1 = avg_div_pct2 = 0
    
    # 创建图表
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Search Strategy Divergence Analysis\n{name1} vs {name2}', 
                 fontsize=14, fontweight='bold', color='#eaeaea')
    
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    
    # ========== 图1: 分歧步数分布直方图 ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#16213e')
    
    if div_steps:
        bins = min(50, len(set(div_steps)))
        ax1.hist(div_steps, bins=bins, color='#4fc3f7', edgecolor='#0288d1', alpha=0.8)
        ax1.axvline(avg_div_step, color='#ff5252', linestyle='--', linewidth=2, 
                    label=f'Mean: {avg_div_step:,.0f}')
        ax1.axvline(median_div_step, color='#69f0ae', linestyle=':', linewidth=2,
                    label=f'Median: {median_div_step:,.0f}')
        ax1.legend(facecolor='#16213e', edgecolor='#e94560')
    else:
        ax1.text(0.5, 0.5, 'No divergence found', transform=ax1.transAxes,
                ha='center', va='center', fontsize=12, color='#eaeaea')
    
    ax1.set_xlabel('Divergence Step', fontsize=11)
    ax1.set_ylabel('Number of Queries', fontsize=11)
    ax1.set_title('Distribution of Divergence Steps', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # ========== 图2: 配对对比图 - 每个查询的两个策略百分比对比 ==========
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#16213e')
    
    if div_pcts1 and div_pcts2:
        # 散点图：X轴是Gamma百分比，Y轴是FixedEF百分比
        ax2.scatter(div_pcts1, div_pcts2, c='#4fc3f7', s=30, alpha=0.6, edgecolors='white', linewidths=0.5)
        
        # 对角线（如果两者相等）
        ax2.plot([0, 100], [0, 100], 'r--', alpha=0.5, label='Equal %')
        
        # 平均值标记线
        ax2.axvline(avg_div_pct1, color='#4fc3f7', linestyle=':', linewidth=2, alpha=0.8)
        ax2.axhline(avg_div_pct2, color='#ce93d8', linestyle=':', linewidth=2, alpha=0.8)
        
        # 平均值交点
        ax2.scatter([avg_div_pct1], [avg_div_pct2], c='#ff5252', s=150, marker='*', 
                   zorder=10, edgecolors='white', linewidths=2,
                   label=f'Mean: ({avg_div_pct1:.1f}%, {avg_div_pct2:.1f}%)')
        
        ax2.set_xlim(0, 105)
        ax2.set_ylim(0, 105)
        ax2.legend(facecolor='#16213e', edgecolor='#e94560', loc='lower right')
        
        # 添加注释
        ax2.annotate(f'{name1}\n{avg_div_pct1:.1f}%', xy=(avg_div_pct1, 5), 
                    fontsize=9, color='#4fc3f7', ha='center')
        ax2.annotate(f'{name2}\n{avg_div_pct2:.1f}%', xy=(5, avg_div_pct2), 
                    fontsize=9, color='#ce93d8', ha='left', va='center')
    else:
        ax2.text(0.5, 0.5, 'No divergence found', transform=ax2.transAxes,
                ha='center', va='center', fontsize=12, color='#eaeaea')
    
    ax2.set_xlabel(f'Divergence % of {name1} total steps', fontsize=11)
    ax2.set_ylabel(f'Divergence % of {name2} total steps', fontsize=11)
    ax2.set_title('Divergence % Comparison (per query)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # ========== 图3: 箱线图 + 配对连线图 ==========
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#16213e')
    
    if div_pcts1 and div_pcts2:
        # 箱线图
        box_data = [list(div_pcts1), list(div_pcts2)]
        bp = ax3.boxplot(box_data, positions=[1, 2], widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor='#16213e', color='#eaeaea'),
                        whiskerprops=dict(color='#eaeaea'),
                        capprops=dict(color='#eaeaea'),
                        medianprops=dict(color='#ff5252', linewidth=2),
                        flierprops=dict(marker='o', markerfacecolor='#4fc3f7', markersize=4, alpha=0.5))
        
        # 设置箱体颜色
        bp['boxes'][0].set_facecolor('#4fc3f7')
        bp['boxes'][0].set_alpha(0.5)
        bp['boxes'][1].set_facecolor('#ce93d8')
        bp['boxes'][1].set_alpha(0.5)
        
        # 配对连线 - 每个查询用线连接两个策略的百分比
        for p1, p2 in zip(div_pcts1, div_pcts2):
            ax3.plot([1, 2], [p1, p2], 'w-', alpha=0.1, linewidth=0.5)
        
        # 标注平均值
        ax3.scatter([1], [avg_div_pct1], c='#4fc3f7', s=100, marker='D', 
                   edgecolors='white', linewidths=2, zorder=10, label=f'{name1}: {avg_div_pct1:.1f}%')
        ax3.scatter([2], [avg_div_pct2], c='#ce93d8', s=100, marker='D',
                   edgecolors='white', linewidths=2, zorder=10, label=f'{name2}: {avg_div_pct2:.1f}%')
        
        # 连接平均值
        ax3.plot([1, 2], [avg_div_pct1, avg_div_pct2], 'r-', linewidth=3, alpha=0.8, zorder=9)
        
        ax3.set_xticks([1, 2])
        ax3.set_xticklabels([name1, name2])
        ax3.set_ylim(-5, 105)
        ax3.legend(facecolor='#16213e', edgecolor='#e94560', loc='lower left')
        
        # 添加差异标注
        diff = avg_div_pct1 - avg_div_pct2
        ax3.annotate(f'Δ = {diff:.1f}%', xy=(1.5, (avg_div_pct1 + avg_div_pct2) / 2),
                    fontsize=12, color='#ff5252', ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#ff5252'))
    
    ax3.set_ylabel('Divergence Point (%)', fontsize=11)
    ax3.set_title('Divergence % Distribution Comparison', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ========== 图4: 统计摘要 + 步数对比 ==========
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#16213e')
    ax4.axis('off')
    
    # 统计文本
    stats_text = f"""
    ═══════════════════════════════════════
              DIVERGENCE STATISTICS
    ═══════════════════════════════════════
    
    Total Queries Analyzed:     {total_queries:,}
    
    Queries with Divergence:    {num_with_div:,} ({100*num_with_div/total_queries:.1f}%)
    Queries without Divergence: {num_no_div:,} ({100*num_no_div/total_queries:.1f}%)
    
    ─────────────────────────────────────────
    Among queries with divergence:
    
      Mean Divergence Step:     {avg_div_step:,.0f}
      Median Divergence Step:   {median_div_step:,.0f}
      
      Mean Divergence % ({name1}): {avg_div_pct1:.1f}%
      Mean Divergence % ({name2}): {avg_div_pct2:.1f}%
    
    ─────────────────────────────────────────
    Step Count Comparison:
    
      {name1}:
        Mean steps:   {np.mean(steps1):,.0f}
        Median steps: {np.median(steps1):,.0f}
      
      {name2}:
        Mean steps:   {np.mean(steps2):,.0f}
        Median steps: {np.median(steps2):,.0f}
    
    ═══════════════════════════════════════
    """
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=9,
            verticalalignment='top', color='#eaeaea', family='monospace')
    
    try:
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    except Exception:
        pass  # Ignore tight_layout warnings for complex layouts
    
    # 保存图片
    if output_path:
        plt.savefig(output_path, dpi=150, facecolor='#1a1a2e', edgecolor='none',
                   bbox_inches='tight')
        print(f"\n[Output] Saved figure to: {output_path}")
    
    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Compare search strategy divergence points')
    parser.add_argument('path1', help='First strategy JSON file (e.g., vis_paths_s.json)')
    parser.add_argument('path2', help='Second strategy JSON file (e.g., vis_paths_fix_s.json)')
    parser.add_argument('--name1', default='Strategy 1', help='Name for first strategy')
    parser.add_argument('--name2', default='Strategy 2', help='Name for second strategy')
    parser.add_argument('--output', '-o', default='strategy_divergence.png', 
                       help='Output image path')
    parser.add_argument('--max-queries', '-n', type=int, default=None,
                       help='Maximum number of queries to analyze')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Search Strategy Divergence Analysis")
    print("=" * 60)
    print(f"\n  File 1: {args.path1}")
    print(f"  File 2: {args.path2}")
    print()
    
    # 分析分歧点
    results = analyze_divergence(args.path1, args.path2, args.max_queries)
    
    # 绘制图表
    plot_divergence_analysis(results, args.name1, args.name2, args.output)
    
    print("\n[Done] Analysis complete!")


if __name__ == '__main__':
    main()
