#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HNSW 图结构和搜索路径可视化工具（优化版）

用法:
    # 完整模式（需要graph.json，内存占用大）
    python visualize_search.py --graph graph.json --paths search_paths.json [--samples 30] [-o output_prefix]
    
    # 轻量模式（只需要paths.json，内存占用小，适合大图）
    python visualize_search.py --paths search_paths.json --light --samples 100 [-o output_prefix]
    
依赖:
    pip install numpy matplotlib scikit-learn
"""

import json
import argparse
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import warnings
warnings.filterwarnings('ignore')

# 尝试导入流式JSON解析器
try:
    import ijson
    HAS_IJSON = True
except ImportError:
    HAS_IJSON = False

# 延迟导入（轻量模式不需要这些）
PCA = None
TSNE = None

# ============================================================
# 可配置参数 - 修改这里来调整可视化行为
# ============================================================

# 散点图采样设置
SCATTER_MAX_POINTS = 100000          # 主图散点最大显示数量（超过则随机采样）
SCATTER_POINT_SIZE = 2             # 散点大小
SCATTER_ALPHA = 0.3                # 散点透明度

# 直方图设置
HISTOGRAM_BINS = 50                # 直方图柱子数量

# 布局设置
FIGURE_SIZE = (16, 10)             # 图像大小 (宽, 高)
DPI_SAVE = 150                     # 保存图像的DPI

# 改进点检测阈值
IMPROVEMENT_THRESHOLD = 0.97       # 距离下降到前一个最优的97%以下视为改进
MAX_IMPROVEMENTS_SHOW = 5          # 最多显示几个改进点
 
# 图结构可视化设置
GRAPH_MAX_NODES_PLOT = 10000       # 图结构可视化最多绘制的节点数
GRAPH_MAX_EDGES = 15000            # 图结构可视化最多绘制的边数

# 搜索路径可视化设置
PATH_MAX_DISPLAY_NODES = 1000       # 单个查询路径图最多显示的节点数

# ============================================================

# 设置中文字体和美观样式
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#16213e'
plt.rcParams['axes.edgecolor'] = '#e94560'
plt.rcParams['text.color'] = '#eaeaea'
plt.rcParams['axes.labelcolor'] = '#eaeaea'
plt.rcParams['xtick.color'] = '#eaeaea'
plt.rcParams['ytick.color'] = '#eaeaea'


def load_graph(path):
    """加载图结构"""
    print(f"Loading graph from {path}...")
    with open(path, 'r') as f:
        return json.load(f)


def load_search_paths(path):
    """加载搜索路径"""
    print(f"Loading search paths from {path}...")
    with open(path, 'r') as f:
        return json.load(f)


def reduce_dimensions(vectors, method='pca', n_components=2, perplexity=30):
    """降维到2D用于可视化"""
    vectors = np.array(vectors)
    if vectors.shape[1] <= n_components:
        return vectors[:, :n_components]
    
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    else:
        reducer = TSNE(n_components=n_components, perplexity=min(perplexity, len(vectors)-1), 
                       random_state=42, n_iter=1000)
    return reducer.fit_transform(vectors)


def plot_graph_structure(ax, coords, adjacency, entry_points, 
                         max_nodes_to_plot=10000, max_edges=15000):
    """绘制图的基本结构（清爽版）"""
    n_nodes = len(coords)
    
    # 采样节点
    if n_nodes > max_nodes_to_plot:
        sample_idx = np.random.choice(n_nodes, max_nodes_to_plot, replace=False)
        sample_set = set(sample_idx)
        plot_coords = coords[sample_idx]
        print(f"  Plotting {max_nodes_to_plot}/{n_nodes} nodes...")
    else:
        sample_idx = np.arange(n_nodes)
        sample_set = set(range(n_nodes))
        plot_coords = coords
    
    # 采样边
    edges = []
    edge_count = 0
    for i in sample_idx:
        if edge_count >= max_edges:
            break
        for j in adjacency[i] if i < len(adjacency) else []:
            if j in sample_set and i < j:
                edges.append([coords[i], coords[j]])
                edge_count += 1
                if edge_count >= max_edges:
                    break
    
    # 绘制边（非常淡）
    if edges:
        lc = LineCollection(edges, colors='#3a4a6b', linewidths=0.15, alpha=0.3)
        ax.add_collection(lc)
    
    # 绘制节点（小而透明）
    ax.scatter(plot_coords[:, 0], plot_coords[:, 1], 
               c='#4fc3f7', s=1, alpha=0.3, rasterized=True)
    
    # 高亮入口点
    valid_eps = [ep for ep in entry_points[:min(20, len(entry_points))] if ep < len(coords)]
    if valid_eps:
        ep_coords = coords[valid_eps]
        ax.scatter(ep_coords[:, 0], ep_coords[:, 1], c='#ffd700', s=80, marker='*', 
                   edgecolors='#ff8c00', linewidths=1.5, label=f'Entry Points ({len(valid_eps)})', zorder=10)
    
    ax.set_title(f'HNSW Graph Structure\n({len(plot_coords):,} nodes, {len(edges):,} edges)', 
                 fontsize=14, fontweight='bold', color='#eaeaea')
    ax.legend(loc='upper right', fontsize=9, facecolor='#16213e', edgecolor='#e94560')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')


def compute_distance_layout(vectors, query_vec, visited_nodes, result_nodes, start_ep, adjacency, max_nodes=100):
    """基于距离计算布局 - 查询点在中心，按距离排列"""
    from sklearn.manifold import MDS
    
    # 收集要显示的节点 - 优先保留关键节点（结果、入口点）
    critical = set(result_nodes) | {start_ep}
    other_visited = [n for n in visited_nodes if n not in critical]
    
    # 先加入关键节点，再填充其他访问节点
    nodes_to_show = list(critical)
    remaining = max_nodes - len(nodes_to_show)
    if remaining > 0:
        nodes_to_show.extend(other_visited[:remaining])
    
    n = len(nodes_to_show)
    
    if n == 0:
        return {}, np.array([0, 0])
    
    # 计算距离矩阵（包括查询点）
    all_vecs = [query_vec] + [vectors[i] for i in nodes_to_show]
    n_total = len(all_vecs)
    
    dist_matrix = np.zeros((n_total, n_total))
    for i in range(n_total):
        for j in range(i+1, n_total):
            d = np.sqrt(np.sum((np.array(all_vecs[i]) - np.array(all_vecs[j]))**2))
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    
    # MDS 降维 - 保持距离关系
    if n_total > 2:
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, max_iter=300)
        coords_2d = mds.fit_transform(dist_matrix)
    else:
        coords_2d = np.array([[0, 0], [1, 0]])[:n_total]
    
    # 返回节点ID到坐标的映射
    node_coords = {nodes_to_show[i]: coords_2d[i+1] for i in range(n)}
    query_coord = coords_2d[0]
    
    return node_coords, query_coord


def plot_single_search_path_full(fig, vectors, path_info, adjacency, query_idx):
    """绘制完整搜索分析图 - 使用steps中的距离信息（内存友好）"""
    visited = path_info['visited']
    steps = path_info['steps']
    results = path_info.get('results', [])
    
    if not steps:
        return
    
    # 直接使用steps中记录的距离（不重新计算，节省内存）
    step_distances = [s['dist'] for s in steps if 'dist' in s]
    
    if not step_distances:
        return
    
    # 计算收敛历史
    min_dist_so_far = float('inf')
    best_dist_history = []
    for d in step_distances:
        min_dist_so_far = min(min_dist_so_far, d)
        best_dist_history.append(min_dist_so_far)
    
    total_steps = len(step_distances)
    x = np.arange(total_steps)
    
    # 入口点距离（用第一步的距离）
    ep_dist = step_distances[0] if step_distances else 1.0
    
    # 最终结果距离（从最小的几个距离中估计）
    final_best = best_dist_history[-1]
    sorted_dists = sorted(step_distances)
    result_dists = sorted_dists[:min(10, len(sorted_dists))]  # Top-10 最小距离作为结果估计
    best_result = result_dists[0] if result_dists else final_best
    worst_result = result_dists[-1] if result_dists else final_best
    
    # 找关键改进点
    improvements = []
    prev_best = ep_dist
    for i, d in enumerate(best_dist_history):
        if d < prev_best * IMPROVEMENT_THRESHOLD:
            improvements.append((i, d, (prev_best - d) / prev_best * 100))
            prev_best = d
    
    final_best = best_dist_history[-1]
    first_best_idx = next((i for i, d in enumerate(best_dist_history) if d == final_best), total_steps-1)
    
    # === 创建双图布局 ===
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1], 
                          hspace=0.15, wspace=0.15)
    
    ax1 = fig.add_subplot(gs[0, 0])  # 主图：散点+收敛曲线
    ax2 = fig.add_subplot(gs[1, 0])  # 下图：距离直方图
    ax3 = fig.add_subplot(gs[0, 1])  # 右图：统计信息
    ax4 = fig.add_subplot(gs[1, 1])  # 右下：结果排名
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor('#16213e')
    
    # ========== 主图：散点 + 收敛曲线 ==========
    # 散点（采样显示）
    if total_steps > SCATTER_MAX_POINTS:
        sample_idx = np.random.choice(total_steps, SCATTER_MAX_POINTS, replace=False)
        sample_idx = np.sort(sample_idx)
        ax1.scatter(sample_idx, [step_distances[i] for i in sample_idx], 
                   c='#4fc3f7', s=SCATTER_POINT_SIZE, alpha=SCATTER_ALPHA, rasterized=True)
    else:
        ax1.scatter(x, step_distances, c='#4fc3f7', s=SCATTER_POINT_SIZE, alpha=SCATTER_ALPHA, rasterized=True)
    
    # 收敛曲线（红色粗线）
    ax1.plot(x, best_dist_history, color='#ff5252', linewidth=2.5, label='Best so far', zorder=5)
    
    # 关键水平线
    ax1.axhline(y=ep_dist, color='#ffd700', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Entry: {ep_dist:.3f}')
    ax1.axhspan(best_result, worst_result, alpha=0.25, color='#00e676')
    ax1.axhline(y=best_result, color='#00e676', linestyle='-', linewidth=1.5, label=f'Best result: {best_result:.3f}')
    
    # 标记关键改进点
    if improvements:
        top_imp = sorted(improvements, key=lambda x: x[2], reverse=True)[:MAX_IMPROVEMENTS_SHOW]
        imp_x, imp_y, imp_pct = zip(*top_imp)
        ax1.scatter(imp_x, imp_y, c='#ff9800', s=80, marker='v', edgecolors='white', linewidths=1.5, zorder=10, label='Improvements')
    
    # 首次达到最优
    ax1.scatter([first_best_idx], [final_best], c='#00e676', s=150, marker='*', 
               edgecolors='white', linewidths=2, zorder=10, label=f'First best @ {first_best_idx}')
    
    ax1.set_xlabel('Search Step', fontsize=10, color='#eaeaea')
    ax1.set_ylabel('Distance to Query', fontsize=10, color='#eaeaea')
    ax1.set_xlim(0, total_steps)
    ax1.tick_params(labelbottom=True)  # 显示X轴刻度
    ax1.legend(loc='upper right', fontsize=7, facecolor='#16213e', edgecolor='#e94560', 
               labelcolor='#eaeaea', framealpha=0.9)
    ax1.grid(True, alpha=0.15, color='#ffffff')
    
    # ========== 下图：距离分布直方图 ==========
    # X轴是距离值，Y轴是出现次数
    ax2.hist(step_distances, bins=HISTOGRAM_BINS, color='#4fc3f7', edgecolor='#0288d1', alpha=0.7, range=(min(step_distances)*0.95, max(step_distances)*1.05))
    ax2.axvline(ep_dist, color='#ffd700', linestyle='--', linewidth=2, label=f'Entry: {ep_dist:.3f}')
    ax2.axvline(best_result, color='#00e676', linestyle='-', linewidth=2, label=f'Best: {best_result:.3f}')
    
    ax2.set_xlabel('Distance to Query', fontsize=10, color='#eaeaea')
    ax2.set_ylabel('Count', fontsize=10, color='#eaeaea')
    ax2.legend(loc='upper right', fontsize=8, facecolor='#16213e', edgecolor='#e94560', labelcolor='#eaeaea')
    ax2.grid(True, alpha=0.15, color='#ffffff')
    
    # 标注统计信息
    above_entry = sum(1 for d in step_distances if d > ep_dist)
    below_entry = sum(1 for d in step_distances if d <= ep_dist)
    in_result_range = sum(1 for d in step_distances if d <= worst_result)
    ax2.text(0.02, 0.95, 
            f'Distance > Entry: {100*above_entry/total_steps:.1f}% (wasted)\n'
            f'Distance <= Entry: {100*below_entry/total_steps:.1f}% (useful)\n'
            f'In result range: {100*in_result_range/total_steps:.1f}%',
            transform=ax2.transAxes, fontsize=8, ha='left', va='top', color='#eaeaea',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', alpha=0.9))
    
    # ========== 右上图：统计信息面板 ==========
    ax3.axis('off')
    
    efficiency = 100 * first_best_idx / total_steps
    improvement_pct = (1 - final_best / ep_dist) * 100
    waste_ratio = 100 * (total_steps - first_best_idx) / total_steps
    
    stats_text = f"""Query #{query_idx}

Total Steps: {total_steps:,}
Visited Nodes: {len(visited):,}

Entry Distance: {ep_dist:.4f}
Best Distance: {final_best:.4f}
Improvement: {improvement_pct:.1f}%

First Best @ Step {first_best_idx:,}
  = {efficiency:.1f}% of search
  
Wasted Steps: {waste_ratio:.1f}%
"""
    
    ax3.text(0.1, 0.95, stats_text, transform=ax3.transAxes, fontsize=10,
            verticalalignment='top', color='#eaeaea', family='monospace')
    
    # ========== 右下图：Top-K 结果距离条形图 ==========
    if result_dists:
        sorted_dists = sorted(result_dists)[:10]
        ranks = [f'#{i+1}' for i in range(len(sorted_dists))]
        
        # 水平条形图
        colors = ['#00e676' if i == 0 else '#4fc3f7' for i in range(len(sorted_dists))]
        bars = ax4.barh(ranks[::-1], sorted_dists[::-1], color=colors[::-1], edgecolor='white', height=0.6)
        
        # 在条形上标注距离值
        for bar, dist in zip(bars, sorted_dists[::-1]):
            ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{dist:.4f}', va='center', ha='left', fontsize=8, color='#eaeaea')
        
        ax4.set_xlabel('Distance', fontsize=9, color='#eaeaea')
        ax4.set_title('Top-K Results', fontsize=10, color='#eaeaea', pad=5)
        ax4.set_xlim(0, max(sorted_dists) * 1.15)
        ax4.tick_params(colors='#eaeaea')
        ax4.grid(True, alpha=0.15, color='#ffffff', axis='x')
    else:
        ax4.axis('off')
        ax4.text(0.5, 0.5, 'No results', transform=ax4.transAxes, ha='center', va='center', color='#888888')


def plot_single_search_path(ax, vectors, path_info, adjacency, title=""):
    """简化版：单个axes绘制（兼容旧调用）"""
    # 这个函数保留但不再使用，实际使用 plot_single_search_path_full
    pass


def plot_search_path_light(fig, path_info, query_idx):
    """轻量版搜索分析图 - 直接使用steps中的距离信息，不需要加载vectors"""
    steps = path_info['steps']
    results = path_info.get('results', [])
    visited = path_info.get('visited', [])
    
    if not steps:
        return
    
    # 直接从steps中提取距离信息（转float以兼容ijson的Decimal类型）
    step_distances = [float(s['dist']) for s in steps if 'dist' in s]
    
    if not step_distances:
        return
    
    # 计算收敛历史
    min_dist_so_far = float('inf')
    best_dist_history = []
    for d in step_distances:
        min_dist_so_far = min(min_dist_so_far, d)
        best_dist_history.append(min_dist_so_far)
    
    total_steps = len(step_distances)
    x = np.arange(total_steps)
    
    # 入口点距离（第一步的距离作为参考）
    ep_dist = step_distances[0] if step_distances else 1.0
    
    # 最终结果距离
    final_best = best_dist_history[-1]
    
    # ===== 计算Top-K结果的发现时间 =====
    # 使用实际的 results 字段（最终 Top-K 结果节点 ID）
    # 在 steps 中查找这些节点首次出现的步数
    results = path_info.get('results', [])
    steps = path_info.get('steps', [])
    
    top_k_info = []  # [(step_idx, distance, node_id), ...]
    
    if results and steps:
        # 为每个结果节点找到它首次出现的步数
        for result_node in results:
            for step_idx, step in enumerate(steps):
                if step.get('to') == result_node:
                    dist = float(step.get('dist', 0))
                    top_k_info.append((step_idx, dist, result_node))
                    break
        # 按步数排序（先发现的排前面）
        top_k_info.sort(key=lambda x: x[0])
    else:
        # 降级：用旧方法（距离最小的10个）
        K = 10
        sorted_with_idx = sorted(enumerate(step_distances), key=lambda x: x[1])
        seen_dists = set()
        for step_idx, dist in sorted_with_idx:
            dist_key = round(dist, 6)
            if dist_key not in seen_dists:
                seen_dists.add(dist_key)
                top_k_info.append((step_idx, dist, -1))
                if len(top_k_info) >= K:
                    break
    
    # 计算Top-K全部找到的步数
    if top_k_info:
        last_topk_step = max(info[0] for info in top_k_info)
        last_topk_dist = max(info[1] for info in top_k_info)
    else:
        last_topk_step = total_steps - 1
        last_topk_dist = final_best
    
    # 找关键改进点
    improvements = []
    prev_best = ep_dist
    for i, d in enumerate(best_dist_history):
        if d < prev_best * IMPROVEMENT_THRESHOLD:
            improvements.append((i, d, (prev_best - d) / prev_best * 100))
            prev_best = d
    
    first_best_idx = next((i for i, d in enumerate(best_dist_history) if d == final_best), total_steps-1)
    
    # ===== 计算"打转"指标 =====
    # 后半段搜索的距离波动
    mid_point = total_steps // 2
    late_distances = step_distances[mid_point:]
    if late_distances:
        late_mean = np.mean(late_distances)
        late_std = np.std(late_distances)
        late_min = min(late_distances)
        # 如果后半段最小距离没有明显改善，说明在打转
        is_spinning = (late_min > final_best * 1.1)  # 后半段最小距离比最优差10%以上
    else:
        late_mean, late_std, is_spinning = 0, 0, False
    
    # === 创建双图布局 ===
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1], 
                          hspace=0.15, wspace=0.15)
    
    ax1 = fig.add_subplot(gs[0, 0])  # 主图
    ax2 = fig.add_subplot(gs[1, 0])  # 直方图
    ax3 = fig.add_subplot(gs[0, 1])  # 统计
    ax4 = fig.add_subplot(gs[1, 1])  # 额外信息
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor('#16213e')
    
    # ========== 主图 ==========
    # 采样显示（避免绘制太多点）
    if total_steps > SCATTER_MAX_POINTS:
        sample_idx = np.random.choice(total_steps, SCATTER_MAX_POINTS, replace=False)
        sample_idx = np.sort(sample_idx)
        ax1.scatter(sample_idx, [step_distances[i] for i in sample_idx], 
                   c='#4fc3f7', s=SCATTER_POINT_SIZE, alpha=SCATTER_ALPHA, rasterized=True)
    else:
        ax1.scatter(x, step_distances, c='#4fc3f7', s=SCATTER_POINT_SIZE, alpha=SCATTER_ALPHA, rasterized=True)
    
    # 收敛曲线
    ax1.plot(x, best_dist_history, color='#ff5252', linewidth=2.5, label='Best so far', zorder=5)
    
    # 关键水平线
    ax1.axhline(y=ep_dist, color='#ffd700', linestyle='--', linewidth=1.5, alpha=0.8, label=f'First step: {ep_dist:.3f}')
    ax1.axhline(y=final_best, color='#00e676', linestyle='-', linewidth=1.5, label=f'Best: {final_best:.3f}')
    
    # 标记改进点
    if improvements:
        top_imp = sorted(improvements, key=lambda x: x[2], reverse=True)[:MAX_IMPROVEMENTS_SHOW]
        imp_x, imp_y, _ = zip(*top_imp)
        ax1.scatter(imp_x, imp_y, c='#ff9800', s=80, marker='v', edgecolors='white', linewidths=1.5, zorder=10, label='Improvements')
    
    # 首次最优 (Top-1)
    ax1.scatter([first_best_idx], [final_best], c='#00e676', s=150, marker='*', 
               edgecolors='white', linewidths=2, zorder=10, label=f'Top-1 @ step {first_best_idx}')
    
    # Top-K 全部找到的位置
    if last_topk_step > first_best_idx:
        ax1.axvline(x=last_topk_step, color='#ba68c8', linestyle=':', linewidth=2, alpha=0.8,
                   label=f'Top-{len(top_k_info)} complete @ {last_topk_step}')
        ax1.scatter([last_topk_step], [last_topk_dist], c='#ba68c8', s=120, marker='D',
                   edgecolors='white', linewidths=2, zorder=10)
    
    ax1.set_xlabel('Search Step', fontsize=10, color='#eaeaea')
    ax1.set_ylabel('Distance to Query', fontsize=10, color='#eaeaea')
    ax1.set_xlim(0, total_steps)
    ax1.tick_params(labelbottom=True)
    ax1.legend(loc='upper right', fontsize=7, facecolor='#16213e', edgecolor='#e94560', 
               labelcolor='#eaeaea', framealpha=0.9)
    ax1.grid(True, alpha=0.15, color='#ffffff')
    
    # 如果在"打转"，添加警告标注
    if is_spinning:
        ax1.text(0.5, 0.02, '⚠ Late search not improving (spinning)', 
                transform=ax1.transAxes, fontsize=9, ha='center', color='#ff9800',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor='#ff9800', alpha=0.9))
    
    # ========== 直方图 ==========
    ax2.hist(step_distances, bins=HISTOGRAM_BINS, color='#4fc3f7', edgecolor='#0288d1', alpha=0.7)
    ax2.axvline(ep_dist, color='#ffd700', linestyle='--', linewidth=2, label=f'First: {ep_dist:.3f}')
    ax2.axvline(final_best, color='#00e676', linestyle='-', linewidth=2, label=f'Best: {final_best:.3f}')
    
    ax2.set_xlabel('Distance to Query', fontsize=10, color='#eaeaea')
    ax2.set_ylabel('Count', fontsize=10, color='#eaeaea')
    ax2.legend(loc='upper right', fontsize=8, facecolor='#16213e', edgecolor='#e94560', labelcolor='#eaeaea')
    ax2.grid(True, alpha=0.15, color='#ffffff')
    
    # 统计
    above_first = sum(1 for d in step_distances if d > ep_dist)
    below_first = sum(1 for d in step_distances if d <= ep_dist)
    ax2.text(0.02, 0.95, 
            f'Dist > First: {100*above_first/total_steps:.1f}%\n'
            f'Dist <= First: {100*below_first/total_steps:.1f}%',
            transform=ax2.transAxes, fontsize=8, ha='left', va='top', color='#eaeaea',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', alpha=0.9))
    
    # ========== 统计面板 ==========
    ax3.axis('off')
    
    efficiency_top1 = 100 * first_best_idx / total_steps
    efficiency_topk = 100 * last_topk_step / total_steps
    improvement_pct = (1 - final_best / ep_dist) * 100 if ep_dist > 0 else 0
    waste_after_topk = 100 * (total_steps - last_topk_step) / total_steps
    
    stats_text = f"""Query #{query_idx}

Total Steps: {total_steps:,}
Visited Nodes: {len(visited):,}

First Step Dist: {ep_dist:.4f}
Best Distance: {final_best:.4f}
Improvement: {improvement_pct:.1f}%

Top-1 found @ step {first_best_idx:,}
  = {efficiency_top1:.1f}% of search

Top-{len(top_k_info)} complete @ step {last_topk_step:,}
  = {efficiency_topk:.1f}% of search
  
Wasted after Top-K: {waste_after_topk:.1f}%
"""
    
    # 添加打转警告
    if is_spinning:
        stats_text += f"\n⚠ Spinning detected!"
        stats_text += f"\n  Late mean dist: {late_mean:.3f}"
    
    ax3.text(0.05, 0.98, stats_text, transform=ax3.transAxes, fontsize=9,
            verticalalignment='top', color='#eaeaea', family='monospace')
    
    # ========== Top-K 结果详情 ==========
    ax4.axis('off')
    
    # 按距离排序显示（最近的排前面）
    top_k_sorted_by_dist = sorted(top_k_info, key=lambda x: x[1])
    
    topk_text = f"Top-{len(top_k_info)} Results:\n"
    topk_text += "─" * 26 + "\n"
    topk_text += f"{'Rank':<5}{'Dist':<10}{'Step':<8}{'%':<6}\n"
    topk_text += "─" * 26 + "\n"
    
    for rank, (step_idx, dist, node_id) in enumerate(top_k_sorted_by_dist):
        pct = 100 * step_idx / total_steps
        topk_text += f"#{rank+1:<4}{dist:<10.4f}{step_idx:<8,}{pct:>5.1f}%\n"
    
    # 添加总结行：Top-K 全部找到所需的步数
    topk_text += "─" * 26 + "\n"
    topk_text += f"★ All Top-{len(top_k_info)} found:\n"
    topk_text += f"   Step {last_topk_step:,} ({efficiency_topk:.1f}%)\n"
    topk_text += f"   Waste: {waste_after_topk:.1f}%\n"
    
    ax4.text(0.05, 0.98, topk_text, transform=ax4.transAxes, fontsize=8,
            verticalalignment='top', color='#eaeaea', family='monospace')


def plot_statistics(all_paths, output_path=None):
    """绘制搜索统计图"""
    visited_counts = [len(p['visited']) for p in all_paths]
    step_counts = [len(p['steps']) for p in all_paths]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')
    
    # 访问节点数分布
    ax1 = axes[0]
    ax1.hist(visited_counts, bins=30, color='#4fc3f7', edgecolor='#0288d1', alpha=0.8)
    ax1.axvline(np.mean(visited_counts), color='#ff5252', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(visited_counts):.1f}')
    ax1.axvline(np.median(visited_counts), color='#69f0ae', linestyle=':', linewidth=2,
                label=f'Median: {np.median(visited_counts):.1f}')
    ax1.set_xlabel('Visited Nodes per Query', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Distribution of Visited Nodes', fontsize=12, fontweight='bold')
    ax1.legend(facecolor='#16213e', edgecolor='#e94560')
    ax1.grid(True, alpha=0.2, color='#4a5a7a')
    
    # 搜索步数分布
    ax2 = axes[1]
    ax2.hist(step_counts, bins=30, color='#ce93d8', edgecolor='#7b1fa2', alpha=0.8)
    ax2.axvline(np.mean(step_counts), color='#ff5252', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(step_counts):.1f}')
    ax2.axvline(np.median(step_counts), color='#69f0ae', linestyle=':', linewidth=2,
                label=f'Median: {np.median(step_counts):.1f}')
    ax2.set_xlabel('Search Steps per Query', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Distribution of Search Steps', fontsize=12, fontweight='bold')
    ax2.legend(facecolor='#16213e', edgecolor='#e94560')
    ax2.grid(True, alpha=0.2, color='#4a5a7a')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(f"{output_path}_stats.png", dpi=200, bbox_inches='tight', 
                    facecolor='#1a1a2e', edgecolor='none')
        print(f"[OK] Saved: {output_path}_stats.png")
    
    return fig


def visualize(graph_path, paths_path, num_samples=30, output_path=None, method='pca', 
              max_nodes=50000, skip_graph=False):
    """主可视化函数"""
    all_paths = load_search_paths(paths_path)
    print(f"Search paths: {len(all_paths)} queries")
    
    # 如果跳过图结构，使用轻量模式绘制搜索路径
    if skip_graph or graph_path is None:
        print("[Skip Graph Mode] Only visualizing search paths...")
        visualize_light(paths_path, num_samples, output_path)
        return
    
    graph = load_graph(graph_path)
    
    vectors = np.array(graph['vectors'], dtype=np.float32)
    adjacency = graph['adjacency']
    entry_points = graph['entry_points']
    
    n_nodes = len(vectors)
    print(f"Graph: {n_nodes:,} nodes, dim={graph['dim']}")
    
    # 大图采样 - 优先保留搜索路径中的关键节点
    sample_idx = None
    old_to_new = None
    if n_nodes > max_nodes:
        print(f"Sampling {max_nodes:,} nodes from {n_nodes:,}...")
        
        # 1. 收集所有搜索路径中涉及的关键节点（必须保留）
        critical_nodes = set()
        for p in all_paths:
            critical_nodes.add(p['start_ep'])
            critical_nodes.update(p['visited'])
            critical_nodes.update(p['results'])
            for s in p['steps']:
                critical_nodes.add(s['from'])
                critical_nodes.add(s['to'])
        # 添加入口点
        critical_nodes.update(entry_points)
        # 过滤掉无效节点
        critical_nodes = {n for n in critical_nodes if 0 <= n < n_nodes}
        
        print(f"  Critical nodes from search paths: {len(critical_nodes):,}")
        
        # 2. 计算还需要随机采样多少节点
        remaining_quota = max_nodes - len(critical_nodes)
        
        if remaining_quota > 0:
            # 从非关键节点中随机采样
            non_critical = np.array([i for i in range(n_nodes) if i not in critical_nodes])
            if len(non_critical) > remaining_quota:
                random_sample = np.random.choice(non_critical, remaining_quota, replace=False)
            else:
                random_sample = non_critical
            sample_idx = np.sort(np.concatenate([np.array(list(critical_nodes)), random_sample]))
        else:
            # 关键节点已超过配额，只保留关键节点（可能超过max_nodes）
            sample_idx = np.sort(np.array(list(critical_nodes)))
            print(f"  Warning: Critical nodes ({len(critical_nodes):,}) exceed max_nodes ({max_nodes:,})")
        
        old_to_new = {old: new for new, old in enumerate(sample_idx)}
        
        vectors = vectors[sample_idx]
        adjacency = [[old_to_new[n] for n in adjacency[i] if n in old_to_new] 
                     for i in sample_idx]
        entry_points = [old_to_new[ep] for ep in entry_points if ep in old_to_new]
        
        # 转换搜索路径中的节点ID（现在所有关键节点都在采样中）
        converted_paths = []
        for p in all_paths:
            new_visited = [old_to_new[v] for v in p['visited'] if v in old_to_new]
            new_results = [old_to_new[r] for r in p['results'] if r in old_to_new]
            new_start_ep = old_to_new.get(p['start_ep'], new_visited[0] if new_visited else 0)
            
            converted_paths.append({
                'query': p['query'],
                'start_ep': new_start_ep,
                'visited': new_visited,
                'steps': [{'from': old_to_new.get(s['from'], -1), 
                           'to': old_to_new.get(s['to'], -1),
                           'dist': s['dist'], 'added': s['added']} 
                          for s in p['steps'] if s['from'] in old_to_new and s['to'] in old_to_new],
                'results': new_results
            })
        all_paths = converted_paths
        n_nodes = len(vectors)
        print(f"  After sampling: {n_nodes:,} nodes, {len(all_paths)} paths with all critical nodes preserved")
    
    # 降维
    print(f"Reducing dimensions using {method.upper()}...")
    coords = reduce_dimensions(vectors, method=method)
    
    # 选择要可视化的查询
    num_samples = min(num_samples, len(all_paths))
    sample_indices = random.sample(range(len(all_paths)), num_samples) if len(all_paths) > num_samples else list(range(len(all_paths)))
    sampled_paths = [all_paths[i] for i in sample_indices]
    
    # 查询向量也降维
    print("Processing query vectors...")
    query_vectors = np.array([p['query'] for p in sampled_paths], dtype=np.float32)
    all_vecs = np.vstack([vectors, query_vectors])
    all_coords = reduce_dimensions(all_vecs, method=method)
    coords = all_coords[:len(vectors)]
    query_coords = all_coords[len(vectors):]
    
    print("Creating visualizations...")
    
    # 自动创建输出目录
    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
            print(f"Created output directory: {out_dir}")
    
    # === 图1: 整体图结构 ===
    fig1, ax1 = plt.subplots(figsize=(14, 12))
    fig1.patch.set_facecolor('#1a1a2e')
    plot_graph_structure(ax1, coords, adjacency, entry_points)
    plt.tight_layout()
    
    if output_path:
        fig1.savefig(f"{output_path}_graph.png", dpi=200, bbox_inches='tight',
                     facecolor='#1a1a2e', edgecolor='none')
        print(f"[OK] Saved: {output_path}_graph.png")
    
    # === 图2: 每个查询单独一张大图 ===
    if output_path:
        paths_dir = f"{output_path}_paths"
        if not os.path.exists(paths_dir):
            os.makedirs(paths_dir)
        print(f"Saving individual search path plots to: {paths_dir}/")
    
    for idx, path_info in enumerate(sampled_paths):
        # 创建大图（双图布局）
        fig2 = plt.figure(figsize=(16, 10))
        fig2.patch.set_facecolor('#1a1a2e')
        
        v_count = len(path_info['visited'])
        r_count = len(path_info['results'])
        steps_count = len(path_info.get('steps', []))
        
        # 使用完整版绘图函数
        plot_single_search_path_full(fig2, vectors, path_info, adjacency, sample_indices[idx]+1)
        
        # 添加总标题
        fig2.suptitle(f"HNSW Search Analysis - Query #{sample_indices[idx]+1}\n"
                      f"Visited: {v_count:,} nodes | Results: {r_count} | Steps: {steps_count:,}",
                      fontsize=14, fontweight='bold', color='#eaeaea', y=0.98)
        
        if output_path:
            fig2.savefig(f"{paths_dir}/query_{sample_indices[idx]+1:03d}.png", 
                        dpi=150, bbox_inches='tight',
                        facecolor='#1a1a2e', edgecolor='none')
            plt.close(fig2)
        else:
            plt.show()
        
        if (idx + 1) % 10 == 0:
            print(f"  Saved {idx + 1}/{num_samples} path images...")
    
    if output_path:
        print(f"[OK] Saved {num_samples} individual path images to: {paths_dir}/")
    
    # === 图3: 统计信息 ===
    plot_statistics(all_paths, output_path)
    
    # 打印统计
    print("\n" + "="*50)
    print("Search Statistics:")
    print("="*50)
    visited_counts = [len(p['visited']) for p in all_paths]
    step_counts = [len(p['steps']) for p in all_paths]
    print(f"Visited nodes: min={min(visited_counts)}, max={max(visited_counts)}, "
          f"mean={np.mean(visited_counts):.1f}, median={np.median(visited_counts):.1f}")
    print(f"Search steps:  min={min(step_counts)}, max={max(step_counts)}, "
          f"mean={np.mean(step_counts):.1f}, median={np.median(step_counts):.1f}")
    print("="*50)
    
    if not output_path:
        plt.show()
    
    print("\n[Done] Visualization complete!")


def visualize_light(paths_path, num_samples=100, output_path=None):
    """轻量版可视化 - 只使用搜索路径数据，不加载图结构，内存占用极小"""
    print(f"[Light Mode] Loading search paths from {paths_path}...")
    
    sampled_paths = []
    sample_indices = []
    total_paths = 0  # 总查询数
    
    if HAS_IJSON:
        # 使用流式解析，内存友好
        print("[Using ijson for streaming JSON parsing]")
        
        # 第一遍：计数
        print("  Pass 1: Counting paths...")
        total_paths = 0
        with open(paths_path, 'rb') as f:
            for _ in ijson.items(f, 'item'):
                total_paths += 1
        print(f"  Total paths: {total_paths}")
        
        # 决定采样哪些
        num_samples = min(num_samples, total_paths)
        sample_set = set(random.sample(range(total_paths), num_samples))
        
        # 第二遍：只加载采样的路径
        print(f"  Pass 2: Loading {num_samples} sampled paths...")
        with open(paths_path, 'rb') as f:
            for idx, path_obj in enumerate(ijson.items(f, 'item')):
                if idx in sample_set:
                    sampled_paths.append(path_obj)
                    sample_indices.append(idx)
        
        # 按原顺序排序
        paired = sorted(zip(sample_indices, sampled_paths))
        sample_indices = [p[0] for p in paired]
        sampled_paths = [p[1] for p in paired]
    else:
        # 无ijson，尝试增量解析
        print("[Warning] ijson not installed. Trying incremental parsing...")
        print("  Install for better performance: pip install ijson")
        
        # 获取文件大小
        file_size = os.path.getsize(paths_path)
        print(f"  File size: {file_size / (1024*1024):.1f} MB")
        
        if file_size > 500 * 1024 * 1024:  # 500MB
            print("  [Error] File too large without ijson. Please install: pip install ijson")
            return
        
        # 尝试加载（小文件）
        with open(paths_path, 'r') as f:
            all_paths = json.load(f)
        
        total_paths = len(all_paths)
        print(f"  Loaded {total_paths} search paths")
        num_samples = min(num_samples, total_paths)
        sample_indices = random.sample(range(total_paths), num_samples) if total_paths > num_samples else list(range(total_paths))
        sample_indices.sort()
        sampled_paths = [all_paths[i] for i in sample_indices]
        del all_paths  # 释放内存
    
    print(f"Loaded {len(sampled_paths)} paths for visualization")
    
    print(f"Visualizing {num_samples} queries...")
    
    # 自动创建输出目录
    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
            print(f"Created output directory: {out_dir}")
        
        paths_dir = f"{output_path}_paths"
        if not os.path.exists(paths_dir):
            os.makedirs(paths_dir)
        print(f"Saving to: {paths_dir}/")
    
    # 为每个查询生成图
    for idx, path_info in enumerate(sampled_paths):
        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor('#1a1a2e')
        
        v_count = len(path_info.get('visited', []))
        steps_count = len(path_info.get('steps', []))
        
        # 使用轻量版绘图
        plot_search_path_light(fig, path_info, sample_indices[idx]+1)
        
        # 标题
        fig.suptitle(f"HNSW Search Analysis - Query #{sample_indices[idx]+1}\n"
                     f"Visited: {v_count:,} nodes | Steps: {steps_count:,}",
                     fontsize=14, fontweight='bold', color='#eaeaea', y=0.98)
        
        if output_path:
            fig.savefig(f"{paths_dir}/query_{sample_indices[idx]+1:03d}.png", 
                       dpi=150, bbox_inches='tight',
                       facecolor='#1a1a2e', edgecolor='none')
            plt.close(fig)
        else:
            plt.show()
        
        if (idx + 1) % 20 == 0:
            print(f"  Progress: {idx + 1}/{num_samples}")
    
    # 统计信息
    if output_path:
        print(f"[OK] Saved {num_samples} images to: {paths_dir}/")
    
    # 汇总统计（使用采样数据）
    print("\n" + "="*50)
    print(f"Search Statistics (Sampled {len(sampled_paths)} Queries):")
    print("="*50)
    
    all_step_counts = [len(p.get('steps', [])) for p in sampled_paths]
    all_visited = [len(p.get('visited', [])) for p in sampled_paths]
    
    # 计算效率指标
    efficiencies = []
    for p in sampled_paths:
        steps = p.get('steps', [])
        if not steps:
            continue
        dists = [float(s['dist']) for s in steps if 'dist' in s]
        if not dists:
            continue
        final_best = min(dists)
        first_best_idx = next((i for i, d in enumerate(dists) if d == final_best), len(dists)-1)
        eff = 100 * first_best_idx / len(dists)
        efficiencies.append(eff)
    
    print(f"Total queries: {total_paths}")
    print(f"Steps:   min={min(all_step_counts):,}, max={max(all_step_counts):,}, "
          f"mean={np.mean(all_step_counts):,.1f}, median={np.median(all_step_counts):,.1f}")
    print(f"Visited: min={min(all_visited):,}, max={max(all_visited):,}, "
          f"mean={np.mean(all_visited):,.1f}, median={np.median(all_visited):,.1f}")
    if efficiencies:
        print(f"Efficiency (first best %): min={min(efficiencies):.1f}%, max={max(efficiencies):.1f}%, "
              f"mean={np.mean(efficiencies):.1f}%, median={np.median(efficiencies):.1f}%")
    print("="*50)
    
    # 生成汇总图
    if output_path:
        fig_summary, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig_summary.patch.set_facecolor('#1a1a2e')
        
        for ax in axes.flat:
            ax.set_facecolor('#16213e')
        
        # 步数分布
        axes[0, 0].hist(all_step_counts, bins=50, color='#4fc3f7', edgecolor='#0288d1', alpha=0.7)
        axes[0, 0].axvline(np.mean(all_step_counts), color='#ff5252', linestyle='--', label=f'Mean: {np.mean(all_step_counts):,.0f}')
        axes[0, 0].set_xlabel('Steps per Query', color='#eaeaea')
        axes[0, 0].set_ylabel('Count', color='#eaeaea')
        axes[0, 0].set_title('Distribution of Search Steps', color='#eaeaea')
        axes[0, 0].legend(facecolor='#16213e', edgecolor='#e94560', labelcolor='#eaeaea')
        axes[0, 0].grid(True, alpha=0.15)
        
        # 访问节点数分布
        axes[0, 1].hist(all_visited, bins=50, color='#ce93d8', edgecolor='#7b1fa2', alpha=0.7)
        axes[0, 1].axvline(np.mean(all_visited), color='#ff5252', linestyle='--', label=f'Mean: {np.mean(all_visited):,.0f}')
        axes[0, 1].set_xlabel('Visited Nodes per Query', color='#eaeaea')
        axes[0, 1].set_ylabel('Count', color='#eaeaea')
        axes[0, 1].set_title('Distribution of Visited Nodes', color='#eaeaea')
        axes[0, 1].legend(facecolor='#16213e', edgecolor='#e94560', labelcolor='#eaeaea')
        axes[0, 1].grid(True, alpha=0.15)
        
        # 效率分布
        if efficiencies:
            axes[1, 0].hist(efficiencies, bins=50, color='#81c784', edgecolor='#388e3c', alpha=0.7)
            axes[1, 0].axvline(np.mean(efficiencies), color='#ff5252', linestyle='--', label=f'Mean: {np.mean(efficiencies):.1f}%')
            axes[1, 0].set_xlabel('First Best Position (%)', color='#eaeaea')
            axes[1, 0].set_ylabel('Count', color='#eaeaea')
            axes[1, 0].set_title('Search Efficiency Distribution', color='#eaeaea')
            axes[1, 0].legend(facecolor='#16213e', edgecolor='#e94560', labelcolor='#eaeaea')
            axes[1, 0].grid(True, alpha=0.15)
        
        # 步数 vs 效率 散点图
        if efficiencies and len(efficiencies) == len(all_step_counts):
            axes[1, 1].scatter(all_step_counts, efficiencies, c='#4fc3f7', s=10, alpha=0.5)
            axes[1, 1].set_xlabel('Total Steps', color='#eaeaea')
            axes[1, 1].set_ylabel('First Best Position (%)', color='#eaeaea')
            axes[1, 1].set_title('Steps vs Efficiency', color='#eaeaea')
            axes[1, 1].grid(True, alpha=0.15)
        
        plt.tight_layout()
        fig_summary.savefig(f"{output_path}_summary.png", dpi=200, bbox_inches='tight',
                           facecolor='#1a1a2e', edgecolor='none')
        plt.close(fig_summary)
        print(f"[OK] Saved summary: {output_path}_summary.png")
    
    print("\n[Done] Light mode visualization complete!")


def main():
    parser = argparse.ArgumentParser(description='HNSW Graph and Search Path Visualization')
    parser.add_argument('--graph', '-g', help='Path to graph JSON file (optional)')
    parser.add_argument('--paths', '-p', required=True, help='Path to search paths JSON file')
    parser.add_argument('--samples', '-n', type=int, default=30, 
                        help='Number of queries to visualize (default: 30)')
    parser.add_argument('--output', '-o', default=None, 
                        help='Output file prefix')
    parser.add_argument('--method', '-m', choices=['pca', 'tsne'], default='pca', 
                        help='Dimensionality reduction method (default: pca)')
    parser.add_argument('--max-nodes', type=int, default=50000,
                        help='Max nodes for graph structure visualization (default: 50000)')
    parser.add_argument('--light', action='store_true',
                        help='Light mode: only use search paths, no graph loading (memory efficient)')
    parser.add_argument('--skip-graph', action='store_true',
                        help='Skip graph structure visualization (use with --graph to still load graph)')
    
    args = parser.parse_args()
    
    if args.light or not args.graph:
        # 轻量模式（不需要graph.json）
        print("="*60)
        print("LIGHT MODE - Memory efficient, no graph loading")
        print("="*60)
        visualize_light(args.paths, args.samples, args.output)
    else:
        # 完整模式
        print("="*60)
        print("FULL MODE - With graph structure visualization")
        print("="*60)
        visualize(args.graph, args.paths, args.samples, args.output, args.method, args.max_nodes, args.skip_graph)


if __name__ == '__main__':
    main()
