#!/usr/bin/env python3
"""
实验结果分析可视化脚本
每个图表单独保存为一个文件

图表内容：
1. 不同搜索策略和模块对 Recall/QPS 的影响
2. M 和 efC 对 QPS 和召回率的影响
3. 全优化 vs 无优化的距离计算次数和 QPS
4. QPS 提升率分析
5. 达成目标召回率所需参数分析

5位二进制码含义（从左到右）:
bit0: single_layer (0=多层, 1=单层)
bit1: onng (0=关闭, 1=开启)
bit2: bfs (0=关闭, 1=开启)
bit3: simd (0=关闭, 1=开启)
bit4: neg_ip (0=L2距离, 1=负内积)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.facecolor'] = 'white'

# 定义颜色方案
STRATEGY_COLORS = {
    'gamma': '#E63946',           # 红色 - 动态gamma
    'gamma-static': '#457B9D',    # 蓝色 - 静态gamma
    'fixed': '#2A9D8F',           # 绿色 - 固定ef
}

STRATEGY_NAMES = {
    'gamma': '动态Gamma',
    'gamma-static': '静态Gamma',
    'fixed': '固定EF'
}

# 模块颜色
MODULE_COLORS = {
    'single_layer': '#FF6B6B',
    'onng': '#4ECDC4',
    'bfs': '#45B7D1',
    'simd': '#96CEB4',
    'neg_ip': '#DDA0DD',
}

MODULE_NAMES = {
    'single_layer': '单层图',
    'onng': 'ONNG',
    'bfs': 'BFS重排',
    'simd': 'SIMD',
    'neg_ip': 'NegIP',
}

# M值颜色
M_COLORS = {24: '#E63946', 48: '#F4A261', 96: '#2A9D8F', 128: '#457B9D'}

# 配置代码映射
CONFIG_NAMES = {
    '00000': '无优化',
    '11110': '全优化(L2)',
    '11111': '全优化(NegIP)',
    '01111': '无单层图',
    '10111': '无ONNG',
    '11011': '无BFS',
    '11101': '无SIMD',
}


def load_data(csv_path='experiment_results.csv'):
    """加载并预处理数据"""
    df = pd.read_csv(csv_path)
    
    # 解析 group_name 获取 code_bits
    df['code_bits'] = df['group_name'].str.extract(r'^(\d{5})_')[0]
    
    # 解析各个特征
    df['bit_single_layer'] = df['code_bits'].str[0].astype(int)
    df['bit_onng'] = df['code_bits'].str[1].astype(int)
    df['bit_bfs'] = df['code_bits'].str[2].astype(int)
    df['bit_simd'] = df['code_bits'].str[3].astype(int)
    df['bit_neg_ip'] = df['code_bits'].str[4].astype(int)
    
    # 创建配置名称
    df['config_name'] = df['code_bits'].map(CONFIG_NAMES).fillna(df['code_bits'])
    
    return df


def transform_recall_axis(recall):
    """
    将 recall 转换为对数尺度，使高召回率区域放大
    transform: -log10(1 - recall)
    98% -> 1.7, 99% -> 2, 99.5% -> 2.3, 99.9% -> 3
    """
    # 避免 log(0)
    recall_clipped = np.clip(recall, 0, 0.9999)
    return -np.log10(1 - recall_clipped)


def inverse_transform_recall(x):
    """逆变换：从对数尺度恢复 recall"""
    return 1 - 10**(-x)


def setup_recall_axis(ax, x_min=0.5, x_max=0.9999):
    """设置 recall 轴的刻度和标签"""
    # 主要刻度点
    major_recalls = [0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
    major_ticks = [transform_recall_axis(r) for r in major_recalls if x_min <= r <= x_max]
    major_labels = [f'{r*100:.1f}%' if r >= 0.99 else f'{r*100:.0f}%' for r in major_recalls if x_min <= r <= x_max]
    
    ax.set_xticks(major_ticks)
    ax.set_xticklabels(major_labels)
    ax.set_xlim(transform_recall_axis(x_min), transform_recall_axis(x_max))
    
    # 添加垂直参考线
    for recall_target in [0.98, 0.99, 0.995]:
        if x_min <= recall_target <= x_max:
            ax.axvline(x=transform_recall_axis(recall_target), 
                      color='gray', linestyle='--', alpha=0.3, linewidth=0.8)


def save_figure(fig, output_dir, filename):
    """保存图表"""
    filepath = output_dir / filename
    fig.savefig(filepath, bbox_inches='tight', dpi=200, facecolor='white')
    plt.close(fig)
    print(f"[OK] 已保存: {filename}")


# ============================================================================
# 图1-2: 不同搜索策略和模块对 Recall/QPS 的影响
# ============================================================================

def plot_fig01_strategy_recall_qps(df, output_dir):
    """图1: 三种搜索策略的 Recall-QPS 曲线（全优化配置）"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 使用全优化L2配置 (11110)
    config_df = df[df['code_bits'] == '11110']
    
    for strategy in ['gamma', 'gamma-static', 'fixed']:
        strategy_df = config_df[config_df['strategy'] == strategy]
        if len(strategy_df) > 0:
            # 聚合数据
            agg = strategy_df.groupby('param_val').agg({
                'recall10': 'mean',
                'qps': 'mean'
            }).reset_index().sort_values('recall10')
            
            # 转换 x 轴
            x_transformed = transform_recall_axis(agg['recall10'])
            
            ax.plot(x_transformed, agg['qps'], 'o-',
                   color=STRATEGY_COLORS[strategy],
                   label=STRATEGY_NAMES[strategy],
                   markersize=6, linewidth=2, alpha=0.8)
    
    setup_recall_axis(ax, x_min=0.5, x_max=0.999)
    ax.set_xlabel('Recall@10', fontsize=12)
    ax.set_ylabel('QPS', fontsize=12)
    ax.set_title('三种搜索策略的 Recall-QPS 曲线\n(全优化配置 11110)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    save_figure(fig, output_dir, 'fig01_strategy_recall_qps.png')


def plot_fig02_module_ablation_recall_qps(df, output_dir):
    """图2: 各模块消融对比（11110 vs 去掉某模块）"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 使用动态gamma策略
    strategy_df = df[df['strategy'] == 'gamma']
    
    # 基准配置和消融配置
    configs = {
        '11110': ('全优化(L2)', '#2A9D8F', '-', 'o'),
        '01111': ('去掉单层图', '#FF6B6B', '--', 's'),
        '10111': ('去掉ONNG', '#4ECDC4', '--', '^'),
        '11011': ('去掉BFS', '#45B7D1', '--', 'D'),
        '11101': ('去掉SIMD', '#96CEB4', '--', 'v'),
    }
    
    for code, (name, color, linestyle, marker) in configs.items():
        config_df = strategy_df[strategy_df['code_bits'] == code]
        if len(config_df) > 0:
            agg = config_df.groupby('param_val').agg({
                'recall10': 'mean',
                'qps': 'mean'
            }).reset_index().sort_values('recall10')
            
            x_transformed = transform_recall_axis(agg['recall10'])
            
            linewidth = 2.5 if code == '11110' else 1.8
            ax.plot(x_transformed, agg['qps'], linestyle=linestyle, marker=marker,
                   color=color, label=name,
                   markersize=5, linewidth=linewidth, alpha=0.85)
    
    setup_recall_axis(ax, x_min=0.5, x_max=0.999)
    ax.set_xlabel('Recall@10', fontsize=12)
    ax.set_ylabel('QPS', fontsize=12)
    ax.set_title('模块消融实验: 全优化 vs 去掉某模块\n(动态Gamma策略)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    save_figure(fig, output_dir, 'fig02_module_ablation_recall_qps.png')


# ============================================================================
# 图3-5: M 和 efC 对 QPS 和召回率的影响
# ============================================================================

def plot_fig03_M_impact_recall_qps(df, output_dir):
    """图3: 不同M值的 Recall-QPS 曲线"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 使用全优化配置、动态gamma、固定efC=300
    config_df = df[(df['code_bits'] == '11110') & (df['strategy'] == 'gamma') & (df['efC'] == 300)]
    
    M_values = sorted(df['M'].unique())
    
    for M in M_values:
        m_df = config_df[config_df['M'] == M]
        if len(m_df) > 0:
            agg = m_df.groupby('param_val').agg({
                'recall10': 'mean',
                'qps': 'mean'
            }).reset_index().sort_values('recall10')
            
            x_transformed = transform_recall_axis(agg['recall10'])
            
            ax.plot(x_transformed, agg['qps'], 'o-',
                   color=M_COLORS[M], label=f'M={M}',
                   markersize=5, linewidth=2, alpha=0.8)
    
    setup_recall_axis(ax, x_min=0.7, x_max=0.999)
    ax.set_yscale('log')  # 使用对数坐标使高召回率区域更清晰
    ax.set_xlabel('Recall@10', fontsize=12)
    ax.set_ylabel('QPS (对数坐标)', fontsize=12)
    ax.set_title('不同 M 值对 Recall-QPS 的影响\n(efC=300, 全优化配置, 动态Gamma)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    save_figure(fig, output_dir, 'fig03_M_impact_recall_qps.png')


def plot_fig04_efC_impact_recall_qps(df, output_dir):
    """图4: 不同efC值的 Recall-QPS 曲线"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 使用全优化配置、动态gamma、M=96
    config_df = df[(df['code_bits'] == '11110') & (df['strategy'] == 'gamma') & (df['M'] == 96)]
    
    efC_values = sorted(df['efC'].unique())
    efC_colors = {200: '#E63946', 300: '#F4A261', 400: '#2A9D8F', 500: '#457B9D'}
    
    for efC in efC_values:
        efc_df = config_df[config_df['efC'] == efC]
        if len(efc_df) > 0:
            agg = efc_df.groupby('param_val').agg({
                'recall10': 'mean',
                'qps': 'mean'
            }).reset_index().sort_values('recall10')
            
            x_transformed = transform_recall_axis(agg['recall10'])
            
            ax.plot(x_transformed, agg['qps'], 'o-',
                   color=efC_colors[efC], label=f'efC={efC}',
                   markersize=5, linewidth=2, alpha=0.8)
    
    setup_recall_axis(ax, x_min=0.7, x_max=0.999)
    ax.set_yscale('log')  # 使用对数坐标使高召回率区域更清晰
    ax.set_xlabel('Recall@10', fontsize=12)
    ax.set_ylabel('QPS (对数坐标)', fontsize=12)
    ax.set_title('不同 efC 值对 Recall-QPS 的影响\n(M=96, 全优化配置, 动态Gamma)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    save_figure(fig, output_dir, 'fig04_efC_impact_recall_qps.png')


def plot_fig05_M_efC_heatmap(df, output_dir):
    """图5: M×efC 对高召回率下 QPS 的热力图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    M_values = sorted(df['M'].unique())
    efC_values = sorted(df['efC'].unique())
    
    # 左图: 98%召回率下的最大QPS
    ax = axes[0]
    recall_98_df = df[(df['recall10'] >= 0.98) & (df['code_bits'] == '11110') & (df['strategy'] == 'gamma')]
    qps_98 = recall_98_df.groupby(['M', 'efC'])['qps'].max().unstack(fill_value=0)
    # 确保所有M和efC值都存在
    qps_98 = qps_98.reindex(index=M_values, columns=efC_values, fill_value=0)
    
    im = ax.imshow(qps_98.values, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(efC_values)))
    ax.set_xticklabels(efC_values)
    ax.set_yticks(range(len(M_values)))
    ax.set_yticklabels(M_values)
    ax.set_xlabel('efC', fontsize=11)
    ax.set_ylabel('M', fontsize=11)
    ax.set_title('Recall≥98% 时的最大 QPS', fontsize=12, fontweight='bold')
    
    max_val = qps_98.values.max() if qps_98.values.max() > 0 else 1
    for i in range(len(M_values)):
        for j in range(len(efC_values)):
            val = qps_98.iloc[i, j]
            color = 'white' if val > max_val * 0.6 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=10, color=color)
    
    plt.colorbar(im, ax=ax, label='Max QPS')
    
    # 右图: 99%召回率下的最大QPS
    ax = axes[1]
    recall_99_df = df[(df['recall10'] >= 0.99) & (df['code_bits'] == '11110') & (df['strategy'] == 'gamma')]
    
    if len(recall_99_df) > 0:
        qps_99 = recall_99_df.groupby(['M', 'efC'])['qps'].max().unstack(fill_value=0)
        # 确保所有M和efC值都存在
        qps_99 = qps_99.reindex(index=M_values, columns=efC_values, fill_value=0)
        
        im = ax.imshow(qps_99.values, aspect='auto', cmap='YlOrRd')
        ax.set_xticks(range(len(efC_values)))
        ax.set_xticklabels(efC_values)
        ax.set_yticks(range(len(M_values)))
        ax.set_yticklabels(M_values)
        ax.set_xlabel('efC', fontsize=11)
        ax.set_ylabel('M', fontsize=11)
        ax.set_title('Recall≥99% 时的最大 QPS', fontsize=12, fontweight='bold')
        
        max_val = qps_99.values.max() if qps_99.values.max() > 0 else 1
        for i in range(len(M_values)):
            for j in range(len(efC_values)):
                val = qps_99.iloc[i, j]
                color = 'white' if val > max_val * 0.6 else 'black'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=10, color=color)
        
        plt.colorbar(im, ax=ax, label='Max QPS')
    
    plt.suptitle('M × efC 对高召回率 QPS 的影响 (全优化配置, 动态Gamma)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, output_dir, 'fig05_M_efC_heatmap.png')


# ============================================================================
# 图6-7: 距离计算次数与 QPS/Recall 分析
# ============================================================================

def plot_fig06_dist_calcs_vs_qps(df, output_dir):
    """图6: 全优化 vs 无优化的距离计算-QPS关系"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    strategy_df = df[df['strategy'] == 'gamma']
    
    configs = {
        '00000': ('无优化', '#E63946', 'o'),
        '11110': ('全优化(L2)', '#2A9D8F', 's'),
        '11111': ('全优化(NegIP)', '#457B9D', '^'),
    }
    
    for code, (name, color, marker) in configs.items():
        config_df = strategy_df[strategy_df['code_bits'] == code]
        if len(config_df) > 0:
            agg = config_df.groupby('param_val').agg({
                'dist_calcs': 'mean',
                'qps': 'mean',
                'recall10': 'mean'
            }).reset_index()
            
            # 用颜色深浅表示recall
            scatter = ax.scatter(agg['dist_calcs'], agg['qps'], 
                               c=agg['recall10'], cmap='viridis',
                               marker=marker, s=80, alpha=0.7,
                               label=name, edgecolors='white', linewidths=0.5)
    
    ax.set_xscale('log')  # 距离计算次数使用对数坐标
    ax.set_yscale('log')  # QPS 使用对数坐标
    ax.set_xlabel('距离计算次数 (对数坐标)', fontsize=12)
    ax.set_ylabel('QPS (对数坐标)', fontsize=12)
    ax.set_title('距离计算次数 vs QPS\n(动态Gamma策略)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Recall@10', fontsize=10)
    
    save_figure(fig, output_dir, 'fig06_dist_calcs_vs_qps.png')


def plot_fig07_dist_calcs_vs_recall(df, output_dir):
    """图7: 距离计算次数与Recall的关系"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    strategy_df = df[df['strategy'] == 'gamma']
    
    configs = {
        '00000': ('无优化', '#E63946', '-', 'o'),
        '11110': ('全优化(L2)', '#2A9D8F', '-', 's'),
        '11111': ('全优化(NegIP)', '#457B9D', '-', '^'),
    }
    
    for code, (name, color, linestyle, marker) in configs.items():
        config_df = strategy_df[strategy_df['code_bits'] == code]
        if len(config_df) > 0:
            agg = config_df.groupby('param_val').agg({
                'recall10': 'mean',
                'dist_calcs': 'mean'
            }).reset_index().sort_values('recall10')
            
            x_transformed = transform_recall_axis(agg['recall10'])
            
            ax.plot(x_transformed, agg['dist_calcs'], linestyle=linestyle, marker=marker,
                   color=color, label=name,
                   markersize=6, linewidth=2, alpha=0.8)
    
    setup_recall_axis(ax, x_min=0.5, x_max=0.999)
    ax.set_xlabel('Recall@10', fontsize=12)
    ax.set_ylabel('距离计算次数', fontsize=12)
    ax.set_title('Recall vs 距离计算次数\n(动态Gamma策略)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    save_figure(fig, output_dir, 'fig07_dist_calcs_vs_recall.png')


def plot_fig07b_strategy_dist_calcs_vs_recall(df, output_dir):
    """图7b/c/d: 三种搜索策略的 Recall-距离计算次数 对比 (M=96, 不同efC)"""
    
    # 使用全优化配置 (11110), M=96
    base_df = df[(df['code_bits'] == '11110') & (df['M'] == 96)]
    
    efC_values = [200, 300, 400]
    
    for efC in efC_values:
        fig, ax = plt.subplots(figsize=(10, 7))
        
        config_df = base_df[base_df['efC'] == efC]
        
        for strategy in ['gamma', 'gamma-static', 'fixed']:
            strategy_df = config_df[config_df['strategy'] == strategy]
            if len(strategy_df) > 0:
                agg = strategy_df.groupby('param_val').agg({
                    'recall10': 'mean',
                    'dist_calcs': 'mean'
                }).reset_index().sort_values('recall10')
                
                x_transformed = transform_recall_axis(agg['recall10'])
                
                ax.plot(x_transformed, agg['dist_calcs'], 'o-',
                       color=STRATEGY_COLORS[strategy],
                       label=STRATEGY_NAMES[strategy],
                       markersize=6, linewidth=2, alpha=0.8)
        
        setup_recall_axis(ax, x_min=0.5, x_max=0.999)
        ax.set_xlabel('Recall@10', fontsize=12)
        ax.set_ylabel('距离计算次数', fontsize=12)
        ax.set_title(f'三种搜索策略的 Recall-距离计算次数 对比\n(M=96, efC={efC}, 全优化配置)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3)
        
        suffix = {200: 'b', 300: 'c', 400: 'd'}[efC]
        save_figure(fig, output_dir, f'fig07{suffix}_strategy_dist_calcs_efC{efC}.png')


# ============================================================================
# 图8-10: QPS 提升率分析
# ============================================================================

def plot_fig08_module_qps_improvement(df, output_dir):
    """图8: 各模块的QPS提升率（全优化1111x相对于去掉某模块的提升）"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 在不同召回率目标下计算提升率
    recall_targets = [0.90, 0.95, 0.98, 0.99]
    
    # 模块消融对比: 全优化(1111x) vs 去掉某模块
    # 最后一位(L2/NegIP)忽略，11110和11111视为相同
    ablation_configs = {
        '单层图': '01111',    # 去掉单层图 (0xxxx)
        'ONNG': '10111',      # 去掉ONNG (x0xxx)
        'BFS重排': '11011',   # 去掉BFS (xx0xx)
        'SIMD': '11101',      # 去掉SIMD (xxx0x)
    }
    
    strategy_df = df[df['strategy'] == 'gamma']
    
    # 全优化基准: 11110 和 11111 取最大值
    def get_full_opt_qps(data, target):
        full_df = data[(data['code_bits'].isin(['11110', '11111'])) & (data['recall10'] >= target)]
        return full_df['qps'].max() if len(full_df) > 0 else 0
    
    x = np.arange(len(recall_targets))
    width = 0.18
    
    for i, (module_name, ablation_code) in enumerate(ablation_configs.items()):
        improvements = []
        for target in recall_targets:
            full_qps = get_full_opt_qps(strategy_df, target)
            ablation_df = strategy_df[(strategy_df['code_bits'] == ablation_code) & (strategy_df['recall10'] >= target)]
            ablation_qps = ablation_df['qps'].max() if len(ablation_df) > 0 else 0
            
            if ablation_qps > 0:
                # 全优化相对于去掉该模块的提升
                improvement = (full_qps - ablation_qps) / ablation_qps * 100
            else:
                improvement = 0
            improvements.append(improvement)
        
        bars = ax.bar(x + i * width, improvements, width, label=module_name, alpha=0.8)
        
        # 添加数值标签
        for bar, val in zip(bars, improvements):
            y_pos = val + 1 if val >= 0 else val - 3
            ax.text(bar.get_x() + bar.get_width()/2, y_pos, 
                   f'{val:.1f}%', ha='center', fontsize=8, rotation=0)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('目标 Recall', fontsize=12)
    ax.set_ylabel('QPS 提升率 (%)', fontsize=12)
    ax.set_title('各优化模块的 QPS 贡献率\n(全优化1111x 相对于去掉该模块的提升, 动态Gamma)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f'{t*100:.0f}%' for t in recall_targets])
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    save_figure(fig, output_dir, 'fig08_module_qps_improvement.png')


def plot_fig09_strategy_qps_comparison(df, output_dir):
    """图9: 三种搜索策略在全优化配置下的QPS对比"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    recall_targets = [0.90, 0.95, 0.98, 0.99, 0.995]
    config_df = df[df['code_bits'] == '11110']  # 全优化配置
    
    x = np.arange(len(recall_targets))
    width = 0.25
    
    for i, strategy in enumerate(['gamma', 'gamma-static', 'fixed']):
        qps_values = []
        for target in recall_targets:
            strategy_df = config_df[(config_df['strategy'] == strategy) & (config_df['recall10'] >= target)]
            max_qps = strategy_df['qps'].max() if len(strategy_df) > 0 else 0
            qps_values.append(max_qps)
        
        bars = ax.bar(x + i * width, qps_values, width, 
                     label=STRATEGY_NAMES[strategy], 
                     color=STRATEGY_COLORS[strategy], alpha=0.8)
        
        # 添加数值标签
        for bar, val in zip(bars, qps_values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, val + 20, 
                       f'{val:.0f}', ha='center', fontsize=8)
    
    ax.set_xlabel('目标 Recall', fontsize=12)
    ax.set_ylabel('最大 QPS', fontsize=12)
    ax.set_title('三种搜索策略在全优化配置下的 QPS 对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{t*100:.1f}%' if t >= 0.99 else f'{t*100:.0f}%' for t in recall_targets])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    save_figure(fig, output_dir, 'fig09_strategy_qps_comparison.png')


def plot_fig10_strategy_improvement_matrix(df, output_dir):
    """图10: 策略间相互提升率对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    strategies = ['gamma', 'gamma-static', 'fixed']
    strategy_labels = [STRATEGY_NAMES[s] for s in strategies]
    config_df = df[df['code_bits'] == '11110']
    
    # 左图: 98%召回率下的策略间提升率矩阵
    ax = axes[0]
    recall_98_df = config_df[config_df['recall10'] >= 0.98]
    
    matrix_98 = np.zeros((3, 3))
    for i, s1 in enumerate(strategies):
        for j, s2 in enumerate(strategies):
            qps1 = recall_98_df[recall_98_df['strategy'] == s1]['qps'].max()
            qps2 = recall_98_df[recall_98_df['strategy'] == s2]['qps'].max()
            if qps2 > 0:
                matrix_98[i, j] = (qps1 - qps2) / qps2 * 100
    
    im = ax.imshow(matrix_98, cmap='RdYlGn', vmin=-30, vmax=30)
    ax.set_xticks(range(3))
    ax.set_xticklabels(strategy_labels, fontsize=10)
    ax.set_yticks(range(3))
    ax.set_yticklabels(strategy_labels, fontsize=10)
    ax.set_xlabel('基准策略', fontsize=11)
    ax.set_ylabel('对比策略', fontsize=11)
    ax.set_title('Recall≥98% 策略间 QPS 提升率 (%)', fontsize=12, fontweight='bold')
    
    for i in range(3):
        for j in range(3):
            color = 'white' if abs(matrix_98[i, j]) > 15 else 'black'
            ax.text(j, i, f'{matrix_98[i, j]:.1f}%', ha='center', va='center', fontsize=11, color=color)
    
    plt.colorbar(im, ax=ax, label='提升率 (%)')
    
    # 右图: 99%召回率下的策略间提升率矩阵
    ax = axes[1]
    recall_99_df = config_df[config_df['recall10'] >= 0.99]
    
    matrix_99 = np.zeros((3, 3))
    for i, s1 in enumerate(strategies):
        for j, s2 in enumerate(strategies):
            qps1 = recall_99_df[recall_99_df['strategy'] == s1]['qps'].max()
            qps2 = recall_99_df[recall_99_df['strategy'] == s2]['qps'].max()
            if qps2 > 0:
                matrix_99[i, j] = (qps1 - qps2) / qps2 * 100
    
    im = ax.imshow(matrix_99, cmap='RdYlGn', vmin=-30, vmax=30)
    ax.set_xticks(range(3))
    ax.set_xticklabels(strategy_labels, fontsize=10)
    ax.set_yticks(range(3))
    ax.set_yticklabels(strategy_labels, fontsize=10)
    ax.set_xlabel('基准策略', fontsize=11)
    ax.set_ylabel('对比策略', fontsize=11)
    ax.set_title('Recall≥99% 策略间 QPS 提升率 (%)', fontsize=12, fontweight='bold')
    
    for i in range(3):
        for j in range(3):
            color = 'white' if abs(matrix_99[i, j]) > 15 else 'black'
            ax.text(j, i, f'{matrix_99[i, j]:.1f}%', ha='center', va='center', fontsize=11, color=color)
    
    plt.colorbar(im, ax=ax, label='提升率 (%)')
    
    plt.tight_layout()
    save_figure(fig, output_dir, 'fig10_strategy_improvement_matrix.png')


# ============================================================================
# 图11-13: 达成目标召回率所需参数分析
# ============================================================================

def plot_fig11_params_for_target_recall(df, output_dir):
    """图11: 不同策略达成目标召回率所需的最小参数"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    recall_targets = [0.90, 0.95, 0.98, 0.99, 0.995]
    recall_labels = [f'{t*100:.1f}%' if t >= 0.99 else f'{t*100:.0f}%' for t in recall_targets]
    x_indices = np.arange(len(recall_targets))  # 使用等距的整数索引
    config_df = df[df['code_bits'] == '11110']
    
    # 左图: gamma 和 gamma-static 的参数对比
    ax = axes[0]
    for strategy in ['gamma', 'gamma-static']:
        min_params = []
        for target in recall_targets:
            strategy_df = config_df[(config_df['strategy'] == strategy) & (config_df['recall10'] >= target)]
            if len(strategy_df) > 0:
                min_params.append(strategy_df['param_val'].min())
            else:
                min_params.append(np.nan)
        
        ax.plot(x_indices, min_params, 'o-', 
               color=STRATEGY_COLORS[strategy],
               label=STRATEGY_NAMES[strategy],
               markersize=8, linewidth=2)
    
    ax.set_xlabel('目标 Recall', fontsize=11)
    ax.set_ylabel('最小 Gamma 参数', fontsize=11)
    ax.set_title('Gamma 策略达成目标所需的最小参数', fontsize=12, fontweight='bold')
    ax.set_xticks(x_indices)
    ax.set_xticklabels(recall_labels)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 右图: fixed 策略的 ef 参数
    ax = axes[1]
    fixed_df = config_df[config_df['strategy'] == 'fixed']
    
    min_efs = []
    for target in recall_targets:
        target_df = fixed_df[fixed_df['recall10'] >= target]
        if len(target_df) > 0:
            min_efs.append(target_df['param_val'].min())
        else:
            min_efs.append(np.nan)
    
    ax.plot(x_indices, min_efs, 'o-', 
           color=STRATEGY_COLORS['fixed'],
           label=STRATEGY_NAMES['fixed'],
           markersize=8, linewidth=2)
    
    ax.set_xlabel('目标 Recall', fontsize=11)
    ax.set_ylabel('最小 ef 参数', fontsize=11)
    ax.set_title('固定EF策略达成目标所需的最小参数', fontsize=12, fontweight='bold')
    ax.set_xticks(x_indices)
    ax.set_xticklabels(recall_labels)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, output_dir, 'fig11_params_for_target_recall.png')


def plot_fig12_params_by_module(df, output_dir):
    """图12: 不同模块配置达成目标召回率所需参数"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    recall_targets = [0.90, 0.98, 0.99, 0.995]
    strategy_df = df[df['strategy'] == 'gamma']
    
    configs = {
        '11110': ('全优化(L2)', '#2A9D8F'),
        '01111': ('去掉单层图', '#FF6B6B'),
        '10111': ('去掉ONNG', '#4ECDC4'),
        '11011': ('去掉BFS', '#45B7D1'),
        '11101': ('去掉SIMD', '#96CEB4'),
        '00000': ('无优化', '#888888'),
    }
    
    x = np.arange(len(recall_targets))
    width = 0.12
    
    for i, (code, (name, color)) in enumerate(configs.items()):
        min_params = []
        for target in recall_targets:
            config_df = strategy_df[(strategy_df['code_bits'] == code) & (strategy_df['recall10'] >= target)]
            if len(config_df) > 0:
                min_params.append(config_df['param_val'].min())
            else:
                min_params.append(np.nan)
        
        bars = ax.bar(x + i * width, min_params, width, label=name, color=color, alpha=0.8)
    
    ax.set_xlabel('目标 Recall', fontsize=12)
    ax.set_ylabel('最小 Gamma 参数', fontsize=12)
    ax.set_title('不同配置达成目标召回率所需的最小 Gamma 参数\n(动态Gamma策略)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels([f'{t*100:.1f}%' if t >= 0.99 else f'{t*100:.0f}%' for t in recall_targets])
    ax.legend(fontsize=9, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    save_figure(fig, output_dir, 'fig12_params_by_module.png')


def plot_fig13_param_recall_summary_table(df, output_dir):
    """图13: 参数-召回率汇总表格"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    recall_targets = [0.90, 0.98, 0.99, 0.995]
    strategies = ['gamma', 'gamma-static', 'fixed']
    config_df = df[df['code_bits'] == '11110']
    
    # 构建表格数据
    table_data = [['策略', '目标Recall', '最小参数', '对应QPS', '距离计算次数']]
    
    for strategy in strategies:
        strategy_df = config_df[config_df['strategy'] == strategy]
        for target in recall_targets:
            target_df = strategy_df[strategy_df['recall10'] >= target]
            if len(target_df) > 0:
                # 找最小参数及其对应的性能
                min_param_idx = target_df['param_val'].idxmin()
                row = target_df.loc[min_param_idx]
                table_data.append([
                    STRATEGY_NAMES[strategy],
                    f'{target*100:.1f}%' if target >= 0.99 else f'{target*100:.0f}%',
                    f'{row["param_val"]:.2f}' if strategy != 'fixed' else f'{row["param_val"]:.0f}',
                    f'{row["qps"]:.0f}',
                    f'{row["dist_calcs"]:.0f}'
                ])
            else:
                table_data.append([
                    STRATEGY_NAMES[strategy],
                    f'{target*100:.1f}%' if target >= 0.99 else f'{target*100:.0f}%',
                    'N/A', 'N/A', 'N/A'
                ])
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.18, 0.14, 0.14, 0.14, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    
    # 设置表头样式
    for i in range(5):
        table[(0, i)].set_facecolor('#2A9D8F')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # 根据策略设置行背景色
    colors = {'动态Gamma': '#FFEEEE', '静态Gamma': '#EEEEFF', '固定EF': '#EEFFEE'}
    for row_idx in range(1, len(table_data)):
        strategy_name = table_data[row_idx][0]
        if strategy_name in colors:
            for col_idx in range(5):
                table[(row_idx, col_idx)].set_facecolor(colors[strategy_name])
    
    ax.set_title('达成目标召回率所需参数汇总表\n(全优化配置 11110)', fontsize=14, fontweight='bold', pad=20)
    
    save_figure(fig, output_dir, 'fig13_param_recall_summary_table.png')


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("=" * 60)
    print("实验结果分析可视化")
    print("=" * 60)
    
    # 创建输出目录
    output_dir = Path('analysis_plots')
    output_dir.mkdir(exist_ok=True)
    
    # 加载数据
    print("\n加载数据...")
    df = load_data('experiment_results.csv')
    print(f"[OK] 已加载 {len(df)} 条记录")
    print(f"  配置数量: {df['code_bits'].nunique()}")
    print(f"  策略类型: {df['strategy'].unique().tolist()}")
    print(f"  M 值: {sorted(df['M'].unique())}")
    print(f"  efC 值: {sorted(df['efC'].unique())}")
    
    # 绘制所有图表
    print("\n" + "=" * 60)
    print("开始绘制图表...")
    print("=" * 60)
    
    # 图1-2: 搜索策略和模块对 Recall/QPS 的影响
    print("\n--- 搜索策略和模块分析 ---")
    plot_fig01_strategy_recall_qps(df, output_dir)
    plot_fig02_module_ablation_recall_qps(df, output_dir)
    
    # 图3-5: M 和 efC 参数影响
    print("\n--- M 和 efC 参数影响分析 ---")
    plot_fig03_M_impact_recall_qps(df, output_dir)
    plot_fig04_efC_impact_recall_qps(df, output_dir)
    plot_fig05_M_efC_heatmap(df, output_dir)
    
    # 图6-7: 距离计算分析
    print("\n--- 距离计算分析 ---")
    plot_fig06_dist_calcs_vs_qps(df, output_dir)
    plot_fig07_dist_calcs_vs_recall(df, output_dir)
    plot_fig07b_strategy_dist_calcs_vs_recall(df, output_dir)
    
    # 图8-10: QPS 提升率分析
    print("\n--- QPS 提升率分析 ---")
    plot_fig08_module_qps_improvement(df, output_dir)
    plot_fig09_strategy_qps_comparison(df, output_dir)
    plot_fig10_strategy_improvement_matrix(df, output_dir)
    
    # 图11-13: 参数分析
    print("\n--- 达成目标召回率参数分析 ---")
    plot_fig11_params_for_target_recall(df, output_dir)
    plot_fig12_params_by_module(df, output_dir)
    plot_fig13_param_recall_summary_table(df, output_dir)
    
    print("\n" + "=" * 60)
    print(f"[OK] 所有图表已保存到 {output_dir.absolute()}")
    print("=" * 60)


if __name__ == '__main__':
    main()

