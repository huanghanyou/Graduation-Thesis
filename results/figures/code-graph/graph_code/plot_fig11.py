# -*- coding: utf-8 -*-
"""
图11：敏感度逐样本得分分布箱线图

从 sensitivity_results.json 的 per_sample_scores 字段中提取三种方法
（注意力权重、集成梯度、LIME）在 SST-2 和 CWRU 数据集上各10个逐样本
敏感度分数，以箱线图展示分布情况。

数据来源：
    - sensitivity_results.json（per_sample_scores 字段）

输出文件名：图7-敏感度逐样本得分分布箱线图.svg / .png

作者：Kris
"""

import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams

rcParams['font.family'] = ['Times New Roman', 'SimSun']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 11

DATA_DIR = r"D:\systemfiles\ccf-shap\Results"
FIGURES_DIR_SVG = r"D:\systemfiles\ccf-shap\Results\figures\code-graph\Figures\svg"
FIGURES_DIR_PNG = r"D:\systemfiles\ccf-shap\Results\figures\code-graph\Figures\png"


def load_sensitivity_data():
    """
    加载敏感度评估结果。

    返回：
        data: dict, 敏感度评估结果
    """
    with open(os.path.join(DATA_DIR, "sensitivity_results.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def main():
    """
    绘制敏感度逐样本得分分布箱线图（SST-2 和 CWRU 两个子图），并保存。
    """
    os.makedirs(FIGURES_DIR_SVG, exist_ok=True)
    os.makedirs(FIGURES_DIR_PNG, exist_ok=True)

    data = load_sensitivity_data()
    per_sample = data['per_sample_scores']

    method_names = ['注意力权重', '集成梯度', 'LIME']
    # 箱线图填充色
    box_colors = ['#A8C8E8', '#F5C7A0', '#A8D8A8']

    # 两个数据集的数据键
    datasets = {
        'SST-2': ['attention_sst2', 'integrated_gradients_sst2', 'lime_sst2'],
        'CWRU': ['attention_cwru', 'integrated_gradients_cwru', 'lime_cwru'],
    }
    subtitles = {
        'SST-2': 'SST-2 数据集敏感度逐样本分布',
        'CWRU': 'CWRU 数据集敏感度逐样本分布',
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor('white')

    for ax_idx, (ds_name, keys) in enumerate(datasets.items()):
        ax = axes[ax_idx]
        ax.set_facecolor('white')

        box_data = [per_sample[k] for k in keys]

        bp = ax.boxplot(box_data, labels=method_names, patch_artist=True,
                        widths=0.5, showfliers=True,
                        flierprops=dict(marker='o', markersize=4, alpha=0.6))

        # 设置箱体颜色
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('#333333')
        for median in bp['medians']:
            median.set_color('#333333')
            median.set_linewidth(1.5)
        for whisker in bp['whiskers']:
            whisker.set_color('#666666')
        for cap in bp['caps']:
            cap.set_color('#666666')

        # 水平参考线 y=0
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

        ax.set_ylabel('余弦相似度', fontsize=11)
        ax.set_ylim(-0.3, 1.1)
        ax.set_title(subtitles[ds_name], fontsize=11)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.yaxis.grid(True, color='#CCCCCC', linewidth=0.6, alpha=0.4)
        ax.set_axisbelow(True)

    fig.suptitle('各方法敏感度逐样本得分分布', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(os.path.join(FIGURES_DIR_SVG, "图7-敏感度逐样本得分分布箱线图.svg"),
                format='svg', bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR_PNG, "图7-敏感度逐样本得分分布箱线图.png"),
                format='png', dpi=300, bbox_inches='tight')

    plt.close('all')


if __name__ == '__main__':
    main()
