# -*- coding: utf-8 -*-
"""
图5：各方法敏感度得分跨数据集对比（分组柱状图）

从 sensitivity_results.json 的 results 字段中读取三种可解释性方法（注意力权重、
集成梯度、LIME）在 SST-2 和 CWRU 数据集上的敏感度均值得分，进行分组柱状图对比。
SHAP 未参与敏感度评估，不显示。

数据来源：
    - sensitivity_results.json（results 字段）

输出文件名：图8-各方法敏感度得分对比.svg / .png

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

COLOR_SST2 = '#4878CF'
COLOR_CWRU = '#E07B39'


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
    绘制各方法敏感度得分跨数据集对比分组柱状图，并保存。
    """
    os.makedirs(FIGURES_DIR_SVG, exist_ok=True)
    os.makedirs(FIGURES_DIR_PNG, exist_ok=True)

    data = load_sensitivity_data()
    results = data['results']

    # 仅三种方法（SHAP 未参与敏感度评估）
    method_keys = ['attention', 'integrated_gradients', 'lime']
    method_names = ['注意力权重', '集成梯度', 'LIME']

    sst2_scores = [results[k]['sst2'] for k in method_keys]
    cwru_scores = [results[k]['cwru'] for k in method_keys]

    x = np.arange(len(method_names))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    bars1 = ax.bar(x - bar_width / 2, sst2_scores, bar_width,
                   label='SST-2（教育场景）', color=COLOR_SST2, edgecolor='white')
    bars2 = ax.bar(x + bar_width / 2, cwru_scores, bar_width,
                   label='CWRU（工程场景）', color=COLOR_CWRU, edgecolor='white')

    # 标注数值
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.012,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.012,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('敏感度得分（余弦相似度均值）', fontsize=11)
    ax.set_title('各方法敏感度得分对比', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_ylim(0, 1.05)

    ax.legend(loc='upper right', fontsize=10)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.yaxis.grid(True, color='#CCCCCC', linewidth=0.6, alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()

    fig.savefig(os.path.join(FIGURES_DIR_SVG, "图8-各方法敏感度得分对比.svg"),
                format='svg', bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR_PNG, "图8-各方法敏感度得分对比.png"),
                format='png', dpi=300, bbox_inches='tight')

    plt.close('all')


if __name__ == '__main__':
    main()
