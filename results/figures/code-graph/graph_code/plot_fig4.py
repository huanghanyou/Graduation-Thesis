# -*- coding: utf-8 -*-
"""
图4：各方法忠实度综合得分跨数据集对比（分组柱状图）

从 faithfulness_results.json 的 results 字段中读取四种可解释性方法在
SST-2 和 CWRU 两个数据集上的忠实度综合得分（AUC-Drop），进行分组柱状图对比。

数据来源：
    - faithfulness_results.json（results 字段）

输出文件名：图6-各方法忠实度综合得分对比.svg / .png

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


def load_faithfulness_data():
    """
    加载忠实度评估结果。

    返回：
        data: dict, 忠实度评估结果
    """
    with open(os.path.join(DATA_DIR, "faithfulness_results.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def main():
    """
    绘制各方法忠实度综合得分跨数据集对比分组柱状图，并保存。
    """
    os.makedirs(FIGURES_DIR_SVG, exist_ok=True)
    os.makedirs(FIGURES_DIR_PNG, exist_ok=True)

    data = load_faithfulness_data()
    results = data['results']

    # 方法键与中文名称的映射
    method_keys = ['attention', 'integrated_gradients', 'shap', 'lime']
    method_names = ['注意力权重', '集成梯度', 'SHAP', 'LIME']

    sst2_scores = [results[k]['sst2'] for k in method_keys]
    cwru_scores = [results[k]['cwru'] for k in method_keys]

    x = np.arange(len(method_names))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    bars1 = ax.bar(x - bar_width / 2, sst2_scores, bar_width,
                   label='SST-2（教育场景）', color=COLOR_SST2, edgecolor='white')
    bars2 = ax.bar(x + bar_width / 2, cwru_scores, bar_width,
                   label='CWRU（工程场景）', color=COLOR_CWRU, edgecolor='white')

    # 标注数值
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.008,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.008,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('忠实度得分（AUC-Drop）', fontsize=11)
    ax.set_title('各方法忠实度综合得分对比', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, fontsize=10)
    ax.tick_params(axis='y', labelsize=10)

    # 纵轴从0开始，上限略高于最大值
    max_val = max(max(sst2_scores), max(cwru_scores))
    ax.set_ylim(0, max_val * 1.15)

    ax.legend(loc='upper right', fontsize=10)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.yaxis.grid(True, color='#CCCCCC', linewidth=0.6, alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()

    fig.savefig(os.path.join(FIGURES_DIR_SVG, "图6-各方法忠实度综合得分对比.svg"),
                format='svg', bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR_PNG, "图6-各方法忠实度综合得分对比.png"),
                format='png', dpi=300, bbox_inches='tight')

    plt.close('all')


if __name__ == '__main__':
    main()
