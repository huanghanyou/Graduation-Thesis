# -*- coding: utf-8 -*-
"""
图3：CWRU 数据集忠实度 AUC 下降曲线（折线图）

从 faithfulness_results.json 中读取 CWRU 数据集上四种可解释性方法的
遮蔽比例-预测概率下降幅度曲线，展示各方法的忠实度表现。

数据来源：
    - faithfulness_results.json（auc_drop_curves 中 *_cwru 字段，masking_ratios 字段）

输出文件名：图5-CWRU数据集忠实度AUC下降曲线.svg / .png

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
    绘制 CWRU 数据集忠实度 AUC 下降曲线，并保存为 SVG 和 PNG 格式。
    """
    os.makedirs(FIGURES_DIR_SVG, exist_ok=True)
    os.makedirs(FIGURES_DIR_PNG, exist_ok=True)

    data = load_faithfulness_data()
    masking_ratios = data['masking_ratios']
    curves = data['auc_drop_curves']

    # 使用 tab10 调色板前四色
    cmap = plt.cm.tab10
    colors = [cmap(i) for i in range(4)]

    # 四种方法配置：名称、数据键、线型、标记
    methods = [
        ('注意力权重', 'attention_cwru', '-', 'o'),
        ('集成梯度', 'integrated_gradients_cwru', '--', '^'),
        ('SHAP', 'shap_cwru', '-.', 's'),
        ('LIME', 'lime_cwru', (0, (5, 3)), 'D'),  # 长虚线
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for i, (name, key, ls, marker) in enumerate(methods):
        ax.plot(masking_ratios, curves[key], linestyle=ls, marker=marker,
                color=colors[i], linewidth=1.8, markersize=6, label=name)

    # 横轴百分号格式
    ax.set_xticks(masking_ratios)
    ax.set_xticklabels([f'{int(r * 100)}%' for r in masking_ratios], fontsize=10)
    ax.set_xlabel('遮蔽比例', fontsize=11)
    ax.set_ylabel('预测概率下降幅度', fontsize=11)
    ax.set_title('CWRU 数据集忠实度 AUC 下降曲线', fontsize=12)
    ax.tick_params(axis='y', labelsize=10)

    # 纵轴从0开始，上限略高于最大值
    all_values = []
    for _, key, _, _ in methods:
        all_values.extend(curves[key])
    ax.set_ylim(0, max(all_values) * 1.1)

    ax.legend(loc='lower right', fontsize=10)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.yaxis.grid(True, color='#CCCCCC', linewidth=0.6, alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()

    fig.savefig(os.path.join(FIGURES_DIR_SVG, "图5-CWRU数据集忠实度AUC下降曲线.svg"),
                format='svg', bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR_PNG, "图5-CWRU数据集忠实度AUC下降曲线.png"),
                format='png', dpi=300, bbox_inches='tight')

    plt.close('all')


if __name__ == '__main__':
    main()
