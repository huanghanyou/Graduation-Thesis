# -*- coding: utf-8 -*-
"""
图1：两数据集分类性能指标对比（分组柱状图）

从 sst2_classification.json 和 cwru_classification.json 中读取准确率、精确率、
召回率、F1值四项分类指标，以分组柱状图形式展示 SST-2（教育场景）与 CWRU（工程场景）
的性能对比。

数据来源：
    - sst2_classification.json
    - cwru_classification.json

输出文件名：图1-两数据集分类性能指标对比.svg / .png

作者：Kris
"""

import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams

# 字体配置：中文使用宋体，英文与数字使用 Times New Roman
rcParams['font.family'] = ['Times New Roman', 'SimSun']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 11

# 路径配置
DATA_DIR = r"D:\systemfiles\ccf-shap\Results"
FIGURES_DIR_SVG = r"D:\systemfiles\ccf-shap\Results\figures\code-graph\Figures\svg"
FIGURES_DIR_PNG = r"D:\systemfiles\ccf-shap\Results\figures\code-graph\Figures\png"

# 颜色配置
COLOR_SST2 = '#4878CF'   # 钢蓝色，SST-2 / 教育场景
COLOR_CWRU = '#E07B39'   # 橙褐色，CWRU / 工程场景


def load_classification_data():
    """
    加载两个数据集的分类结果。

    返回：
        sst2_data: dict, SST-2 分类结果
        cwru_data: dict, CWRU 分类结果
    """
    with open(os.path.join(DATA_DIR, "sst2_classification.json"), "r", encoding="utf-8") as f:
        sst2_data = json.load(f)
    with open(os.path.join(DATA_DIR, "cwru_classification.json"), "r", encoding="utf-8") as f:
        cwru_data = json.load(f)
    return sst2_data, cwru_data


def main():
    """
    绘制两数据集分类性能指标对比分组柱状图，并保存为 SVG 和 PNG 格式。
    """
    os.makedirs(FIGURES_DIR_SVG, exist_ok=True)
    os.makedirs(FIGURES_DIR_PNG, exist_ok=True)

    sst2_data, cwru_data = load_classification_data()

    # 提取四项指标
    metrics_names = ['准确率', '精确率', '召回率', 'F1值']
    keys = ['accuracy', 'precision', 'recall', 'f1']
    sst2_values = [sst2_data[k] for k in keys]
    cwru_values = [cwru_data[k] for k in keys]

    x = np.arange(len(metrics_names))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    bars1 = ax.bar(x - bar_width / 2, sst2_values, bar_width,
                   label='SST-2（教育场景）', color=COLOR_SST2, edgecolor='white')
    bars2 = ax.bar(x + bar_width / 2, cwru_values, bar_width,
                   label='CWRU（工程场景）', color=COLOR_CWRU, edgecolor='white')

    # 柱子上方标注数值，保留三位小数
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.003,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.003,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('指标值', fontsize=11)
    ax.set_title('两数据集分类性能指标对比', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_ylim(0.85, 1.02)
    ax.legend(loc='upper right', fontsize=10)

    # 隐藏上边框和右边框
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # 网格线
    ax.yaxis.grid(True, color='#CCCCCC', linewidth=0.6, alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()

    fig.savefig(os.path.join(FIGURES_DIR_SVG, "图1-两数据集分类性能指标对比.svg"),
                format='svg', bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR_PNG, "图1-两数据集分类性能指标对比.png"),
                format='png', dpi=300, bbox_inches='tight')

    plt.close('all')


if __name__ == '__main__':
    main()
