# -*- coding: utf-8 -*-
"""
图8：CWRU 样本注意力权重逐层分布热力图

从 attention_cwru.json 中取索引为 0 的样本，提取 BERT 12 层的注意力权重分布，
以热力图形式展示各层对各 token 位置的注意力分配情况。使用 Oranges colormap。

数据来源：
    - attention_cwru.json（samples[0]）

输出文件名：图8-CWRU样本注意力权重逐层分布热力图.svg / .png

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


def load_attention_data():
    """
    加载 CWRU 注意力数据。

    返回：
        data: dict, 注意力可视化数据
    """
    with open(os.path.join(DATA_DIR, "attention_cwru.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def main():
    """
    绘制 CWRU 样本注意力权重逐层分布热力图，并保存。
    """
    os.makedirs(FIGURES_DIR_SVG, exist_ok=True)
    os.makedirs(FIGURES_DIR_PNG, exist_ok=True)

    data = load_attention_data()
    sample = data['samples'][0]
    tokens = sample['tokens']
    layer_attention = sample['layer_attention']

    num_layers = 12
    num_tokens = len(tokens)

    # 若 token 数量超过20个，只显示前20个
    max_display_tokens = 20
    display_tokens = tokens[:max_display_tokens]
    display_num = len(display_tokens)

    attention_matrix = np.zeros((num_layers, display_num))
    for layer_idx in range(num_layers):
        layer_key = str(layer_idx)
        layer_values = layer_attention[layer_key][:display_num]
        attention_matrix[layer_idx, :len(layer_values)] = layer_values

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # CWRU 使用 Oranges colormap
    im = ax.imshow(attention_matrix, aspect='auto', cmap='Oranges', interpolation='nearest')

    ax.set_xticks(np.arange(display_num))
    ax.set_xticklabels(display_tokens, rotation=45, ha='right', fontsize=9)

    layer_labels = [f'第{i}层' for i in range(num_layers)]
    ax.set_yticks(np.arange(num_layers))
    ax.set_yticklabels(layer_labels, fontsize=10)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('注意力权重', fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    ax.set_title('CWRU 样本注意力权重逐层分布（样本索引：0）', fontsize=12)

    plt.tight_layout()

    fig.savefig(os.path.join(FIGURES_DIR_SVG, "图8-CWRU样本注意力权重逐层分布热力图.svg"),
                format='svg', bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR_PNG, "图8-CWRU样本注意力权重逐层分布热力图.png"),
                format='png', dpi=300, bbox_inches='tight')

    plt.close('all')


if __name__ == '__main__':
    main()
