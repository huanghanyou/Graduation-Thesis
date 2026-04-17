# -*- coding: utf-8 -*-
"""
图10：CWRU 样本多方法归因结果对比（多子图水平柱状图）

取索引为 0 的样本，分别从 attention_cwru.json（第11层注意力）、ig_cwru.json、
shap_cwru.json、lime_cwru.json 中提取归因分数，以 4x1 的水平柱状图子图布局
展示四种方法的归因结果对比。

数据来源：
    - attention_cwru.json（索引0，第11层）
    - ig_cwru.json（索引0）
    - shap_cwru.json（索引0）
    - lime_cwru.json（索引0）

输出文件名：图10-CWRU样本多方法归因结果对比.svg / .png

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

MAX_TOKENS = 25


def load_all_attribution_data():
    """
    加载 CWRU 上四种方法的归因数据。

    返回：
        attn_sample: dict, 注意力样本数据
        ig_sample: dict, 集成梯度样本数据
        shap_sample: dict, SHAP 样本数据
        lime_sample: dict, LIME 样本数据
    """
    with open(os.path.join(DATA_DIR, "attention_cwru.json"), "r", encoding="utf-8") as f:
        attn_data = json.load(f)
    with open(os.path.join(DATA_DIR, "ig_cwru.json"), "r", encoding="utf-8") as f:
        ig_data = json.load(f)
    with open(os.path.join(DATA_DIR, "shap_cwru.json"), "r", encoding="utf-8") as f:
        shap_data = json.load(f)
    with open(os.path.join(DATA_DIR, "lime_cwru.json"), "r", encoding="utf-8") as f:
        lime_data = json.load(f)

    return (attn_data['samples'][0], ig_data['samples'][0],
            shap_data['samples'][0], lime_data['samples'][0])


def main():
    """
    绘制 CWRU 样本多方法归因结果对比水平柱状图，并保存。
    """
    os.makedirs(FIGURES_DIR_SVG, exist_ok=True)
    os.makedirs(FIGURES_DIR_PNG, exist_ok=True)

    attn_sample, ig_sample, shap_sample, lime_sample = load_all_attribution_data()

    # 注意力权重：取第11层
    attn_tokens = attn_sample['tokens'][:MAX_TOKENS]
    attn_scores = attn_sample['layer_attention']['11'][:MAX_TOKENS]

    ig_tokens = ig_sample['tokens'][:MAX_TOKENS]
    ig_scores = ig_sample['attribution_scores'][:MAX_TOKENS]

    shap_tokens = shap_sample['tokens'][:MAX_TOKENS]
    shap_scores = shap_sample['attribution_scores'][:MAX_TOKENS]

    lime_tokens = lime_sample['tokens'][:MAX_TOKENS]
    lime_scores = lime_sample['attribution_scores'][:MAX_TOKENS]

    methods_data = [
        ('注意力权重', attn_tokens, attn_scores),
        ('集成梯度', ig_tokens, ig_scores),
        ('SHAP', shap_tokens, shap_scores),
        ('LIME', lime_tokens, lime_scores),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(10, 14))
    fig.patch.set_facecolor('white')

    for idx, (name, tokens, scores) in enumerate(methods_data):
        ax = axes[idx]
        ax.set_facecolor('white')

        scores_arr = np.array(scores)
        colors = ['#4878CF' if s >= 0 else '#D04040' for s in scores_arr]

        y_pos = np.arange(len(tokens))
        ax.barh(y_pos, scores_arr, color=colors, height=0.7, edgecolor='white')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(tokens, fontsize=9)
        ax.invert_yaxis()

        ax.set_title(name, fontsize=11)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.xaxis.grid(True, color='#CCCCCC', linewidth=0.6, alpha=0.4)
        ax.set_axisbelow(True)

    axes[-1].set_xlabel('归因分数', fontsize=11)

    fig.suptitle('CWRU 样本多方法归因结果对比（样本索引：0）', fontsize=12, y=1.01)
    plt.tight_layout(pad=1.5)

    fig.savefig(os.path.join(FIGURES_DIR_SVG, "图10-CWRU样本多方法归因结果对比.svg"),
                format='svg', bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR_PNG, "图10-CWRU样本多方法归因结果对比.png"),
                format='png', dpi=300, bbox_inches='tight')

    plt.close('all')


if __name__ == '__main__':
    main()
