# -*- coding: utf-8 -*-
"""
图6：忠实度与敏感度综合评估雷达图

从 faithfulness_results.json 和 sensitivity_results.json 中读取各方法在两个
数据集上的忠实度和敏感度得分，以雷达图形式展示综合评估结果。共七个评估维度，
两个并排子图分别对应 SST-2 和 CWRU 数据集。

对每个维度取所有得分中的最大值进行归一化至 [0, 1] 区间，使各维度在雷达图中
具有可比性。若某方法在某数据集上缺少敏感度数据（如 SHAP），该维度标注为 0。

数据来源：
    - faithfulness_results.json（results 字段）
    - sensitivity_results.json（results 字段）

输出文件名：图6-可解释性方法综合评估雷达图.svg / .png

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


def load_data():
    """
    加载忠实度和敏感度评估结果。

    返回：
        faith_data: dict, 忠实度评估结果
        sens_data: dict, 敏感度评估结果
    """
    with open(os.path.join(DATA_DIR, "faithfulness_results.json"), "r", encoding="utf-8") as f:
        faith_data = json.load(f)
    with open(os.path.join(DATA_DIR, "sensitivity_results.json"), "r", encoding="utf-8") as f:
        sens_data = json.load(f)
    return faith_data, sens_data


def main():
    """
    绘制忠实度与敏感度综合评估雷达图（两个并排子图），并保存。
    """
    os.makedirs(FIGURES_DIR_SVG, exist_ok=True)
    os.makedirs(FIGURES_DIR_PNG, exist_ok=True)

    faith_data, sens_data = load_data()
    faith_results = faith_data['results']
    sens_results = sens_data['results']

    # 七个评估维度
    dimension_labels = [
        '忠实-注意力', '忠实-集成梯度', '忠实-SHAP', '忠实-LIME',
        '敏感-注意力', '敏感-集成梯度', '敏感-LIME'
    ]

    # 提取各数据集在各维度上的原始得分
    def get_scores(dataset_key):
        """
        获取某数据集在七个维度上的得分。

        参数：
            dataset_key: str, 数据集标识（'sst2' 或 'cwru'）

        返回：
            scores: list, 七维得分列表
        """
        scores = []
        # 忠实度四维
        for method_key in ['attention', 'integrated_gradients', 'shap', 'lime']:
            scores.append(faith_results[method_key][dataset_key])
        # 敏感度三维（SHAP 缺失时填0）
        for method_key in ['attention', 'integrated_gradients', 'lime']:
            if method_key in sens_results and dataset_key in sens_results[method_key]:
                scores.append(sens_results[method_key][dataset_key])
            else:
                scores.append(0.0)
        return scores

    sst2_scores = get_scores('sst2')
    cwru_scores = get_scores('cwru')

    # 归一化至 [0, 1]：对每个维度取两个数据集中较大的值作为归一化基准
    all_scores = np.array([sst2_scores, cwru_scores])
    max_per_dim = np.max(all_scores, axis=0)
    # 避免除以零
    max_per_dim[max_per_dim == 0] = 1.0
    sst2_norm = (np.array(sst2_scores) / max_per_dim).tolist()
    cwru_norm = (np.array(cwru_scores) / max_per_dim).tolist()

    num_dims = len(dimension_labels)
    # 雷达图角度
    angles = np.linspace(0, 2 * np.pi, num_dims, endpoint=False).tolist()
    # 闭合多边形
    angles += angles[:1]
    sst2_norm += sst2_norm[:1]
    cwru_norm += cwru_norm[:1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6),
                             subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('white')

    titles = ['SST-2 数据集', 'CWRU 数据集']
    data_pairs = [
        (sst2_norm, cwru_norm),
        (sst2_norm, cwru_norm),
    ]

    for idx, ax in enumerate(axes):
        ax.set_facecolor('white')

        # 两条轮廓线
        ax.plot(angles, sst2_norm, color=COLOR_SST2, linewidth=1.8,
                label='SST-2（教育场景）')
        ax.fill(angles, sst2_norm, color=COLOR_SST2, alpha=0.15)

        ax.plot(angles, cwru_norm, color=COLOR_CWRU, linewidth=1.8,
                label='CWRU（工程场景）')
        ax.fill(angles, cwru_norm, color=COLOR_CWRU, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimension_labels, fontsize=9)
        ax.set_title(titles[idx], fontsize=12, pad=20)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)

    # 共享图例，放置在整体图下方中央
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('可解释性方法综合评估雷达图', fontsize=12, y=1.02)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])

    fig.savefig(os.path.join(FIGURES_DIR_SVG, "图6-可解释性方法综合评估雷达图.svg"),
                format='svg', bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR_PNG, "图6-可解释性方法综合评估雷达图.png"),
                format='png', dpi=300, bbox_inches='tight')

    plt.close('all')


if __name__ == '__main__':
    main()
