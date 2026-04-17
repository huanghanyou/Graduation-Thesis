# -*- coding: utf-8 -*-
"""
一键批量执行所有绘图脚本

运行方式：
    python plot_all.py

功能：
    依次调用 graph_code 目录下的全部独立绘图脚本，
    将所有图像保存至 Figures/svg 和 Figures/png 目录。

作者：Kris
"""

import sys
import os

# 将 graph_code 目录加入模块搜索路径
GRAPH_CODE_DIR = r"D:\systemfiles\ccf-shap\Results\figures\code-graph\graph_code"
sys.path.insert(0, GRAPH_CODE_DIR)

# 使用非交互式后端，避免弹出窗口
import matplotlib
matplotlib.use('Agg')

# 依次导入并执行各脚本的 main 函数
import plot_fig1; plot_fig1.main()
print("[1/11] 图1-两数据集分类性能指标对比 已生成")

import plot_fig2; plot_fig2.main()
print("[2/11] 图4-SST2数据集忠实度AUC下降曲线 已生成")

import plot_fig3; plot_fig3.main()
print("[3/11] 图5-CWRU数据集忠实度AUC下降曲线 已生成")

import plot_fig4; plot_fig4.main()
print("[4/11] 图6-各方法忠实度综合得分对比 已生成")

import plot_fig5; plot_fig5.main()
print("[5/11] 图8-各方法敏感度得分对比 已生成")

import plot_fig6; plot_fig6.main()
print("[6/11] 图6-可解释性方法综合评估雷达图 已生成")

import plot_fig7; plot_fig7.main()
print("[7/11] 图2-SST2样本注意力权重逐层分布热力图 已生成")

import plot_fig8; plot_fig8.main()
print("[8/11] 图8-CWRU样本注意力权重逐层分布热力图 已生成")

import plot_fig9; plot_fig9.main()
print("[9/11] 图3-SST2样本多方法归因结果对比 已生成")

import plot_fig10; plot_fig10.main()
print("[10/11] 图10-CWRU样本多方法归因结果对比 已生成")

import plot_fig11; plot_fig11.main()
print("[11/11] 图7-敏感度逐样本得分分布箱线图 已生成")

print("\n所有图像已生成完毕。")
