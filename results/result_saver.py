"""
实验结果保存工具模块

模块功能：
    提供统一的实验结果保存接口，将数据字典序列化为 JSON 格式并写入
    Results 目录。所有结果文件包含统一的元数据字段，方便后续绘图脚本读取。

各结果文件的字段结构：
    所有 JSON 文件均包含以下元数据字段：
    - experiment_name (str): 实验名称，如 "classification"、"attention_viz" 等
    - dataset (str): 数据集名称，如 "sst2"、"cwru"
    - timestamp (str): 结果保存的时间戳，格式为 ISO 8601
    - author (str): 固定为 "Kris"

    分类结果文件额外包含：accuracy, precision, recall, f1, per_class_f1,
    predictions, true_labels

    归因结果文件额外包含：samples 列表，每个样本含 text, tokens,
    attribution_scores, predicted_label, true_label

    评估结果文件额外包含：results 字典（各方法的评估分数）

依赖模块：
    - config.py：结果目录路径配置

作者：Kris
"""

import os
import json
import numpy as np
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR


class NumpyEncoder(json.JSONEncoder):
    """
    自定义 JSON 编码器，处理 numpy 数据类型的序列化

    numpy 的数组和标量类型（如 np.int64、np.float32）不能被标准 JSON
    编码器直接序列化，本编码器将其转换为 Python 原生类型。
    浮点数精度保留至小数点后六位。
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return round(float(obj), 6)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_result(data, filename):
    """
    将实验结果保存为 JSON 文件

    参数：
        data (dict): 实验结果字典，需包含实验特定的数据字段。
                     函数会自动补充 timestamp 和 author 元数据字段
                     （若字典中未包含）。
        filename (str): 输出文件名（含 .json 后缀），如 "sst2_classification.json"

    输出：
        JSON 文件写入 RESULTS_DIR 目录下

    注意：
        - 若 RESULTS_DIR 目录不存在，函数会自动创建
        - 浮点数保留六位小数
        - numpy 数组自动转换为 Python 列表
    """
    # 自动创建结果目录
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 补充元数据字段
    if "timestamp" not in data:
        data["timestamp"] = datetime.now().isoformat()
    if "author" not in data:
        data["author"] = "Kris"

    # 构建输出路径
    output_path = os.path.join(RESULTS_DIR, filename)

    # 序列化为 JSON 并写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, cls=NumpyEncoder, ensure_ascii=False, indent=2)

    print(f"结果已保存至 {output_path}")


def load_result(filename):
    """
    从 JSON 文件加载实验结果

    参数：
        filename (str): 文件名（含 .json 后缀）

    返回值：
        dict: 实验结果字典

    异常：
        FileNotFoundError: 文件不存在时抛出
    """
    filepath = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"结果文件不存在: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data
