"""
CWRU 轴承故障文本描述数据集生成脚本

模块功能：
    生成模拟的 CWRU 轴承故障文本描述数据集。原始 CWRU 数据集为振动信号数据，
    本模块将不同故障类型的信号特征转化为结构化文本描述，用于文本分类任务。

类别对应关系：
    - 0: Normal（正常状态）—— 振动幅度小、频谱平稳、无异常频率成分
    - 1: Inner Race Fault（内圈故障）—— 高频周期性冲击、内圈通过频率及其谐波突出
    - 2: Outer Race Fault（外圈故障）—— 低频周期性冲击、外圈通过频率突出、振幅调制
    - 3: Ball Fault（滚动体故障）—— 不规则冲击、滚动体自转频率成分、振幅不稳定

文本构造逻辑：
    每条文本由多个描述片段随机组合而成，涵盖振动幅度特征、频谱特征词、
    运行状态描述及故障特征词等工程语义词汇，使不同类别在语义上具有区分度。

输出：
    CSV 格式文件，保存至 data/cwru_text_data.csv，字段为 text 和 label

作者：Kris
"""

import os
import random
import csv

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CWRU_TEXT_DATA_PATH, SEED

# 每个类别生成的样本数量
SAMPLES_PER_CLASS = 300

# 各类别的文本模板片段
# 每条文本通过从对应类别的模板库中随机采样多个片段拼接而成

NORMAL_TEMPLATES = {
    "vibration": [
        "The vibration amplitude remains at a low and stable level.",
        "Vibration signals show minimal fluctuation during operation.",
        "The overall vibration intensity is within the normal operating range.",
        "No significant vibration spikes are detected in the time domain.",
        "The bearing vibration is smooth with consistent amplitude.",
        "RMS vibration values are well below the warning threshold.",
        "Peak-to-peak vibration amplitude is stable across measurement cycles.",
    ],
    "frequency": [
        "The frequency spectrum shows no abnormal peaks or harmonic components.",
        "Spectral analysis reveals a flat and uniform frequency distribution.",
        "No characteristic fault frequencies are present in the spectrum.",
        "The frequency domain is clean with only shaft rotation frequency visible.",
        "Spectral energy is concentrated at the fundamental rotation frequency.",
        "No sideband frequencies are observed around the main spectral peaks.",
    ],
    "status": [
        "The bearing operates under normal conditions with no signs of degradation.",
        "Operating temperature and noise levels are within acceptable limits.",
        "The machinery runs smoothly without any unusual acoustic emissions.",
        "All monitoring parameters indicate healthy bearing condition.",
        "The bearing lubrication condition is adequate and effective.",
        "No abnormal wear patterns are detected during inspection.",
    ],
}

INNER_RACE_TEMPLATES = {
    "vibration": [
        "High-frequency periodic impulses are detected in the vibration signal.",
        "The vibration amplitude shows modulation at the shaft rotation frequency.",
        "Impulsive vibration components appear at regular intervals corresponding to inner race defect.",
        "The time-domain signal exhibits sharp spikes with high crest factor.",
        "Vibration energy increases notably in the high-frequency band.",
        "Periodic transient impacts are observed in the vibration waveform.",
        "The kurtosis value of vibration signal is elevated indicating impulsive behavior.",
    ],
    "frequency": [
        "The inner race fault characteristic frequency and its harmonics are prominent.",
        "Ball pass frequency inner race (BPFI) components dominate the spectrum.",
        "Sidebands spaced at shaft speed appear around the BPFI peaks.",
        "The frequency spectrum reveals inner race defect frequency at multiple harmonics.",
        "Envelope analysis highlights the inner race fault frequency.",
        "High-frequency resonance bands are excited by inner race fault impacts.",
    ],
    "status": [
        "An inner race defect is developing on the bearing surface.",
        "The bearing shows signs of inner race spalling or pitting.",
        "Inner race fault progression is indicated by increasing vibration severity.",
        "The fault is located on the inner raceway of the bearing.",
        "Localized damage on the inner race causes periodic impact excitation.",
        "Inner race surface degradation is detected through vibration monitoring.",
    ],
}

OUTER_RACE_TEMPLATES = {
    "vibration": [
        "Low-frequency periodic impulses characterize the vibration pattern.",
        "The vibration signal shows amplitude modulation at outer race defect frequency.",
        "Repetitive impacts at constant intervals indicate outer race damage.",
        "The time-domain signal contains periodic transients with moderate amplitude.",
        "Vibration levels increase in the low-to-mid frequency range.",
        "The shock pulse pattern is consistent with outer race surface defects.",
        "Periodic impulsive behavior is observed without shaft speed modulation.",
    ],
    "frequency": [
        "The outer race fault characteristic frequency (BPFO) is clearly visible.",
        "Ball pass frequency outer race dominates the lower frequency spectrum.",
        "Multiple harmonics of the outer race defect frequency are present.",
        "The frequency spectrum shows BPFO with limited sideband activity.",
        "Envelope spectrum peaks correspond to outer race characteristic frequency.",
        "Outer race defect frequency components maintain stable amplitude over time.",
    ],
    "status": [
        "An outer race defect has developed on the bearing.",
        "The bearing exhibits outer raceway surface damage.",
        "Outer race fault is identified through characteristic frequency analysis.",
        "Stationary outer race damage produces consistent impact patterns.",
        "The fault location on the outer race is confirmed by directional vibration analysis.",
        "Outer race pitting creates periodic loading variation on rolling elements.",
    ],
}

BALL_FAULT_TEMPLATES = {
    "vibration": [
        "Irregular and erratic impulses appear in the vibration signal.",
        "The vibration amplitude fluctuates unpredictably during operation.",
        "Non-periodic transient spikes indicate rolling element surface damage.",
        "The time-domain signal shows intermittent bursts of high vibration.",
        "Vibration characteristics change with bearing cage rotation.",
        "The impact pattern is irregular due to rolling element defect orientation changes.",
        "Amplitude instability in vibration suggests rolling element surface irregularity.",
    ],
    "frequency": [
        "Ball spin frequency (BSF) components appear in the frequency spectrum.",
        "The spectrum contains rolling element defect frequency with cage frequency modulation.",
        "Frequency components related to ball spin frequency and its harmonics are detected.",
        "Cage frequency sidebands surround the ball defect frequency peaks.",
        "The envelope spectrum reveals ball fault frequency modulated by cage speed.",
        "Spectral components at twice the ball spin frequency indicate rolling element damage.",
    ],
    "status": [
        "A rolling element defect is present in the bearing.",
        "Ball surface damage causes intermittent contact anomalies.",
        "The rolling element fault produces variable vibration depending on defect orientation.",
        "Surface spalling on rolling elements leads to unstable vibration patterns.",
        "Rolling element damage creates complex fault signatures in vibration data.",
        "Ball fault severity is assessed through changes in vibration statistical indicators.",
    ],
}


def generate_text_sample(template_dict):
    """
    从模板字典中随机采样片段，拼接生成一条文本描述

    参数：
        template_dict: 按特征类别（vibration、frequency、status）组织的模板字典，
                       每个类别包含多条候选描述片段

    返回值：
        str: 拼接后的文本描述
    """
    parts = []
    for category, templates in template_dict.items():
        # 每个特征类别随机选取 1-2 条描述
        num_select = random.randint(1, 2)
        selected = random.sample(templates, min(num_select, len(templates)))
        parts.extend(selected)

    # 打乱片段顺序，增加文本多样性
    random.shuffle(parts)
    return " ".join(parts)


def generate_cwru_text_dataset():
    """
    生成完整的 CWRU 文本描述数据集并保存为 CSV 文件

    生成流程：
        1. 为四个类别分别生成 SAMPLES_PER_CLASS 条文本描述
        2. 打乱所有样本顺序
        3. 以 CSV 格式写入文件，字段为 text 和 label

    输出文件路径：由 config.CWRU_TEXT_DATA_PATH 指定
    """
    random.seed(SEED)

    all_samples = []

    # 类别 0：正常状态
    for _ in range(SAMPLES_PER_CLASS):
        text = generate_text_sample(NORMAL_TEMPLATES)
        all_samples.append({"text": text, "label": 0})

    # 类别 1：内圈故障
    for _ in range(SAMPLES_PER_CLASS):
        text = generate_text_sample(INNER_RACE_TEMPLATES)
        all_samples.append({"text": text, "label": 1})

    # 类别 2：外圈故障
    for _ in range(SAMPLES_PER_CLASS):
        text = generate_text_sample(OUTER_RACE_TEMPLATES)
        all_samples.append({"text": text, "label": 2})

    # 类别 3：滚动体故障
    for _ in range(SAMPLES_PER_CLASS):
        text = generate_text_sample(BALL_FAULT_TEMPLATES)
        all_samples.append({"text": text, "label": 3})

    # 打乱样本顺序
    random.shuffle(all_samples)

    # 确保目标目录存在
    os.makedirs(os.path.dirname(CWRU_TEXT_DATA_PATH), exist_ok=True)

    # 写入 CSV 文件
    with open(CWRU_TEXT_DATA_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(all_samples)

    print(f"CWRU 文本数据集已生成，共 {len(all_samples)} 条样本，保存至 {CWRU_TEXT_DATA_PATH}")


if __name__ == "__main__":
    generate_cwru_text_dataset()
