"""
CWRU 文本数据集的 Dataset 类封装

模块功能：
    将 generate_cwru_text.py 生成的 CSV 格式 CWRU 文本数据加载为 PyTorch Dataset，
    使用 BertTokenizer 进行 WordPiece 分词，并提供 DataLoader 创建接口。

依赖模块：
    - config.py：路径配置与超参数
    - data/generate_cwru_text.py：数据生成脚本（若 CSV 不存在则自动调用）

作者：Kris
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODEL_NAME, MAX_SEQ_LEN, BATCH_SIZE, SEED, CWRU_TEXT_DATA_PATH
)
from data.generate_cwru_text import generate_cwru_text_dataset


class CWRUTextDataset(Dataset):
    """
    CWRU 文本数据集的 PyTorch Dataset 封装

    参数：
        encodings: BertTokenizer 分词后的编码结果
        labels: 标签列表，取值为 0-3，分别对应正常、内圈故障、外圈故障、滚动体故障
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        返回单条样本的张量字典

        返回值：
            dict: 包含 input_ids、attention_mask、token_type_ids、labels
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def get_cwru_dataloaders():
    """
    加载 CWRU 文本数据集并返回训练、验证、测试三个 DataLoader

    处理流程：
        1. 检查 CSV 文件是否存在，不存在则自动调用生成脚本
        2. 读取 CSV 数据
        3. 按 70%/15%/15% 比例拆分为训练集、验证集、测试集
        4. 使用 BertTokenizer 进行分词
        5. 封装为 CWRUTextDataset 并创建 DataLoader

    返回值：
        tuple: (train_loader, val_loader, test_loader)
    """
    # 若 CSV 文件不存在，自动生成
    if not os.path.exists(CWRU_TEXT_DATA_PATH):
        print("CWRU 文本数据集不存在，正在自动生成...")
        generate_cwru_text_dataset()

    # 读取 CSV 数据
    df = pd.read_csv(CWRU_TEXT_DATA_PATH)
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # 拆分数据集：先拆出测试集(15%)，再从剩余中拆出验证集(15%/85% ≈ 17.6%)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=SEED, stratify=labels
    )
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.176, random_state=SEED, stratify=train_labels
    )

    # 初始化 BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # WordPiece 分词
    train_encodings = tokenizer(
        train_texts, padding="max_length", truncation=True,
        max_length=MAX_SEQ_LEN, return_token_type_ids=True
    )
    val_encodings = tokenizer(
        val_texts, padding="max_length", truncation=True,
        max_length=MAX_SEQ_LEN, return_token_type_ids=True
    )
    test_encodings = tokenizer(
        test_texts, padding="max_length", truncation=True,
        max_length=MAX_SEQ_LEN, return_token_type_ids=True
    )

    # 封装为 Dataset
    train_dataset = CWRUTextDataset(train_encodings, train_labels)
    val_dataset = CWRUTextDataset(val_encodings, val_labels)
    test_dataset = CWRUTextDataset(test_encodings, test_labels)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


def get_cwru_raw_texts_and_labels(split="test"):
    """
    获取 CWRU 文本数据集的原始文本与标签，用于可解释性分析

    参数：
        split: 数据集分割名，"train"、"val" 或 "test"

    返回值：
        tuple: (texts, labels)
    """
    if not os.path.exists(CWRU_TEXT_DATA_PATH):
        generate_cwru_text_dataset()

    df = pd.read_csv(CWRU_TEXT_DATA_PATH)
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # 复现拆分逻辑以获取对应分割
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=SEED, stratify=labels
    )
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.176, random_state=SEED, stratify=train_labels
    )

    if split == "train":
        return train_texts, train_labels
    elif split == "val":
        return val_texts, val_labels
    else:
        return test_texts, test_labels
