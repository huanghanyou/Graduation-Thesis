"""
SST-2 数据集加载与预处理模块

模块功能：
    通过 HuggingFace datasets 库加载 SST-2（GLUE 基准中的 sst2 子集），
    使用 bert-base-uncased 对应的 BertTokenizer 进行 WordPiece 分词，
    将文本转换为模型可接受的张量格式，并封装为 PyTorch DataLoader。

WordPiece 分词说明：
    WordPiece 是 BERT 使用的子词分词算法。对于词表中不存在的词，
    WordPiece 会将其拆分为若干已知的子词片段（以 "##" 前缀标记非首片段）。
    这种策略在保持词表规模可控的同时，能够处理未登录词（OOV）。

[CLS] token 说明：
    BERT 输入序列的首位为特殊标记 [CLS]。经过 Transformer 编码后，
    [CLS] 位置的隐藏状态被用作整个序列的聚合表示，用于下游分类任务。

填充与截断策略：
    所有输入序列统一填充或截断至 MAX_SEQ_LEN 长度。短于该长度的序列
    在右侧补 [PAD] token，长于该长度的序列从右侧截断。attention_mask
    标记有效 token 位置为 1，填充位置为 0。

依赖模块：config.py 中的路径与超参数配置

作者：Kris
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from datasets import load_dataset

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_NAME, MAX_SEQ_LEN, BATCH_SIZE, SEED


class SST2Dataset(Dataset):
    """
    SST-2 数据集的 PyTorch Dataset 封装

    参数：
        encodings: 分词后的编码结果，包含 input_ids、attention_mask、token_type_ids
        labels: 标签列表，0 表示负面情感，1 表示正面情感
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
            dict: 包含 input_ids、attention_mask、token_type_ids、labels 四个键，
                  值均为 PyTorch 张量
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def get_sst2_dataloaders():
    """
    加载 SST-2 数据集并返回训练、验证、测试三个 DataLoader

    处理流程：
        1. 通过 HuggingFace datasets 库加载 GLUE 的 sst2 子集
        2. 使用 BertTokenizer 对文本进行 WordPiece 分词
        3. 将分词结果封装为 SST2Dataset
        4. 创建 DataLoader，训练集打乱顺序，验证集和测试集不打乱

    返回值：
        tuple: (train_loader, val_loader, test_loader)
            - train_loader: 训练集 DataLoader
            - val_loader: 验证集 DataLoader
            - test_loader: 测试集 DataLoader

    注意：
        SST-2 的官方测试集不包含标签，因此此处将验证集同时作为测试集使用。
        若后续有需要，可将训练集进一步拆分出验证子集。
    """
    # 加载 SST-2 数据集
    dataset = load_dataset("glue", "sst2")

    # 初始化 BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # 提取各分割的文本与标签，转换为 Python 原生列表
    train_texts = list(dataset["train"]["sentence"])
    train_labels = list(dataset["train"]["label"])

    val_texts = list(dataset["validation"]["sentence"])
    val_labels = list(dataset["validation"]["label"])

    # SST-2 官方测试集标签为 -1（不可用），因此使用验证集作为测试集
    test_texts = val_texts
    test_labels = val_labels

    # 对文本进行 WordPiece 分词，统一填充和截断至 MAX_SEQ_LEN
    # padding="max_length" 确保所有序列长度一致
    # truncation=True 截断超过最大长度的序列
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

    # 封装为 PyTorch Dataset
    train_dataset = SST2Dataset(train_encodings, train_labels)
    val_dataset = SST2Dataset(val_encodings, val_labels)
    test_dataset = SST2Dataset(test_encodings, test_labels)

    # 创建 DataLoader
    # 训练集打乱顺序以提升泛化能力，验证集和测试集保持原始顺序
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


def get_sst2_raw_texts_and_labels(split="validation"):
    """
    获取 SST-2 数据集的原始文本与标签，用于可解释性分析

    参数：
        split: 数据集分割名，默认为 "validation"

    返回值：
        tuple: (texts, labels)
            - texts: 文本字符串列表
            - labels: 标签整数列表
    """
    dataset = load_dataset("glue", "sst2")
    texts = dataset[split]["sentence"]
    labels = dataset[split]["label"]
    return texts, labels
