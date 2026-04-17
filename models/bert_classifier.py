"""
基于 BERT 的文本分类模型定义

模块功能：
    定义 BertTextClassifier 类，使用 HuggingFace 的 BertModel 作为文本编码器，
    在 [CLS] token 的最终隐藏状态之上接全连接分类头，完成文本分类任务。

[CLS] 表示的获取方式：
    BERT 的输出 last_hidden_state 形状为 (batch_size, seq_len, hidden_size)。
    取序列中第 0 个位置（即 [CLS] token）的隐藏状态作为整个输入序列的聚合表示，
    记为 h ∈ R^d，其中 d = 768（BERT-base 的隐藏维度）。

分类头结构：
    由一层 Dropout（防止过拟合）和一层全连接层（Linear: d → num_labels）组成，
    将 [CLS] 表示映射为各类别的 logits。

依赖模块：config.py 中的模型名称配置

作者：Kris
"""

import torch
import torch.nn as nn
from transformers import BertModel

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_NAME


class BertTextClassifier(nn.Module):
    """
    基于 BERT 的文本分类模型

    参数：
        num_labels (int): 分类类别数。SST-2 为 2，CWRU 为 4。
        model_name (str): 预训练 BERT 模型名称，默认使用 config 中的 MODEL_NAME。
        dropout_rate (float): Dropout 概率，默认 0.1。

    前向传播输入：
        input_ids: token 编码序列，形状 (batch_size, seq_len)
        attention_mask: 注意力掩码，形状 (batch_size, seq_len)
        token_type_ids: token 类型编码，形状 (batch_size, seq_len)

    前向传播输出：
        logits: 各类别的未归一化分数，形状 (batch_size, num_labels)
        cls_hidden_state: [CLS] 位置的隐藏表示，形状 (batch_size, hidden_size)
    """

    def __init__(self, num_labels, model_name=MODEL_NAME, dropout_rate=0.1):
        super(BertTextClassifier, self).__init__()

        # 加载预训练 BERT 编码器
        # output_attentions=True 使模型输出注意力权重，供可解释性分析使用
        self.bert = BertModel.from_pretrained(model_name, output_attentions=True)

        # 隐藏层维度，BERT-base 为 768
        self.hidden_size = self.bert.config.hidden_size

        # Dropout 层，用于正则化，防止分类头过拟合
        self.dropout = nn.Dropout(dropout_rate)

        # 全连接分类头，将 [CLS] 的 d 维表示映射为 num_labels 维 logits
        self.classifier = nn.Linear(self.hidden_size, num_labels)

        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        前向传播

        参数：
            input_ids: token 编码序列，形状 (batch_size, seq_len)
            attention_mask: 注意力掩码，1 表示有效 token，0 表示填充位置
            token_type_ids: 段落类型编码，单句输入全为 0

        返回值：
            logits: 分类 logits，形状 (batch_size, num_labels)
            cls_hidden_state: [CLS] 隐藏状态，形状 (batch_size, hidden_size)
        """
        # BERT 编码器前向传播
        # outputs.last_hidden_state: (batch_size, seq_len, hidden_size)
        # outputs.attentions: 元组，每层注意力权重 (batch_size, num_heads, seq_len, seq_len)
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # 取 [CLS] token（序列第 0 个位置）的隐藏状态作为序列表示
        # pooled_output 形状: (batch_size, hidden_size)
        cls_hidden_state = outputs.last_hidden_state[:, 0, :]

        # 经过 Dropout 后送入分类头
        pooled = self.dropout(cls_hidden_state)
        logits = self.classifier(pooled)

        return logits, cls_hidden_state

    def get_attentions(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        获取 BERT 所有层的注意力权重

        参数：
            input_ids: token 编码序列
            attention_mask: 注意力掩码
            token_type_ids: 段落类型编码

        返回值：
            attentions: 元组，长度为 num_layers (12 for BERT-base)，
                        每个元素形状为 (batch_size, num_heads, seq_len, seq_len)
                        BERT-base 有 12 层 Transformer，每层 12 个注意力头
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return outputs.attentions
