"""
Streamlit 交互式可视化分析界面

模块功能：
    提供基于 Streamlit 的 Web 交互界面，供研究者对单条文本样本运行
    多种可解释性分析方法，并以热力表格形式展示归因结果。支持多方法
    并排对比模式。

UI 组件说明：
    - 侧边栏：数据集选择（SST-2 / CWRU）与可解释性方法选择
    - 主区域：文本输入框、"运行分析"按钮、预测结果展示、归因结果热力图
    - 多方法对比：选择多个方法时，以并列列方式展示各方法的归因分布

数据流向：
    用户输入文本 -> 分词与编码 -> 模型推理（预测类别与置信度）
    -> 选定的可解释性方法计算归因分数 -> 归因结果表格化渲染

依赖模块：
    - models/bert_classifier.py：文本分类模型
    - train/trainer.py：模型权重加载
    - explainability/ 各归因模块
    - config.py：全局配置

作者：Kris
"""

import sys
import os

# 确保可以导入项目根目录的模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import BertTokenizer
import time

from config import MODEL_NAME, MAX_SEQ_LEN, DEVICE, SST2_NUM_LABELS, CWRU_NUM_LABELS
from models.bert_classifier import BertTextClassifier
from train.trainer import load_trained_model


# ==================== 页面配置 ====================

st.set_page_config(page_title="Transformer 可解释性分析工具", layout="wide")

st.title("🔍 Transformer 可解释性分析工具 (CCF-SHAP)")
st.caption("基于可解释性方法的 Transformer 模型可视化研究 | 作者：Kris")
st.markdown("---")

# 页面说明
with st.expander("ℹ️ 使用说明"):
    st.markdown("""
    ### 工具功能
    本工具提供对 BERT 文本分类模型的多种可解释性分析方法：
    - **注意力可视化**：展示模型各层对输入词语的关注权重
    - **Grad-CAM**：梯度加权的特征图激活，用于生成 token 级别的重要性分数
    - **Integrated Gradients (IG)**：基于梯度的输入属性方法，展示各词对预测的贡献
    - **SHAP**：基于合作博弈论的特征归因方法
    - **LIME**：基于局部线性代理模型的解释
    
    ### 使用步骤
    1. 在左侧选择数据集（SST-2 或 CWRU）和可解释性方法
    2. 输入或选择示例文本
    3. 点击"运行分析"按钮
    4. 查看预测结果和各方法的归因分析结果
    5. 可同时选择多个方法进行并排对比
    
    ### 数据集说明
    - **SST-2**：教育文本情感分类（二分类）：负面/正面
    - **CWRU**：轴承故障文本分类（四分类）：正常/内圈故障/外圈故障/滚动体故障
    """)


# ==================== 侧边栏配置 ====================

st.sidebar.header("分析配置")

# 数据集选择
dataset_option = st.sidebar.selectbox(
    "选择数据集",
    options=["SST-2", "CWRU"],
    help="SST-2 为教育文本情感分类（二分类），CWRU 为轴承故障文本分类（四分类）",
)

# 可解释性方法选择（支持多选）
method_options = st.sidebar.multiselect(
    "选择可解释性方法",
    options=["注意力可视化", "Grad-CAM", "Integrated Gradients", "SHAP", "LIME"],
    default=["注意力可视化", "Integrated Gradients"],
    help="可同时选择多个方法进行并排对比",
)

# 可视化选项
st.sidebar.markdown("### 可视化选项")
show_heatmap = st.sidebar.checkbox("显示热力图", value=True)
show_bar_chart = st.sidebar.checkbox("显示条形图（Top-10）", value=True)
highlight_style = st.sidebar.radio(
    "Token 高亮样式", ["背景色", "条形"], horizontal=True
)

# 数据集对应的标签信息
if dataset_option == "SST-2":
    num_labels = SST2_NUM_LABELS
    label_names = {0: "负面情感", 1: "正面情感"}
    dataset_key = "sst2"
else:
    num_labels = CWRU_NUM_LABELS
    label_names = {0: "正常", 1: "内圈故障", 2: "外圈故障", 3: "滚动体故障"}
    dataset_key = "cwru"


# ==================== 模型加载（缓存） ====================


@st.cache_resource
def load_model_and_tokenizer(ds_key, n_labels):
    """
    加载模型和分词器，使用 Streamlit 缓存避免重复加载

    参数：
        ds_key: 数据集标识符
        n_labels: 类别数

    返回值：
        model: 加载了训练权重的 BertTextClassifier
        tokenizer: BertTokenizer
    """
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertTextClassifier(num_labels=n_labels)
    try:
        model = load_trained_model(model, dataset_name=ds_key)
    except FileNotFoundError:
        st.error(
            f"未找到 {ds_key} 的训练模型权重，请先运行训练：python main.py --mode train_{ds_key}"
        )
        st.stop()
    return model, tokenizer


# ==================== 主区域 ====================

# 文本输入
st.subheader("输入文本")

# 示例文本
if dataset_option == "SST-2":
    example_text = "This movie is wonderfully entertaining and deeply moving."
else:
    example_text = (
        "The vibration amplitude remains at a low and stable level. "
        "The frequency spectrum shows no abnormal peaks or harmonic components. "
        "The bearing operates under normal conditions with no signs of degradation."
    )

text_input = st.text_area(
    "请输入要分析的文本（或使用下方示例）",
    value="",
    height=100,
    placeholder="在此输入文本...",
)

if st.button("使用示例文本"):
    text_input = example_text
    st.rerun()

if not text_input:
    text_input = example_text

st.info(f"当前文本：{text_input[:200]}{'...' if len(text_input) > 200 else ''}")

# ==================== 运行分析 ====================

if st.button("运行分析", type="primary"):
    if not method_options:
        st.warning("请在侧边栏选择至少一种可解释性方法")
    else:
        # 加载模型
        model, tokenizer = load_model_and_tokenizer(dataset_key, num_labels)

        # 模型预测
        encoding = tokenizer(
            text_input,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
            return_token_type_ids=True,
        )
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)
        token_type_ids = encoding["token_type_ids"].to(DEVICE)

        model.eval()
        with torch.no_grad():
            logits, _ = model(input_ids, attention_mask, token_type_ids)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])

        # 展示预测结果
        st.markdown("---")
        st.subheader("预测结果")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "预测类别", label_names.get(predicted_class, str(predicted_class))
            )
        with col2:
            st.metric("置信度", f"{confidence:.4f}")

        # 展示各类别概率
        prob_df = pd.DataFrame(
            {
                "类别": [label_names.get(i, str(i)) for i in range(num_labels)],
                "概率": [f"{p:.4f}" for p in probs],
            }
        )
        st.table(prob_df)

        # ==================== 归因分析 ====================

        st.markdown("---")
        st.subheader("归因分析结果")

        # 根据选择的方法数量决定列布局
        if len(method_options) > 1:
            cols = st.columns(len(method_options))
        else:
            cols = [st.container()]

        for idx, method_name in enumerate(method_options):
            with cols[idx]:
                st.markdown(f"**{method_name}**")

                try:
                    if method_name == "注意力可视化":
                        from explainability.attention_viz import explain_attention

                        result = explain_attention(model, text_input, tokenizer)
                        tokens = result["tokens"]
                        # 使用最后一层的注意力分数
                        last_layer = str(max(int(k) for k in result["layer_attention"]))
                        scores = result["layer_attention"][last_layer]

                    elif method_name == "Grad-CAM":
                        from explainability.gradcam_bert import (
                            explain_sample as gradcam_explain,
                        )

                        result = gradcam_explain(model, text_input, tokenizer)
                        tokens = result["tokens"]
                        scores = result["token_gradcam_scores"]

                    elif method_name == "Integrated Gradients":
                        from explainability.integrated_gradients import explain_sample

                        result = explain_sample(model, text_input, tokenizer=tokenizer)
                        tokens = result["tokens"]
                        scores = result["attribution_scores"]

                    elif method_name == "SHAP":
                        from explainability.shap_explainer_optimized import (
                            explain_sample_fast,
                        )

                        start_time = time.time()
                        result = explain_sample_fast(model, text_input, tokenizer)
                        elapsed_time = time.time() - start_time

                        if result:
                            tokens = result["tokens"]
                            scores = result["attribution_scores"]
                            # 显示计算时间
                            st.caption(f"⏱️ 计算时间: {elapsed_time:.2f}s")
                        else:
                            st.warning("SHAP 分析失败")
                            continue

                    elif method_name == "LIME":
                        from explainability.lime_explainer import (
                            explain_sample as lime_explain,
                        )

                        result = lime_explain(
                            model,
                            text_input,
                            num_labels=num_labels,
                            tokenizer=tokenizer,
                        )
                        tokens = result["tokens"]
                        scores = result["attribution_scores"]

                    else:
                        st.warning(f"未知方法: {method_name}")
                        continue

                    # 构建归因结果表格
                    if tokens and scores:
                        from explainability.attribution_unified import (
                            unify_attribution_result,
                            AttributionNormalizer,
                        )

                        # 统一处理归因结果
                        method_key = method_name.lower().replace("可视化", "").strip()
                        if method_key == "注意力":
                            method_key = "attention"
                        elif method_key == "grad-cam":
                            method_key = "gradcam"
                        elif method_key == "integrated gradients":
                            method_key = "ig"

                        unified_result = unify_attribution_result(
                            tokens=tokens,
                            method=method_key,
                            scores=scores,
                            normalize_method="minmax",
                        )

                        # 创建 DataFrame
                        df = pd.DataFrame(
                            {
                                "Token": unified_result.tokens[
                                    : len(unified_result.normalized_scores)
                                ],
                                "原始分数": [
                                    f"{s:.6f}"
                                    for s in unified_result.raw_scores[
                                        : len(unified_result.normalized_scores)
                                    ]
                                ],
                                "归一化分数": [
                                    f"{s:.6f}"
                                    for s in unified_result.normalized_scores[
                                        : len(unified_result.normalized_scores)
                                    ]
                                ],
                                "数值": unified_result.normalized_scores[
                                    : len(unified_result.normalized_scores)
                                ],
                            }
                        )

                        # 显示统计信息
                        st.markdown("**分数统计信息：**")
                        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                        with col_stats1:
                            st.metric(
                                "均值",
                                f"{unified_result.statistics['mean']:.4f}",
                            )
                        with col_stats2:
                            st.metric(
                                "标准差",
                                f"{unified_result.statistics['std']:.4f}",
                            )
                        with col_stats3:
                            st.metric(
                                "最小值",
                                f"{unified_result.statistics['min']:.4f}",
                            )
                        with col_stats4:
                            st.metric(
                                "最大值",
                                f"{unified_result.statistics['max']:.4f}",
                            )

                        # 显示热力图（可选）
                        if show_heatmap:
                            st.markdown("**Token 重要性热力图（统一尺度 [0, 1]）：**")
                            # 创建 Plotly 条形图，使用统一的颜色方案
                            fig = go.Figure()
                            fig.add_trace(
                                go.Bar(
                                    x=df["Token"],
                                    y=df["数值"],
                                    marker=dict(
                                        color=df["数值"],
                                        colorscale="RdYlGn",  # 统一颜色方案：红-黄-绿
                                        colorbar=dict(
                                            title="重要性<br>[0-1]",
                                            thickness=20,
                                            len=0.7,
                                        ),
                                        cmin=0,
                                        cmax=1,
                                    ),
                                    text=[f"{x:.4f}" for x in df["数值"]],
                                    textposition="outside",
                                    hovertemplate="<b>Token:</b> %{x}<br><b>重要性分数:</b> %{y:.6f}<extra></extra>",
                                )
                            )
                            fig.update_layout(
                                title=f"{method_name} - Token 重要性排序（归一化）",
                                xaxis_title="Token",
                                yaxis_title="重要性分数 [0-1]",
                                height=400,
                                showlegend=False,
                                yaxis=dict(range=[0, 1]),
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        # 显示 Top-10 token
                        if show_bar_chart:
                            st.markdown("**Top-10 重要 Token：**")
                            df_sorted = (
                                df.sort_values(by="数值", ascending=False)
                                .head(10)
                                .reset_index(drop=True)
                            )
                            df_sorted.index = df_sorted.index + 1

                            fig_top10 = go.Figure()
                            fig_top10.add_trace(
                                go.Bar(
                                    y=df_sorted["Token"],
                                    x=df_sorted["数值"],
                                    orientation="h",
                                    marker=dict(
                                        color=df_sorted["数值"],
                                        colorscale="RdYlGn",  # 统一颜色方案
                                        colorbar=dict(
                                            title="重要性<br>[0-1]",
                                            thickness=15,
                                            len=0.7,
                                        ),
                                        cmin=0,
                                        cmax=1,
                                    ),
                                    text=[f"{x:.4f}" for x in df_sorted["数值"]],
                                    textposition="outside",
                                    hovertemplate="<b>Token:</b> %{y}<br><b>重要性分数:</b> %{x:.6f}<extra></extra>",
                                )
                            )
                            fig_top10.update_layout(
                                title="",
                                xaxis_title="重要性分数 [0-1]",
                                yaxis_title="Token",
                                height=400,
                                showlegend=False,
                                margin=dict(l=150),
                                xaxis=dict(range=[0, 1]),
                            )
                            st.plotly_chart(fig_top10, use_container_width=True)

                        # 显示详细表格
                        st.markdown("**详细数据表：**")
                        df_display = df[
                            ["Token", "原始分数", "归一化分数"]
                        ].sort_values(
                            by="归一化分数",
                            key=lambda x: x.astype(float),
                            ascending=False,
                        )
                        st.dataframe(df_display, use_container_width=True, height=300)
                    else:
                        st.warning("未获取到有效的归因结果")

                except Exception as e:
                    st.error(f"{method_name} 分析出错: {str(e)}")

        # ==================== 多方法对比 ====================
        if len(method_options) > 1:
            st.markdown("---")
            st.subheader("📊 多方法对比分析")

            try:
                from explainability.attribution_unified import (
                    unify_multiple_methods,
                    AttributionComparator,
                    format_for_visualization,
                )

                # 收集所有方法的结果
                comparison_results = {}

                # 重新运行所有方法以获取结果（优化版本应该缓存）
                with st.spinner("正在对比各方法结果..."):
                    for method_name in method_options:
                        if method_name == "注意力可视化":
                            from explainability.attention_viz import explain_attention

                            result = explain_attention(model, text_input, tokenizer)
                            tokens = result["tokens"]
                            last_layer = str(
                                max(int(k) for k in result["layer_attention"])
                            )
                            scores = result["layer_attention"][last_layer]

                        elif method_name == "Grad-CAM":
                            from explainability.gradcam_bert import (
                                explain_sample as gradcam_explain,
                            )

                            result = gradcam_explain(model, text_input, tokenizer)
                            tokens = result["tokens"]
                            scores = result["token_gradcam_scores"]

                        elif method_name == "Integrated Gradients":
                            from explainability.integrated_gradients import (
                                explain_sample,
                            )

                            result = explain_sample(
                                model, text_input, tokenizer=tokenizer
                            )
                            tokens = result["tokens"]
                            scores = result["attribution_scores"]

                        elif method_name == "SHAP":
                            from explainability.shap_explainer_optimized import (
                                explain_sample_fast,
                            )

                            result = explain_sample_fast(model, text_input, tokenizer)
                            tokens = result["tokens"]
                            scores = result["attribution_scores"]

                        elif method_name == "LIME":
                            from explainability.lime_explainer import (
                                explain_sample as lime_explain,
                            )

                            result = lime_explain(
                                model,
                                text_input,
                                num_labels=num_labels,
                                tokenizer=tokenizer,
                            )
                            tokens = result["tokens"]
                            scores = result["attribution_scores"]

                        method_key = method_name.lower().replace("可视化", "").strip()
                        if method_key == "注意力":
                            method_key = "attention"
                        elif method_key == "grad-cam":
                            method_key = "gradcam"
                        elif method_key == "integrated gradients":
                            method_key = "ig"

                        comparison_results[method_key] = {
                            "tokens": tokens,
                            "scores": scores,
                        }

                # 统一处理所有结果
                unified_results = unify_multiple_methods(
                    comparison_results, normalize_method="minmax"
                )

                # 创建对比热力图
                st.markdown("**统一尺度对比热力图 [0-1]：**")

                # 提取数据用于热力图
                all_tokens = list(unified_results.values())[0].tokens
                heatmap_data = []
                method_names_short = []

                for method, result in unified_results.items():
                    heatmap_data.append(result.normalized_scores[: len(all_tokens)])
                    method_names_short.append(result.method.upper())

                # 创建热力图
                heatmap_data_array = np.array(heatmap_data)

                fig_heatmap = go.Figure(
                    data=go.Heatmap(
                        z=heatmap_data_array,
                        x=all_tokens[: heatmap_data_array.shape[1]],
                        y=method_names_short,
                        colorscale="RdYlGn",
                        colorbar=dict(title="重要性<br>[0-1]"),
                        hovertemplate="<b>方法:</b> %{y}<br><b>Token:</b> %{x}<br><b>分数:</b> %{z:.4f}<extra></extra>",
                    )
                )
                fig_heatmap.update_layout(
                    title="各方法的 Token 重要性对比（归一化）",
                    xaxis_title="Token",
                    yaxis_title="方法",
                    height=300,
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

                # 共识分析
                st.markdown("**共识分析（多方法达成一致的重要 Token）：**")

                comparator = AttributionComparator()
                consensus_tokens = comparator.get_consensus_important_tokens(
                    unified_results, top_k=10, consensus_threshold=0.5
                )

                if consensus_tokens:
                    consensus_df = pd.DataFrame(
                        [
                            {
                                "Token": item["token"],
                                "共识强度": f"{item['consensus_strength']:.2%}",
                                "平均分数": f"{item['average_score']:.4f}",
                                "位置": int(item["average_position"]),
                            }
                            for item in consensus_tokens[:10]
                        ]
                    )
                    st.dataframe(consensus_df, use_container_width=True)
                else:
                    st.info("未找到达成共识的重要 Token")

                # 方法相关性分析
                try:
                    st.markdown("**方法相关性分析（Spearman 相关系数）：**")
                    correlations = comparator.compute_method_correlation(
                        unified_results
                    )

                    corr_data = []
                    for (method1, method2), corr_value in correlations.items():
                        corr_data.append(
                            {
                                "方法对": f"{method1.upper()} vs {method2.upper()}",
                                "相关系数": f"{corr_value:.4f}",
                            }
                        )

                    corr_df = pd.DataFrame(corr_data)
                    st.dataframe(corr_df, use_container_width=True)
                except Exception as e:
                    st.warning(f"相关性分析失败: {str(e)}")

            except Exception as e:
                st.error(f"多方法对比出错: {str(e)}")

st.markdown("---")
st.caption("CCF-SHAP - 基于可解释性方法的 Transformer 模型可视化研究 | 作者：Kris")

# 页脚信息
col1, col2, col3 = st.columns(3)
with col1:
    st.info(f"**设备**: {DEVICE}")
with col2:
    st.info(f"**模型**: {MODEL_NAME}")
with col3:
    st.info(f"**最大序列长度**: {MAX_SEQ_LEN}")
