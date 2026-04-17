"""
CCF-SHAP 项目一键启动入口

模块功能：
    通过命令行参数控制运行模式，支持模型训练、可解释性分析、
    评估指标计算和 Streamlit 界面启动。

运行模式说明：
    - train_sst2：在 SST-2 数据集上训练 BERT 文本分类模型并评估，
                   保存分类性能指标至 Results/sst2_classification.json
    - train_cwru：在 CWRU 文本数据集上训练 BERT 文本分类模型并评估，
                   保存分类性能指标至 Results/cwru_classification.json
    - explain_sst2：对 SST-2 测试集运行注意力可视化、Integrated Gradients、
                     SHAP、LIME 四种可解释性方法，保存归因结果
    - explain_cwru：对 CWRU 文本测试集运行四种可解释性方法，保存归因结果
    - evaluate_explainability：运行忠实度与敏感度评估，保存评估指标
    - app：启动 Streamlit 交互式可视化分析界面
    - all：依次执行上述所有步骤

调用关系：
    main.py
    ├── train_sst2 / train_cwru
    │   ├── data/dataset_loader.py 或 data/cwru_text_dataset.py（数据加载）
    │   ├── models/bert_classifier.py（模型构建）
    │   ├── train/trainer.py（训练循环）
    │   ├── train/evaluator.py（性能评估）
    │   └── results/result_saver.py（结果保存）
    ├── explain_sst2 / explain_cwru
    │   ├── explainability/attention_viz.py
    │   ├── explainability/integrated_gradients.py
    │   ├── explainability/shap_explainer.py
    │   ├── explainability/lime_explainer.py
    │   └── results/result_saver.py
    ├── evaluate_explainability
    │   ├── evaluation/faithfulness.py
    │   ├── evaluation/sensitivity.py
    │   └── results/result_saver.py
    └── app
        └── app/streamlit_app.py（通过 subprocess 启动）

作者：Kris
"""

import argparse
import os
import sys
import random
import subprocess

import numpy as np
import torch
from tqdm import tqdm

# 将项目根目录加入路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import (
    SEED,
    DEVICE,
    SST2_NUM_LABELS,
    CWRU_NUM_LABELS,
    EXPLAIN_SAMPLE_SIZE,
    SHAP_SAMPLE_SIZE,
    MASKING_RATIOS,
)
from models.bert_classifier import BertTextClassifier
from results.result_saver import save_result


def set_seed(seed=SEED):
    """设置全局随机种子以保证实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==================== 训练阶段 ====================


def run_train_sst2():
    """
    在 SST-2 数据集上训练 BERT 文本分类模型

    流程：
        1. 加载 SST-2 数据集并创建 DataLoader
        2. 构建 BertTextClassifier（二分类）
        3. 执行训练循环
        4. 在测试集上评估
        5. 保存分类结果至 Results/sst2_classification.json
    """
    print("=" * 60)
    print("阶段：SST-2 模型训练与评估")
    print("=" * 60)

    from data.dataset_loader import get_sst2_dataloaders
    from train.trainer import train_model
    from train.evaluator import evaluate_model

    # 加载数据
    train_loader, val_loader, test_loader = get_sst2_dataloaders()
    print(
        f"数据集加载完成 - 训练批次: {len(train_loader)}, "
        f"验证批次: {len(val_loader)}, 测试批次: {len(test_loader)}"
    )

    # 构建模型
    model = BertTextClassifier(num_labels=SST2_NUM_LABELS)
    print(f"模型构建完成 - 类别数: {SST2_NUM_LABELS}, 设备: {DEVICE}")

    # 训练
    model, _ = train_model(model, train_loader, val_loader, dataset_name="sst2")

    # 在测试集上评估
    test_metrics = evaluate_model(model, test_loader)
    print(f"\n测试集评估结果:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")

    # 保存结果
    result_data = {
        "experiment_name": "classification",
        "dataset": "sst2",
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "per_class_f1": test_metrics["per_class_f1"],
        "predictions": test_metrics["predictions"],
        "true_labels": test_metrics["true_labels"],
    }
    save_result(result_data, "sst2_classification.json")


def run_train_cwru():
    """
    在 CWRU 文本数据集上训练 BERT 文本分类模型

    流程：
        1. 自动生成 CWRU 文本数据集（若不存在）
        2. 加载数据集并创建 DataLoader
        3. 构建 BertTextClassifier（四分类）
        4. 执行训练循环
        5. 保存分类结果至 Results/cwru_classification.json
    """
    print("=" * 60)
    print("阶段：CWRU 文本模型训练与评估")
    print("=" * 60)

    from data.cwru_text_dataset import get_cwru_dataloaders
    from train.trainer import train_model
    from train.evaluator import evaluate_model

    # 加载数据（会自动检查并生成 CSV 文件）
    train_loader, val_loader, test_loader = get_cwru_dataloaders()
    print(
        f"数据集加载完成 - 训练批次: {len(train_loader)}, "
        f"验证批次: {len(val_loader)}, 测试批次: {len(test_loader)}"
    )

    # 构建模型
    model = BertTextClassifier(num_labels=CWRU_NUM_LABELS)
    print(f"模型构建完成 - 类别数: {CWRU_NUM_LABELS}, 设备: {DEVICE}")

    # 训练
    model, _ = train_model(model, train_loader, val_loader, dataset_name="cwru")

    # 在测试集上评估
    test_metrics = evaluate_model(model, test_loader)
    print(f"\n测试集评估结果:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")

    # 保存结果
    result_data = {
        "experiment_name": "classification",
        "dataset": "cwru",
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "per_class_f1": test_metrics["per_class_f1"],
        "predictions": test_metrics["predictions"],
        "true_labels": test_metrics["true_labels"],
    }
    save_result(result_data, "cwru_classification.json")


# ==================== 可解释性分析阶段 ====================


def run_explain(dataset_name):
    """
    对指定数据集运行全部可解释性方法

    参数：
        dataset_name: "sst2" 或 "cwru"

    流程：
        1. 加载训练好的模型
        2. 获取测试集样本
        3. 依次运行注意力可视化、Grad-CAM、IG、SHAP、LIME
        4. 保存各方法的归因结果至 Results 目录
    """
    print("=" * 60)
    print(f"阶段：{dataset_name.upper()} 可解释性分析")
    print("=" * 60)

    from transformers import BertTokenizer
    from train.trainer import load_trained_model
    from explainability.attention_viz import explain_attention
    from explainability.gradcam_bert import explain_sample as gradcam_explain
    from explainability.integrated_gradients import explain_sample as ig_explain
    from explainability.shap_explainer import explain_batch as shap_explain
    from explainability.lime_explainer import explain_sample as lime_explain

    # 确定类别数
    num_labels = SST2_NUM_LABELS if dataset_name == "sst2" else CWRU_NUM_LABELS

    # 加载模型
    model = BertTextClassifier(num_labels=num_labels)
    model = load_trained_model(model, dataset_name=dataset_name)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # 获取测试集文本和标签
    if dataset_name == "sst2":
        from data.dataset_loader import get_sst2_raw_texts_and_labels

        texts, labels = get_sst2_raw_texts_and_labels(split="validation")
    else:
        from data.cwru_text_dataset import get_cwru_raw_texts_and_labels

        texts, labels = get_cwru_raw_texts_and_labels(split="test")

    # 采样用于分析的样本
    sample_size = min(EXPLAIN_SAMPLE_SIZE, len(texts))
    indices = random.sample(range(len(texts)), sample_size)
    sampled_texts = [texts[i] for i in indices]
    sampled_labels = [labels[i] for i in indices]

    print(f"采样 {sample_size} 条样本进行可解释性分析")

    # --- 注意力可视化 ---
    print("\n[1/5] 运行注意力可视化...")
    attention_samples = []
    for i, text in enumerate(tqdm(sampled_texts, desc="注意力分析")):
        result = explain_attention(model, text, tokenizer)
        attention_samples.append(result)

    save_result(
        {
            "experiment_name": "attention_viz",
            "dataset": dataset_name,
            "samples": attention_samples,
        },
        f"attention_{dataset_name}.json",
    )

    # --- Grad-CAM ---
    print("\n[2/5] 运行 Grad-CAM...")
    gradcam_samples = []
    for i, (text, label) in enumerate(
        tqdm(
            zip(sampled_texts, sampled_labels),
            total=len(sampled_texts),
            desc="Grad-CAM 分析",
        )
    ):
        result = gradcam_explain(model, text, tokenizer)
        result["true_label"] = label
        gradcam_samples.append(result)

    save_result(
        {
            "experiment_name": "gradcam",
            "dataset": dataset_name,
            "samples": gradcam_samples,
        },
        f"gradcam_{dataset_name}.json",
    )

    # --- Integrated Gradients ---
    print("\n[3/5] 运行 Integrated Gradients...")
    ig_samples = []
    for i, (text, label) in enumerate(
        tqdm(
            zip(sampled_texts, sampled_labels), total=len(sampled_texts), desc="IG 分析"
        )
    ):
        result = ig_explain(model, text, tokenizer=tokenizer)
        result["true_label"] = label
        ig_samples.append(result)

    save_result(
        {
            "experiment_name": "integrated_gradients",
            "dataset": dataset_name,
            "samples": ig_samples,
        },
        f"ig_{dataset_name}.json",
    )

    # --- SHAP ---
    print("\n[4/5] 运行 SHAP...")
    shap_sample_size = min(SHAP_SAMPLE_SIZE, len(sampled_texts))
    shap_texts = sampled_texts[:shap_sample_size]
    shap_labels = sampled_labels[:shap_sample_size]

    shap_results = shap_explain(model, shap_texts, tokenizer)
    for i, result in enumerate(shap_results):
        if i < len(shap_labels):
            result["true_label"] = shap_labels[i]

    save_result(
        {
            "experiment_name": "shap",
            "dataset": dataset_name,
            "samples": shap_results,
        },
        f"shap_{dataset_name}.json",
    )

    # --- LIME ---
    print("\n[5/5] 运行 LIME...")
    lime_samples = []
    for i, (text, label) in enumerate(
        tqdm(
            zip(sampled_texts, sampled_labels),
            total=len(sampled_texts),
            desc="LIME 分析",
        )
    ):
        result = lime_explain(model, text, num_labels=num_labels, tokenizer=tokenizer)
        result["true_label"] = label
        lime_samples.append(result)

    save_result(
        {
            "experiment_name": "lime",
            "dataset": dataset_name,
            "samples": lime_samples,
        },
        f"lime_{dataset_name}.json",
    )

    print(f"\n{dataset_name.upper()} 可解释性分析完成")


def run_explain_sst2():
    """对 SST-2 测试集运行全部可解释性方法"""
    run_explain("sst2")


def run_explain_cwru():
    """对 CWRU 文本测试集运行全部可解释性方法"""
    run_explain("cwru")


# ==================== 可解释性评估阶段 ====================


def run_evaluate_explainability():
    """
    运行忠实度与敏感度评估

    流程：
        1. 加载两个数据集的训练好的模型和归因结果
        2. 对注意力、IG、SHAP、LIME 计算忠实度（Comprehensiveness）
        3. 对注意力、IG、LIME 计算敏感度（余弦相似度）
        4. 保存评估结果
    """
    print("=" * 60)
    print("阶段：可解释性评估（忠实度与敏感度）")
    print("=" * 60)

    from transformers import BertTokenizer
    from train.trainer import load_trained_model
    from evaluation.faithfulness import evaluate_faithfulness
    from evaluation.sensitivity import evaluate_sensitivity
    from results.result_saver import load_result

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    faithfulness_results = {}
    faithfulness_curves = {}
    sensitivity_results = {}
    all_per_sample_scores = {}

    for dataset_name in ["sst2", "cwru"]:
        num_labels = SST2_NUM_LABELS if dataset_name == "sst2" else CWRU_NUM_LABELS

        # 加载模型
        model = BertTextClassifier(num_labels=num_labels)
        try:
            model = load_trained_model(model, dataset_name=dataset_name)
        except FileNotFoundError:
            print(f"跳过 {dataset_name}：未找到训练好的模型")
            continue

        # 加载各方法的归因结果
        methods_for_faithfulness = {
            "attention": f"attention_{dataset_name}.json",
            "gradcam": f"gradcam_{dataset_name}.json",
            "integrated_gradients": f"ig_{dataset_name}.json",
            "shap": f"shap_{dataset_name}.json",
            "lime": f"lime_{dataset_name}.json",
        }

        # --- 忠实度评估 ---
        print(f"\n评估 {dataset_name} 的忠实度...")
        for method_name, result_file in methods_for_faithfulness.items():
            try:
                result_data = load_result(result_file)
                samples = result_data.get("samples", [])

                # 提取文本和归因分数
                texts = [s["text"] for s in samples]
                attr_results = samples

                # 注意力方法使用最后一层的注意力分数作为归因
                if method_name == "attention":
                    for s in attr_results:
                        if "layer_attention" in s:
                            last_layer = str(max(int(k) for k in s["layer_attention"]))
                            # 排除 [CLS] 和 [SEP] 的注意力分数
                            attn = s["layer_attention"][last_layer]
                            s["attribution_scores"] = (
                                attn[1:-1] if len(attn) > 2 else attn
                            )

                # Grad-CAM 方法使用 token_gradcam_scores 作为归因
                elif method_name == "gradcam":
                    for s in attr_results:
                        if "token_gradcam_scores" in s:
                            s["attribution_scores"] = s["token_gradcam_scores"]

                faith_result = evaluate_faithfulness(
                    model,
                    texts[:20],
                    attr_results[:20],
                    method_name,
                    dataset_name,
                    tokenizer,
                )

                if method_name not in faithfulness_results:
                    faithfulness_results[method_name] = {}
                faithfulness_results[method_name][dataset_name] = faith_result[
                    "mean_auc_drop"
                ]

                curve_key = f"{method_name}_{dataset_name}"
                faithfulness_curves[curve_key] = faith_result["auc_drop_curves"]

                print(
                    f"  {method_name} - AUC-Drop: {faith_result['mean_auc_drop']:.4f}"
                )

            except FileNotFoundError:
                print(f"  {method_name} - 归因结果文件不存在，跳过")
            except Exception as e:
                print(f"  {method_name} - 评估出错: {e}")

        # --- 敏感度评估 ---
        print(f"\n评估 {dataset_name} 的敏感度...")
        methods_for_sensitivity = ["attention", "integrated_gradients", "lime"]

        for method_name in methods_for_sensitivity:
            try:
                result_file = methods_for_faithfulness[method_name]
                result_data = load_result(result_file)
                samples = result_data.get("samples", [])
                texts = [s["text"] for s in samples[:10]]

                # 构造归因函数
                if method_name == "attention":
                    from explainability.attention_viz import explain_attention

                    def explain_func(text, _m=model, _t=tokenizer):
                        r = explain_attention(_m, text, _t)
                        last_layer = str(max(int(k) for k in r["layer_attention"]))
                        attn = r["layer_attention"][last_layer]
                        return {
                            "attribution_scores": attn[1:-1] if len(attn) > 2 else attn
                        }

                elif method_name == "integrated_gradients":
                    from explainability.integrated_gradients import (
                        explain_sample as ig_explain,
                    )

                    def explain_func(text, _m=model, _t=tokenizer):
                        return ig_explain(_m, text, tokenizer=_t)

                elif method_name == "lime":
                    from explainability.lime_explainer import (
                        explain_sample as lime_explain,
                    )

                    def explain_func(text, _m=model, _t=tokenizer, _nl=num_labels):
                        return lime_explain(_m, text, num_labels=_nl, tokenizer=_t)

                sens_result = evaluate_sensitivity(
                    explain_func, texts, method_name, dataset_name
                )

                if method_name not in sensitivity_results:
                    sensitivity_results[method_name] = {}
                sensitivity_results[method_name][dataset_name] = sens_result[
                    "mean_sensitivity"
                ]

                score_key = f"{method_name}_{dataset_name}"
                all_per_sample_scores[score_key] = sens_result["per_sample_scores"]

                print(
                    f"  {method_name} - 平均余弦相似度: {sens_result['mean_sensitivity']:.4f}"
                )

            except FileNotFoundError:
                print(f"  {method_name} - 归因结果文件不存在，跳过")
            except Exception as e:
                print(f"  {method_name} - 评估出错: {e}")

    # 保存忠实度结果
    save_result(
        {
            "experiment_name": "faithfulness_evaluation",
            "results": faithfulness_results,
            "masking_ratios": MASKING_RATIOS,
            "auc_drop_curves": faithfulness_curves,
        },
        "faithfulness_results.json",
    )

    # 保存敏感度结果
    save_result(
        {
            "experiment_name": "sensitivity_evaluation",
            "results": sensitivity_results,
            "per_sample_scores": all_per_sample_scores,
        },
        "sensitivity_results.json",
    )

    print("\n可解释性评估完成")


# ==================== Streamlit 界面启动 ====================


def run_app():
    """
    启动 Streamlit 交互式可视化分析界面

    通过 subprocess 调用 streamlit run 命令启动 Web 服务。
    """
    print("=" * 60)
    print("启动 Streamlit 可视化界面")
    print("=" * 60)

    app_path = os.path.join(PROJECT_ROOT, "app", "streamlit_app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])


# ==================== 主入口 ====================


def main():
    """
    命令行参数解析与模式分发

    支持的运行模式：
        train_sst2, train_cwru, explain_sst2, explain_cwru,
        evaluate_explainability, app, all
    """
    parser = argparse.ArgumentParser(
        description="CCF-SHAP: 基于可解释性方法的 Transformer 模型可视化研究"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=[
            "train_sst2",
            "train_cwru",
            "explain_sst2",
            "explain_cwru",
            "evaluate_explainability",
            "app",
            "all",
        ],
        help="运行模式选择（默认: all）",
    )
    args = parser.parse_args()

    # 设置随机种子
    set_seed()

    print(f"CCF-SHAP 项目 - 运行模式: {args.mode}")
    print(f"计算设备: {DEVICE}")
    print()

    if args.mode == "train_sst2":
        run_train_sst2()

    elif args.mode == "train_cwru":
        run_train_cwru()

    elif args.mode == "explain_sst2":
        run_explain_sst2()

    elif args.mode == "explain_cwru":
        run_explain_cwru()

    elif args.mode == "evaluate_explainability":
        run_evaluate_explainability()

    elif args.mode == "app":
        run_app()

    elif args.mode == "all":
        # 依次执行所有步骤
        run_train_sst2()
        run_train_cwru()
        run_explain_sst2()
        run_explain_cwru()
        run_evaluate_explainability()
        print("\n所有实验步骤已完成。")
        print("如需启动可视化界面，请运行: python main.py --mode app")


if __name__ == "__main__":
    main()
