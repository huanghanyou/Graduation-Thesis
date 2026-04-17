"""
五种可解释性方法初步验证脚本

运行本脚本以验证以下方法在 SST-2 和 CWRU 数据集上的正常运行：
  1. 注意力可视化（Attention Visualization）
  2. Integrated Gradients（IG）
  3. SHAP（KernelSHAP）
  4. LIME
  5. Grad-CAM for ViT（演示模块）

使用方式：
  python test_methods.py

输出：
  - 验证每个方法是否能成功加载模型、分词和执行推理
  - 打印每个方法的输出形状和数值示例
  - 生成 test_results.json 包含所有验证结果
"""

import sys
import os
import json
import torch
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_NAME,
    DEVICE,
    SST2_NUM_LABELS,
    CWRU_NUM_LABELS,
    MAX_SEQ_LEN,
    IG_N_STEPS,
    SHAP_SAMPLE_SIZE,
)
from models.bert_classifier import BertTextClassifier
from transformers import BertTokenizer
from train.trainer import load_trained_model

# 测试样本（两个数据集各一句）
TEST_SAMPLES = {
    "sst2": [
        "This movie is absolutely fantastic and amazing!",
        "I really did not like this film at all.",
    ],
    "cwru": [
        "normal bearing vibration signal with low noise",
        "inner race fault detected at operating frequency",
    ],
}


def test_attention_viz(model, texts, tokenizer, dataset_name):
    """测试注意力可视化"""
    print("\n" + "=" * 60)
    print("[1/5] 测试注意力可视化（Attention Visualization）")
    print("=" * 60)
    try:
        from explainability.attention_viz import explain_attention

        results = []
        for text in texts:
            result = explain_attention(model, text, tokenizer)
            results.append(result)
            print(f"✓ 文本: '{text[:50]}...'")
            print(f"  Tokens: {result['tokens'][:5]}...")
            print(f"  Scores shape: {len(result['layer_attention']['11'])}")

        return {
            "method": "attention_viz",
            "status": "success",
            "num_samples": len(results),
            "sample_output": results[0] if results else None,
        }
    except Exception as e:
        print(f"✗ 注意力可视化测试失败: {e}")
        return {"method": "attention_viz", "status": "failed", "error": str(e)}


def test_integrated_gradients(model, texts, tokenizer, dataset_name):
    """测试 Integrated Gradients"""
    print("\n" + "=" * 60)
    print("[2/5] 测试 Integrated Gradients (IG)")
    print("=" * 60)
    try:
        from explainability.integrated_gradients import explain_sample as ig_explain

        results = []
        for text in texts:
            result = ig_explain(model, text, tokenizer=tokenizer)
            results.append(result)
            print(f"✓ 文本: '{text[:50]}...'")
            print(f"  Tokens: {result['tokens'][:5]}...")
            print(f"  Attribution scores: {result['attribution_scores'][:5]}")

        return {
            "method": "integrated_gradients",
            "status": "success",
            "num_samples": len(results),
            "sample_output": results[0] if results else None,
        }
    except Exception as e:
        print(f"✗ Integrated Gradients 测试失败: {e}")
        return {"method": "integrated_gradients", "status": "failed", "error": str(e)}


def test_shap(model, texts, tokenizer, dataset_name):
    """测试 SHAP"""
    print("\n" + "=" * 60)
    print("[3/5] 测试 SHAP (KernelSHAP)")
    print("=" * 60)
    try:
        from explainability.shap_explainer import explain_batch

        # SHAP 处理整批文本
        results = explain_batch(
            model, texts[: min(SHAP_SAMPLE_SIZE, len(texts))], tokenizer
        )

        for result in results:
            print(f"✓ 文本: '{result['text'][:50]}...'")
            print(f"  Tokens: {result['tokens'][:5]}...")
            print(f"  Attribution scores (前5个): {result['attribution_scores'][:5]}")

        return {
            "method": "shap",
            "status": "success",
            "num_samples": len(results),
            "sample_output": results[0] if results else None,
        }
    except Exception as e:
        print(f"✗ SHAP 测试失败: {e}")
        return {"method": "shap", "status": "failed", "error": str(e)}


def test_lime(model, texts, tokenizer, num_labels):
    """测试 LIME"""
    print("\n" + "=" * 60)
    print("[4/5] 测试 LIME")
    print("=" * 60)
    try:
        from explainability.lime_explainer import explain_sample as lime_explain

        results = []
        for text in texts:
            result = lime_explain(
                model, text, num_labels=num_labels, tokenizer=tokenizer
            )
            results.append(result)
            print(f"✓ 文本: '{text[:50]}...'")
            print(f"  Tokens: {result['tokens'][:5]}...")
            print(f"  Attribution scores (前5个): {result['attribution_scores'][:5]}")

        return {
            "method": "lime",
            "status": "success",
            "num_samples": len(results),
            "sample_output": results[0] if results else None,
        }
    except Exception as e:
        print(f"✗ LIME 测试失败: {e}")
        return {"method": "lime", "status": "failed", "error": str(e)}


def test_gradcam_vit():
    """测试 Grad-CAM for ViT（演示模块）"""
    print("\n" + "=" * 60)
    print("[5/5] 测试 Grad-CAM for ViT（演示模块）")
    print("=" * 60)
    try:
        from explainability.gradcam_vit import demo_gradcam

        result = demo_gradcam()
        print(f"✓ Grad-CAM 演示成功")
        print(f"  预测类别: {result['predicted_class']}")
        print(f"  激活图形状: {result['cam_shape']}")

        return {"method": "gradcam_vit", "status": "success", "sample_output": result}
    except Exception as e:
        print(f"✗ Grad-CAM ViT 测试失败: {e}")
        return {"method": "gradcam_vit", "status": "failed", "error": str(e)}


def main():
    print("\n" + "=" * 60)
    print("CCF-SHAP 五种可解释性方法初步验证")
    print("=" * 60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {DEVICE}")

    test_results = {
        "timestamp": datetime.now().isoformat(),
        "device": str(DEVICE),
        "sst2": {},
        "cwru": {},
    }

    # ==================== SST-2 数据集测试 ====================
    print("\n\n" + "#" * 60)
    print("# SST-2 数据集（二分类）")
    print("#" * 60)

    try:
        model_sst2 = BertTextClassifier(num_labels=SST2_NUM_LABELS)
        model_sst2 = load_trained_model(model_sst2, dataset_name="sst2")
        model_sst2 = model_sst2.to(DEVICE)
        model_sst2.eval()
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

        texts_sst2 = TEST_SAMPLES["sst2"]

        test_results["sst2"]["attention"] = test_attention_viz(
            model_sst2, texts_sst2, tokenizer, "sst2"
        )
        test_results["sst2"]["integrated_gradients"] = test_integrated_gradients(
            model_sst2, texts_sst2, tokenizer, "sst2"
        )
        test_results["sst2"]["shap"] = test_shap(
            model_sst2, texts_sst2, tokenizer, "sst2"
        )
        test_results["sst2"]["lime"] = test_lime(
            model_sst2, texts_sst2, tokenizer, SST2_NUM_LABELS
        )

    except Exception as e:
        print(f"❌ SST-2 模型加载失败: {e}")
        test_results["sst2"]["error"] = str(e)

    # ==================== CWRU 数据集测试 ====================
    print("\n\n" + "#" * 60)
    print("# CWRU 数据集（四分类）")
    print("#" * 60)

    try:
        model_cwru = BertTextClassifier(num_labels=CWRU_NUM_LABELS)
        model_cwru = load_trained_model(model_cwru, dataset_name="cwru")
        model_cwru = model_cwru.to(DEVICE)
        model_cwru.eval()
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

        texts_cwru = TEST_SAMPLES["cwru"]

        test_results["cwru"]["attention"] = test_attention_viz(
            model_cwru, texts_cwru, tokenizer, "cwru"
        )
        test_results["cwru"]["integrated_gradients"] = test_integrated_gradients(
            model_cwru, texts_cwru, tokenizer, "cwru"
        )
        test_results["cwru"]["shap"] = test_shap(
            model_cwru, texts_cwru, tokenizer, "cwru"
        )
        test_results["cwru"]["lime"] = test_lime(
            model_cwru, texts_cwru, tokenizer, CWRU_NUM_LABELS
        )

    except Exception as e:
        print(f"❌ CWRU 模型加载失败: {e}")
        test_results["cwru"]["error"] = str(e)

    # ==================== Grad-CAM ViT 测试（独立） ====================
    test_results["gradcam_vit"] = test_gradcam_vit()

    # ==================== 汇总报告 ====================
    print("\n\n" + "=" * 60)
    print("验证汇总报告")
    print("=" * 60)

    total_tests = 0
    passed_tests = 0

    for dataset in ["sst2", "cwru"]:
        print(f"\n{dataset.upper()}:")
        if "error" in test_results[dataset]:
            print(f"  ✗ 数据集加载失败")
        else:
            for method, result in test_results[dataset].items():
                status = result.get("status", "unknown")
                if status == "success":
                    print(f"  ✓ {method}")
                    passed_tests += 1
                else:
                    print(f"  ✗ {method}")
                total_tests += 1

    print(f"\nGrad-CAM ViT: ", end="")
    if test_results["gradcam_vit"]["status"] == "success":
        print("✓")
        passed_tests += 1
    else:
        print("✗")
    total_tests += 1

    print(f"\n总计: {passed_tests}/{total_tests} 个方法验证通过")

    # 保存结果
    output_file = os.path.join(os.path.dirname(__file__), "test_results.json")
    # 清理不可序列化的内容
    for ds in ["sst2", "cwru"]:
        for method in list(test_results[ds].keys()):
            if (
                isinstance(test_results[ds][method], dict)
                and "sample_output" in test_results[ds][method]
            ):
                test_results[ds][method]["sample_output"] = str(
                    test_results[ds][method]["sample_output"]
                )[:200]

    if "sample_output" in test_results["gradcam_vit"]:
        test_results["gradcam_vit"]["sample_output"] = str(
            test_results["gradcam_vit"]["sample_output"]
        )[:200]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 验证结果已保存至: {output_file}")


if __name__ == "__main__":
    main()
