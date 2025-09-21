# comprehensive_evaluation.py

import os
import time
import torch
import json
import numpy as np
import torch.nn as nn
import inspect
from sklearn.metrics import classification_report
from openprompt.utils.logging import logger
from openprompt.plms import load_plm
from openprompt.prompts import PtuningTemplate, ManualVerbalizer
from openprompt.prompt_base import Template
from openprompt import PromptDataLoader
from openprompt.data_utils.text_classification_dataset import CnClickbaitProcessor
from tqdm import tqdm

# ==================================================================================================
#                                 1. 所有配置参数区域
# ==================================================================================================
# --- 在这里修改所有评估参数 ---

# 1.1 项目与路径配置
MODEL_TYPE = "bert"
MODEL_BASE_PATH = "model/chinese-roberta-wwm-ext"
DATASET_NAME = "toutiao"
MODEL_CHECKPOINT_NAME = "20250911-153849-toutiao622-Acc0.9677-F1s0.9357.ckpt" # 要评估的.ckpt模型

# 1.2 Prompt 相关配置
TEMPLATE_ID = 0
VERBALIZER_FILE_NAME = "manual_verbalizer.txt"
MAX_SEQ_LENGTH = 128

# 1.3 性能基准测试配置
DEVICE = 'cpu'              # 统一使用 CPU 进行评估
BATCH_SIZE = 32
WARMUP_STEPS = 10           # 预热批次数（不计入统计）
MEASURE_STEPS = 100         # 性能统计批次数（可以小于总批次数，以节省时间）

# ==================================================================================================
#                                 2. PromptBasedModel 定义
# ==================================================================================================
# ==================================================================================================
#                                 2. PromptBasedModel 定义 (最终修正版)
# ==================================================================================================
class PromptBasedModel(nn.Module):
    def __init__(self, plm, template: Template, verbalizer: ManualVerbalizer):
        super().__init__()
        self.plm = plm
        self.template = template
        self.verbalizer = verbalizer
        self.plm_forward_keys = inspect.signature(self.plm.forward).parameters.keys()
        self.mask_idx_in_template = -1
        
        # ####################################################################################
        # ### 最终修正：PtuningTemplate 在创建时已自动解析，直接使用 .text 属性即可 ###
        # ####################################################################################
        for i, token_info in enumerate(self.template.text): 
            if 'mask' in token_info:
                self.mask_idx_in_template = i
                break
        
        if self.mask_idx_in_template == -1:
            raise ValueError("错误：模板中未找到 {'mask': ...} 标记。")

    def forward(self, batch):
        batch = self.template.process_batch(batch)
        inputs = {key: batch[key] for key in batch if key in self.plm_forward_keys}
        outputs = self.plm(**inputs)
        raw_logits = outputs.logits
        all_special_token_ids = batch['loss_ids']
        mask_indices = all_special_token_ids[:, self.mask_idx_in_template]
        batch_size = raw_logits.size(0)
        batch_indices = torch.arange(batch_size, device=raw_logits.device)
        masked_logits = raw_logits[batch_indices, mask_indices]
        label_words_logits = self.verbalizer.process_outputs(masked_logits, batch=batch)
        return label_words_logits


# ==================================================================================================
#                                 3. 性能统计辅助函数
# ==================================================================================================
def summarize_latency(name, vals, bs):
    if not vals: return f"{name}: n/a"
    arr = np.array(vals)
    per_sample_arr = arr / bs
    return (
        f"{name:<15} | "
        f"Batch Avg: {arr.mean()*1000:>7.2f}ms | "
        f"Sample Avg: {per_sample_arr.mean()*1000:>7.2f}ms | "
        f"P50: {np.percentile(per_sample_arr, 50)*1000:>6.2f}ms | "
        f"P99: {np.percentile(per_sample_arr, 99)*1000:>6.2f}ms"
    )

# ==================================================================================================
#                                 4. 综合评估主逻辑
# ==================================================================================================
def main():
    print("=" * 30 + " 综合评估脚本 " + "=" * 30)
    print(f"配置: Device={DEVICE} | BS={BATCH_SIZE} | Warmup={WARMUP_STEPS} | Measure={MEASURE_STEPS}")

    # --- 4.1. 初始化模型和数据加载器 ---
    project_root = "."
    model_checkpoint_path = os.path.join(project_root, "ckpts", MODEL_CHECKPOINT_NAME)
    model_base_path = os.path.join(project_root, MODEL_BASE_PATH)
    scripts_base_dir = os.path.join(project_root, "scripts", f"TextClassification/{DATASET_NAME}")
    data_path = os.path.join(project_root, "datasets", f"TextClassification/{DATASET_NAME}/")

    plm, tokenizer, model_config, WrapperClass = load_plm(MODEL_TYPE, model_base_path)
    processor = CnClickbaitProcessor()
    class_labels = processor.get_labels()

    template_path = os.path.join(scripts_base_dir, "ptuning_template.txt")
    with open(template_path, 'r', encoding='utf-8') as f: text = f.readlines()[TEMPLATE_ID].rstrip()
    mytemplate = PtuningTemplate(model=plm, tokenizer=tokenizer, text=text)

    verbalizer_path = os.path.join(scripts_base_dir, VERBALIZER_FILE_NAME)
    with open(verbalizer_path, 'r', encoding='utf-8') as f: v_dict = json.load(f)
    myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels, label_words=v_dict['label_words'])

    model = PromptBasedModel(plm=plm, template=mytemplate, verbalizer=myverbalizer).to(DEVICE)
    if os.path.exists(model_checkpoint_path):
        # map_location=DEVICE 确保权重被加载到正确的设备上（此处为CPU）
        model.load_state_dict(torch.load(model_checkpoint_path, map_location=DEVICE), strict=False)
        print(f"✅ 成功加载模型权重: {model_checkpoint_path}")
    else:
        print(f"⚠️ 警告: 未找到模型权重文件 {model_checkpoint_path}, 将使用基础PLM进行评估。")
    model.eval()

    test_set = processor.get_test_examples(data_path)
    test_dataloader = PromptDataLoader(dataset=test_set, template=mytemplate, tokenizer=tokenizer,
                                       tokenizer_wrapper_class=WrapperClass, max_seq_length=MAX_SEQ_LENGTH,
                                       batch_size=BATCH_SIZE, shuffle=False, truncate_method="tail")
    print(f"✅ 数据和模型准备就绪。测试样本数: {len(test_set)}")

    # --- 4.2. 执行基准测试和评估 ---
    # 记录数组
    latencies_e2e, latencies_fwd = [], []
    all_preds, all_labels = [], []
    
    steps_done = 0
    total_samples_measured = 0
    
    with torch.no_grad():
        # 迭代次数最多为 warmup + measure，或数据加载器的大小
        pbar_total = min(len(test_dataloader), WARMUP_STEPS + MEASURE_STEPS)
        pbar = tqdm(test_dataloader, total=pbar_total, desc="正在评估")

        for step, inputs in enumerate(pbar):
            t0_e2e = time.perf_counter()
            inputs = inputs.to(DEVICE)
            
            # --- 模型前向传播 ---
            t0_fwd = time.perf_counter()
            
            logits = model(inputs)
            
            t1_fwd = time.perf_counter()
            
            # --- 结果处理 ---
            labels = inputs['label']
            preds = torch.argmax(logits, dim=-1)
            t1_e2e = time.perf_counter()

            # --- 统计数据 (预热后开始) ---
            if step >= WARMUP_STEPS:
                latencies_e2e.append(t1_e2e - t0_e2e)
                latencies_fwd.append(t1_fwd - t0_fwd)
                
                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
                
                total_samples_measured += labels.size(0)
                steps_done += 1

            if steps_done >= MEASURE_STEPS:
                break
    
    # --- 4.3. 打印报告 ---
    print("\n" + "="*80)
    print(" " * 32 + "综合评估报告")
    print("="*80)

    # 性能报告
    print("\n--- 性能指标 (Performance Metrics) ---")
    print(summarize_latency("End-to-End", latencies_e2e, BATCH_SIZE))
    print(summarize_latency("Model Forward", latencies_fwd, BATCH_SIZE))

    if steps_done > 0:
        total_time = sum(latencies_e2e)
        throughput = total_samples_measured / total_time if total_time > 0 else 0
        print(f"\n吞吐量 (Throughput): {throughput:.2f} samples/s (基于 {total_samples_measured} 个样本, {steps_done} 个批次)")

    # 精度报告
    print("\n--- 精度指标 (Accuracy Metrics) ---")
    if all_labels:
        print(classification_report(all_labels, all_preds, target_names=class_labels, digits=4, zero_division=0))
    else:
        print("未收集到精度数据 (可能是因为 MEASURE_STEPS 设置为 0)。")
    print("="*80)

if __name__ == "__main__":
    main()