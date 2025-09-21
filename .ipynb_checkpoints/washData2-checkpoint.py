# comprehensive_evaluation_with_cleaning.py (robust_loader_version)

import os
import time
import torch
import json
import numpy as np
import torch.nn as nn
import inspect
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from openprompt.utils.logging import logger
from openprompt.plms import load_plm
from openprompt.prompts import PtuningTemplate, ManualVerbalizer
from openprompt.prompt_base import Template
from openprompt import PromptDataLoader
from openprompt.data_utils.text_classification_dataset import CnClickbaitProcessor
from openprompt.data_utils.utils import InputExample
from tqdm import tqdm
from collections import OrderedDict

# ==================================================================================================
#                                 1. 配置参数 (保持不变)
# ==================================================================================================
MODEL_TYPE = "bert"
MODEL_BASE_PATH = "model/chinese-roberta-wwm-ext"
DATASET_NAME = "toutiao"
MODEL_CHECKPOINT_NAME = "20250901-142404-toutiao622-Acc0.9312-F1s0.9213.ckpt"
TEMPLATE_ID = 0
VERBALIZER_FILE_NAME = "manual_verbalizer.txt"
MAX_SEQ_LENGTH = 128
DEVICE = 'cpu'
BATCH_SIZE = 32
TARGET_CLASS_INDEX = 2
CLEANED_TRAIN_CSV_NAME = "train_cleaned.csv"
CLEANED_TEST_CSV_NAME = "test_cleaned.csv"

# ==================================================================================================
#                                 2. PromptBasedModel 定义 (保持不变)
# ==================================================================================================
class PromptBasedModel(nn.Module):
    def __init__(self, plm, template: Template, verbalizer: ManualVerbalizer):
        super().__init__()
        self.plm = plm
        self.template = template
        self.verbalizer = verbalizer
        self.plm_forward_keys = inspect.signature(self.plm.forward).parameters.keys()
        self.mask_idx_in_template = -1
        try:
            parsed_template_text = self.template.parsed_text
        except AttributeError:
            parsed_template_text = self.template.text
        if not isinstance(parsed_template_text, list):
             raise RuntimeError("无法在 PtuningTemplate 对象中找到解析后的模板列表。")
        for i, token_info in enumerate(parsed_template_text): 
            if 'mask' in token_info: self.mask_idx_in_template = i; break
        if self.mask_idx_in_template == -1: raise ValueError("错误：模板中未找到 {'mask': ...} 标记。")

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
#                           3. 【新增】健壮的模型加载器 (核心修改)
# ==================================================================================================
def load_model_weights_robustly(model: PromptBasedModel, checkpoint_path: str, device: str):
    """
    一个健壮的权重加载器，能自动处理常见的前缀问题，并分部加载权重。
    """
    print("\n" + "="*80)
    print(" " * 25 + "启动健壮的权重加载与诊断程序")
    print("="*80)

    if not os.path.exists(checkpoint_path):
        print(f"❌ 错误: 权重文件未找到于 {checkpoint_path}")
        return False

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # --- 自动检测并剥离前缀 ---
    # 常见的错误前缀
    prefixes_to_try = ["prompt_model.", "model."] 
    
    # 检查是否存在任何一个前缀
    detected_prefix = ""
    for prefix in prefixes_to_try:
        if any(key.startswith(prefix) for key in checkpoint.keys()):
            detected_prefix = prefix
            break
    
    if detected_prefix:
        print(f"🔍 检测到并准备移除键名前缀: '{detected_prefix}'")
        cleaned_checkpoint = OrderedDict()
        for key, value in checkpoint.items():
            if key.startswith(detected_prefix):
                cleaned_checkpoint[key[len(detected_prefix):]] = value
            else:
                cleaned_checkpoint[key] = value
        checkpoint = cleaned_checkpoint
    else:
        print("ℹ️ 未检测到常见的多余键名前缀。")

    # --- 分部加载与诊断 ---
    def load_and_report(sub_module_name: str, sub_module: nn.Module):
        print(f"\n--- 正在尝试加载 '{sub_module_name}' 部分的权重 ---")
        
        # 提取只属于这个子模块的权重
        sub_module_checkpoint = OrderedDict()
        for key, value in checkpoint.items():
            if key.startswith(sub_module_name + "."):
                sub_module_checkpoint[key[len(sub_module_name)+1:]] = value
        
        if not sub_module_checkpoint:
            print(f"❌ [严重问题] 在CKPT文件中没有找到任何属于 '{sub_module_name}' 的权重！")
            return False

        print(f"  > 在CKPT中找到 {len(sub_module_checkpoint)} 个相关权重。")
        
        # 尝试加载
        missing_keys, unexpected_keys = sub_module.load_state_dict(sub_module_checkpoint, strict=False)
        
        # 报告
        if not missing_keys and not unexpected_keys:
            print("  > ✅ 完美匹配！所有相关权重已成功加载。")
            return True
        else:
            print(f"  > ⚠️  加载不完全匹配:")
            if missing_keys:
                print(f"    - 模型 '{sub_module_name}' 中有 {len(missing_keys)} 个权重在文件中缺失。示例: {missing_keys[:3]}")
            if unexpected_keys:
                print(f"    - 文件中有 {len(unexpected_keys)} 个权重在模型 '{sub_module_name}' 中不存在。示例: {unexpected_keys[:3]}")
            # 只要有任何权重被加载，我们就认为它部分成功
            if len(unexpected_keys) < len(sub_module_checkpoint):
                 print("  > ✅ 部分权重已成功加载。")
                 return True
            else:
                 print("  > ❌ 加载完全失败，所有键名都不匹配。")
                 return False

    # 依次加载 plm 和 template
    plm_loaded = load_and_report("plm", model.plm)
    template_loaded = load_and_report("template", model.template)
    
    print("="*80)
    
    if template_loaded:
        print("✅ [诊断结论] 关键的 'template' 权重已成功加载！模型应该可以正常工作。")
        return True
    else:
        print("❌ [诊断结论] 未能加载任何 'template' 权重。模型将使用随机初始化的prompt，预测结果将不可靠。")
        return False


# ==================================================================================================
#                                 4. 数据清洗主逻辑 (使用新加载器)
# ==================================================================================================
def clean_dataset():
    """使用FP32模型作为裁判，清洗数据集中被误判的第二类样本。"""
    print("=" * 30 + " 数据清洗脚本 " + "=" * 30)
    
    # --- 4.1. 初始化模型骨架 ---
    project_root = "."
    model_checkpoint_path = os.path.join(project_root, "ckpts", MODEL_CHECKPOINT_NAME)
    model_base_path = os.path.join(project_root, MODEL_BASE_PATH)
    scripts_base_dir = os.path.join(project_root, "scripts", f"TextClassification/{DATASET_NAME}")
    data_path = os.path.join(project_root, "datasets", "TextClassification", DATASET_NAME)

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
    
    # --- 4.2. 使用新的健壮加载器加载权重 ---
    success = load_model_weights_robustly(model, model_checkpoint_path, DEVICE)
    
    if not success:
        print("\n由于关键权重加载失败，清洗任务已中止。")
        return
        
    print(f"\n✅ 裁判模型初始化完成，准备开始清洗数据...")
    model.eval()

    # --- 4.3. 清洗流程 (与之前相同) ---
    def process_and_clean_split(split_name: str, examples: list[InputExample]):
        print(f"\n--- 正在处理 {split_name} 集 ---")
        dataloader = PromptDataLoader(dataset=examples, template=mytemplate, tokenizer=tokenizer,
                                      tokenizer_wrapper_class=WrapperClass, max_seq_length=MAX_SEQ_LENGTH,
                                      batch_size=BATCH_SIZE, shuffle=False)
        guids_to_remove, total_class2_samples, misclassified_count = set(), 0, 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"正在用裁判模型预测 {split_name} 集"):
                batch = batch.to(DEVICE)
                logits, preds, labels, guids = model(batch), torch.argmax(model(batch), dim=-1), batch['label'], batch['guid']
                for i in range(len(labels)):
                    if labels[i].item() == TARGET_CLASS_INDEX:
                        total_class2_samples += 1
                        if preds[i].item() != labels[i].item():
                            misclassified_count += 1; guids_to_remove.add(guids[i])
        print(f"在 {split_name} 集中找到 {total_class2_samples} 个第二类样本。")
        print(f"其中，裁判模型将其中的 {misclassified_count} 个错误分类，将被删除。")
        cleaned_examples = [ex for ex in examples if ex.guid not in guids_to_remove]
        print(f"{split_name} 集原始大小: {len(examples)}, 清洗后大小: {len(cleaned_examples)}")
        return cleaned_examples

    train_examples = processor.get_train_examples(data_path)
    test_examples = processor.get_test_examples(data_path)
    cleaned_train_examples = process_and_clean_split("train", train_examples)
    cleaned_test_examples = process_and_clean_split("test", test_examples)
    
    def write_to_csv(examples: list[InputExample], output_path: str):
        data_to_write = [[ex.label, ex.text_a, ex.text_b] for ex in examples]
        df = pd.DataFrame(data_to_write, columns=['label', 'title', 'content'])
        df.to_csv(output_path, index=False, header=False)
        print(f"✅ 清洗后的数据已保存到: {output_path}")

    write_to_csv(cleaned_train_examples, os.path.join(data_path, CLEANED_TRAIN_CSV_NAME))
    write_to_csv(cleaned_test_examples, os.path.join(data_path, CLEANED_TEST_CSV_NAME))
    print("\n" + "=" * 30 + " 数据清洗完成 " + "=" * 30)

# ==================================================================================================
#                                 5. 主程序入口
# ==================================================================================================
if __name__ == "__main__":
    # 为了专注于清洗任务，我们直接调用 clean_dataset()
    # 如果您想保留任务选择功能，可以恢复使用 argparse
    clean_dataset()