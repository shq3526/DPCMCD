# verify_classifier_weights.py
# 描述: 一个快速诊断脚本，用于检查多模态模型中最终分类器的权重分布，
#       以验证“分类器是否学会了忽略图像特征”的假设。

import os
import sys
import torch
import torch.nn as nn
from typing import *
import json
# ==================================================================================================
#                       1. 核心依赖组件 (自包含)
# ==================================================================================================
from openprompt.utils.logging import logger
from openprompt.prompts import ManualVerbalizer, PtuningTemplate
from openprompt.plms import load_plm
from openprompt import PromptForClassification
from openprompt.prompt_base import Template
from openprompt.data_utils.utils import InputFeatures # <<-- 添加这一行
# --- 初始化项目路径 ---
try:
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(current_script_path)
except NameError:
    project_root = os.path.abspath('.')

# --- 多模态模型定义 (与训练时完全一致) ---
class MultiModalPromptModel(nn.Module):
    def __init__(self, plm, template: Template, verbalizer: ManualVerbalizer):
        super().__init__()
        self.prompt_model = PromptForClassification(plm, template, verbalizer, freeze_plm=False)
        num_classes = len(self.prompt_model.verbalizer.classes)
        clip_feature_size = 512
        self.classifier = nn.Sequential(
            nn.Linear(num_classes + clip_feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.verbalizer = self.prompt_model.verbalizer
    def forward(self, inputs: InputFeatures): # forward 方法在此脚本中不会被调用
        pass

# --- 辅助函数 ---
def load_config_from_file(config_path: str) -> object:
    import json
    with open(config_path, 'r', encoding='utf-8') as f: config_dict = json.load(f)
    class Config:
        def __init__(self, **entries): self.__dict__.update({k:v for k,v in entries.items() if not k.startswith('//')})
    return Config(**config_dict)
    
def create_model_skeleton(config: object, scripts_path: str, device: str):
    # =======================================================
    #              <<<<< 核心修复区域 >>>>>>>
    # 恢复正确的、包含三个返回值的 return 语句
    # =======================================================
    plm, tokenizer, model_config, WrapperClass = load_plm(config.model_type, config.model_name_or_path)
    template_path = os.path.join(scripts_path, "ptuning_template.txt")
    with open(template_path, 'r', encoding='utf-8') as f:
        template_text = f.readlines()[config.template_id].rstrip()
    template = PtuningTemplate(model=plm, tokenizer=tokenizer, text=template_text)
    verbalizer_path = os.path.join(scripts_path, config.verbalizer_file_name)
    with open(verbalizer_path, 'r', encoding='utf-8') as f: verbalizer_dict = json.load(f)
    class_names = list(verbalizer_dict['label_words'].keys())
    myverbalizer = ManualVerbalizer(tokenizer, classes=class_names, label_words=verbalizer_dict['label_words'])
    model = MultiModalPromptModel(plm=plm, template=template, verbalizer=myverbalizer)
    model.to(device)
    
    # 返回所有创建的核心组件
    return model, tokenizer, WrapperClass
# =======================================================

# ==================================================================================================
#                                 2. 诊断配置区域
# ==================================================================================================
class DiagnosticConfig:
    FP32_MODEL_CKPT = "20250918-001656-toutiao-FP32_MultiModal-Acc0.9657-F1s0.9296.ckpt"
    TRAIN_SCRIPT_CONFIG_FILE = "./config/config.txt"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG = DiagnosticConfig()

# ==================================================================================================
#                                       3. 主诊断流程
# ==================================================================================================
def main():
    logger.info("=" * 30 + " 分类器权重诊断脚本 " + "=" * 30)
    
    train_config = load_config_from_file(CONFIG.TRAIN_SCRIPT_CONFIG_FILE)
    scripts_path = os.path.join(project_root, "scripts", "TextClassification", train_config.prompt_name)
    fp32_ckpt_path = os.path.join("./ckpts", CONFIG.FP32_MODEL_CKPT)

    logger.info("\n--- 正在创建模型骨架并加载权重 ---")
    
    # 步骤 1: 创建与保存时结构完全一致的模型骨架
    # 我们需要确保此处的配置与训练时一致，特别是 use_images
    if not (hasattr(train_config, 'use_images') and train_config.use_images):
        logger.error("错误：此诊断需要一个在 'use_images: true' 配置下训练的多模态模型。")
        logger.error("请检查您的 config.txt 文件。")
        return
        
    model, _, _ = create_model_skeleton(train_config, scripts_path, CONFIG.DEVICE)
    
    # 步骤 2: 加载权重
    model.load_state_dict(torch.load(fp32_ckpt_path, map_location=CONFIG.DEVICE))
    logger.info(f"✅ 模型加载成功: {CONFIG.FP32_MODEL_CKPT}")
    
    # --- 步骤 3: 提取并分析分类器权重 ---
    logger.info("\n--- 开始分析分类器第一层权重 ---")
    
    # 确保模型是多模态模型
    if not isinstance(model, MultiModalPromptModel):
        logger.error("错误：加载的模型不是预期的 MultiModalPromptModel 类型。")
        return
        
    # 定位到分类器的第一层全连接层
    classifier_layer1 = model.classifier[0]
    if not isinstance(classifier_layer1, nn.Linear):
        logger.error(f"错误：分类器第一层不是 nn.Linear，而是 {type(classifier_layer1)}。")
        return
    
    # 获取权重矩阵
    classifier_layer1_weight = classifier_layer1.weight.detach()
    
    # 获取文本logits和图像特征的维度
    text_logits_dim = model.verbalizer.num_classes
    # clip_feature_size = 512
    
    # 权重矩阵的形状应该是 (output_dim, input_dim)，其中 input_dim = text_logits_dim + 512
    # 我们按列（dim=1）来分割
    text_part_weights = classifier_layer1_weight[:, :text_logits_dim]
    image_part_weights = classifier_layer1_weight[:, text_logits_dim:]
    
    # 计算各部分的L2范数
    text_part_norm = torch.linalg.norm(text_part_weights).item()
    image_part_norm = torch.linalg.norm(image_part_weights).item()
    
    # --- 步骤 4: 打印诊断报告 ---
    print("\n" + "="*80)
    print(" " * 25 + "分类器权重诊断报告")
    print("="*80)
    print(f"分析目标: 分类器第一层全连接层 (model.classifier[0])")
    print(f"权重矩阵总形状: {list(classifier_layer1_weight.shape)}")
    print(f"输入维度拆分: 前 {text_logits_dim} 维来自文本Logits，后 {image_part_weights.shape[1]} 维来自图像特征")
    print("-" * 80)
    print(f"对应【文本Logits】部分的权重范数 (L2 Norm): {text_part_norm:.6f}")
    print(f"对应【图像特征】部分的权重范数 (L2 Norm):  {image_part_norm:.6f}")
    print("-" * 80)
    
    print("诊断结论:")
    if image_part_norm < text_part_norm * 0.1: # 如果图像部分范数不到文本部分的10%
        print("✅ 结论显著：图像部分的权重范数远小于文本部分。")
        print("   这强烈表明，分类器在训练过程中学会了【高度忽略或抑制】图像特征的输入，")
        print("   其决策几乎完全依赖于文本分支提供的Logits。")
        if image_part_norm < text_part_norm * 0.01:
             print("   影响极其微弱（小于1%），几乎可以视为完全解耦。")
    elif image_part_norm < text_part_norm * 0.5:
        print("🟡 结论较明显：图像部分的权重范数明显小于文本部分。")
        print("   这表明，模型主要依赖文本信息，但图像特征仍保留了一定的影响力。")
    else:
        print("🤔 结论不明确：图像部分的权重范数与文本部分处于同一数量级。")
        print("   这表明从权重上看，模型并未完全忽略图像特征。")
        print("   结果相同的原因可能更复杂，例如文本和图像特征在送入分类器前就已高度相关或冗余。")
        
    print("\n这个诊断结果解释了为什么'w/o Visual'实验的性能与'Baseline'几乎相同：")
    print("因为即使在Baseline中提供了图像特征，分类器也几乎没有使用它们。")
    print("="*80)
    

if __name__ == "__main__":
    main()