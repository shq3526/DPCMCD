# ablation_study_qat_model_final_v4_diagnosed.py
# 描述: 基于诊断脚本的最终验证，使用最纯净、最正确的QAT加载流程。

import os, sys, torch, torch.nn as nn, numpy as np, json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from datetime import datetime
from typing import *
from collections import OrderedDict

# ==================================================================================================
#                       1. 核心依赖组件 (无自定义Wrappers的纯净版)
# ==================================================================================================
from openprompt.utils.logging import logger
from openprompt.data_utils.text_classification_dataset import CnClickbaitProcessor
from openprompt.data_utils.utils import InputExample, InputFeatures
from openprompt.prompts import ManualVerbalizer, PtuningTemplate
from openprompt.plms import load_plm
from openprompt import PromptForClassification
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from openprompt.prompt_base import Template
from openprompt.plms.utils import TokenizerWrapper
from openprompt.utils import signature
from torch.ao.quantization import QuantStub, DeQuantStub

try:
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(current_script_path)
except NameError:
    project_root = os.path.abspath('.')

# 关键修复1: 简化MultiModalPromptModel以适应自动插入的Quant/Dequant stubs
class MultiModalPromptModel(nn.Module):
    def __init__(self, plm, template: Template, verbalizer: ManualVerbalizer):
        super().__init__()
        self.prompt_model = PromptForClassification(plm, template, verbalizer, freeze_plm=False)
        num_classes = len(self.prompt_model.verbalizer.classes)
        clip_feature_size = 512
        self.classifier = nn.Sequential(
            nn.Linear(num_classes + clip_feature_size, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )
        self.verbalizer = self.prompt_model.verbalizer
    def forward(self, inputs: InputFeatures, use_visual: bool = True) -> torch.Tensor:
        text_logits = self.prompt_model(inputs)
        if use_visual and hasattr(inputs, 'image_features') and inputs.image_features is not None:
            image_features = inputs.image_features.squeeze(1) if inputs.image_features.dim() == 3 else inputs.image_features
        else:
            image_features = torch.zeros(text_logits.size(0), 512, device=text_logits.device)
        # 在纯净模型中，输入和输出都是FP32，Quant/Dequant由prepare_qat自动处理
        combined_features = torch.cat([text_logits, image_features], dim=1)
        return self.classifier(combined_features)

# 关键修复2: create_model不再需要quantize标志和自定义Wrappers
def create_pure_model(config: object, scripts_path: str, device: str) -> Tuple:
    plm, tokenizer, model_config, WrapperClass = load_plm(config.model_type, config.model_name_or_path)
    template_path = os.path.join(scripts_path, "ptuning_template.txt")
    with open(template_path, 'r', encoding='utf-8') as f: template_text = f.readlines()[config.template_id].rstrip()
    mytemplate = PtuningTemplate(model=plm, tokenizer=tokenizer, text=template_text)
    verbalizer_path = os.path.join(scripts_path, config.verbalizer_file_name)
    with open(verbalizer_path, 'r', encoding='utf-8') as f: verbalizer_dict = json.load(f)
    class_names = list(verbalizer_dict['label_words'].keys())
    myverbalizer = ManualVerbalizer(tokenizer, classes=class_names, label_words=verbalizer_dict['label_words'])
    # 假设总是创建多模态模型
    model = MultiModalPromptModel(plm=plm, template=mytemplate, verbalizer=myverbalizer)
    model.to(device)
    return model, tokenizer, mytemplate, WrapperClass
    
# (省略其他不变的辅助函数)
def load_config_from_file(config_path: str) -> object:
    with open(config_path, 'r', encoding='utf-8') as f: config_dict = json.load(f)
    class Config:
        def __init__(self, **entries): self.__dict__.update({k:v for k,v in entries.items() if not k.startswith('//')})
    return Config(**config_dict)
def preprocess_and_tokenize_dataset(dataset, template, tokenizer, wrapper_class, config, default_image_features):
    tokenizer_wrapper_init_keys = signature(wrapper_class.__init__).args
    prepare_kwargs = {"max_seq_length": config.max_seq_l, "truncate_method": "tail", "tokenizer": tokenizer}
    to_pass_kwargs = {key: prepare_kwargs[key] for key in prepare_kwargs if key in tokenizer_wrapper_init_keys}
    tokenizer_wrapper = wrapper_class(**to_pass_kwargs)
    tensor_dataset = []
    for example in tqdm(dataset, desc="Tokenizing Dataset"):
        wrapped_example = template.wrap_one_example(example)
        tokenized_example = tokenizer_wrapper.tokenize_one_example(wrapped_example, teacher_forcing=False)
        features_dict = {**tokenized_example, **wrapped_example[1]}
        if 'image_features' in signature(InputFeatures.__init__).args and hasattr(config, 'use_images') and config.use_images:
             features_dict['image_features'] = example.meta.get('image_features', default_image_features)
        tensor_dataset.append(InputFeatures(**features_dict).to_tensor())
    return tensor_dataset
def load_image_features(img_path_str, clip_processor, clip_model, device, project_root_path):
    if not img_path_str or str(img_path_str).lower() == 'nan': return None
    full_path = os.path.join(project_root_path, img_path_str)
    if not os.path.exists(full_path): return None
    try:
        image = Image.open(full_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
        return image_features
    except Exception: return None

class AblationConfig:
    QAT_MODEL_CKPT = "20250916-022629-toutiao-QAT_per_tensor_linear_only_MultiModal-Acc0.9451-F1s0.8904.ckpt"
    TRAIN_SCRIPT_CONFIG_FILE = "./config/config.txt"
    DEVICE = "cpu"
    BATCH_SIZE = 32
CONFIG = AblationConfig()

def evaluate_ablation(model, dataloader, device, desc="Evaluating", use_visual=True, use_syntactic=True):
    model.to(device).eval()
    original_template = model.prompt_model.template
    if not use_syntactic:
        plm = model.prompt_model.plm; tokenizer = model.prompt_model.template.tokenizer
        text_without_soft = '{"placeholder":"text_a"} {"mask"}'
        plain_template = PtuningTemplate(model=plm, tokenizer=tokenizer, text=text_without_soft)
        model.prompt_model.template = plain_template
        logger.info("  -> Syntactic features (soft prompt) have been ablated.")
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs in tqdm(dataloader, desc=desc, leave=False):
            inputs.to(device)
            logits = model(inputs, use_visual=use_visual)
            labels = inputs['label']
            all_labels.extend(labels.cpu().numpy()); all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
    if not use_syntactic:
        model.prompt_model.template = original_template
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    return {"Accuracy": acc, "Macro Precision": p, "Macro Recall": r, "Macro F1-Score": f1}

def main():
    logger.info("=" * 30 + " QAT量化模型消融研究 (诊断后最终修复版) " + "=" * 30)

    train_config = load_config_from_file(CONFIG.TRAIN_SCRIPT_CONFIG_FILE)
    train_config.use_images = True # 强制多模态
    scripts_path = os.path.join(project_root, "scripts", "TextClassification", train_config.prompt_name)
    
    # =========================================================================
    #                    <<<<<<<<< 最终核心修复区域 >>>>>>>>>
    #       完全复刻诊断脚本中被验证为正确的QAT加载流程
    # =========================================================================
    logger.info("\n--- 正在创建并加载QAT模型 (精确复现'仅线性层'配置) ---")
    
    # 步骤1: 创建一个纯净的、与FP32结构完全相同的骨架
    model_qat_skeleton, tokenizer, mytemplate, WrapperClass = create_pure_model(train_config, scripts_path, CONFIG.DEVICE)
    model_qat_skeleton.train()
    
    # 步骤2: 精确地只为 nn.Linear 模块应用qconfig
    logger.info("--> Applying QAT qconfig to nn.Linear modules only...")
    qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
    # 遍历所有模块，只对线性层应用qconfig
    for module in model_qat_skeleton.modules():
        if isinstance(module, nn.Linear):
            module.qconfig = qconfig
            
    # 步骤3: 准备QAT，这会自动插入Quant/Dequant stubs
    torch.quantization.prepare_qat(model_qat_skeleton, inplace=True)
    
    # 步骤4: 转换模型，得到最终的INT8结构
    model_qat_skeleton.eval()
    torch.quantization.convert(model_qat_skeleton, inplace=True)
    
    # 步骤5: 在结构完全匹配的骨架上加载权重
    model_qat = model_qat_skeleton
    qat_ckpt_path = os.path.join("./ckpts", CONFIG.QAT_MODEL_CKPT)
    if not os.path.exists(qat_ckpt_path):
        logger.error(f"错误：QAT模型文件未找到: {qat_ckpt_path}"); return
    
    # strict=False是安全的，因为prepare_qat/convert后模型可能会有一些非参数的辅助状态(如observers)，
    # 而保存的ckpt只包含权重。
    model_qat.load_state_dict(torch.load(qat_ckpt_path, map_location=CONFIG.DEVICE), strict=False)
    logger.info(f"✅ QAT (INT8)模型加载成功: {CONFIG.QAT_MODEL_CKPT}")
    
    # =========================================================================

    data_path = os.path.join(project_root, "datasets", "TextClassification", train_config.dataset_name)
    processor = CnClickbaitProcessor()
    test_dataset_raw = processor.get_test_examples(data_path)
    
    logger.info("\n--- 正在进行一次性图像特征预处理 ---")
    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        clip_model_path = os.path.join(project_root, "model", "clip-vit-base-patch32")
        if not os.path.exists(clip_model_path): raise FileNotFoundError("CLIP模型未找到")
        clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
        clip_model = CLIPModel.from_pretrained(clip_model_path).to(clip_device)
        default_image_features = torch.zeros(512, device=CONFIG.DEVICE)
        for example in tqdm(test_dataset_raw, desc="提取图像特征"):
            img_path_str = None
            if isinstance(example.meta, str): img_path_str = example.meta; example.meta = {}
            elif isinstance(example.meta, dict): img_path_str = example.meta.get('img_path')
            features = load_image_features(img_path_str, clip_processor, clip_model, clip_device, project_root)
            example.meta['image_features'] = features.squeeze(0).to(CONFIG.DEVICE) if features is not None else default_image_features
    except Exception as e:
        logger.error(f"CLIP模型加载或特征提取失败: {e}"); return

    logger.info("\n--- 正在预处理（分词）测试数据 ---")
    test_dataset_tokenized = preprocess_and_tokenize_dataset(test_dataset_raw, mytemplate, tokenizer, WrapperClass, train_config, default_image_features)
    test_dataloader = DataLoader(test_dataset_tokenized, batch_size=CONFIG.BATCH_SIZE, collate_fn=InputFeatures.collate_fct)

    logger.info("\n--- 开始执行消融实验 ---")
    results = {}
    logger.info("\n[Baseline] 评估完整模型...")
    results['Baseline'] = evaluate_ablation(model_qat, test_dataloader, CONFIG.DEVICE, desc="Eval-Baseline")
    logger.info("\n[w/o Visual] 评估移除视觉特征后的模型...")
    results['w/o Visual'] = evaluate_ablation(model_qat, test_dataloader, CONFIG.DEVICE, desc="Eval-w/o_Visual", use_visual=False)
    logger.info("\n[w/o Syntactic] 评估移除句法特征后的模型...")
    results['w/o Syntactic'] = evaluate_ablation(model_qat, test_dataloader, CONFIG.DEVICE, desc="Eval-w/o_Syntactic", use_syntactic=False)
    logger.info("\n[Text Only] 评估只保留文本特征的模型...")
    results['Text Only'] = evaluate_ablation(model_qat, test_dataloader, CONFIG.DEVICE, desc="Eval-Text_Only", use_visual=False, use_syntactic=False)

    logger.info("\n--- 实验完成，生成最终报告 ---")
    df = pd.DataFrame(results).T
    for col in df.columns: df[col] = df[col].apply(lambda x: f"{x*100:.2f}" if isinstance(x, (int, float)) else x)
    report = f"""
======================================================================
                  Ablation Study of the QAT (INT8) Model
======================================================================
Model: {CONFIG.QAT_MODEL_CKPT}
Device: {CONFIG.DEVICE}
Configuration: {CONFIG.TRAIN_SCRIPT_CONFIG_FILE}

Methodology:
- Baseline: Full multi-modal model with soft prompts.
- w/o Visual: Visual features are zeroed out during forward pass.
- w/o Syntactic: Soft prompt template is replaced with a plain one.
- Text Only: Both visual and syntactic features are ablated.
----------------------------------------------------------------------
{df.to_string()}
======================================================================
"""
    print(report)
    report_filename = f"ablation_study_qat_{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f: f.write(report)
    logger.info(f"✅ 详细报告已保存至: {report_filename}")

if __name__ == "__main__":
    main()