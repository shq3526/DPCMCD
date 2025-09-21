
import os, sys, torch, torch.nn as nn, numpy as np, json, copy
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from datetime import datetime
from typing import *

from openprompt.utils.logging import logger
from openprompt.data_utils.text_classification_dataset import CnClickbaitProcessor
from openprompt.data_utils.utils import InputExample, InputFeatures
from openprompt.prompts import ManualVerbalizer, PtuningTemplate
from openprompt.plms import load_plm
from openprompt import PromptForClassification, PromptDataLoader
from transformers import PreTrainedTokenizer
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from openprompt.prompt_base import Template
from openprompt.plms.utils import TokenizerWrapper
from openprompt.utils import signature

try:
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(current_script_path)
except NameError:
    project_root = os.path.abspath('.')


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

    def forward(self, inputs: InputFeatures, use_visual: bool = True, use_syntactic: bool = True) -> torch.Tensor:
        # 句法消融
        if not use_syntactic:
            if hasattr(inputs, 'soft_embeds') and inputs.soft_embeds is not None:
                inputs.soft_embeds.zero_()
        
        text_logits = self.prompt_model(inputs)
        
        # 视觉特征处理
        if use_visual and hasattr(inputs, 'image_features') and inputs.image_features is not None:
            image_features = inputs.image_features.squeeze(1) if inputs.image_features.dim() == 3 else inputs.image_features
        else:
            image_features = torch.randn(text_logits.size(0), 512, device=text_logits.device)
            
        combined_features = torch.cat([text_logits, image_features], dim=1)
        return self.classifier(combined_features)
# =========================================================================

def load_config_from_file(config_path: str) -> object:
    with open(config_path, 'r', encoding='utf-8') as f: config_dict = json.load(f)
    class Config:
        def __init__(self, **entries): self.__dict__.update({k:v for k,v in entries.items() if not k.startswith('//')})
    return Config(**config_dict)

def create_model_and_components(config: object, scripts_path: str, device: str):
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
    return model, tokenizer, WrapperClass

def load_image_features(img_path_str, clip_processor, clip_model, device, project_root_path):
    if not img_path_str or str(img_path_str).lower() == 'nan': return None
    full_path = os.path.join(project_root_path, img_path_str)
    if not os.path.exists(full_path): return None
    try:
        image = Image.open(full_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad(): image_features = clip_model.get_image_features(**inputs)
        return image_features
    except Exception: return None

class AblationConfig:
    FP32_MODEL_CKPT = "20250918-001656-toutiao-FP32_MultiModal-Acc0.9657-F1s0.9296.ckpt"
    TRAIN_SCRIPT_CONFIG_FILE = "./config/config.txt"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
CONFIG = AblationConfig()

def evaluate_model(model: nn.Module, dataloader: PromptDataLoader, device: str, desc: str = "Evaluating", 
                     use_visual: bool = True, use_syntactic: bool = True):
    model.to(device).eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            inputs = copy.deepcopy(batch)
            inputs.to(device)
            logits = model(inputs, use_visual=use_visual, use_syntactic=use_syntactic)
            labels = inputs.label
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
    if not all_labels: return {"Accuracy": 0, "Macro Precision": 0, "Macro Recall": 0, "Macro F1-Score": 0}
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    return {"Accuracy": acc, "Macro Precision": p, "Macro Recall": r, "Macro F1-Score": f1}

def main():
    logger.info("=" * 30 + " 教师模型(FP32)消融研究 (随机噪声遮蔽版) " + "=" * 30)
    
    train_config = load_config_from_file(CONFIG.TRAIN_SCRIPT_CONFIG_FILE)
    scripts_path = os.path.join(project_root, "scripts", "TextClassification", train_config.prompt_name)
    data_path = os.path.join(project_root, "datasets", "TextClassification", train_config.dataset_name)
    processor = CnClickbaitProcessor()
    test_dataset_raw = processor.get_test_examples(data_path)
    fp32_ckpt_path = os.path.join("./ckpts", CONFIG.FP32_MODEL_CKPT)

    logger.info("\n--- 正在创建和加载【唯一】的模型实例 ---")
    model, tokenizer, WrapperClass = create_model_and_components(train_config, scripts_path, CONFIG.DEVICE)
    model.load_state_dict(torch.load(fp32_ckpt_path, map_location=CONFIG.DEVICE))
    logger.info(f"✅ 模型加载成功: {CONFIG.FP32_MODEL_CKPT}")
    
    logger.info("\n--- 正在进行【唯一】的图像特征预处理 ---")
    clip_model_path = os.path.join(project_root, "model", "clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
    clip_model = CLIPModel.from_pretrained(clip_model_path).to(CONFIG.DEVICE)
    default_image_features = torch.zeros(512, device=CONFIG.DEVICE)
    for example in tqdm(test_dataset_raw, desc="提取图像特征"):
        img_path_str = example.meta if isinstance(example.meta, str) else (example.meta.get('img_path') if isinstance(example.meta, dict) else None)
        if isinstance(example.meta, str): example.meta = {}
        features = load_image_features(img_path_str, clip_processor, clip_model, CONFIG.DEVICE, project_root)
        example.meta['image_features'] = features.squeeze(0) if features is not None else default_image_features
        
    logger.info("\n--- 正在创建【唯一】的数据加载器 ---")
    dataloader = PromptDataLoader(dataset=test_dataset_raw, template=model.prompt_model.template, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, batch_size=CONFIG.BATCH_SIZE, shuffle=False)
        
    results = {}
    
    logger.info("\n--- 开始执行消融实验 ---")
    
    results['Baseline'] = evaluate_model(model, dataloader, CONFIG.DEVICE, "Eval-Baseline", use_visual=True, use_syntactic=True)
    results['w/o Visual'] = evaluate_model(model, dataloader, CONFIG.DEVICE, "Eval-w/o_Visual", use_visual=False, use_syntactic=True)
    results['w/o Syntactic'] = evaluate_model(model, dataloader, CONFIG.DEVICE, "Eval-w/o_Syntactic", use_visual=True, use_syntactic=False)
    results['Text Only'] = evaluate_model(model, dataloader, CONFIG.DEVICE, "Eval-Text_Only", use_visual=False, use_syntactic=False)
    
    logger.info("\n--- 实验完成，生成最终报告 ---")
    df = pd.DataFrame(results).T
    for col in df.columns: df[col] = df[col].apply(lambda x: f"{x*100:.2f}" if isinstance(x, (int, float)) else x)
    report = f"""
======================================================================
           Ablation Study of the Multi-Modal Teacher Model
                 (Using Random Noise for Ablation)
======================================================================
Model: {CONFIG.FP32_MODEL_CKPT}
Device: {CONFIG.DEVICE}
Configuration: {CONFIG.TRAIN_SCRIPT_CONFIG_FILE}

Methodology:
- A single model and dataloader are used for all tests.
- Ablation is controlled by flags passed to the forward method.
- 'w/o Visual': Visual features are replaced with random Gaussian noise.
- 'w/o Syntactic': Soft prompt embeddings are zeroed out in the forward pass.
----------------------------------------------------------------------
{df.to_string()}
======================================================================
"""
    print(report)
    report_filename = f"ablation_study_noise_{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f: f.write(report)
    logger.info(f"✅ 详细报告已保存至: {report_filename}")

if __name__ == "__main__":
    main()