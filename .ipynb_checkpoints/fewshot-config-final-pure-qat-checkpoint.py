# 描述: 终极版训练与评估一体化脚本 (完整实现，已修复多模态特征融合)。
# - ... (描述保持不变) ...

import os
from tqdm import tqdm
import torch
import torch.nn as nn
import time
import sys
import random
import numpy as np
import json
import argparse
import csv
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from datetime import datetime
from typing import *
from collections import OrderedDict

# --- 初始化项目路径 ---
try:
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(current_script_path)
except NameError:
    project_root = os.path.abspath('.')
sys.path.insert(0, project_root)
print(f"--- 项目根目录已添加至 sys.path: {project_root} ---")

# --- 导入所需库 ---
from openprompt.utils.logging import logger
from openprompt.data_utils.text_classification_dataset import CnClickbaitProcessor
from openprompt.data_utils.utils import InputExample, InputFeatures
from openprompt.prompts import ManualVerbalizer, PtuningTemplate
from openprompt.plms import load_plm
from openprompt.data_utils.data_sampler import FewShotSampler
from openprompt.utils.reproduciblity import set_seed
from openprompt import PromptForClassification
from transformers import AdamW, PreTrainedTokenizer
from torch.utils.data import Dataset, DataLoader, RandomSampler
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from openprompt.prompt_base import Template
from openprompt.plms.utils import TokenizerWrapper
from openprompt.utils import signature
from torch.ao.quantization import QuantStub, DeQuantStub
from torch.ao.nn.quantized import FloatFunctional

# --- 全局设置 ---
csv.field_size_limit(512 * 1024)

# ==================================================================================================
#                                 核心依赖组件 (最终一致性版本)
# ==================================================================================================
class BertEncoderQuantWrapper(nn.Module):
    def __init__(self, module): super().__init__(); self.quant = QuantStub(); self.encoder = module
    def forward(self, hidden_states, **kwargs):
        return self.encoder(self.quant(hidden_states), **kwargs)

class AttentionQuantWrapper(nn.Module):
    def __init__(self, module): super().__init__(); self.dequant = DeQuantStub(); self.attention = module; self.quant = QuantStub()
    def forward(self, *args, **kwargs):
        dq_args = (self.dequant(args[0]),) + args[1:]
        outputs = self.attention(*dq_args, **kwargs)
        return (self.quant(outputs[0]),) + outputs[1:]

class BertOutputQuantWrapper(nn.Module):
    def __init__(self, module): super().__init__(); self.output = module; self.add = FloatFunctional()
    def forward(self, hidden_states, input_tensor):
        dense_out = self.output.dropout(self.output.dense(hidden_states))
        return self.output.LayerNorm(self.add.add(dense_out, input_tensor))

class ClsQuantWrapper(nn.Module):
    def __init__(self, module): super().__init__(); self.cls = module; self.dequant = DeQuantStub()
    def forward(self, *args, **kwargs):
        return self.dequant(self.cls(*args, **kwargs))
        
# ==================================================================================================
#                       【新增】多模态模型定义 (融合图像与文本)
# ==================================================================================================
class MultiModalPromptModel(nn.Module):
    def __init__(self, plm, template: Template, verbalizer: ManualVerbalizer):
        super().__init__()
        self.prompt_model = PromptForClassification(plm, template, verbalizer, freeze_plm=False)
        num_classes = len(self.prompt_model.verbalizer.classes)
        clip_feature_size = 512
        classifier_input_dim = num_classes + clip_feature_size
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.verbalizer = self.prompt_model.verbalizer

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, inputs: InputFeatures) -> torch.Tensor:
        text_logits = self.prompt_model(inputs)
        if hasattr(inputs, 'image_features') and inputs.image_features is not None:
            image_features = inputs.image_features
            if image_features.dim() == 3 and image_features.size(1) == 1:
                image_features = image_features.squeeze(1)
        else:
            batch_size = text_logits.size(0)
            image_features = torch.zeros(batch_size, 512, device=text_logits.device)
        combined_features = torch.cat([text_logits, image_features], dim=1)
        final_logits = self.classifier(combined_features)
        return final_logits

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.prompt_model = self.prompt_model.to(*args, **kwargs)
        self.classifier = self.classifier.to(*args, **kwargs)
        return self

# ==================================================================================================
#                                        辅助函数定义
# ==================================================================================================
def load_config_from_file(config_path: str) -> object:
    """从 JSON 格式的 .txt 文件加载配置。"""
    print(f"--> 正在从 {config_path} 加载配置...")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    class Config:
        def __init__(self, **entries):
            entries = {k: v for k, v in entries.items() if not k.startswith('//')}
            self.__dict__.update(entries)
    return Config(**config_dict)

def read_readme(path: str, component_name: str) -> None:
    """读取并打印指定路径下的 readme.txt 文件。"""
    readme_path = os.path.join(path, "readme.txt")
    print("-" * 60)
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            print(f"✔️  读取到 {component_name} 信息 (readme.txt):\n" + f.read())
    else:
        print(f"⚠️  警告：在 {path} 中未找到 {component_name} 的说明文件 (readme.txt)。")
    print("-" * 60)

def load_image_features(img_path_str: Optional[str], clip_processor: CLIPProcessor, clip_model: CLIPModel, 
                        device: str, project_root_path: str) -> Optional[torch.Tensor]:
    """加载单张图片并使用CLIP提取特征。(最终修复版)"""
    if not img_path_str or str(img_path_str).lower() == 'nan':
        return None
    full_path = os.path.join(project_root_path, img_path_str)
    if not os.path.exists(full_path):
        return None
    try:
        image = Image.open(full_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
        return image_features
    except Exception as e:
        return None

def preprocess_and_tokenize_dataset(dataset: List[InputExample], template: Template, tokenizer: PreTrainedTokenizer,
                                    wrapper_class, config: object, default_image_features: torch.Tensor) -> List[InputFeatures]:
    """一次性完成数据集的 wrapping 和 tokenizing。"""
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

def create_prompt_model(config: object, scripts_path: str, device: str, quantize: bool = False) -> Tuple:
    """创建模型实例，并根据 quantize 参数和图像使用情况决定模型类型。"""
    plm, tokenizer, model_config, WrapperClass = load_plm(config.model_type, config.model_name_or_path)
    template_path = os.path.join(scripts_path, "ptuning_template.txt")
    with open(template_path, 'r', encoding='utf-8') as f:
        template_text = f.readlines()[config.template_id].rstrip()
    mytemplate = PtuningTemplate(model=plm, tokenizer=tokenizer, text=template_text)
    verbalizer_path = os.path.join(scripts_path, config.verbalizer_file_name)
    with open(verbalizer_path, 'r', encoding='utf-8') as f:
        verbalizer_dict = json.load(f)
    class_names = list(verbalizer_dict['label_words'].keys())
    myverbalizer = ManualVerbalizer(tokenizer, classes=class_names)
    myverbalizer.label_words = verbalizer_dict['label_words']

    if hasattr(config, 'use_images') and config.use_images:
        logger.info("--> 创建多模态模型 (MultiModalPromptModel)...")
        prompt_model = MultiModalPromptModel(plm=plm, template=mytemplate, verbalizer=myverbalizer)
    else:
        logger.info("--> 创建标准文本模型 (PromptForClassification)...")
        prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)

    if quantize:
        logger.info("--> 正在为量化准备模型骨架 (植入所有Wrappers)...")
        if isinstance(prompt_model, MultiModalPromptModel):
            plm_model = prompt_model.prompt_model.plm
        else:
            plm_model = prompt_model.plm
        
        plm_model.bert.encoder = BertEncoderQuantWrapper(plm_model.bert.encoder)
        for layer in plm_model.bert.encoder.encoder.layer:
            layer.attention = AttentionQuantWrapper(layer.attention)
            layer.output = BertOutputQuantWrapper(layer.output)
        plm_model.cls = ClsQuantWrapper(plm_model.cls)
        logger.info("--> 模型骨架准备完毕。")

    prompt_model.to(device)
    return prompt_model, tokenizer, mytemplate, WrapperClass

# ==================================================================================================
#                                      评测 & 保存函数 (整合版)
# ==================================================================================================
def evaluate(p_model, dataloader, desc="Evaluating"):
    """(用于epoch间验证)评估模型性能。"""
    p_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs in tqdm(dataloader, desc=desc):
            inputs.to(p_model.device)
            logits = p_model(inputs)
            labels = inputs['label']
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    pre, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    return {'acc': acc, 'pre': pre, 'recall': recall, 'f1': f1}

def get_model_stats(model, state_dict_path=None):
    """计算模型大小和参数量。"""
    size_mb = 0
    if state_dict_path and os.path.exists(state_dict_path):
        size_mb = os.path.getsize(state_dict_path) / (1024 ** 2)
    num_params = sum(p.numel() for p in model.parameters())
    return {"size_mb": size_mb, "num_params": num_params}

def measure_full_performance(model, test_dataset, config, device):
    """对模型进行完整的精度和性能评估，并提供详细的延迟分析。"""
    logger.info("\n" + "#"*20 + " 开始最终模型评估 " + "#"*20)
    model.eval()
    
    accuracy_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=InputFeatures.collate_fct, shuffle=False)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs in tqdm(accuracy_dataloader, desc="正在评估最终精度"):
            inputs.to(device)
            logits = model(inputs)
            labels = inputs['label']
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
    accuracy_metrics = classification_report(all_labels, all_preds, output_dict=True, zero_division=0, digits=4, target_names=model.verbalizer.classes)
    accuracy_metrics['overall_accuracy'] = accuracy_score(all_labels, all_preds)

    WARMUP_STEPS, MEASURE_SAMPLES = 20, min(len(test_dataset), 100)
    latency_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=InputFeatures.collate_fct, shuffle=False)
    
    logger.info(f"--> [性能] 正在预热 ({WARMUP_STEPS} 个样本)...")
    with torch.no_grad():
        for i, inputs in enumerate(latency_dataloader):
            if i >= WARMUP_STEPS: break
            inputs.to(device); _ = model(inputs)
            
    e2e_latencies, fwd_latencies = [], []
    logger.info(f"--> [性能] 正在测量延迟 (batch_size=1, 测量 {MEASURE_SAMPLES} 个样本)...")
    with torch.no_grad():
        for i, inputs in enumerate(tqdm(latency_dataloader, total=MEASURE_SAMPLES, desc="测量延迟")):
            if i >= MEASURE_SAMPLES: break
            start_e2e = time.perf_counter()
            inputs.to(device)
            torch.cuda.synchronize() if device == 'cuda' else None
            start_fwd = time.perf_counter()
            _ = model(inputs)
            torch.cuda.synchronize() if device == 'cuda' else None
            end_fwd = time.perf_counter()
            end_e2e = time.perf_counter()
            e2e_latencies.append((end_e2e - start_e2e) * 1000)
            fwd_latencies.append((end_fwd - start_fwd) * 1000)

    total_time, total_samples = 0, 0
    logger.info(f"--> [性能] 正在测量吞吐量 (batch_size={config.batch_size})...")
    with torch.no_grad():
        for inputs in tqdm(accuracy_dataloader, desc="测量吞吐量"):
            inputs.to(device)
            torch.cuda.synchronize() if device == 'cuda' else None
            start = time.perf_counter()
            _ = model(inputs)
            torch.cuda.synchronize() if device == 'cuda' else None
            total_time += time.perf_counter() - start
            total_samples += len(inputs['label'])
            
    def get_latency_stats(latencies_ms: List[float]) -> Dict:
        if not latencies_ms: return { 'avg': 0, 'p50': 0, 'p90': 0, 'p95': 0, 'p99': 0 }
        arr = np.array(latencies_ms)
        return {'avg': np.mean(arr), 'p50': np.percentile(arr, 50), 'p90': np.percentile(arr, 90), 'p95': np.percentile(arr, 95), 'p99': np.percentile(arr, 99)}

    performance_metrics = {
        "e2e_latency_ms": get_latency_stats(e2e_latencies),
        "fwd_latency_ms": get_latency_stats(fwd_latencies),
        "throughput_sps": total_samples / total_time if total_time > 0 else 0
    }
    
    if device == 'cuda':
        performance_metrics["gpu_peak_mem_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
    return accuracy_metrics, performance_metrics

def save_model_and_report(final_model, accuracy_metrics, performance_metrics, config, start_time, fold_results=None, mean_acc=0, std_acc=0, best_val_acc=0):
    """保存最终模型(state_dict)并生成详细报告。"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    is_qat = hasattr(config, 'use_qat') and config.use_qat
    qat_str = "-QAT" if is_qat else "-FP32"
    
    # 增加多模态标识
    if hasattr(config, 'use_images') and config.use_images:
        qat_str += "_MultiModal"
        
    base_filename = f"{timestamp}-{config.dataset_name}{qat_str}-Acc{accuracy_metrics['overall_accuracy']:.4f}-F1s{accuracy_metrics['macro avg']['f1-score']:.4f}"
    
    final_model_path = None
    if final_model:
        if is_qat:
            logger.info("\n" + "="*20 + " QAT模型最终转换 " + "="*20)
            final_model.cpu().eval()
            torch.quantization.convert(final_model, inplace=True)
            logger.info("--> 模型已成功转换为最终INT8格式。")
        
        final_model_path = os.path.join("./ckpts", f"{base_filename}.ckpt")
        torch.save(final_model.to('cpu').state_dict(), final_model_path)
        logger.info(f"\n--> ✅ 模型 state_dict 已保存至: {final_model_path}")

    model_stats = get_model_stats(final_model, final_model_path)
    report_file_path = os.path.join("./ckpts", f"{base_filename}.txt")
    end_time = time.time(); duration_minutes = (end_time - start_time) / 60
    
    report_content = [
        "=" * 25 + f" 最终训练评估报告 " + "=" * 25,
        f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"总运行时长: {duration_minutes:.2f} 分钟",
        f"模型文件: {os.path.basename(final_model_path) if final_model_path else '未保存'}",
        "\n" + "-" * 15 + " 1. 实验配置 " + "-" * 15,
        json.dumps({k: v for k, v in config.__dict__.items() if not k.startswith('//')}, indent=4, ensure_ascii=False),
    ]
    if config.use_k_fold and fold_results is not None:
        report_content.extend(["\n" + "-" * 15 + " 2. K-折交叉验证结果 " + "-" * 15, f"各折最佳验证准确率: {[round(r, 4) for r in fold_results]}", f"平均验证准确率: {mean_acc:.4f} (±{std_acc:.4f})"])
    else:
        report_content.extend(["\n" + "-" * 15 + " 2. 验证集表现 " + "-" * 15, f"最佳验证准确率: {best_val_acc:.4f}"])
        
    report_content.extend(["\n" + "-" * 15 + " 3. 最终测试集精度指标 " + "-" * 15, f"整体准确率: {accuracy_metrics['overall_accuracy']:.4f}"])
    class_names_from_report = final_model.verbalizer.classes
    report_content.append(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    for class_name in class_names_from_report:
        if class_name in accuracy_metrics:
            metrics = accuracy_metrics[class_name]
            report_content.append(f"{class_name:<15} {metrics['precision']:.4f}     {metrics['recall']:.4f}     {metrics['f1-score']:.4f}")
    report_content.append("-" * 50)
    macro_avg = accuracy_metrics['macro avg']
    report_content.append(f"{'Macro Avg':<15} {macro_avg['precision']:.4f}     {macro_avg['recall']:.4f}     {macro_avg['f1-score']:.4f}")

    report_content.extend(["\n" + "-" * 15 + " 4. 最终模型性能指标 " + "-" * 15, f"评测设备: {final_model.device}", f"模型磁盘大小: {model_stats['size_mb']:.2f} MB", f"参数量: {model_stats['num_params'] / 1e6:.2f} M", f"吞吐量: {performance_metrics['throughput_sps']:.2f} samples/s (在 batch_size={config.batch_size} 下测得)"])
    if "gpu_peak_mem_mb" in performance_metrics:
        report_content.append(f"GPU峰值内存: {performance_metrics['gpu_peak_mem_mb']:.1f} MB")
    report_content.append("\n详细延迟分析 (ms/sample, batch_size=1):")
    report_content.append(f"{'Metric':<18} | {'Average':<10} | {'P50':<10} | {'P90':<10} | {'P95':<10} | {'P99':<10}")
    report_content.append("-" * 80)
    e2e, fwd = performance_metrics['e2e_latency_ms'], performance_metrics['fwd_latency_ms']
    report_content.append(f"{'End-to-End':<18} | {e2e['avg']:<10.2f} | {e2e['p50']:<10.2f} | {e2e['p90']:<10.2f} | {e2e['p95']:<10.2f} | {e2e['p99']:<10.2f}")
    report_content.append(f"{'Model Forward':<18} | {fwd['avg']:<10.2f} | {fwd['p50']:<10.2f} | {fwd['p90']:<10.2f} | {fwd['p95']:<10.2f} | {fwd['p99']:<10.2f}")
    report_content.append("=" * 80)
    
    final_report = "\n".join(report_content)
    with open(report_file_path, 'w', encoding='utf-8') as f: f.write(final_report)
    logger.info(f"--> ✅ 详细报告已保存至: {report_file_path}"); print("\n" + final_report)

# ==================================================================================================
#                                      核心训练流程
# ==================================================================================================
def run_k_fold_training(config, dataset, scripts_path, device, default_image_features):
    full_dataset, labels = dataset['train'] + dataset['test'], [ex.label for ex in dataset['train'] + dataset['test']]
    train_val_pool, final_test_set = train_test_split(full_dataset, test_size=config.k_fold_test_split_size, random_state=config.seed, stratify=labels)
    skf = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=config.seed)
    fold_results, train_val_pool_np, train_val_pool_labels = [], np.array(train_val_pool), [ex.label for ex in train_val_pool]
    is_qat = hasattr(config, 'use_qat') and config.use_qat

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_pool_np, train_val_pool_labels)):
        logger.info(f"\n========== 开始第 {fold + 1}/{config.k_folds} 折 ==========")
        prompt_model, tokenizer, mytemplate, WrapperClass = create_prompt_model(config, scripts_path, device, quantize=is_qat)
        if is_qat:
            qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
            if isinstance(prompt_model, MultiModalPromptModel): plm_model_ref = prompt_model.prompt_model
            else: plm_model_ref = prompt_model
            modules_to_skip = [m.attention for m in plm_model_ref.modules() if isinstance(m, AttentionQuantWrapper)]
            for name, module in prompt_model.named_modules():
                if any(module in parent.modules() for parent in modules_to_skip): continue
                if 'embeddings' in name and isinstance(module, nn.LayerNorm): continue
                if isinstance(module, (nn.Linear, QuantStub, DeQuantStub, FloatFunctional, nn.LayerNorm)):
                    module.qconfig = qconfig
            prompt_model.train()
            torch.quantization.prepare_qat(prompt_model, inplace=True)
            logger.info("--> QAT 模型已为本折训练准备好。")
        
        train_data_fold, val_data_fold = train_val_pool_np[train_idx].tolist(), train_val_pool_np[val_idx].tolist()
        tokenized_train_fold = preprocess_and_tokenize_dataset(train_data_fold, mytemplate, tokenizer, WrapperClass, config, default_image_features)
        tokenized_val_fold = preprocess_and_tokenize_dataset(val_data_fold, mytemplate, tokenizer, WrapperClass, config, default_image_features)
        val_dataloader_fold = DataLoader(tokenized_val_fold, batch_size=config.batch_size, collate_fn=InputFeatures.collate_fct)
        loss_func, optimizer = torch.nn.CrossEntropyLoss(), AdamW(prompt_model.parameters(), lr=config.learning_rate)
        best_val_acc_fold = 0
        
        for epoch in range(config.max_epochs):
            prompt_model.train()
            train_dataloader_fold = DataLoader(tokenized_train_fold, sampler=RandomSampler(tokenized_train_fold), batch_size=config.batch_size, collate_fn=InputFeatures.collate_fct)
            pbar = tqdm(train_dataloader_fold, desc=f"Fold {fold + 1} Epoch {epoch + 1}")
            for inputs in pbar:
                inputs.to(device); logits = prompt_model(inputs)
                loss = loss_func(logits, inputs['label']); loss.backward(); optimizer.step(); optimizer.zero_grad()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            val_metrics = evaluate(prompt_model, val_dataloader_fold, desc=f"Fold {fold+1} Validating")
            if val_metrics['acc'] > best_val_acc_fold: best_val_acc_fold = val_metrics['acc']
        logger.info(f"--> 第 {fold + 1} 折结束, 最佳验证准确率: {best_val_acc_fold:.4f}"); fold_results.append(best_val_acc_fold)
    
    mean_acc, std_acc = np.mean(fold_results), np.std(fold_results)
    logger.info(f"\n--> {config.k_folds}-折交叉验证平均准确率: {mean_acc:.4f} ± {std_acc:.4f}")
    logger.info("\n" + "=" * 20 + " 正在使用全部数据训练最终模型 " + "=" * 20)
    
    final_model, tokenizer, mytemplate, WrapperClass = create_prompt_model(config, scripts_path, device, quantize=is_qat)
    if is_qat:
        qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
        if isinstance(final_model, MultiModalPromptModel): plm_model_ref = final_model.prompt_model
        else: plm_model_ref = final_model
        modules_to_skip = [m.attention for m in plm_model_ref.modules() if isinstance(m, AttentionQuantWrapper)]
        for name, module in final_model.named_modules():
            if any(module in parent.modules() for parent in modules_to_skip): continue
            if 'embeddings' in name and isinstance(module, nn.LayerNorm): continue
            if isinstance(module, (nn.Linear, QuantStub, DeQuantStub, FloatFunctional, nn.LayerNorm)):
                module.qconfig = qconfig
        final_model.train()
        torch.quantization.prepare_qat(final_model, inplace=True)
        logger.info("--> QAT 最终模型已准备好。")
    
    tokenized_train_val_pool = preprocess_and_tokenize_dataset(train_val_pool, mytemplate, tokenizer, WrapperClass, config, default_image_features)
    final_train_dataloader = DataLoader(tokenized_train_val_pool, sampler=RandomSampler(tokenized_train_val_pool), batch_size=config.batch_size, collate_fn=InputFeatures.collate_fct)
    loss_func, optimizer = torch.nn.CrossEntropyLoss(), AdamW(final_model.parameters(), lr=config.learning_rate)
    for epoch in range(config.max_epochs):
        final_model.train()
        pbar = tqdm(final_train_dataloader, desc=f"Final Model Epoch {epoch + 1}")
        for inputs in pbar:
            inputs.to(device); logits = final_model(inputs)
            loss = loss_func(logits, inputs['label']); loss.backward(); optimizer.step(); optimizer.zero_grad()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    tokenized_final_test_set = preprocess_and_tokenize_dataset(final_test_set, mytemplate, tokenizer, WrapperClass, config, default_image_features)
    return final_model, tokenized_final_test_set, fold_results, mean_acc, std_acc

def run_single_shot_training(config, dataset, scripts_path, device, default_image_features):
    is_qat = hasattr(config, 'use_qat') and config.use_qat
    prompt_model, tokenizer, mytemplate, WrapperClass = create_prompt_model(config, scripts_path, device, quantize=is_qat)
    
    if is_qat:
        logger.info("--> 启用量化感知训练 (QAT)。正在准备模型...")
        qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
        
        # 适配多模态模型
        if isinstance(prompt_model, MultiModalPromptModel):
            plm_model_ref = prompt_model.prompt_model
        else:
            plm_model_ref = prompt_model
            
        modules_to_skip = [m.attention for m in plm_model_ref.modules() if isinstance(m, AttentionQuantWrapper)]
        for name, module in prompt_model.named_modules(): # 遍历整个大模型
            if any(module in parent.modules() for parent in modules_to_skip): continue
            if 'embeddings' in name and isinstance(module, nn.LayerNorm): continue
            if isinstance(module, (nn.Linear, QuantStub, DeQuantStub, FloatFunctional, nn.LayerNorm)):
                module.qconfig = qconfig
        prompt_model.train()
        torch.quantization.prepare_qat(prompt_model, inplace=True)
        logger.info("--> QAT 模型准备完毕。")

    sampler = FewShotSampler(num_examples_per_label=config.shot, also_sample_dev=True, num_examples_per_label_dev=config.shot)
    train_set, val_set = sampler(dataset['train'], seed=config.seed)
    test_set = dataset['test']
    tokenized_train_set = preprocess_and_tokenize_dataset(train_set, mytemplate, tokenizer, WrapperClass, config, default_image_features)
    tokenized_val_set = preprocess_and_tokenize_dataset(val_set, mytemplate, tokenizer, WrapperClass, config, default_image_features)
    tokenized_test_set = preprocess_and_tokenize_dataset(test_set, mytemplate, tokenizer, WrapperClass, config, default_image_features)
    val_dataloader = DataLoader(tokenized_val_set, batch_size=config.batch_size, collate_fn=InputFeatures.collate_fct)
    
    loss_func, optimizer = torch.nn.CrossEntropyLoss(), AdamW(prompt_model.parameters(), lr=config.learning_rate)
    best_val_acc, temp_ckpt_path = 0, f"./ckpts/temp_{config.dataset_name}_{config.seed}.ckpt"
    
    for epoch in range(config.max_epochs):
        prompt_model.train()
        train_dataloader = DataLoader(tokenized_train_set, sampler=RandomSampler(tokenized_train_set), batch_size=config.batch_size, collate_fn=InputFeatures.collate_fct)
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config.max_epochs}")
        for inputs in pbar:
            inputs.to(device); logits = prompt_model(inputs)
            loss = loss_func(logits, inputs['label']); loss.backward(); optimizer.step(); optimizer.zero_grad()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        val_metrics = evaluate(prompt_model, val_dataloader, desc="Validating")
        if val_metrics['acc'] >= best_val_acc:
            torch.save(prompt_model.state_dict(), temp_ckpt_path); best_val_acc = val_metrics['acc']
        print(f"--> Epoch {epoch + 1} 结束, Val Acc: {val_metrics['acc']:.4f}, Best Val Acc: {best_val_acc:.4f}", flush=True)
    
    logger.info(f"--> 加载最佳验证模型 (Acc: {best_val_acc:.4f})")
    prompt_model.load_state_dict(torch.load(temp_ckpt_path))
    if os.path.exists(temp_ckpt_path): os.remove(temp_ckpt_path)
    return prompt_model, tokenized_test_set, best_val_acc

# ==================================================================================================
#                                      程序主入口
# ==================================================================================================
def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="统一训练与评估脚本 (最终完整版)")
    parser.add_argument("--config_file", type=str, default="./config/config.txt", help="配置文件路径")
    args = parser.parse_args()
    config = load_config_from_file(args.config_file)
    set_seed(config.seed)

    is_qat = hasattr(config, 'use_qat') and config.use_qat

    if torch.cuda.is_available(): device = "cuda"; logger.info(f"--> 检测到CUDA设备，将在GPU上进行训练。")
    else: device = "cpu"; logger.info(f"--> 未检测到CUDA设备，将在CPU上进行训练和评估。")
    
    if is_qat:
        torch.backends.quantized.engine = "fbgemm"
        logger.info(f"--> QAT 已启用，设置量化后端引擎为 'fbgemm'。")

    os.makedirs("./ckpts", exist_ok=True)
    
    print("\n" + "=" * 20 + " 实验配置检查 " + "=" * 20)
    data_path = os.path.join(project_root, "datasets", "TextClassification", config.dataset_name)
    scripts_path = os.path.join(project_root, "scripts", "TextClassification", config.prompt_name)
    read_readme(data_path, f"数据集 ({config.dataset_name})"); read_readme(scripts_path, f"提示 ({config.prompt_name})")
    print("=" * 66 + "\n")
    
    processor = CnClickbaitProcessor()
    dataset = {'train': processor.get_train_examples(data_path), 'test': processor.get_test_examples(data_path)}
    default_image_features = torch.zeros(512, device=device)
    
    if hasattr(config, 'use_images') and config.use_images:
        logger.info("--> 启用多模态图片处理。正在进行一次性图像特征提取...")
        try:
            clip_model_path = os.path.join(project_root, "model", "clip-vit-base-patch32")
            if not os.path.exists(clip_model_path):
                 raise FileNotFoundError("CLIP模型未在本地路径 'model/clip-vit-base-patch32' 找到。")
            clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
            clip_model = CLIPModel.from_pretrained(clip_model_path).to(device)
            
            all_examples = dataset['train'] + dataset['test']
            for example in tqdm(all_examples, desc="提取所有图像特征"):
                img_path_str = None
                if isinstance(example.meta, str): img_path_str = example.meta; example.meta = {}
                elif isinstance(example.meta, dict): img_path_str = example.meta.get('img_path')
                
                features = load_image_features(img_path_str, clip_processor, clip_model, device, project_root)
                example.meta['image_features'] = features.squeeze(0) if features is not None else default_image_features
            logger.info("--> 所有图像特征提取完毕。")
        except Exception as e:
            logger.error(f"CLIP模型加载或特征提取失败，将关闭图片处理功能。错误: {e}")
            config.use_images = False

    final_model, test_dataset, fold_results, mean_acc, std_acc, best_val_acc = None, None, None, 0, 0, 0
    if config.use_k_fold:
        final_model, test_dataset, fold_results, mean_acc, std_acc = run_k_fold_training(config, dataset, scripts_path, device, default_image_features)
    else:
        final_model, test_dataset, best_val_acc = run_single_shot_training(config, dataset, scripts_path, device, default_image_features)

    if final_model and test_dataset:
        accuracy_metrics, performance_metrics = measure_full_performance(final_model, test_dataset, config, device)
        save_model_and_report(final_model, accuracy_metrics, performance_metrics, config, start_time, 
                              fold_results=fold_results, mean_acc=mean_acc, std_acc=std_acc, best_val_acc=best_val_acc)
    else:
        logger.error("训练流程未能返回有效的最终模型或测试数据集，无法进行评估和保存。")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("脚本执行过程中发生未捕获的异常:")
        logger.exception(e)