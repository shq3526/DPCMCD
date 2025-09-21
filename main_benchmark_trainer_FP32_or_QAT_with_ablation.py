# ==================================================================================================
# Unified Training and Evaluation Script (FP32 / QAT)
#
# This is the main entry point for the Quantization route of the project.
# It is driven by 'config_main.txt' and supports:
#   - Standard FP32 training.
#   - Quantization-Aware Training (QAT) with various ablation settings.
#   - Multi-modal (Text+Image) and Text-only configurations.
#   - K-fold cross-validation or few-shot training splits.
#   - Comprehensive performance benchmarking (accuracy, latency, throughput, memory).
#
# Author: [Haoqian Song, Haoran Yin, Fuwen Zhao]
# Date: September 18, 2025
# ==================================================================================================

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

# --- Initialize Project Path ---
try:
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(current_script_path)
except NameError:
    project_root = os.path.abspath('.')
sys.path.insert(0, project_root)
print(f"--- Project root added to sys.path: {project_root} ---")

# --- Import Required Libraries ---
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

# --- Global Settings ---
csv.field_size_limit(512 * 1024)

# ==================================================================================================
#                                 Quantization Wrapper Components
# These wrappers are inserted into the model to mark where quantization/dequantization should occur.
# ==================================================================================================
class BertEncoderQuantWrapper(nn.Module):
    def __init__(self, module): super().__init__(); self.quant = QuantStub(); self.encoder = module
    def forward(self, hidden_states, **kwargs): return self.encoder(self.quant(hidden_states), **kwargs)

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
    def forward(self, *args, **kwargs): return self.dequant(self.cls(*args, **kwargs))

# ==================================================================================================
#                       Multi-Modal Model Definition (Image & Text)
# ==================================================================================================
class MultiModalPromptModel(nn.Module):
    """A multi-modal model combining a text prompt model with visual features."""
    def __init__(self, plm, template: Template, verbalizer: ManualVerbalizer):
        super().__init__()
        self.prompt_model = PromptForClassification(plm, template, verbalizer, freeze_plm=False)
        num_classes = len(self.prompt_model.verbalizer.classes)
        clip_feature_size = 512  # CLIP-base feature dimension
        classifier_input_dim = num_classes + clip_feature_size
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.verbalizer = self.prompt_model.verbalizer

    @property
    def device(self): return next(self.parameters()).device

    def forward(self, inputs: InputFeatures) -> torch.Tensor:
        text_logits = self.prompt_model(inputs)
        if hasattr(inputs, 'image_features') and inputs.image_features is not None:
            image_features = inputs.image_features
            if image_features.dim() == 3 and image_features.size(1) == 1:
                image_features = image_features.squeeze(1)
        else: # Fallback to zero tensor if no image features are provided
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
#                                        Helper Functions
# ==================================================================================================
def load_config_from_file(config_path: str) -> object:
    """Loads configuration from a JSON-formatted .txt file."""
    print(f"--> Loading configuration from {config_path}...")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    class Config:
        def __init__(self, **entries):
            entries = {k: v for k, v in entries.items() if not k.startswith('//')}
            self.__dict__.update(entries)
    return Config(**config_dict)

def read_readme(path: str, component_name: str) -> None:
    """Reads and prints the readme.txt file from a given path."""
    readme_path = os.path.join(path, "readme.txt")
    print("-" * 60)
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            print(f"✔️  Reading {component_name} info (readme.txt):\n" + f.read())
    else:
        print(f"⚠️  Warning: Description file (readme.txt) not found for {component_name} in {path}.")
    print("-" * 60)

def load_image_features(img_path_str: Optional[str], clip_processor: CLIPProcessor, clip_model: CLIPModel,
                        device: str, project_root_path: str) -> Optional[torch.Tensor]:
    """Loads a single image and extracts features using CLIP."""
    if not img_path_str or str(img_path_str).lower() == 'nan': return None
    full_path = os.path.join(project_root_path, img_path_str)
    if not os.path.exists(full_path): return None
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
    """Performs one-shot wrapping and tokenizing of the entire dataset."""
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

def create_prompt_model(config: object, scripts_path: str, device: str) -> Tuple:
    """Creates the model instance and inserts QAT wrappers if needed."""
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
        logger.info("--> Creating MultiModalPromptModel...")
        prompt_model = MultiModalPromptModel(plm=plm, template=mytemplate, verbalizer=myverbalizer)
    else:
        logger.info("--> Creating standard text-only PromptForClassification model...")
        prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)

    # Insert wrappers only if a full-module quantization strategy is selected
    if config.qat_ablation_type in ["per_channel_full", "per_tensor_full"]:
        logger.info(f"--> Preparing model backbone for QAT strategy '{config.qat_ablation_type}' (inserting all wrappers)...")
        plm_model = prompt_model.prompt_model.plm if isinstance(prompt_model, MultiModalPromptModel) else prompt_model.plm
        plm_model.bert.encoder = BertEncoderQuantWrapper(plm_model.bert.encoder)
        for layer in plm_model.bert.encoder.encoder.layer:
            layer.attention = AttentionQuantWrapper(layer.attention)
            layer.output = BertOutputQuantWrapper(layer.output)
        plm_model.cls = ClsQuantWrapper(plm_model.cls)
        logger.info("--> Model backbone preparation complete.")

    prompt_model.to(device)
    return prompt_model, tokenizer, mytemplate, WrapperClass

# ==================================================================================================
#                                 QAT Preparation Function
# ==================================================================================================
def prepare_model_for_qat(model, experiment_type: str):
    """Prepares the model for QAT based on the specified ablation type."""
    if experiment_type == "none":
        return  # Do nothing for FP32 mode

    qconfig = None
    if experiment_type == "per_tensor_linear_only":
        logger.info("--> [QAT Prep] Per-Tensor + Linear Layers Only...")
        qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
        for module in model.modules():
            module.qconfig = qconfig if isinstance(module, torch.nn.Linear) else None
    elif experiment_type == "per_channel_full":
        logger.info("--> [QAT Prep] Per-Channel + Full Modules...")
        from torch.ao.quantization.observer import PerChannelMinMaxObserver
        base_qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
        per_channel_weight_observer = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
        qconfig = base_qconfig._replace(weight=per_channel_weight_observer)
        if not any(isinstance(m, BertEncoderQuantWrapper) for m in model.modules()):
            raise RuntimeError("Per-Channel Full quantization requires QuantWrappers. Check create_prompt_model logic.")
        modules_to_skip = [m.attention for m in model.modules() if isinstance(m, AttentionQuantWrapper)]
        for name, module in model.named_modules():
            if any(module in parent.modules() for parent in modules_to_skip): continue
            if 'embeddings' in name and isinstance(module, nn.LayerNorm): continue
            if isinstance(module, (nn.Linear, QuantStub, DeQuantStub, FloatFunctional, nn.LayerNorm)):
                module.qconfig = qconfig
    elif experiment_type == "per_tensor_full":
        logger.info("--> [QAT Prep] Per-Tensor + Full Modules (Default)...")
        qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
        if not any(isinstance(m, BertEncoderQuantWrapper) for m in model.modules()):
            raise RuntimeError("Per-Tensor Full quantization requires QuantWrappers. Check create_prompt_model logic.")
        modules_to_skip = [m.attention for m in model.modules() if isinstance(m, AttentionQuantWrapper)]
        for name, module in model.named_modules():
            if any(module in parent.modules() for parent in modules_to_skip): continue
            if 'embeddings' in name and isinstance(module, nn.LayerNorm): continue
            if isinstance(module, (nn.Linear, QuantStub, DeQuantStub, FloatFunctional, nn.LayerNorm)):
                module.qconfig = qconfig
    else:
        raise ValueError(f"Unknown QAT experiment type: {experiment_type}")

    model.train()
    torch.quantization.prepare_qat(model, inplace=True)
    logger.info("--> QAT model is now prepared for the current experiment.")

# ==================================================================================================
#                                      Evaluation & Saving Functions
# ==================================================================================================
def evaluate(p_model, dataloader, desc="Evaluating"):
    """Evaluates model performance (for inter-epoch validation)."""
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
    """Calculates model size and number of parameters."""
    size_mb = os.path.getsize(state_dict_path) / (1024 ** 2) if state_dict_path and os.path.exists(state_dict_path) else 0
    num_params = sum(p.numel() for p in model.parameters())
    return {"size_mb": size_mb, "num_params": num_params}

def measure_full_performance(model, test_dataset, config, device):
    """Performs a complete accuracy and performance evaluation with detailed latency analysis."""
    logger.info("\n" + "#" * 20 + " Starting Final Model Evaluation " + "#" * 20)
    model.eval()

    accuracy_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=InputFeatures.collate_fct, shuffle=False)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs in tqdm(accuracy_dataloader, desc="Evaluating final accuracy"):
            inputs.to(device)
            logits = model(inputs)
            labels = inputs['label']
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
    accuracy_metrics = classification_report(all_labels, all_preds, output_dict=True, zero_division=0, digits=4, target_names=model.verbalizer.classes)
    accuracy_metrics['overall_accuracy'] = accuracy_score(all_labels, all_preds)

    WARMUP_STEPS, MEASURE_SAMPLES = 20, min(len(test_dataset), 100)
    latency_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=InputFeatures.collate_fct, shuffle=False)

    logger.info(f"--> [Performance] Warming up ({WARMUP_STEPS} samples)...")
    with torch.no_grad():
        for i, inputs in enumerate(latency_dataloader):
            if i >= WARMUP_STEPS: break
            inputs.to(device); _ = model(inputs)

    e2e_latencies, fwd_latencies = [], []
    logger.info(f"--> [Performance] Measuring latency (batch_size=1, on {MEASURE_SAMPLES} samples)...")
    with torch.no_grad():
        for i, inputs in enumerate(tqdm(latency_dataloader, total=MEASURE_SAMPLES, desc="Measuring latency")):
            if i >= MEASURE_SAMPLES: break
            start_e2e = time.perf_counter()
            inputs.to(device)
            if device == 'cuda': torch.cuda.synchronize()
            start_fwd = time.perf_counter()
            _ = model(inputs)
            if device == 'cuda': torch.cuda.synchronize()
            end_fwd = time.perf_counter()
            end_e2e = time.perf_counter()
            e2e_latencies.append((end_e2e - start_e2e) * 1000)
            fwd_latencies.append((end_fwd - start_fwd) * 1000)

    total_time, total_samples = 0, 0
    logger.info(f"--> [Performance] Measuring throughput (batch_size={config.batch_size})...")
    with torch.no_grad():
        for inputs in tqdm(accuracy_dataloader, desc="Measuring throughput"):
            inputs.to(device)
            if device == 'cuda': torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(inputs)
            if device == 'cuda': torch.cuda.synchronize()
            total_time += time.perf_counter() - start
            total_samples += len(inputs['label'])

    def get_latency_stats(latencies_ms: List[float]) -> Dict:
        if not latencies_ms: return {'avg': 0, 'p50': 0, 'p90': 0, 'p95': 0, 'p99': 0}
        arr = np.array(latencies_ms)
        return {'avg': np.mean(arr), 'p50': np.percentile(arr, 50), 'p90': np.percentile(arr, 90), 'p95': np.percentile(arr, 95), 'p99': np.percentile(arr, 99)}

    performance_metrics = {
        "e2e_latency_ms": get_latency_stats(e2e_latencies), "fwd_latency_ms": get_latency_stats(fwd_latencies),
        "throughput_sps": total_samples / total_time if total_time > 0 else 0
    }
    if device == 'cuda':
        performance_metrics["gpu_peak_mem_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return accuracy_metrics, performance_metrics

def save_model_and_report(final_model, accuracy_metrics, performance_metrics, config, start_time, fold_results=None, mean_acc=0, std_acc=0, best_val_acc=0):
    """Saves the final model state_dict and generates a detailed report."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ablation_type = config.qat_ablation_type
    model_type_str = "-FP32" if ablation_type == "none" else f"-QAT_{ablation_type}"
    if hasattr(config, 'use_images') and config.use_images: model_type_str += "_MultiModal"
    base_filename = f"{timestamp}-{config.dataset_name}{model_type_str}-Acc{accuracy_metrics['overall_accuracy']:.4f}-F1s{accuracy_metrics['macro avg']['f1-score']:.4f}"

    final_model_path = None
    if final_model:
        if ablation_type != "none":
            logger.info("\n" + "=" * 20 + " Final QAT Model Conversion " + "=" * 20)
            final_model.cpu().eval()
            torch.quantization.convert(final_model, inplace=True)
            logger.info("--> Model successfully converted to final INT8 format.")
        final_model_path = os.path.join("./ckpts", f"{base_filename}.ckpt")
        torch.save(final_model.to('cpu').state_dict(), final_model_path)
        logger.info(f"\n--> ✅ Model state_dict saved to: {final_model_path}")

    model_stats = get_model_stats(final_model, final_model_path)
    report_file_path = os.path.join("./ckpts", f"{base_filename}.txt")
    duration_minutes = (time.time() - start_time) / 60

    report_content = [
        "=" * 25 + " Final Training & Evaluation Report " + "=" * 25,
        f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Runtime: {duration_minutes:.2f} minutes",
        f"Model File: {os.path.basename(final_model_path) if final_model_path else 'Not Saved'}",
        "\n" + "-" * 15 + " 1. Experiment Configuration " + "-" * 15,
        json.dumps({k: v for k, v in config.__dict__.items() if not k.startswith('//')}, indent=4, ensure_ascii=False),
    ]
    if config.use_k_fold and fold_results is not None:
        report_content.extend(["\n" + "-" * 15 + " 2. K-Fold Cross-Validation Results " + "-" * 15,
                               f"Best validation accuracy per fold: {[round(r, 4) for r in fold_results]}",
                               f"Average validation accuracy: {mean_acc:.4f} (±{std_acc:.4f})"])
    else:
        report_content.extend(["\n" + "-" * 15 + " 2. Validation Set Performance " + "-" * 15, f"Best validation accuracy: {best_val_acc:.4f}"])

    report_content.extend(["\n" + "-" * 15 + " 3. Final Test Set Accuracy Metrics " + "-" * 15,
                           f"Overall Accuracy: {accuracy_metrics['overall_accuracy']:.4f}"])
    class_names_from_report = final_model.verbalizer.classes
    report_content.append(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    for class_name in class_names_from_report:
        if class_name in accuracy_metrics:
            metrics = accuracy_metrics[class_name]
            report_content.append(f"{class_name:<15} {metrics['precision']:.4f}     {metrics['recall']:.4f}     {metrics['f1-score']:.4f}")
    report_content.append("-" * 50)
    macro_avg = accuracy_metrics['macro avg']
    report_content.append(f"{'Macro Avg':<15} {macro_avg['precision']:.4f}     {macro_avg['recall']:.4f}     {macro_avg['f1-score']:.4f}")

    report_content.extend(["\n" + "-" * 15 + " 4. Final Model Performance Metrics " + "-" * 15, f"Evaluation Device: {final_model.device}",
                           f"Model Size on Disk: {model_stats['size_mb']:.2f} MB",
                           f"Number of Parameters: {model_stats['num_params'] / 1e6:.2f} M",
                           f"Throughput: {performance_metrics['throughput_sps']:.2f} samples/s (measured at batch_size={config.batch_size})"])
    if "gpu_peak_mem_mb" in performance_metrics: report_content.append(f"GPU Peak Memory: {performance_metrics['gpu_peak_mem_mb']:.1f} MB")
    report_content.append("\nDetailed Latency Analysis (ms/sample, batch_size=1):")
    report_content.append(f"{'Metric':<18} | {'Average':<10} | {'P50':<10} | {'P90':<10} | {'P95':<10} | {'P99':<10}")
    report_content.append("-" * 80)
    e2e, fwd = performance_metrics['e2e_latency_ms'], performance_metrics['fwd_latency_ms']
    report_content.append(f"{'End-to-End':<18} | {e2e['avg']:<10.2f} | {e2e['p50']:<10.2f} | {e2e['p90']:<10.2f} | {e2e['p95']:<10.2f} | {e2e['p99']:<10.2f}")
    report_content.append(f"{'Model Forward':<18} | {fwd['avg']:<10.2f} | {fwd['p50']:<10.2f} | {fwd['p90']:<10.2f} | {fwd['p95']:<10.2f} | {fwd['p99']:<10.2f}")
    report_content.append("=" * 80)

    final_report = "\n".join(report_content)
    with open(report_file_path, 'w', encoding='utf-8') as f: f.write(final_report)
    logger.info(f"--> ✅ Detailed report saved to: {report_file_path}"); print("\n" + final_report)

# ==================================================================================================
#                                      Training Workflows
# ==================================================================================================
def run_k_fold_training(config, dataset, scripts_path, device, default_image_features):
    full_dataset, labels = dataset['train'] + dataset['test'], [ex.label for ex in dataset['train'] + dataset['test']]
    train_val_pool, final_test_set = train_test_split(full_dataset, test_size=config.k_fold_test_split_size, random_state=config.seed, stratify=labels)
    skf = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=config.seed)
    fold_results, train_val_pool_np, train_val_pool_labels = [], np.array(train_val_pool), [ex.label for ex in train_val_pool]

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_pool_np, train_val_pool_labels)):
        logger.info(f"\n========== Starting Fold {fold + 1}/{config.k_folds} ==========")
        prompt_model, tokenizer, mytemplate, WrapperClass = create_prompt_model(config, scripts_path, device)
        prepare_model_for_qat(prompt_model, config.qat_ablation_type)
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
            val_metrics = evaluate(prompt_model, val_dataloader_fold, desc=f"Fold {fold + 1} Validating")
            if val_metrics['acc'] > best_val_acc_fold: best_val_acc_fold = val_metrics['acc']
        logger.info(f"--> Fold {fold + 1} finished. Best validation accuracy: {best_val_acc_fold:.4f}"); fold_results.append(best_val_acc_fold)

    mean_acc, std_acc = np.mean(fold_results), np.std(fold_results)
    logger.info(f"\n--> {config.k_folds}-Fold cross-validation average accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    logger.info("\n" + "=" * 20 + " Training final model on all data " + "=" * 20)
    final_model, tokenizer, mytemplate, WrapperClass = create_prompt_model(config, scripts_path, device)
    prepare_model_for_qat(final_model, config.qat_ablation_type)
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
    prompt_model, tokenizer, mytemplate, WrapperClass = create_prompt_model(config, scripts_path, device)
    prepare_model_for_qat(prompt_model, config.qat_ablation_type)
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
        print(f"--> Epoch {epoch + 1} complete. Val Acc: {val_metrics['acc']:.4f}, Best Val Acc: {best_val_acc:.4f}", flush=True)

    logger.info(f"--> Loading best validation model (Acc: {best_val_acc:.4f})")
    prompt_model.load_state_dict(torch.load(temp_ckpt_path))
    if os.path.exists(temp_ckpt_path): os.remove(temp_ckpt_path)
    return prompt_model, tokenized_test_set, best_val_acc

# ==================================================================================================
#                                      Program Entry Point
# ==================================================================================================
def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Unified training and evaluation script (config-driven, with QAT ablations).")
    parser.add_argument("--config_file", type=str, default="./config/config_main.txt", help="Path to the configuration file.")
    args = parser.parse_args()
    config = load_config_from_file(args.config_file)
    set_seed(config.seed)

    if not hasattr(config, 'use_qat') or not config.use_qat:
        config.qat_ablation_type = "none"
    elif not hasattr(config, 'qat_ablation_type'):
        logger.warning("Config 'use_qat' is true, but 'qat_ablation_type' is not defined. Defaulting to 'per_tensor_full'.")
        config.qat_ablation_type = "per_tensor_full"

    is_qat = config.qat_ablation_type != "none"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"--> {'CUDA device detected, training on GPU.' if device == 'cuda' else 'No CUDA device detected, training and evaluating on CPU.'}")

    if is_qat:
        torch.backends.quantized.engine = "fbgemm"
        logger.info(f"--> QAT enabled (Strategy: {config.qat_ablation_type}). Quantization backend set to 'fbgemm'.")
    else:
        logger.info("--> This run is for standard FP32 training.")

    os.makedirs("./ckpts", exist_ok=True)

    print("\n" + "=" * 20 + " Experiment Configuration Check " + "=" * 20)
    data_path = os.path.join(project_root, "datasets", "TextClassification", config.dataset_name)
    scripts_path = os.path.join(project_root, "scripts", "TextClassification", config.prompt_name)
    read_readme(data_path, f"Dataset ({config.dataset_name})"); read_readme(scripts_path, f"Prompt ({config.prompt_name})")
    print("=" * 66 + "\n")

    processor = CnClickbaitProcessor()
    dataset = {'train': processor.get_train_examples(data_path), 'test': processor.get_test_examples(data_path)}
    default_image_features = torch.zeros(512, device=device)

    if hasattr(config, 'use_images') and config.use_images:
        logger.info("--> Multi-modal image processing enabled. Performing one-time feature extraction...")
        try:
            clip_model_path = os.path.join(project_root, "model", "clip-vit-base-patch32")
            if not os.path.exists(clip_model_path):
                raise FileNotFoundError("CLIP model not found at local path 'model/clip-vit-base-patch32'.")
            clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
            clip_model = CLIPModel.from_pretrained(clip_model_path).to(device)
            all_examples = dataset['train'] + dataset['test']
            for example in tqdm(all_examples, desc="Extracting all image features"):
                img_path_str = example.meta if isinstance(example.meta, str) else (example.meta.get('img_path') if isinstance(example.meta, dict) else None)
                if isinstance(example.meta, str): example.meta = {}
                features = load_image_features(img_path_str, clip_processor, clip_model, device, project_root)
                example.meta['image_features'] = features.squeeze(0) if features is not None else default_image_features
            logger.info("--> All image features extracted.")
        except Exception as e:
            logger.error(f"Failed to load CLIP model or extract features. Disabling image processing. Error: {e}")
            config.use_images = False

    final_model, test_dataset, fold_results, mean_acc, std_acc, best_val_acc = None, None, None, 0, 0, 0
    if config.use_k_fold:
        final_model, test_dataset, fold_results, mean_acc, std_acc = run_k_fold_training(config, dataset, scripts_path, device, default_image_features)
    else:
        final_model, test_dataset, best_val_acc = run_single_shot_training(config, dataset, scripts_path, device, default_image_features)

    if final_model and test_dataset:
        accuracy_metrics, performance_metrics = measure_full_performance(final_model, test_dataset, config, device)
        save_model_and_report(final_model, accuracy_metrics, performance_metrics, config, start_time, fold_results=fold_results, mean_acc=mean_acc, std_acc=std_acc, best_val_acc=best_val_acc)
    else:
        logger.error("Training workflow did not return a valid final model or test set. Evaluation and saving are skipped.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("An uncaught exception occurred during script execution:")
        logger.exception(e)