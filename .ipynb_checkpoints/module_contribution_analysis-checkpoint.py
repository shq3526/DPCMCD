# standalone_module_analysis_final_v2.py

import os, sys, torch, torch.nn as nn, numpy as np, json, copy
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pandas as pd
from datetime import datetime
from typing import *
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
from torch.ao.nn.quantized import FloatFunctional


try:
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(current_script_path)
except NameError:
    project_root = os.path.abspath('.')
class BertEncoderQuantWrapper(nn.Module):
    def __init__(self, module): super().__init__(); self.quant = QuantStub(); self.encoder = module
    def forward(self, hidden_states, **kwargs): return self.encoder(self.quant(hidden_states), **kwargs)
class AttentionQuantWrapper(nn.Module):
    def __init__(self, module): super().__init__(); self.dequant = DeQuantStub(); self.attention = module; self.quant = QuantStub()
    def forward(self, *args, **kwargs):
        dq_args = (self.dequant(args[0]),) + args[1:]; outputs = self.attention(*dq_args, **kwargs)
        return (self.quant(outputs[0]),) + outputs[1:]
class BertOutputQuantWrapper(nn.Module):
    def __init__(self, module): super().__init__(); self.output = module; self.add = FloatFunctional()
    def forward(self, hidden_states, input_tensor):
        dense_out = self.output.dropout(self.output.dense(hidden_states))
        return self.output.LayerNorm(self.add.add(dense_out, input_tensor))
class ClsQuantWrapper(nn.Module):
    def __init__(self, module): super().__init__(); self.cls = module; self.dequant = DeQuantStub()
    def forward(self, *args, **kwargs): return self.dequant(self.cls(*args, **kwargs))
class MultiModalPromptModel(nn.Module):
    def __init__(self, plm, template: Template, verbalizer: ManualVerbalizer):
        super().__init__()
        self.prompt_model = PromptForClassification(plm, template, verbalizer, freeze_plm=False)
        num_classes = len(self.prompt_model.verbalizer.classes)
        self.classifier = nn.Sequential(
            nn.Linear(num_classes + 512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )
        self.verbalizer = self.prompt_model.verbalizer
    def forward(self, inputs: InputFeatures) -> torch.Tensor:
        text_logits = self.prompt_model(inputs)
        if hasattr(inputs, 'image_features') and inputs.image_features is not None:
            image_features = inputs.image_features.squeeze(1) if inputs.image_features.dim() == 3 else inputs.image_features
        else:
            image_features = torch.zeros(text_logits.size(0), 512, device=text_logits.device)
        combined_features = torch.cat([text_logits, image_features], dim=1)
        return self.classifier(combined_features)
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
def create_prompt_model(config, scripts_path, device, quantize=False):
    plm, tokenizer, model_config, WrapperClass = load_plm(config.model_type, config.model_name_or_path)
    template_path = os.path.join(scripts_path, "ptuning_template.txt")
    with open(template_path, 'r') as f: template_text = f.readlines()[config.template_id].rstrip()
    mytemplate = PtuningTemplate(model=plm, tokenizer=tokenizer, text=template_text)
    verbalizer_path = os.path.join(scripts_path, config.verbalizer_file_name)
    with open(verbalizer_path, 'r') as f: verbalizer_dict = json.load(f)
    class_names = list(verbalizer_dict['label_words'].keys())
    myverbalizer = ManualVerbalizer(tokenizer, classes=class_names, label_words=verbalizer_dict['label_words'])
    if hasattr(config, 'use_images') and config.use_images:
        prompt_model = MultiModalPromptModel(plm=plm, template=mytemplate, verbalizer=myverbalizer)
    else:
        prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
    if quantize:
        plm_model = prompt_model.prompt_model.plm if isinstance(prompt_model, MultiModalPromptModel) else prompt_model.plm
        plm_model.bert.encoder = BertEncoderQuantWrapper(plm_model.bert.encoder)
        for layer in plm_model.bert.encoder.encoder.layer:
            layer.attention = AttentionQuantWrapper(layer.attention)
            layer.output = BertOutputQuantWrapper(layer.output)
        plm_model.cls = ClsQuantWrapper(plm_model.cls)
    prompt_model.to(device)
    return prompt_model, tokenizer, mytemplate, WrapperClass

class AnalysisConfig:
    FP32_MODEL_CKPT = "20250912-152324-toutiao-FP32-Acc0.9734-F1s0.9439.ckpt"
    QAT_MODEL_CKPT = "20250916-200312-toutiao-QAT-Acc0.9084-F1s0.8370.ckpt"
    TRAIN_SCRIPT_CONFIG_FILE = "./config/config.txt"
    FP32_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    QAT_DEVICE = "cpu"
    BATCH_SIZE = 32
    # 使用FP32模型的原生路径，因为我们的目标是在FP32骨架上替换权重
    MODULES_TO_ANALYZE = {
        "Soft_Prompt": "prompt_model.template",
        "Word_Embeddings": "prompt_model.plm.bert.embeddings",
        "Layer_0_Attention": "prompt_model.plm.bert.encoder.layer.0.attention",
        "Layer_0_FFN": "prompt_model.plm.bert.encoder.layer.0.output",
        "Layer_6_Attention": "prompt_model.plm.bert.encoder.layer.6.attention",
        "Layer_6_FFN": "prompt_model.plm.bert.encoder.layer.6.output",
        "Layer_11_Attention": "prompt_model.plm.bert.encoder.layer.11.attention",
        "Layer_11_FFN": "prompt_model.plm.bert.encoder.layer.11.output",
        "Prediction_Head": "prompt_model.plm.cls",
        "Final_Classifier": "classifier"
    }
CONFIG = AnalysisConfig()

def evaluate_model_accuracy(model, dataloader, device, desc="Evaluating"):
    model.to(device).eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs in tqdm(dataloader, desc=desc, leave=False):
            inputs.to(device); logits = model(inputs)
            labels = inputs['label']
            all_labels.extend(labels.cpu().numpy()); all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
    return accuracy_score(all_labels, all_preds)

def get_unpacked_qat_state_dict(qat_model_converted):
    """
    最终修正版 v2：遍历一个转换后的QAT模型，通过调用 module.weight().dequantize()
    将其参数【完全反量化】为与FP32模型键名兼容的state_dict。
    """
    logger.info("--> 正在解包QAT模型的权重 (最终修正版 v2)...")
    
    temp_config = load_config_from_file(CONFIG.TRAIN_SCRIPT_CONFIG_FILE)
    temp_scripts_path = os.path.join(project_root, "scripts", "TextClassification", temp_config.prompt_name)
    fp32_skeleton, _, _, _ = create_prompt_model(temp_config, temp_scripts_path, "cpu", quantize=False)
    unpacked_state_dict = fp32_skeleton.state_dict()
    
    for name, module in qat_model_converted.named_modules():
        fp32_name = name.replace('.encoder.encoder', '.encoder').replace('.attention.attention','.attention').replace('.output.output','.output').replace('.cls.cls','.cls')

        # 如果是量化模块 (Linear 或 Conv)
        if hasattr(module, 'weight') and callable(module.weight) and hasattr(module.weight(), 'dequantize'):
            
            if fp32_name + '.weight' in unpacked_state_dict:
                unpacked_state_dict[fp32_name + '.weight'] = module.weight().dequantize()
            
            if hasattr(module, 'bias') and module.bias() is not None:
                if fp32_name + '.bias' in unpacked_state_dict:
                    # Bias 也是一个量化张量，同样需要反量化
                    unpacked_state_dict[fp32_name + '.bias'] = module.bias().dequantize()

        # 对于其他非量化叶子模块
        elif len(list(module.children())) == 0 and not isinstance(module, (torch.ao.quantization.QuantStub, torch.ao.quantization.DeQuantStub)):
             for param_name, param in module.named_parameters():
                full_key = fp32_name + '.' + param_name if fp32_name else param_name
                if full_key in unpacked_state_dict:
                    unpacked_state_dict[full_key] = param
                    
    logger.info("--> QAT模型权重解包完成。")
    return unpacked_state_dict

def run_weight_swap_analysis(base_model_fp32, fp32_state_dict, qat_unpacked_state_dict, module_prefix, dataloader, device):
    hybrid_state_dict = copy.deepcopy(fp32_state_dict)
    keys_to_replace = [k for k in qat_unpacked_state_dict if k.startswith(module_prefix)]
    if not keys_to_replace:
        logger.warning(f"模块 '{module_prefix}' 未找到可替换的权重。跳过分析...")
        return evaluate_model_accuracy(base_model_fp32, dataloader, device, desc=f"Eval-Skip-{module_prefix.split('.')[-1]}")

    logger.info(f"  > 正在为模块 '{module_prefix}' 替换 {len(keys_to_replace)} 个权重...")
    for key in keys_to_replace:
        hybrid_state_dict[key] = qat_unpacked_state_dict[key]
    
    hybrid_model = copy.deepcopy(base_model_fp32)
    hybrid_model.load_state_dict(hybrid_state_dict, strict=True) # 使用strict=True确保所有键都匹配
    
    desc = f"Eval-WeightSwap-{module_prefix.split('.')[-1]}"
    accuracy = evaluate_model_accuracy(hybrid_model, dataloader, device, desc=desc)
    del hybrid_model
    if device == "cuda": torch.cuda.empty_cache()
    return accuracy

def main():
    logger.info("=" * 30 + " 独立模块贡献度分析实验 (权重替换最终版) " + "=" * 30)
    train_config = load_config_from_file(CONFIG.TRAIN_SCRIPT_CONFIG_FILE)
    scripts_path = os.path.join(project_root, "scripts", "TextClassification", train_config.prompt_name)
    data_path = os.path.join(project_root, "datasets", "TextClassification", train_config.dataset_name)
    processor = CnClickbaitProcessor()
    test_dataset_raw = processor.get_test_examples(data_path)
    
    logger.info("\n--- 正在创建和加载模型 (最终修正流程) ---")
    model_fp32, tokenizer, mytemplate, WrapperClass = create_prompt_model(train_config, scripts_path, CONFIG.FP32_DEVICE, quantize=False)
    fp32_ckpt_path = os.path.join("./ckpts", CONFIG.FP32_MODEL_CKPT)
    model_fp32.load_state_dict(torch.load(fp32_ckpt_path, map_location=CONFIG.FP32_DEVICE))
    logger.info(f"✅ FP32模型加载成功: {CONFIG.FP32_MODEL_CKPT}")
    
    model_int8_skeleton, _, _, _ = create_prompt_model(train_config, scripts_path, CONFIG.QAT_DEVICE, quantize=True)
    model_int8_skeleton.train()
    qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
    if isinstance(model_int8_skeleton, MultiModalPromptModel): plm_model_ref = model_int8_skeleton.prompt_model
    else: plm_model_ref = model_int8_skeleton
    modules_to_skip = [m.attention for m in plm_model_ref.modules() if isinstance(m, AttentionQuantWrapper)]
    for name, module in model_int8_skeleton.named_modules():
        if any(module in parent.modules() for parent in modules_to_skip): continue
        if 'embeddings' in name and isinstance(module, nn.LayerNorm): continue
        if isinstance(module, (nn.Linear, QuantStub, DeQuantStub, FloatFunctional, nn.LayerNorm)):
            module.qconfig = qconfig
    torch.quantization.prepare_qat(model_int8_skeleton, inplace=True)
    model_int8_skeleton.eval()
    torch.quantization.convert(model_int8_skeleton, inplace=True)
    model_qat = model_int8_skeleton
    qat_ckpt_path = os.path.join("./ckpts", CONFIG.QAT_MODEL_CKPT)
    model_qat.load_state_dict(torch.load(qat_ckpt_path, map_location=CONFIG.QAT_DEVICE))
    logger.info(f"✅ INT8模型已正确加载: {CONFIG.QAT_MODEL_CKPT}")

    logger.info("\n--- 正在预处理测试数据 ---")
    default_image_features = torch.zeros(512, device=CONFIG.FP32_DEVICE)
    test_dataset_tokenized = preprocess_and_tokenize_dataset(test_dataset_raw, mytemplate, tokenizer, WrapperClass, train_config, default_image_features)
    test_dataloader = DataLoader(test_dataset_tokenized, batch_size=CONFIG.BATCH_SIZE, collate_fn=InputFeatures.collate_fct)

    logger.info("\n--- 正在评估基准模型精度 ---")
    results_data = []
    fp32_baseline_acc = evaluate_model_accuracy(model_fp32, test_dataloader, CONFIG.FP32_DEVICE, "Eval-FP32_Baseline")
    qat_baseline_acc = evaluate_model_accuracy(model_qat, test_dataloader, CONFIG.QAT_DEVICE, "Eval-INT8_Baseline")
    results_data.append({"Experiment": "FP32 Baseline", "Module": "N/A", "Accuracy": fp32_baseline_acc, "Delta (vs FP32)": 0.0})
    results_data.append({"Experiment": "INT8 Baseline", "Module": "N/A", "Accuracy": qat_baseline_acc, "Delta (vs FP32)": qat_baseline_acc - fp32_baseline_acc})

    logger.info("\n--- 开始量化影响分析 (权重替换法) ---")
    fp32_state_dict = model_fp32.state_dict()
    qat_unpacked_state_dict = get_unpacked_qat_state_dict(model_qat)
    
    if not (hasattr(train_config, 'use_images') and train_config.use_images):
        CONFIG.MODULES_TO_ANALYZE.pop("Final_Classifier", None)
        logger.info("--> 检测到非多模态配置，已从分析列表中移除 'Final_Classifier'。")
    
    for name, path in tqdm(CONFIG.MODULES_TO_ANALYZE.items(), desc="Analyzing Modules"):
        try:
            hybrid_acc = run_weight_swap_analysis(model_fp32, fp32_state_dict, qat_unpacked_state_dict, path, test_dataloader, CONFIG.FP32_DEVICE)
            results_data.append({"Experiment": f"FP32 with INT8 {name}", "Module": name, "Accuracy": hybrid_acc, "Delta (vs FP32)": hybrid_acc - fp32_baseline_acc})
        except Exception as e:
            logger.error(f"分析模块 {name} ({path}) 时出错: {e}")

    logger.info("\n--- 实验完成，生成最终报告 ---")
    df = pd.DataFrame(results_data)
    df["Delta (vs FP32)"] = df["Delta (vs FP32)"].round(5)
    df_sorted = df.sort_values(by="Delta (vs FP32)", ascending=True)
    report = f"""
================================================================================
                        模块贡献度与量化影响分析报告 (权重替换最终版)
================================================================================
基准模型:
  - FP32模型: {CONFIG.FP32_MODEL_CKPT} (精度: {fp32_baseline_acc:.4f}, 设备: {CONFIG.FP32_DEVICE})
  - INT8模型: {CONFIG.QAT_MODEL_CKPT} (精度: {qat_baseline_acc:.4f}, 设备: {CONFIG.QAT_DEVICE})
  - 总精度下降: {fp32_baseline_acc - qat_baseline_acc:.4f}

实验方法:
  在一个纯FP32模型骨架上，将单个模块的权重替换为从INT8模型中解包出的对应FP32权重，
  并评估其精度。"Delta"值越小（负得越多），表示该模块被量化后对整体精度的负面影响越大。
--------------------------------------------------------------------------------
{df_sorted.to_string(index=False)}
================================================================================
"""
    print(report)
    report_filename = f"module_analysis_{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f: f.write(report)
    logger.info(f"✅ 详细报告已保存至: {report_filename}")

if __name__ == "__main__":
    main()