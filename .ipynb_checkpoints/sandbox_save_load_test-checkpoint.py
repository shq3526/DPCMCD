# sandbox_save_load_test.py
# 描述: 一个用于快速验证QAT模型“转换后状态字典”保存与加载流程的沙箱脚本。
# 它将在几分钟内完成一次迷你的 转换->保存->加载->验证 闭环，以避免耗时的完整重训练。

import torch
import os
import json
import torch.nn as nn
from typing import *
from tqdm import tqdm
import random

# 必要的 Imports
from openprompt.plms import load_plm
from openprompt.prompts import ManualVerbalizer, PtuningTemplate
from openprompt import PromptForClassification
from openprompt.data_utils.text_classification_dataset import CnClickbaitProcessor
from openprompt.data_utils.utils import InputExample, InputFeatures
from openprompt.prompt_base import Template
from openprompt.plms.utils import TokenizerWrapper
from openprompt.utils import signature
from transformers import PreTrainedTokenizer
from torch.ao.quantization import QuantStub

# ==============================================================================
# 1. 核心依赖组件 (从最终版脚本中固化)
# ==============================================================================
class BertEncoderQuantWrapper(torch.nn.Module):
    def __init__(self, encoder_module):
        super().__init__()
        self.quant = QuantStub()
        self.encoder = encoder_module
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        q_hidden_states = self.quant(hidden_states)
        return self.encoder(q_hidden_states, attention_mask=attention_mask, **kwargs)

def create_prompt_model(config: object, scripts_path: str, device: str) -> Tuple:
    plm, tokenizer, model_config, WrapperClass = load_plm(config.model_type, config.model_name_or_path)
    template_path = os.path.join(scripts_path, "ptuning_template.txt")
    with open(template_path, 'r', encoding='utf-8') as f: template_text = f.readlines()[config.template_id].rstrip()
    mytemplate = PtuningTemplate(model=plm, tokenizer=tokenizer, text=template_text)
    verbalizer_path = os.path.join(scripts_path, config.verbalizer_file_name)
    with open(verbalizer_path, 'r', encoding='utf-8') as f: verbalizer_dict = json.load(f)
    class_names = list(verbalizer_dict['label_words'].keys())
    myverbalizer = ManualVerbalizer(tokenizer, classes=class_names)
    myverbalizer.label_words = verbalizer_dict['label_words']
    prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
    # 在QAT流程中，我们会在创建后立即应用Wrapper，所以这里先不进行prepare
    prompt_model.to(device)
    return prompt_model, tokenizer, mytemplate, WrapperClass

def preprocess_and_tokenize_dataset(dataset: List[InputExample], template: Template, tokenizer: PreTrainedTokenizer,
                                    wrapper_class, config: object) -> List[InputFeatures]:
    tokenizer_wrapper_init_keys = signature(wrapper_class.__init__).args
    prepare_kwargs = {"max_seq_length": config.max_seq_l, "truncate_method": "tail", "tokenizer": tokenizer}
    to_pass_kwargs = {key: prepare_kwargs[key] for key in prepare_kwargs if key in tokenizer_wrapper_init_keys}
    tokenizer_wrapper = wrapper_class(**to_pass_kwargs)
    tensor_dataset = []
    for example in tqdm(dataset, desc="Tokenizing Dataset"):
        wrapped_example = template.wrap_one_example(example)
        tokenized_example = tokenizer_wrapper.tokenize_one_example(wrapped_example, teacher_forcing=False)
        features_dict = {**tokenized_example, **wrapped_example[1]}
        if 'image_features' in signature(InputFeatures.__init__).args:
             features_dict['image_features'] = torch.zeros(512)
        tensor_dataset.append(InputFeatures(**features_dict).to_tensor())
    return tensor_dataset

# ==============================================================================
# 2. 主验证流程
# ==============================================================================
def main():
    # --- 第0步: 配置 ---
    FP32_CHECKPOINT_PATH = os.path.join("ckpts", "20250911-153849-toutiao622-Acc0.9677-F1s0.9357.ckpt")
    CONFIG_FILE_PATH = "./config/config.txt"
    TEMP_SAVE_PATH = "./ckpts/sandbox_validation_temp.pth"
    CALIBRATION_SIZE = 128  # 使用少量数据即可，目标是验证流程而非精度
    VALIDATION_BATCH_SIZE = 32

    print("="*30 + "\n沙箱验证流程启动\n" + "="*30)
    
    # --- 准备工作 ---
    torch.backends.quantized.engine = 'fbgemm'
    device = "cpu"
    with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f: config_dict = json.load(f)
    class Config:
        def __init__(self, **entries): self.__dict__.update(entries)
    config = Config(**config_dict)
    scripts_path = os.path.join("./scripts/TextClassification", config.prompt_name)

    # --- 第1步: 加载基础FP32模型 ---
    print("\n--- 步骤 1: 加载预训练的FP32模型 ---")
    fp32_model, tokenizer, template, wrapper_class = create_prompt_model(config, scripts_path, device)
    fp32_model.load_state_dict(torch.load(FP32_CHECKPOINT_PATH, map_location=device))
    print("--> FP32模型加载成功。")

    # --- 第2步: 准备少量数据 ---
    processor = CnClickbaitProcessor()
    data_path = os.path.join("./datasets/TextClassification", config.dataset_name)
    all_train_examples = processor.get_train_examples(data_path)
    random.shuffle(all_train_examples)
    calibration_examples = all_train_examples[:CALIBRATION_SIZE]
    validation_examples = all_train_examples[CALIBRATION_SIZE : CALIBRATION_SIZE + VALIDATION_BATCH_SIZE]
    
    tokenized_calib_set = preprocess_and_tokenize_dataset(calibration_examples, template, tokenizer, wrapper_class, config)
    tokenized_valid_set = preprocess_and_tokenize_dataset(validation_examples, template, tokenizer, wrapper_class, config)
    calib_dataloader = torch.utils.data.DataLoader(tokenized_calib_set, batch_size=VALIDATION_BATCH_SIZE, collate_fn=InputFeatures.collate_fct)
    valid_dataloader = torch.utils.data.DataLoader(tokenized_valid_set, batch_size=VALIDATION_BATCH_SIZE, collate_fn=InputFeatures.collate_fct)

    # --- 第3步: 在内存中生成“完美的”QAT模型 (模型A) ---
    print("\n--- 步骤 3: 在内存中生成已校准的QAT模型 (模型A) ---")
    model_a = fp32_model # 我们直接在加载的fp32模型上操作
    
    # 应用外科手术
    bert_model_a = model_a.plm.bert
    encoder_wrapper_a = BertEncoderQuantWrapper(bert_model_a.encoder)
    bert_model_a.encoder = encoder_wrapper_a
    
    # 应用量化配置
    qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
    for name, module in bert_model_a.named_modules():
        if isinstance(module, (torch.nn.Linear, QuantStub)): module.qconfig = qconfig
        elif isinstance(module, torch.nn.LayerNorm) and 'embeddings' not in name: module.qconfig = qconfig
    
    model_a.train()
    torch.quantization.prepare_qat(model_a, inplace=True)
    
    # 校准
    print("--> 正在对模型A进行快速校准...")
    model_a.eval()
    with torch.no_grad():
        for inputs in calib_dataloader:
            _ = model_a(inputs)
    
    # 转换
    torch.quantization.convert(model_a, inplace=True)
    print("--> 模型A (内存中) 已成功转换为INT8。")

    # --- 第4步: 保存并加载模型 (核心测试) ---
    print(f"\n--- 步骤 4: 保存模型A的状态字典到 {TEMP_SAVE_PATH} ---")
    torch.save(model_a.state_dict(), TEMP_SAVE_PATH)
    print("--> 保存成功。")

    print("\n--- 步骤 5: 重建空白INT8骨架并加载状态字典 (模型B) ---")
    # 创建一个全新的空白骨架
    model_b, _, _, _ = create_prompt_model(config, scripts_path, device)
    # 对这个空白骨架执行完全相同的量化准备和转换流程（无校准）
    bert_model_b = model_b.plm.bert
    encoder_wrapper_b = BertEncoderQuantWrapper(bert_model_b.encoder)
    bert_model_b.encoder = encoder_wrapper_b
    for name, module in bert_model_b.named_modules():
        if isinstance(module, (torch.nn.Linear, QuantStub)): module.qconfig = qconfig
        elif isinstance(module, torch.nn.LayerNorm) and 'embeddings' not in name: module.qconfig = qconfig
    model_b.train(); torch.quantization.prepare_qat(model_b, inplace=True); model_b.eval()
    torch.quantization.convert(model_b, inplace=True)
    
    # 加载状态字典
    model_b.load_state_dict(torch.load(TEMP_SAVE_PATH))
    print("--> 模型B (从文件加载) 已成功重建。")

    # --- 第6步: 验证一致性 ---
    print("\n--- 步骤 6: 验证模型A和模型B的预测一致性 ---")
    validation_batch = next(iter(valid_dataloader))
    
    model_a.eval()
    model_b.eval()
    
    with torch.no_grad():
        logits_a = model_a(validation_batch)
        logits_b = model_b(validation_batch)
    
    preds_a = torch.argmax(logits_a, dim=-1)
    preds_b = torch.argmax(logits_b, dim=-1)
    
    print(f"\n内存中模型A的预测: {preds_a.tolist()}")
    print(f"加载后模型B的预测: {preds_b.tolist()}")
    
    # 比较预测结果
    are_preds_equal = torch.equal(preds_a, preds_b)
    # 比较原始logits（由于浮点误差，使用allclose）
    are_logits_close = torch.allclose(logits_a, logits_b, atol=1e-4)

    print("\n" + "="*30 + "\n沙箱验证结果\n" + "="*30)
    if are_preds_equal:
        print("✅ **成功!** 加载后的模型预测结果与内存中的模型**完全一致**。")
        print("✅ **验证通过!** “转换后状态字典”的保存与加载方案是可靠的。")
    else:
        print("❌ **失败!** 加载后的模型预测结果与内存中的模型**不一致**。")
        print("❌ **验证失败!** 该保存/加载方案存在信息丢失。")

    print(f"Logits 是否接近: {are_logits_close}")

    # --- 第7步: 清理 ---
    os.remove(TEMP_SAVE_PATH)
    print(f"\n--- 清理临时文件 {TEMP_SAVE_PATH} ---")
    print("\n沙箱验证流程结束。")

if __name__ == "__main__":
    main()