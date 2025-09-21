# -*- coding: utf-8 -*-

"""
@Project: 多模态标题党检测模型 (Multimodal Clickbait Detection Model)
@File: fewshot-config.py
@Author: Your Name
@Date: 2025-08-31

@Description:
本脚本是项目的主训练程序，具备高度的灵活性和可配置性。主要功能包括：
1.  通过外部 config.txt 文件驱动，实现代码与配置分离。
2.  支持两种训练模式：K-折交叉验证（用于严谨评估）和少样本单次训练（用于快速实验）。
3.  支持多模态输入，可选择性地融合文本和图像特征进行训练。
4.  采用一次性预处理机制，在训练开始前完成所有图像特征提取和文本分词，大幅提升训练效率。
5.  自动化的结果保存机制，将模型权重和详细的实验报告以时间戳和关键指标命名，便于追溯和比较。
"""
# --- 在你 fewshot-config-final.py 脚本的最开头 ---

print("--- 脚本启动，开始逐行诊断 ---")

try:
    print("步骤 1: 导入 torch")
    import torch
    print("✅ torch 导入成功")

    print("步骤 2: 导入 transformers")
    import transformers
    print("✅ transformers 导入成功")

    print("步骤 3: 导入 torch_geometric")
    import torch_geometric
    print("✅ torch_geometric 导入成功")
    
    # 这是一个基础的健全性检查，看看PyG的基本功能是否正常
    print("步骤 4: 尝试创建一个最简单的 torch_geometric.data.Data 对象")
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    x = torch.randn(3, 16) # 3个节点, 每个节点16维特征
    data = torch_geometric.data.Data(x=x, edge_index=edge_index)
    print("✅ torch_geometric.data.Data 对象创建成功")

    # ===================================================================
    # 在这里，请继续用 print 语句包裹住你脚本里的每一个关键操作
    # 比如：
    # print("步骤 5: 准备加载我自己的数据集...")
    # my_dataset = load_my_dataset(...)
    # print("✅ 数据集加载成功")
    #
    # print("步骤 6: 准备创建GNN模型...")
    # my_model = create_my_model(...)
    # print("✅ GNN模型创建成功")
    # ===================================================================

except Exception as e:
    # 注意：Segmentation fault 是底层错误，通常不会被 try...except 捕获，
    # 但这可以帮我们排除所有Python层面的错误。
    print(f"❌ 脚本在某个阶段出现了Python级别的错误: {e}")


# --- 你脚本的其余主要逻辑代码放在这里 ---
# ...
# ...
import os
import tqdm
import torch
import textwrap
import time  # 用于计算训练时长
from torch.cuda.amp import autocast, GradScaler
import sys

# --- 初始化项目路径 ---
# 获取当前脚本文件 (fewshot-config.py) 的绝对路径
current_script_path = os.path.abspath(__file__)
# 推断项目根目录 (即脚本所在目录)
project_root = os.path.dirname(current_script_path)
# 将项目根目录添加到 Python 的模块搜索路径中，确保可以导入项目内的其他模块
sys.path.insert(0, project_root)
print(f"--- 项目根目录已添加至 sys.path: {project_root} ---")

# --- 导入所需库 ---
import random
import numpy as np
import json
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split
from datetime import datetime
from typing import *
import csv

# --- OpenPrompt & Transformers 相关导入 ---
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

# --- 多模态图片处理依赖 ---
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from openprompt.prompt_base import Template
from openprompt.plms.utils import TokenizerWrapper
from openprompt.utils import signature

# --- 全局设置 ---
# 提高csv字段大小限制以处理可能包含超长文本的数据集
csv.field_size_limit(512 * 1024)


# ==================================================================================================
#                                        辅助函数定义
# ==================================================================================================

def load_config_from_file(config_path: str) -> object:
    """
    从 JSON 格式的 .txt 文件加载配置。
    Args:
        config_path (str): 配置文件的路径。
    Returns:
        object: 一个包含所有配置项作为属性的类实例。
    """
    print(f"--> 正在从 {config_path} 加载配置...")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    class Config:
        def __init__(self, **entries):
            entries = {k: v for k, v in entries.items() if not k.startswith('//')}
            self.__dict__.update(entries)

    return Config(**config_dict)


def read_readme(path: str, component_name: str) -> None:
    """
    读取并打印指定路径下的 readme.txt 文件。
    Args:
        path (str): 要检查的文件夹路径。
        component_name (str): 组件的名称（如 "数据集", "提示"），用于打印信息。
    """
    readme_path = os.path.join(path, "readme.txt")
    print("-" * 60)
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            print(f"✔️  读取到 {component_name} 信息 (readme.txt):")
            print(f.read())
    else:
        print(f"⚠️  警告：在 {path} 中未找到 {component_name} 的说明文件 (readme.txt)。")
    print("-" * 60)


def load_image_features(image_path_string: str, processor, model, device: str, base_path: str) -> Optional[
    torch.Tensor]:
    """
    加载并处理单个样本的所有图片，提取并合并其CLIP特征。
    """
    if not image_path_string or not isinstance(image_path_string, str) or image_path_string == '[]':
        return None

    image_paths = [p.strip() for p in image_path_string.strip('[]').replace("'", "").split('|')]
    image_features_list = []

    for relative_path in image_paths:
        try:
            full_path = os.path.join(base_path, relative_path.replace("./", ""))
            image = Image.open(full_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                image_features_list.append(image_features)
        except Exception as e:
            logger.error(f"加载或处理图片失败: {relative_path}, 错误: {e}")
            continue

    if not image_features_list:
        return None

    combined_image_features = torch.cat(image_features_list, dim=0)
    merged_features = torch.mean(combined_image_features, dim=0, keepdim=True)
    return merged_features


def preprocess_and_tokenize_dataset(dataset: List[InputExample], template: Template, tokenizer: PreTrainedTokenizer,
                                    wrapper_class: Type[TokenizerWrapper], config: object,
                                    default_image_features: torch.Tensor) -> List[InputFeatures]:
    """
    一次性完成数据集的 wrapping 和 tokenizing，将 InputExample 列表转换为 InputFeatures 列表。
    """
    tokenizer_wrapper_init_keys = signature(wrapper_class.__init__).args
    prepare_kwargs = {
        "max_seq_length": config.max_seq_l,
        "truncate_method": "tail",
        "tokenizer": tokenizer,
    }
    to_pass_kwargs = {key: prepare_kwargs[key] for key in prepare_kwargs if key in tokenizer_wrapper_init_keys}
    tokenizer_wrapper = wrapper_class(**to_pass_kwargs)

    tensor_dataset = []
    for example in tqdm.tqdm(dataset, desc="Tokenizing Dataset"):
        wrapped_example = template.wrap_one_example(example)
        tokenized_example = tokenizer_wrapper.tokenize_one_example(wrapped_example, teacher_forcing=False)

        features_dict = {**tokenized_example, **wrapped_example[1]}

        if config.use_images:
            features_dict['image_features'] = example.meta.get('image_features', default_image_features)

        input_features = InputFeatures(**features_dict).to_tensor()
        tensor_dataset.append(input_features)

    return tensor_dataset


# ==================================================================================================
#                                      程序主逻辑 (Main Logic)
# ==================================================================================================
def main():
    # --- 1. 初始化与配置加载 ---
    start_time = time.time()
    parser = argparse.ArgumentParser(description="多模态标题党检测模型训练脚本")
    parser.add_argument("--config_file", type=str, default="./config/config.txt", help="指向 JSON 格式的配置文件路径")
    args = parser.parse_args()

    config = load_config_from_file(args.config_file)
    set_seed(config.seed)

    # --- 2. 动态路径构建与信息展示 ---
    print("\n" + "=" * 20 + " 实验配置检查 " + "=" * 20)
    data_path = os.path.join(project_root, "datasets", "TextClassification", config.dataset_name)
    scripts_path = os.path.join(project_root, "scripts", "TextClassification", config.prompt_name)

    read_readme(data_path, f"数据集 ({config.dataset_name})")
    read_readme(scripts_path, f"提示 ({config.prompt_name})")
    print("=" * 66 + "\n")

    os.makedirs("./ckpts", exist_ok=True)

    # --- 3. 数据加载 ---
    logger.info(f"--> 正在从 {data_path} 加载数据集...")
    processor = CnClickbaitProcessor()
    dataset = {
        'train': processor.get_train_examples(data_path),
        'test': processor.get_test_examples(data_path)
    }
    logger.info(f"--> 数据集加载完成，共 {len(dataset['train']) + len(dataset['test'])} 条样本。")
    class_labels = processor.get_labels()

    # --- 4. 多模态特征预处理 (一次性) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    default_image_features = torch.zeros(512, device=device)

    if config.use_images:
        logger.info("--> 启用多模态图片处理。正在进行一次性图像特征提取...")
        try:
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            logger.info("--> CLIP模型加载成功。开始提取所有样本的图像特征...")

            all_examples = dataset['train'] + dataset['test']

            for example in tqdm.tqdm(all_examples, desc="提取所有图像特征"):
                image_path_string = None
                if isinstance(example.meta, str):
                    image_path_string = example.meta

                extracted_features = load_image_features(
                    image_path_string=image_path_string,
                    processor=clip_processor,
                    model=clip_model,
                    device=device,
                    base_path=data_path
                )

                current_meta = {}
                if isinstance(example.meta, dict):
                    current_meta = example.meta

                current_meta['image_features'] = extracted_features.squeeze(
                    0) if extracted_features is not None else default_image_features
                example.meta = current_meta

            logger.info("--> 所有图像特征提取完毕。")
        except Exception as e:
            logger.error(f"--> CLIP模型加载或特征提取失败，将关闭图片处理功能。错误: {e}")
            config.use_images = False

    # --- 5. 评估函数定义 ---
    def evaluate(p_model, dataloader, desc="Evaluating"):
        p_model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs in tqdm.tqdm(dataloader, desc=desc, leave=True):
                if torch.cuda.is_available():
                    inputs.cuda()
                logits = p_model(inputs)
                labels = inputs['label']
                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

        acc = accuracy_score(all_labels, all_preds)
        pre, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
        return {'acc': acc, 'pre': pre, 'recall': recall, 'f1': f1}

    # ======================== K-折交叉验证模式 ========================
    if config.use_k_fold:
        logger.info("\n" + "=" * 20 + " K-折交叉验证模式启动 " + "=" * 20)

        # 6.1 数据集划分
        full_dataset = dataset['train'] + dataset['test']
        full_labels = [ex.label for ex in full_dataset]
        train_val_pool, final_test_set = train_test_split(
            full_dataset, test_size=config.k_fold_test_split_size, random_state=config.seed, stratify=full_labels
        )
        logger.info(f"--> 数据池划分为: {len(train_val_pool)} 用于K-折, {len(final_test_set)} 用于最终测试。")

        skf = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=config.seed)
        train_val_pool_labels = [ex.label for ex in train_val_pool]
        train_val_pool_np = np.array(train_val_pool)
        fold_results = []

        # 6.2 K-折交叉验证循环
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_pool_np, train_val_pool_labels)):
            logger.info(f"\n========== 开始第 {fold + 1}/{config.k_folds} 折 ==========")

            # 为每一折重新初始化模型和组件
            plm, tokenizer, model_config, WrapperClass = load_plm(config.model_type, config.model_name_or_path)
            template_path = os.path.join(scripts_path, "ptuning_template.txt")
            with open(template_path, 'r', encoding='utf-8') as f:
                template_text = f.readlines()[config.template_id].rstrip()
            mytemplate = PtuningTemplate(model=plm, tokenizer=tokenizer, text=template_text)
            verbalizer_path = os.path.join(scripts_path, config.verbalizer_file_name)
            with open(verbalizer_path, 'r', encoding='utf-8') as f:
                verbalizer_dict = json.load(f)
            myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels)
            myverbalizer.label_words = verbalizer_dict['label_words']
            prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer,
                                                   freeze_plm=False).to(device)

            train_data_fold = train_val_pool_np[train_idx].tolist()
            val_data_fold = train_val_pool_np[val_idx].tolist()

            # 一次性预处理该折的数据
            tokenized_train_fold = preprocess_and_tokenize_dataset(train_data_fold, mytemplate, tokenizer, WrapperClass,
                                                                   config, default_image_features)
            tokenized_val_fold = preprocess_and_tokenize_dataset(val_data_fold, mytemplate, tokenizer, WrapperClass,
                                                                 config, default_image_features)
            val_dataloader_fold = DataLoader(tokenized_val_fold, batch_size=config.batch_size,
                                             collate_fn=InputFeatures.collate_fct)

            # 初始化优化器和损失函数
            loss_func = torch.nn.CrossEntropyLoss()
            optimizer_grouped_parameters = [{'params': [p for n, p in prompt_model.named_parameters() if 'plm' in n]}]
            optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
            scaler = GradScaler()

            best_val_acc_fold = 0
            # Epoch 循环
            for epoch in range(config.max_epochs):
                prompt_model.train()

                train_sampler = RandomSampler(tokenized_train_fold)
                train_dataloader_fold = DataLoader(tokenized_train_fold, sampler=train_sampler,
                                                   batch_size=config.batch_size, collate_fn=InputFeatures.collate_fct)

                pbar = tqdm.tqdm(train_dataloader_fold, desc=f"Fold {fold + 1} Epoch {epoch + 1}")
                for step, inputs in enumerate(pbar):
                    inputs.to(device)
                    with autocast():
                        logits = prompt_model(inputs)
                        labels = inputs['label']
                        loss = loss_func(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

                val_metrics = evaluate(prompt_model, val_dataloader_fold)
                if val_metrics['acc'] > best_val_acc_fold:
                    best_val_acc_fold = val_metrics['acc']

            logger.info(f"--> 第 {fold + 1} 折结束, 最佳验证准确率: {best_val_acc_fold:.4f}")
            fold_results.append(best_val_acc_fold)

        # 6.3 汇总 K-折结果
        mean_acc = np.mean(fold_results)
        std_acc = np.std(fold_results)
        logger.info("\n" + "=" * 20 + " K-折交叉验证结果汇总 " + "=" * 20)
        logger.info(f"--> 所有折的最佳验证准确率: {[round(r, 4) for r in fold_results]}")
        logger.info(f"--> {config.k_folds}-折交叉验证平均准确率: {mean_acc:.4f} ± {std_acc:.4f}")

        # 6.4 训练最终模型
        logger.info("\n" + "=" * 20 + " 正在使用全部数据训练最终模型 " + "=" * 20)
        plm, tokenizer, model_config, WrapperClass = load_plm(config.model_type, config.model_name_or_path)
        mytemplate = PtuningTemplate(model=plm, tokenizer=tokenizer, text=template_text)
        myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels)
        myverbalizer.label_words = verbalizer_dict['label_words']
        final_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer,
                                              freeze_plm=False).to(device)

        tokenized_train_val_pool = preprocess_and_tokenize_dataset(train_val_pool, mytemplate, tokenizer, WrapperClass,
                                                                   config, default_image_features)
        final_train_sampler = RandomSampler(tokenized_train_val_pool)
        final_train_dataloader = DataLoader(tokenized_train_val_pool, sampler=final_train_sampler,
                                            batch_size=config.batch_size, collate_fn=InputFeatures.collate_fct)

        optimizer = AdamW(final_model.parameters(), lr=config.learning_rate)

        for epoch in range(config.max_epochs):
            final_model.train()
            pbar = tqdm.tqdm(final_train_dataloader, desc=f"Final Model Epoch {epoch + 1}")
            for step, inputs in enumerate(pbar):
                inputs.to(device)
                with autocast():
                    logits = final_model(inputs)
                    labels = inputs['label']
                    loss = loss_func(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        # 6.5 最终测试与保存
        logger.info("--> 最终模型训练完成，正在进行最终测试...")
        tokenized_final_test_set = preprocess_and_tokenize_dataset(final_test_set, mytemplate, tokenizer, WrapperClass,
                                                                   config, default_image_features)
        final_test_dataloader = DataLoader(tokenized_final_test_set, batch_size=config.batch_size,
                                           collate_fn=InputFeatures.collate_fct)
        test_metrics = evaluate(final_model, final_test_dataloader)

    # ======================== 原始单次训练模式 ========================
    else:
        logger.info("\n" + "=" * 20 + " 单次少样本训练模式启动 " + "=" * 20)

        # 6.1 初始化模型和组件
        plm, tokenizer, model_config, WrapperClass = load_plm(config.model_type, config.model_name_or_path)
        template_path = os.path.join(scripts_path, "ptuning_template.txt")
        with open(template_path, 'r', encoding='utf-8') as f:
            template_text = f.readlines()[config.template_id].rstrip()
        mytemplate = PtuningTemplate(model=plm, tokenizer=tokenizer, text=template_text)
        verbalizer_path = os.path.join(scripts_path, config.verbalizer_file_name)
        with open(verbalizer_path, 'r', encoding='utf-8') as f:
            verbalizer_dict = json.load(f)
        myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels)
        myverbalizer.label_words = verbalizer_dict['label_words']
        prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer,
                                               freeze_plm=False).to(device)

        # 6.2 数据集划分
        sampler = FewShotSampler(num_examples_per_label=config.shot, also_sample_dev=True)
        train_set, val_set = sampler(dataset['train'], seed=config.seed)
        test_set = dataset['test']
        logger.info(
            f"--> 少样本采样完成: {len(train_set)} 条训练样本, {len(val_set)} 条验证样本, {len(test_set)} 条测试样本。")

        # 6.3 一次性预处理数据
        tokenized_train_set = preprocess_and_tokenize_dataset(train_set, mytemplate, tokenizer, WrapperClass, config,
                                                              default_image_features)
        tokenized_val_set = preprocess_and_tokenize_dataset(val_set, mytemplate, tokenizer, WrapperClass, config,
                                                            default_image_features)
        tokenized_test_set = preprocess_and_tokenize_dataset(test_set, mytemplate, tokenizer, WrapperClass, config,
                                                             default_image_features)
        val_dataloader = DataLoader(tokenized_val_set, batch_size=config.batch_size,
                                    collate_fn=InputFeatures.collate_fct)
        test_dataloader = DataLoader(tokenized_test_set, batch_size=config.batch_size,
                                     collate_fn=InputFeatures.collate_fct)

        # 6.4 训练与评估循环
        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = AdamW(prompt_model.parameters(), lr=config.learning_rate)
        scaler = GradScaler()
        best_val_acc = 0
        temp_ckpt_path = f"./ckpts/temp_{config.dataset_name}_{config.seed}.ckpt"

        for epoch in range(config.max_epochs):
            prompt_model.train()
            train_sampler = RandomSampler(tokenized_train_set)
            train_dataloader = DataLoader(tokenized_train_set, sampler=train_sampler, batch_size=config.batch_size,
                                          collate_fn=InputFeatures.collate_fct)
            pbar = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config.max_epochs}")

            for step, inputs in enumerate(pbar):
                inputs.to(device)
                with autocast():
                    logits = prompt_model(inputs)
                    labels = inputs['label']
                    loss = loss_func(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            val_metrics = evaluate(prompt_model, val_dataloader)
            if val_metrics['acc'] >= best_val_acc:
                logger.info(
                    f"--> 新高！验证集准确率从 {best_val_acc:.4f} 提升至 {val_metrics['acc']:.4f}。保存临时模型...")
                torch.save(prompt_model.state_dict(), temp_ckpt_path)
                best_val_acc = val_metrics['acc']
            print(f"--> Epoch {epoch + 1} 结束, Val Acc: {val_metrics['acc']:.4f}, Best Val Acc: {best_val_acc:.4f}",
                  flush=True)

        # 6.5 最终测试与保存
        logger.info("\n" + "=" * 20 + " 单次训练最终测试 " + "=" * 20)
        prompt_model.load_state_dict(torch.load(temp_ckpt_path))
        test_metrics = evaluate(prompt_model, test_dataloader)
        os.remove(temp_ckpt_path)  # 清理临时模型
        final_model = prompt_model  # 最终模型即为在验证集上表现最好的模型

    # --- 7. 新的保存逻辑 ---
    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # 构建包含关键指标的基础文件名
    base_filename = f"{timestamp}-{config.dataset_name}-Acc{test_metrics['acc']:.4f}-F1s{test_metrics['f1']:.4f}"

    # 定义模型和报告的保存路径
    final_model_path = os.path.join("./ckpts", f"{base_filename}.ckpt")
    report_file_path = os.path.join("./ckpts", f"{base_filename}.txt")

    # 保存模型权重
    torch.save(final_model.state_dict(), final_model_path)
    logger.info(f"\n--> ✅ 模型已保存至: {final_model_path}")

    # 生成并保存详细的实验报告
    end_time = time.time()
    training_duration_minutes = (end_time - start_time) / 60

    # 创建报告内容
    report_content = []
    report_content.append("=" * 20 + " 实验报告 " + "=" * 20)
    report_content.append(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append(f"总训练时长: {training_duration_minutes:.2f} 分钟")
    report_content.append("\n" + "-" * 15 + " 1. 实验配置 " + "-" * 15)
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('//')}
    report_content.append(json.dumps(config_dict, indent=4, ensure_ascii=False))

    if config.use_k_fold:
        report_content.append("\n" + "-" * 15 + " 2. K-折交叉验证结果 " + "-" * 15)
        report_content.append(f"各折最佳验证准确率: {[round(r, 4) for r in fold_results]}")
        report_content.append(f"平均验证准确率: {mean_acc:.4f} (±{std_acc:.4f})")
    else:
        report_content.append("\n" + "-" * 15 + " 2. 验证集表现 " + "-" * 15)
        report_content.append(f"最佳验证准确率: {best_val_acc:.4f}")

    report_content.append("\n" + "-" * 15 + " 3. 最终测试结果 " + "-" * 15)
    report_content.append(f"准确率 (Accuracy): {test_metrics['acc']:.4f}")
    report_content.append(f"宏平均精确率 (Macro Precision): {test_metrics['pre']:.4f}")
    report_content.append(f"宏平均召回率 (Macro Recall): {test_metrics['recall']:.4f}")
    report_content.append(f"宏平均 F1-Score: {test_metrics['f1']:.4f}")
    report_content.append("=" * 50)

    final_report = "\n".join(report_content)

    # 写入文件
    with open(report_file_path, 'w', encoding='utf-8') as f:
        f.write(final_report)
    logger.info(f"--> ✅ 详细报告已保存至: {report_file_path}")

    # 在控制台打印最终报告
    print("\n" + final_report)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("脚本执行过程中发生未捕获的异常:")
        logger.exception(e)