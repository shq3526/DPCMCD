import os
import tqdm
import torch
from torch.cuda.amp import autocast, GradScaler
import sys
# 获取当前脚本文件 (fewshot-config.py) 的绝对路径
current_script_path = os.path.abspath(__file__)
# 从脚本路径获取其所在的目录，即项目根目录
project_root = os.path.dirname(current_script_path)
# 将项目根目录添加到 Python 模块搜索路径列表的最前端
sys.path.insert(0, project_root)
print(f"--- [诊断探针] 已将项目根目录添加到 sys.path: {project_root} ---")

import random
import numpy as np
import pandas as pd
import json
import argparse
from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold, train_test_split
from datetime import datetime

from openprompt.utils.logging import logger
from openprompt.data_utils.text_classification_dataset import CnClickbaitProcessor
from openprompt.data_utils.utils import InputExample
from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer
from openprompt.prompts import PtuningTemplate
from openprompt.plms import load_plm
from openprompt.data_utils.data_sampler import FewShotSampler
from openprompt.utils.reproduciblity import set_seed
from openprompt import PromptForClassification
from transformers import AdamW



# ==================================================================================================
#                                        参数加载函数
# ==================================================================================================
def load_config_from_file(config_path):
    """从 .txt (JSON格式) 文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    class Config:
        def __init__(self, **entries):
            entries = {k: v for k, v in entries.items() if not k.startswith('//')}
            self.__dict__.update(entries)

    return Config(**config_dict)


# ==================================================================================================
#                                      程序主逻辑 (Main Logic)
# ==================================================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="./config/config.txt")
    args = parser.parse_args()

    config = load_config_from_file(args.config_file)
    set_seed(config.seed)

    os.makedirs(os.path.dirname(config.result_file), exist_ok=True)
    os.makedirs("./ckpts", exist_ok=True)

    if config.dataset in ["zongxiang", "tencent", "wangyi", "Toutiao_GPT"]:
        data_path = f"datasets/TextClassification/{config.dataset}/"
        scriptsbase = f"TextClassification/{config.dataset}"
        class_labels = CnClickbaitProcessor().get_labels()
    else:
        raise NotImplementedError(f"数据集 '{config.dataset}' 的配置未定义!")

    logger.info(f"正在从 {data_path} 加载数据集 '{config.dataset}'...")
    dataset = {}
    dataset['train'] = CnClickbaitProcessor().get_train_examples(data_path)
    dataset['test'] = CnClickbaitProcessor().get_test_examples(data_path)
    logger.info(f"数据集加载完成，共 {len(dataset['train']) + len(dataset['test'])} 条样本。")

    def evaluate(p_model, dataloader, desc):
        p_model.eval()
        allpreds = []
        alllabels = []
        with torch.no_grad():
            for inputs in tqdm.tqdm(dataloader, desc=desc):
                if torch.cuda.is_available():
                    inputs.cuda()
                logits = p_model(inputs)
                labels = inputs['label']
                alllabels.extend(labels.cpu().tolist())
                allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

        acc = accuracy_score(alllabels, allpreds)
        pre, recall, f1, _ = precision_recall_fscore_support(alllabels, allpreds, average='macro', zero_division=0)
        return {'acc': acc, 'pre': pre, 'recall': recall, 'f1': f1}

    if config.use_k_fold:
        logger.info("=" * 20 + " K-折交叉验证模式 " + "=" * 20)

        full_dataset = dataset['train'] + dataset['test']
        full_labels = [ex.label for ex in full_dataset]
        train_val_pool, final_test_set = train_test_split(
            full_dataset, test_size=config.k_fold_test_split_size, random_state=config.seed, stratify=full_labels
        )
        logger.info(f"数据池划分为: {len(train_val_pool)} 用于K-折, {len(final_test_set)} 用于最终测试。")

        skf = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=config.seed)
        train_val_pool_labels = [ex.label for ex in train_val_pool]
        train_val_pool_np = np.array(train_val_pool)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_pool_np, train_val_pool_labels)):
            logger.info(f"========== 开始第 {fold + 1}/{config.k_folds} 折 ==========")

            plm, tokenizer, model_config, WrapperClass = load_plm(config.model_type, config.model_name_or_path)

            template_path = f"./scripts/{scriptsbase}/ptuning_template.txt"
            with open(template_path, 'r', encoding='utf-8') as f:
                template_text = f.readlines()[config.template_id].rstrip()
            mytemplate = PtuningTemplate(model=plm, tokenizer=tokenizer, text=template_text)

            verbalizer_path = f"scripts/{scriptsbase}/{config.verbalizer}_verbalizer.txt"
            with open(verbalizer_path, 'r', encoding='utf-8') as f:
                verbalizer_dict = json.load(f)
            myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels)
            myverbalizer.label_words = verbalizer_dict['label_words']

            prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer,
                                                   freeze_plm=False)
            if torch.cuda.is_available():
                prompt_model.cuda()

            train_data_fold = train_val_pool_np[train_idx].tolist()
            val_data_fold = train_val_pool_np[val_idx].tolist()

            train_dataloader_fold = PromptDataLoader(dataset=train_data_fold, template=mytemplate, tokenizer=tokenizer,
                                                     tokenizer_wrapper_class=WrapperClass,
                                                     max_seq_length=config.max_seq_l,
                                                     batch_size=config.batch_size, shuffle=True, truncate_method="tail")

            val_dataloader_fold = PromptDataLoader(dataset=val_data_fold, template=mytemplate, tokenizer=tokenizer,
                                                   tokenizer_wrapper_class=WrapperClass,
                                                   max_seq_length=config.max_seq_l,
                                                   batch_size=config.batch_size, shuffle=False, truncate_method="tail")

            loss_func = torch.nn.CrossEntropyLoss()
            optimizer_grouped_parameters = [{'params': [p for n, p in prompt_model.named_parameters() if 'plm' in n]}]
            optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
            scaler = GradScaler()

            best_val_acc_fold = 0
            for epoch in range(config.max_epochs):
                prompt_model.train()
                pbar = tqdm.tqdm(train_dataloader_fold, desc=f"Fold {fold + 1} Epoch {epoch + 1}")
                for inputs in pbar:
                    if torch.cuda.is_available():
                        inputs.cuda()
                    with autocast():
                        logits = prompt_model(inputs)
                        labels = inputs['label']
                        loss = loss_func(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

                val_metrics = evaluate(prompt_model, val_dataloader_fold, desc="Valid")
                if val_metrics['acc'] > best_val_acc_fold:
                    best_val_acc_fold = val_metrics['acc']

            logger.info(f"第 {fold + 1} 折结束, 最佳验证准确率: {best_val_acc_fold:.4f}")
            fold_results.append(best_val_acc_fold)

        mean_acc = np.mean(fold_results)
        std_acc = np.std(fold_results)
        logger.info("=" * 20 + " K-折交叉验证结果汇总 " + "=" * 20)
        logger.info(f"所有折的最佳验证准确率: {[round(r, 4) for r in fold_results]}")
        logger.info(f"{config.k_folds}-折交叉验证平均准确率: {mean_acc:.4f} ± {std_acc:.4f}")

        logger.info("=" * 20 + " 正在训练最终模型 " + "=" * 20)
        plm, tokenizer, model_config, WrapperClass = load_plm(config.model_type, config.model_name_or_path)
        mytemplate = PtuningTemplate(model=plm, tokenizer=tokenizer, text=template_text)
        myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels)
        myverbalizer.label_words = verbalizer_dict['label_words']

        final_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
        if torch.cuda.is_available():
            final_model.cuda()

        final_train_dataloader = PromptDataLoader(dataset=train_val_pool, template=mytemplate, tokenizer=tokenizer,
                                                  tokenizer_wrapper_class=WrapperClass, max_seq_length=config.max_seq_l,
                                                  batch_size=config.batch_size, shuffle=True, truncate_method="tail")

        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = AdamW(final_model.parameters(), lr=config.learning_rate)
        scaler = GradScaler()

        for epoch in range(config.max_epochs):
            final_model.train()
            pbar = tqdm.tqdm(final_train_dataloader, desc=f"Final Training Epoch {epoch + 1}")
            for inputs in pbar:
                if torch.cuda.is_available():
                    inputs.cuda()
                with autocast():
                    logits = final_model(inputs)
                    labels = inputs['label']
                    loss = loss_func(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        final_model_path = f"./ckpts/{config.final_model_name}"
        torch.save(final_model.state_dict(), final_model_path)
        logger.info(f"最终模型已保存至: {final_model_path}")

        final_test_dataloader = PromptDataLoader(dataset=final_test_set, template=mytemplate, tokenizer=tokenizer,
                                                 tokenizer_wrapper_class=WrapperClass, max_seq_length=config.max_seq_l,
                                                 batch_size=config.batch_size, shuffle=False, truncate_method="tail")

        test_metrics = evaluate(final_model, final_test_dataloader, desc="Final Test")

        content_write = "=" * 20 + "\n"
        content_write += f"K-Fold Run Final Report ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n"
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('//')}
        content_write += json.dumps(config_dict, indent=4) + "\n"
        content_write += "\n--- CV Results ---\n"
        content_write += f"Avg Val Acc: {mean_acc:.4f} (+/- {std_acc:.4f})\n"
        content_write += "\n--- Final Test Results ---\n"
        content_write += f"Acc: {test_metrics['acc']:.4f}\n"
        content_write += f"Pre: {test_metrics['pre']:.4f}\n"
        content_write += f"Rec: {test_metrics['recall']:.4f}\n"
        content_write += f"F1s: {test_metrics['f1']:.4f}\n"
        content_write += "=" * 20 + "\n\n"

        print("\n" + content_write)
        with open(config.result_file, "a", encoding='utf-8') as fout:
            fout.write(content_write)

    else:  # ======================== 原始单次训练模式 ========================
        logger.info("=" * 20 + " 单次训练模式 " + "=" * 20)

        plm, tokenizer, model_config, WrapperClass = load_plm(config.model_type, config.model_name_or_path)

        template_path = f"./scripts/{scriptsbase}/ptuning_template.txt"
        with open(template_path, 'r', encoding='utf-8') as f:
            template_text = f.readlines()[config.template_id].rstrip()
        mytemplate = PtuningTemplate(model=plm, tokenizer=tokenizer, text=template_text)

        verbalizer_path = f"scripts/{scriptsbase}/{config.verbalizer}_verbalizer.txt"
        with open(verbalizer_path, 'r', encoding='utf-8') as f:
            verbalizer_dict = json.load(f)
        myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels)
        myverbalizer.label_words = verbalizer_dict['label_words']

        prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
        if torch.cuda.is_available():
            prompt_model.cuda()

        sampler = FewShotSampler(num_examples_per_label=config.shot, also_sample_dev=True)
        train_set, val_set = sampler(dataset['train'], seed=config.seed)
        test_set = dataset['test']

        train_dataloader = PromptDataLoader(dataset=train_set, template=mytemplate, tokenizer=tokenizer,
                                            tokenizer_wrapper_class=WrapperClass, max_seq_length=config.max_seq_l,
                                            batch_size=config.batch_size, shuffle=True, truncate_method="tail")
        val_dataloader = PromptDataLoader(dataset=val_set, template=mytemplate, tokenizer=tokenizer,
                                          tokenizer_wrapper_class=WrapperClass, max_seq_length=config.max_seq_l,
                                          batch_size=config.batch_size, shuffle=False, truncate_method="tail")
        test_dataloader = PromptDataLoader(dataset=test_set, template=mytemplate, tokenizer=tokenizer,
                                           tokenizer_wrapper_class=WrapperClass, max_seq_length=config.max_seq_l,
                                           batch_size=config.batch_size, shuffle=False, truncate_method="tail")

        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = AdamW(prompt_model.parameters(), lr=config.learning_rate)
        scaler = GradScaler()

        best_val_acc = 0
        temp_ckpt_path = f"./ckpts/temp_{config.dataset}_{config.seed}.ckpt"

        for epoch in range(config.max_epochs):
            prompt_model.train()
            pbar = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config.max_epochs}")
            for inputs in pbar:
                if torch.cuda.is_available():
                    inputs.cuda()
                with autocast():
                    logits = prompt_model(inputs)
                    labels = inputs['label']
                    loss = loss_func(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            val_metrics = evaluate(prompt_model, val_dataloader, desc="Valid")
            if val_metrics['acc'] >= best_val_acc:
                logger.info(f"新高！验证集准确率从 {best_val_acc:.4f} 提升至 {val_metrics['acc']:.4f}。保存临时模型...")
                torch.save(prompt_model.state_dict(), temp_ckpt_path)
                best_val_acc = val_metrics['acc']
            print(f"Epoch {epoch + 1} 结束, Val Acc: {val_metrics['acc']:.4f}, Best Val Acc: {best_val_acc:.4f}",
                  flush=True)

        logger.info("=" * 20 + " 单次训练最终测试 " + "=" * 20)
        prompt_model.load_state_dict(torch.load(temp_ckpt_path))
        test_metrics = evaluate(prompt_model, test_dataloader, desc="Test")

        final_model_path = f"./ckpts/{config.final_model_name}"
        torch.save(prompt_model.state_dict(), final_model_path)
        logger.info(f"最终模型已保存至: {final_model_path}")

        os.remove(temp_ckpt_path)

        content_write = "=" * 20 + "\n"
        content_write += f"Single Run Final Report ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n"
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('//')}
        content_write += json.dumps(config_dict, indent=4) + "\n"

        content_write += "\n--- Final Test Results ---\n"
        content_write += f"Acc: {test_metrics['acc']:.4f}\n"
        content_write += f"Pre: {test_metrics['pre']:.4f}\n"
        content_write += f"Rec: {test_metrics['recall']:.4f}\n"
        content_write += f"F1s: {test_metrics['f1']:.4f}\n"
        content_write += "=" * 20 + "\n\n"

        print("\n" + content_write)
        with open(config.result_file, "a", encoding='utf-8') as fout:
            fout.write(content_write)


if __name__ == "__main__":
    # 添加一个顶层的 try-except 块来捕获并打印任何未预料的异常
    try:
        main()
    except Exception as e:
        logger.error("脚本执行过程中发生未捕获的异常:")
        logger.exception(e)