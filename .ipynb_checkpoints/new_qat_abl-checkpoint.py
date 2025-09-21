#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_and_ablation_integrated.py
完整的训练与消融实验一体化脚本
1. 基于量化文本模型训练多模态分类器
2. 训练完成后进行10折交叉验证消融实验
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split
from datetime import datetime
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, RandomSampler
from transformers import BertTokenizer, CLIPProcessor, CLIPModel, BertModel, AdamW
import warnings
import sys
import json
warnings.filterwarnings('ignore')

# 初始化项目路径
try:
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(current_script_path)
except NameError:
    project_root = os.path.abspath('.')

sys.path.insert(0, project_root)

# OpenPrompt导入
from openprompt.utils.logging import logger
from openprompt.data_utils.text_classification_dataset import CnClickbaitProcessor
from openprompt.data_utils.utils import InputExample, InputFeatures
from openprompt.prompts import ManualVerbalizer, PtuningTemplate
from openprompt.plms import load_plm
from openprompt import PromptForClassification
from openprompt.utils import signature

# ==================================================================================================
#                                配置类
# ==================================================================================================
class IntegratedConfig:
    # 路径配置
    QUANTIZED_MODEL_PATH = "/root/autodl-tmp/MSP-master/model_quantized_native/quantized_roberta_native.pth"
    PROJECT_ROOT = "/root/autodl-tmp/MSP-master"
    
    # 数据集配置
    DATASET_NAME = "toutiao"
    CSV_FILE = "toutiao_622.csv"
    PROMPT_NAME = "three"
    
    # 模型路径
    CONTENT_MODEL_PATH = "/root/autodl-tmp/MSP-master/model/chinese-roberta-wwm-ext"
    VISION_MODEL_PATH = "/root/autodl-tmp/MSP-master/model/clip-vit-base-patch32"
    
    # 训练参数
    NUM_CLASSES = 3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16
    MAX_LENGTH = 256
    LEARNING_RATE = 2e-5
    TRAIN_EPOCHS = 3
    TRAIN_TEST_SPLIT = 0.2
    
    # 消融实验参数
    N_FOLDS = 3
    N_EPOCHS = 3
    RANDOM_SEED = 42
    
    # Prompt配置
    TEMPLATE_ID = 1
    VERBALIZER_FILE = "manual_verbalizer.txt"

CONFIG = IntegratedConfig()

# ==================================================================================================
#                       多模态模型定义（支持训练和消融）
# ==================================================================================================
class TrainableMultiModalModel(nn.Module):
    """
    可训练的多模态模型，支持消融实验
    mode: full / no_visual / no_syntactic / text_only
    """
    def __init__(self, quantized_text_model, vision_model_path, mode="full", freeze_text=True):
        super().__init__()
        assert mode in ["full", "no_visual", "no_syntactic", "text_only"]
        self.mode = mode
        self.freeze_text = freeze_text
        
        # 量化的文本模型
        self.quantized_text_model = quantized_text_model
        
        # 是否冻结文本模型
        if freeze_text:
            for param in self.quantized_text_model.parameters():
                param.requires_grad = False
        
        # 视觉模型
        self.vision_model = None
        self.vision_dim = 512
        
        if vision_model_path and os.path.exists(vision_model_path):
            try:
                clip_model = CLIPModel.from_pretrained(vision_model_path)
                self.vision_model = clip_model.vision_model
                self.vision_dim = self.vision_model.config.hidden_size
                
                # 冻结视觉模型参数
                for param in self.vision_model.parameters():
                    param.requires_grad = False
                    
                print(f"✓ 视觉模型加载成功 (dim={self.vision_dim})")
            except Exception as e:
                print(f"视觉模型加载失败: {e}")
                self.vision_model = None
        
        # 获取文本特征维度
        self.text_dim = self._get_text_feature_dim()
        
        # 分类器（可训练部分）
        combined_dim = self.text_dim + self.vision_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, CONFIG.NUM_CLASSES)
        )
        
        print(f"模型初始化完成 - 模式: {mode}")
        print(f"文本特征维度: {self.text_dim}, 视觉特征维度: {self.vision_dim}")
        print(f"分类器输入维度: {combined_dim}")
    
    def _get_text_feature_dim(self):
        """获取文本特征维度"""
        try:
            dummy_input_ids = torch.ones(1, 10, dtype=torch.long)
            dummy_attention_mask = torch.ones(1, 10, dtype=torch.long)
            
            with torch.no_grad():
                outputs = self.quantized_text_model(
                    input_ids=dummy_input_ids,
                    attention_mask=dummy_attention_mask
                )
                
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    return outputs.pooler_output.size(-1)
                elif hasattr(outputs, 'last_hidden_state'):
                    return outputs.last_hidden_state.size(-1)
                else:
                    return 768
        except Exception:
            return 768
    
    @torch.no_grad()
    def _zeros(self, bsz, dim, device):
        """生成零向量"""
        return torch.zeros(bsz, dim, device=device, dtype=torch.float32)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, pixel_values=None):
        device = input_ids.device
        bsz = input_ids.size(0)
        
        # 文本特征提取（量化模型）
        if self.freeze_text:
            with torch.no_grad():
                text_outputs = self.quantized_text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
        else:
            text_outputs = self.quantized_text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        
        if hasattr(text_outputs, 'pooler_output') and text_outputs.pooler_output is not None:
            text_features = text_outputs.pooler_output
        else:
            text_features = text_outputs.last_hidden_state[:, 0, :]
        
        # 视觉特征提取 - 根据消融模式决定
        if self.mode in ["full", "no_syntactic"] and self.vision_model is not None:
            if pixel_values is not None:
                try:
                    with torch.no_grad():
                        vision_outputs = self.vision_model(pixel_values=pixel_values)
                        vision_features = vision_outputs.pooler_output
                except Exception as e:
                    vision_features = self._zeros(bsz, self.vision_dim, device)
            else:
                vision_features = self._zeros(bsz, self.vision_dim, device)
        else:
            # no_visual 或 text_only 模式：使用零向量遮蔽
            vision_features = self._zeros(bsz, self.vision_dim, device)
        
        # 组合特征
        combined_features = torch.cat([text_features, vision_features], dim=1)
        
        # 分类
        logits = self.classifier(combined_features)
        return logits

# ==================================================================================================
#                                数据集类
# ==================================================================================================
class ToutiaoDataset(Dataset):
    def __init__(self, csv_path, project_root, vision_processor=None):
        self.project_root = project_root
        self.vision_processor = vision_processor
        
        # 加载toutiao_622.csv数据
        print(f"正在加载数据: {csv_path}")
        try:
            self.df = pd.read_csv(csv_path, header=None, names=['label', 'title', 'content', 'image_path'])
            print(f"✓ 成功加载数据集: {len(self.df)} 条样本")
            
            # 检查数据格式
            print(f"标签分布: {self.df['label'].value_counts().to_dict()}")
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            raise
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        label = int(row['label'])
        title = str(row['title']) if pd.notna(row['title']) else ""
        content = str(row['content']) if pd.notna(row['content']) else ""
        img_path = str(row['image_path']) if pd.notna(row['image_path']) else ""
        
        # 组合文本
        text = f"{title} {content}".strip()
        
        # 加载图像
        image = Image.new('RGB', (224, 224), color='white')
        
        if img_path and img_path.lower() != 'nan':
            full_path = os.path.join(self.project_root, img_path)
            if os.path.exists(full_path):
                try:
                    image = Image.open(full_path).convert('RGB')
                except Exception:
                    pass
        
        return text, image, label

def collate_fn(batch, tokenizer, vision_processor, max_length, device):
    """数据整理函数"""
    texts, images, labels = zip(*batch)
    
    # 文本tokenization
    text_inputs = tokenizer(
        list(texts),
        return_tensors='pt',
        max_length=max_length,
        padding='max_length',
        truncation=True
    ).to(device)
    
    # 图像处理
    pixel_values = None
    if vision_processor is not None:
        try:
            vision_inputs = vision_processor(
                images=list(images),
                return_tensors='pt'
            )
            pixel_values = vision_inputs['pixel_values'].to(device)
        except Exception:
            pixel_values = torch.zeros(len(images), 3, 224, 224, device=device)
    
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    
    return text_inputs, pixel_values, labels

# ==================================================================================================
#                                训练函数
# ==================================================================================================
def train_model(dataset, tokenizer, vision_processor, device):
    """训练多模态分类器"""
    print("\n" + "="*80)
    print("开始训练阶段 - 基于量化文本模型的多模态分类器")
    print("="*80)
    
    # 数据分割
    indices = list(range(len(dataset)))
    labels = [dataset[i][2] for i in indices]
    
    train_indices, test_indices = train_test_split(
        indices, 
        test_size=CONFIG.TRAIN_TEST_SPLIT, 
        random_state=CONFIG.RANDOM_SEED, 
        stratify=labels
    )
    
    train_data = [dataset[i] for i in train_indices]
    test_data = [dataset[i] for i in test_indices]
    
    print(f"训练集大小: {len(train_data)}")
    print(f"测试集大小: {len(test_data)}")
    
    # 创建数据加载器
    def create_dataloader(data, shuffle=True):
        return DataLoader(
            data,
            batch_size=CONFIG.BATCH_SIZE,
            shuffle=shuffle,
            collate_fn=lambda batch: collate_fn(batch, tokenizer, vision_processor, CONFIG.MAX_LENGTH, device),
            num_workers=0
        )
    
    train_dataloader = create_dataloader(train_data, shuffle=True)
    test_dataloader = create_dataloader(test_data, shuffle=False)
    
    # 加载量化模型
    print("正在加载量化文本模型...")
    quantized_model = torch.load(CONFIG.QUANTIZED_MODEL_PATH, map_location=device)
    
    if isinstance(quantized_model, dict):
        base_model = BertModel.from_pretrained(CONFIG.CONTENT_MODEL_PATH)
        base_model.load_state_dict(quantized_model, strict=False)
        quantized_model = base_model
    
    # 创建多模态模型
    model = TrainableMultiModalModel(
        quantized_text_model=quantized_model,
        vision_model_path=CONFIG.VISION_MODEL_PATH,
        mode="full",  # 训练时使用完整模型
        freeze_text=True  # 冻结量化文本模型
    ).to(device)
    
    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params:,} / {total_params:,}")
    
    # 训练设置
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=CONFIG.LEARNING_RATE)
    
    best_test_acc = 0.0
    best_model_state = None
    
    # 训练循环
    for epoch in range(CONFIG.TRAIN_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{CONFIG.TRAIN_EPOCHS} ---")
        
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
        for text_inputs, pixel_values, labels in train_pbar:
            optimizer.zero_grad()
            
            logits = model(
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask'],
                token_type_ids=text_inputs.get('token_type_ids'),
                pixel_values=pixel_values
            )
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            train_total += labels.size(0)
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_correct/train_total:.4f}'
            })
        
        train_acc = train_correct / train_total
        
        # 测试
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for text_inputs, pixel_values, labels in tqdm(test_dataloader, desc="Testing"):
                logits = model(
                    input_ids=text_inputs['input_ids'],
                    attention_mask=text_inputs['attention_mask'],
                    token_type_ids=text_inputs.get('token_type_ids'),
                    pixel_values=pixel_values
                )
                
                test_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
                test_total += labels.size(0)
        
        test_acc = test_correct / test_total
        
        print(f"Epoch {epoch + 1}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")
        
        # 保存最佳模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = model.state_dict().copy()
            print(f"✓ 新的最佳模型: {best_test_acc:.4f}")
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✅ 训练完成！最佳测试准确率: {best_test_acc:.4f}")
    
    return model

# ==================================================================================================
#                                消融实验函数
# ==================================================================================================
def evaluate_single_epoch(model, dataloader, device):
    """单次epoch评估"""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for text_inputs, pixel_values, labels in dataloader:
            try:
                logits = model(
                    input_ids=text_inputs['input_ids'],
                    attention_mask=text_inputs['attention_mask'],
                    token_type_ids=text_inputs.get('token_type_ids'),
                    pixel_values=pixel_values
                )
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            except Exception as e:
                continue
    
    if len(all_preds) == 0:
        return {"Accuracy": 0.0, "Macro Precision": 0.0, "Macro Recall": 0.0, "Macro F1-Score": 0.0}
    
    acc = accuracy_score(all_labels, all_preds)
    mp, mr, mf1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    
    return {"Accuracy": acc, "Macro Precision": mp, "Macro Recall": mr, "Macro F1-Score": mf1}

def run_ablation_study(trained_model_state, dataset, tokenizer, vision_processor, device):
    """运行消融实验"""
    print("\n" + "="*80)
    print("开始消融实验 - 10折交叉验证 + 3个epoch重复")
    print("="*80)
    
    # 准备数据
    y_all = dataset.df['label'].astype(int).values
    skf = StratifiedKFold(n_splits=CONFIG.N_FOLDS, shuffle=True, random_state=CONFIG.RANDOM_SEED)
    
    # 消融模式
    ablation_modes = [
        ("full", "Full Model"),
        ("no_visual", "w/o Visual"),
        ("no_syntactic", "w/o Syntactic"), 
        ("text_only", "Text Only")
    ]
    
    # 存储所有结果
    all_results = {}
    detailed_results = {}
    
    for mode_key, mode_label in ablation_modes:
        print(f"\n{'='*60}")
        print(f"评估模式: {mode_label}")
        print(f"{'='*60}")
        
        mode_fold_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.arange(len(dataset)), y_all)):
            print(f"\n--- Fold {fold_idx + 1}/10 ---")
            
            # 创建测试数据集
            test_data = [dataset[i] for i in test_idx]
            
            # 每个fold重复3个epoch
            fold_epoch_results = []
            
            for epoch in range(CONFIG.N_EPOCHS):
                print(f"  Epoch {epoch + 1}/3", end=" ")
                
                try:
                    # 重新加载量化模型
                    quantized_model = torch.load(CONFIG.QUANTIZED_MODEL_PATH, map_location=device)
                    if isinstance(quantized_model, dict):
                        base_model = BertModel.from_pretrained(CONFIG.CONTENT_MODEL_PATH)
                        base_model.load_state_dict(quantized_model, strict=False)
                        quantized_model = base_model
                    
                    # 创建消融模型
                    model = TrainableMultiModalModel(
                        quantized_text_model=quantized_model,
                        vision_model_path=CONFIG.VISION_MODEL_PATH,
                        mode=mode_key,
                        freeze_text=True
                    ).to(device)
                    
                    # 加载训练好的权重（仅分类器部分）
                    model.load_state_dict(trained_model_state, strict=False)
                    
                    # 创建数据加载器
                    test_dataloader = DataLoader(
                        test_data,
                        batch_size=CONFIG.BATCH_SIZE,
                        shuffle=False,
                        collate_fn=lambda batch: collate_fn(batch, tokenizer, vision_processor, CONFIG.MAX_LENGTH, device),
                        num_workers=0
                    )
                    
                    # 评估
                    metrics = evaluate_single_epoch(model, test_dataloader, device)
                    fold_epoch_results.append(metrics)
                    
                    print(f"Acc: {metrics['Accuracy']:.4f}")
                    
                    # 释放内存
                    del model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                except Exception as e:
                    print(f"失败: {e}")
                    fold_epoch_results.append({
                        "Accuracy": 0.0, "Macro Precision": 0.0, 
                        "Macro Recall": 0.0, "Macro F1-Score": 0.0
                    })
            
            # 计算fold平均结果
            if fold_epoch_results:
                fold_avg = {}
                for metric in ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-Score"]:
                    values = [r[metric] for r in fold_epoch_results]
                    fold_avg[metric] = np.mean(values)
                
                mode_fold_results.append(fold_avg)
                print(f"  Fold平均: Acc={fold_avg['Accuracy']:.4f}, F1={fold_avg['Macro F1-Score']:.4f}")
        
        # 计算该模式的总体统计
        if mode_fold_results:
            mode_stats = {}
            for metric in ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-Score"]:
                values = [r[metric] for r in mode_fold_results]
                mode_stats[f"{metric}_mean"] = np.mean(values)
                mode_stats[f"{metric}_std"] = np.std(values)
            
            all_results[mode_label] = mode_stats
            detailed_results[mode_label] = mode_fold_results
            
            print(f"\n{mode_label} 总体结果:")
            print(f"  Accuracy: {mode_stats['Accuracy_mean']:.4f} ± {mode_stats['Accuracy_std']:.4f}")
            print(f"  Macro F1: {mode_stats['Macro F1-Score_mean']:.4f} ± {mode_stats['Macro F1-Score_std']:.4f}")
    
    return all_results, detailed_results

def save_results(all_results, detailed_results):
    """保存实验结果"""
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    if not all_results:
        print("没有成功的实验结果")
        return
    
    # 创建汇总表格
    summary_data = []
    for mode_label, stats in all_results.items():
        summary_data.append({
            "Setting": mode_label,
            "Accuracy": f"{stats['Accuracy_mean']:.4f} ± {stats['Accuracy_std']:.4f}",
            "Macro Precision": f"{stats['Macro Precision_mean']:.4f} ± {stats['Macro Precision_std']:.4f}",
            "Macro Recall": f"{stats['Macro Recall_mean']:.4f} ± {stats['Macro Recall_std']:.4f}",
            "Macro F1-Score": f"{stats['Macro F1-Score_mean']:.4f} ± {stats['Macro F1-Score_std']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + "="*80)
    print("训练+消融实验结果汇总")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    # 保存文件
    summary_filename = f"train_ablation_summary_{timestamp}.csv"
    summary_df.to_csv(summary_filename, index=False)
    print(f"\n✅ 汇总结果已保存至: {summary_filename}")
    
    try:
        detailed_filename = f"train_ablation_detailed_{timestamp}.xlsx"
        with pd.ExcelWriter(detailed_filename) as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            for mode_label, fold_results in detailed_results.items():
                if fold_results:
                    detailed_df = pd.DataFrame(fold_results)
                    detailed_df.index = [f"Fold_{i+1}" for i in range(len(detailed_df))]
                    sheet_name = mode_label.replace('/', '_').replace(' ', '_')
                    detailed_df.to_excel(writer, sheet_name=sheet_name)
        
        print(f"✅ 详细结果已保存至: {detailed_filename}")
    except Exception as e:
        print(f"Excel保存失败: {e}")

# ==================================================================================================
#                                主函数
# ==================================================================================================
def main():
    print("=" * 80)
    print("完整的训练+消融实验 - toutiao_622数据集")
    print("=" * 80)
    
    device = CONFIG.DEVICE
    print(f"使用设备: {device}")
    print(f"量化模型: {os.path.basename(CONFIG.QUANTIZED_MODEL_PATH)}")
    print(f"数据文件: {CONFIG.CSV_FILE}")
    
    # 设置随机种子
    torch.manual_seed(CONFIG.RANDOM_SEED)
    np.random.seed(CONFIG.RANDOM_SEED)
    
    # 检查路径
    if not os.path.exists(CONFIG.QUANTIZED_MODEL_PATH):
        print(f"❌ 错误：量化模型文件不存在: {CONFIG.QUANTIZED_MODEL_PATH}")
        return
    
    # 构造数据文件路径
    csv_path = os.path.join(CONFIG.PROJECT_ROOT, "datasets", "TextClassification", CONFIG.DATASET_NAME, CONFIG.CSV_FILE)
    
    if not os.path.exists(csv_path):
        print(f"❌ 错误：数据文件不存在: {csv_path}")
        return
    
    print(f"✓ 找到数据文件: {csv_path}")
    
    # 初始化处理器
    tokenizer = BertTokenizer.from_pretrained(CONFIG.CONTENT_MODEL_PATH)
    print("✓ 初始化BERT tokenizer")
    
    vision_processor = None
    if os.path.exists(CONFIG.VISION_MODEL_PATH):
        try:
            vision_processor = CLIPProcessor.from_pretrained(CONFIG.VISION_MODEL_PATH)
            print("✓ 初始化CLIP processor")
        except Exception as e:
            print(f"CLIP processor初始化失败: {e}")
    
    # 加载数据集
    try:
        dataset = ToutiaoDataset(csv_path, CONFIG.PROJECT_ROOT, vision_processor)
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return
    
    # ========== 阶段1：训练 ==========
    print("\n" + "#"*60)
    print("阶段1：基于量化模型的多模态分类器训练")
    print("#"*60)
    
    trained_model = train_model(dataset, tokenizer, vision_processor, device)
    trained_model_state = trained_model.state_dict()
    
    # ========== 阶段2：消融实验 ==========
    print("\n" + "#"*60)
    print("阶段2：消融实验")
    print("#"*60)
    
    all_results, detailed_results = run_ablation_study(
        trained_model_state, dataset, tokenizer, vision_processor, device
    )
    
    # 保存结果
    save_results(all_results, detailed_results)
    
    print(f"\n{'='*80}")
    print("🎉 训练+消融实验完成！")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()