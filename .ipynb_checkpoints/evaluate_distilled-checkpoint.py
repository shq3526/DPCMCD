#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import os
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from transformers import BertModel, CLIPVisionModel, CLIPProcessor, BertTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from tqdm import tqdm
from PIL import Image
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F  # <-- 添加这一行来导入 F

# 配置
CSV_PATH = './datasets/TextClassification/toutiao/toutiao_622.csv'
PREPROCESSED_DATA_DIR = './processed_data'
PROJECT_ROOT = '/workspace'
CONTENT_MODEL_PATH = './lightweight_content_model_distilled'
VISION_MODEL_PATH  = './lightweight_vision_model_distilled'   # 需包含 visual_projection.pt
GAT_MODEL_PATH     = './lightweight_gat_model.pth'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
MODEL_PATH = './final_lightweight_model_origin.pth'  # 需要评估的模型路径
NUM_CLASSES = 3

class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, heads):
        super().__init__()
        self.convs = nn.ModuleList([GATConv(in_channels, hidden_channels, heads=heads)])
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
        self.fc = nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)  # 使用 F.elu()
        x = global_mean_pool(x, batch)
        return self.fc(x)


class FinalDataset(Dataset):
    """读取 CSV + 预处理图数据(data_{idx}.pt) + 加载图片"""
    def __init__(self, csv_path, preprocessed_dir, project_root):
        self.project_root = project_root
        self.preprocessed_dir = preprocessed_dir
        self.df = pd.read_csv(csv_path, header=None, names=['label','title','content','image_path'])
        
        # 过滤掉不存在预处理文件的数据
        self.valid_indices = [
            idx for idx in range(len(self.df)) 
            if os.path.exists(os.path.join(self.preprocessed_dir, f'data_{idx}.pt'))
        ]

    def __len__(self): 
        return len(self.valid_indices)

    def __getitem__(self, index):
        # 映射到原始DataFrame的索引
        idx = self.valid_indices[index]
        
        row = self.df.iloc[idx]
        label, content, path = int(row['label']), str(row['content']), str(row['image_path'])
        
        # 加载图片
        image = Image.new('RGB', (224, 224), color='white')
        if path and path.lower() != 'nan':
            try:
                full_path = os.path.join(self.project_root, path)
                if os.path.exists(full_path): image = Image.open(full_path).convert("RGB")
            except Exception: pass
        
        # 加载预处理好的图数据
        graph_data_path = os.path.join(self.preprocessed_dir, f'data_{idx}.pt')
        graph_data = torch.load(graph_data_path)
        
        return content, image, label, graph_data


def collate_fn_final(batch):
    contents, images, labels, graph_data_list = zip(*batch)
    
    # 使用 torch_geometric.data.Batch 来正确地批处理图数据
    batched_graph = Batch.from_data_list(graph_data_list)

    return list(contents), list(images), torch.tensor(list(labels), dtype=torch.long), batched_graph


class FinalLightweightModel(nn.Module):
    def __init__(self, content_path, vision_path, gat_path, num_classes=3):
        super().__init__()
        self.content_model = BertModel.from_pretrained(content_path)
        self.vision_model = CLIPVisionModel.from_pretrained(vision_path)
        self.gat_model = GATModel(100, 64, 1, 64, 2)
        self.gat_model.load_state_dict(torch.load(gat_path))
        vision_proj = nn.Linear(self.vision_model.config.hidden_size, 512, bias=False)
        vision_proj.load_state_dict(torch.load(os.path.join(vision_path, 'visual_projection.pt')))
        self.visual_projection = vision_proj
        classifier_input_dim = self.content_model.config.hidden_size + 512 + 64
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, content_inputs, pixel_values, graph_batch):
        content_feature = self.content_model(**content_inputs).pooler_output
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        image_feature = self.visual_projection(vision_outputs.pooler_output)
        gat_feature = self.gat_model(graph_batch)
        combined_features = torch.cat([content_feature, image_feature, gat_feature], dim=1)
        return self.classifier(combined_features)


def evaluate(model, dataloader, loss_fn, content_tokenizer, vision_processor):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for contents, images, labels, graph_batch in tqdm(dataloader, desc="评估中"):
            inputs = content_tokenizer(contents, return_tensors='pt', padding='max_length', truncation=True, max_length=256).to(DEVICE)
            pixels = vision_processor(images=images, return_tensors='pt')['pixel_values'].to(DEVICE)
            graph_batch = graph_batch.to(DEVICE)
            labels = labels.to(DEVICE)
            
            logits = model(inputs, pixels, graph_batch)
            preds = torch.argmax(logits, dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    prec, rec, f1_class, _ = precision_recall_fscore_support(all_labels, all_preds, labels=[0, 1, 2], zero_division=0)
    return acc, f1, prec, rec, f1_class


def main():
    print(f"设备: {DEVICE}")
    content_tokenizer = BertTokenizer.from_pretrained(CONTENT_MODEL_PATH)
    vision_processor = CLIPProcessor.from_pretrained(VISION_MODEL_PATH)
    
    # 加载数据集
    dataset = FinalDataset(CSV_PATH, PREPROCESSED_DATA_DIR, PROJECT_ROOT)
    eval_loader = TorchDataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_final, num_workers=4)

    model = FinalLightweightModel(CONTENT_MODEL_PATH, VISION_MODEL_PATH, GAT_MODEL_PATH, NUM_CLASSES).to(DEVICE)

    # 加载模型权重
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"✅ 已加载模型权重: {MODEL_PATH}")
    model.eval()

    # 计算损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 评估
    acc, f1, prec, rec, f1_class = evaluate(model, eval_loader, loss_fn, content_tokenizer, vision_processor)

    # 打印评估结果
    print(f"\nModel Evaluation Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print("\nPer-Class Metrics:")
    for i in range(3):
        print(f"Class {i}: Precision={prec[i]:.4f} Recall={rec[i]:.4f} F1={f1_class[i]:.4f}")


if __name__ == "__main__":
    main()
