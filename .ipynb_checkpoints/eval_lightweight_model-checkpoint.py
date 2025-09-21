#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估蒸馏/微调后的轻量模型:
- 指标: Accuracy, Macro-F1, Weighted-F1, 每类F1, 混淆矩阵
- 额外: 类别分布、推理速度、模型大小
- 与 finetune_final_model.py 的模型/数据约定保持一致（CSV列、预处理图数据、投影层等）
"""

import os, time, json, argparse
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, CLIPVisionModel, CLIPProcessor, BertTokenizer

# ----- 可按需改动的默认参数 -----
DEF_CSV            = './datasets/TextClassification/toutiao/toutiao_622.csv'
DEF_PREPROC_DIR    = './processed_data'           # 存放预处理后的图数据 data_{idx}.pt
DEF_PROJECT_ROOT   = '.'                          # 用于拼接 image_path
DEF_CONTENT_PATH   = './lightweight_content_model_distilled'  # 蒸馏后的文本模型目录（BERT）
DEF_VISION_PATH    = './lightweight_vision_model_distilled'   # 蒸馏后的视觉模型目录（CLIP Vision）
DEF_GAT_PATH       = './lightweight_gat_model.pth'            # 蒸馏后的GAT权重
DEF_FINETUNED_CKPT = './final_lightweight_model_orgin.pth'          # 若微调过，加载此权重
DEF_NUM_CLASSES    = 3
DEF_DEVICE         = 'cuda' if torch.cuda.is_available() else 'cpu'
DEF_BATCH          = 32
DEF_TEST_SIZE      = 0.2
DEF_SEED           = 42

# ----- 与 finetune_final_model.py 保持一致的 GAT 配置 -----
GAT_CONFIG = {"in_channels": 100, "hidden_channels": 64, "num_layers": 1, "out_channels": 64, "heads": 2}

# ---------------- GAT 定义 ----------------
try:
    from torch_geometric.data import Batch
    from torch_geometric.nn import GATConv, global_mean_pool
except Exception as e:
    raise ImportError("需要安装 torch_geometric 及其依赖。") from e

class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, heads):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        for _ in range(num_layers-1):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
        self.fc = nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)

# --------------- 数据集 (与 finetune_final_model.py 兼容) ---------------
class FinalDataset(Dataset):
    def __init__(self, csv_path, preprocessed_dir, project_root):
        self.project_root = project_root
        self.preprocessed_dir = preprocessed_dir
        self.df = pd.read_csv(csv_path, header=None, names=['label','title','content','image_path'])
        # 仅保留已有预处理图数据的样本
        self.valid_indices = [i for i in range(len(self.df))
                              if os.path.exists(os.path.join(self.preprocessed_dir, f'data_{i}.pt'))]

    def __len__(self): return len(self.valid_indices)

    def __getitem__(self, index):
        idx = self.valid_indices[index]
        row = self.df.iloc[idx]
        label, content, path = int(row['label']), str(row['content']), str(row['image_path'])

        # 图片（如果不存在就用白图占位）
        image = Image.new('RGB', (224, 224), color='white')
        if path and path.lower() != 'nan':
            try:
                full_path = os.path.join(self.project_root, path)
                if os.path.exists(full_path):
                    image = Image.open(full_path).convert('RGB')
            except Exception:
                pass

        # 载入预处理的图数据
        graph = torch.load(os.path.join(self.preprocessed_dir, f'data_{idx}.pt'))
        return content, image, label, graph

def collate_fn(batch):
    contents, images, labels, graphs = zip(*batch)
    batched_graph = Batch.from_data_list(list(graphs))
    return list(contents), list(images), torch.tensor(labels, dtype=torch.long), batched_graph

# ---------------- 最终轻量模型 (与 finetune_final_model.py 对齐) ----------------
class FinalLightweightModel(nn.Module):
    def __init__(self, content_path, vision_path, gat_path, num_classes):
        super().__init__()
        self.content_model = BertModel.from_pretrained(content_path)
        self.vision_model  = CLIPVisionModel.from_pretrained(vision_path)
        self.gat_model     = GATModel(**GAT_CONFIG)
        self.gat_model.load_state_dict(torch.load(gat_path, map_location='cpu'))

        # 视觉投影（从蒸馏目录载入）
        vision_proj = nn.Linear(self.vision_model.config.hidden_size, 512, bias=False)
        vision_proj.load_state_dict(torch.load(os.path.join(vision_path, 'visual_projection.pt'), map_location='cpu'))
        self.visual_projection = vision_proj

        in_dim = self.content_model.config.hidden_size + 512 + GAT_CONFIG['out_channels']
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, content_inputs, pixel_values, graph_batch):
        content_feature = self.content_model(**content_inputs).pooler_output
        vision_outputs  = self.vision_model(pixel_values=pixel_values)
        image_feature   = self.visual_projection(vision_outputs.pooler_output)
        gat_feature     = self.gat_model(graph_batch)
        feat = torch.cat([content_feature, image_feature, gat_feature], dim=1)
        return self.classifier(feat)

# ---------------- 评估主流程 ----------------
def evaluate(args):
    device = args.device
    print(f"设备: {device}")

    # 数据
    dataset = FinalDataset(args.csv, args.preprocessed_dir, args.project_root)
    if len(dataset) == 0:
        raise RuntimeError("没有有效样本（未找到任何 data_*.pt）。请检查预处理目录。")

    # 分层划分
    valid_df = dataset.df.iloc[dataset.valid_indices].reset_index(drop=True)
    y_all = valid_df['label'].astype(int).values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, val_idx = next(sss.split(np.arange(len(valid_df)), y_all))

    # 子集 DataLoader（这里只评估验证集；如需也评估训练集可再建一个 loader）
    from torch.utils.data import Subset
    val_ds = Subset(dataset, val_idx.tolist())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # 标注分布
    val_labels_series = valid_df.iloc[val_idx]['label'].value_counts().sort_index()
    print("\n[验证集类别分布]")
    for k, v in val_labels_series.items():
        print(f"  类 {k}: {v}")

    # 模型
    tokenizer = BertTokenizer.from_pretrained(args.content_path)
    processor = CLIPProcessor.from_pretrained(args.vision_path)
    model = FinalLightweightModel(args.content_path, args.vision_path, args.gat_path, args.num_classes).to(device)

    # 若提供了微调权重，则加载
    if args.finetuned_ckpt and os.path.exists(args.finetuned_ckpt):
        ckpt = torch.load(args.finetuned_ckpt, map_location=device)
        model.load_state_dict(ckpt, strict=False)
        print(f"\n已加载微调权重: {args.finetuned_ckpt}")
    else:
        print("\n未提供微调权重，将以“当前权重”直接评估（指标可能较低）。")

    # 统计模型大小
    def count_params(m): return sum(p.numel() for p in m.parameters())
    n_params = count_params(model)
    # 粗略字节估计（未考虑稀疏/量化）
    bytes_est = n_params * 4
    print(f"\n[模型规模] 参数量: {n_params:,} (~{bytes_est/1024/1024:.1f} MB FP32)")

    # 评估
    model.eval()
    all_preds, all_labels = [], []
    total_time, total_samples = 0.0, 0

    with torch.no_grad():
        for contents, images, labels, graph_batch in tqdm(val_loader, desc="评估中"):
            t0 = time.time()
            inputs = tokenizer(contents, return_tensors='pt', max_length=256,
                               padding='max_length', truncation=True).to(device)
            pixel_values = processor(images=images, return_tensors='pt')['pixel_values'].to(device)
            graph_batch = graph_batch.to(device)
            labels = labels.to(device)

            logits = model(inputs, pixel_values, graph_batch)
            preds = torch.argmax(logits, dim=1)

            dt = time.time() - t0
            total_time += dt
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    # 指标
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    report = classification_report(all_labels, all_preds, digits=4, zero_division=0, output_dict=False)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n========== 指标 ==========")
    print(f"Accuracy        : {acc:.4f}")
    print(f"Macro F1        : {f1_macro:.4f}")
    print(f"Weighted F1     : {f1_weighted:.4f}")
    print("\n[每类指标]")
    print(classification_report(all_labels, all_preds, digits=4, zero_division=0))
    print("[混淆矩阵]")
    print(cm)

    # 推理吞吐
    if total_time > 0:
        ips = total_samples / total_time
        print(f"\n[推理速度] {total_samples} 个样本 / {total_time:.2f}s  =>  {ips:.2f} samples/s")

    # 保存报告
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, 'eval_report.json')
    payload = {
        "accuracy": acc,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "confusion_matrix": cm.tolist(),
        "val_class_distribution": {int(k): int(v) for k, v in val_labels_series.items()},
        "params_count": n_params,
        "inference_samples": total_samples,
        "inference_seconds": total_time,
        "samples_per_second": (total_samples / total_time) if total_time > 0 else None,
    }
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n评估报告已写入: {out_json}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, default=DEF_CSV)
    ap.add_argument('--preprocessed_dir', type=str, default=DEF_PREPROC_DIR)
    ap.add_argument('--project_root', type=str, default=DEF_PROJECT_ROOT)
    ap.add_argument('--content_path', type=str, default=DEF_CONTENT_PATH)
    ap.add_argument('--vision_path', type=str, default=DEF_VISION_PATH)
    ap.add_argument('--gat_path', type=str, default=DEF_GAT_PATH)
    ap.add_argument('--finetuned_ckpt', type=str, default=DEF_FINETUNED_CKPT)
    ap.add_argument('--num_classes', type=int, default=DEF_NUM_CLASSES)
    ap.add_argument('--device', type=str, default=DEF_DEVICE)
    ap.add_argument('--batch_size', type=int, default=DEF_BATCH)
    ap.add_argument('--test_size', type=float, default=DEF_TEST_SIZE)
    ap.add_argument('--seed', type=int, default=DEF_SEED)
    ap.add_argument('--out_dir', type=str, default='./eval_outputs')
    return ap.parse_args()

if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
