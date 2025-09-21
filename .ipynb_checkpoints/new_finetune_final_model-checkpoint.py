#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, BertTokenizer, CLIPVisionModel, CLIPProcessor
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# PyG
try:
    from torch_geometric.data import Batch
    from torch_geometric.nn import GATConv, global_mean_pool
except ImportError:
    print("❌ 错误: 未安装 torch_geometric。")
    raise

torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# ----------------- 模型 -----------------
class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, heads):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
        self.fc = nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)


class FinalLightweightModel(nn.Module):
    """ 文本BERT(学生) + 视觉CLIP-Vision(学生)+projection + GAT(学生) -> 分类 """
    def __init__(self, content_path, vision_path, gat_path, gat_cfg, num_classes):
        super().__init__()
        # 文本
        self.content_model = BertModel.from_pretrained(content_path)

        # 视觉 + 自投影
        self.vision_model = CLIPVisionModel.from_pretrained(vision_path)
        self.visual_projection = nn.Linear(self.vision_model.config.hidden_size, 512, bias=False)
        vp = os.path.join(vision_path, "visual_projection.pt")
        if not os.path.isfile(vp):
            raise FileNotFoundError(f"未找到视觉投影权重：{vp}")
        self.visual_projection.load_state_dict(torch.load(vp, map_location="cpu"))

        # GAT
        self.gat_model = GATModel(**gat_cfg)
        gat_state = torch.load(gat_path, map_location="cpu")
        if isinstance(gat_state, dict) and any(k.startswith("module.") for k in gat_state.keys()):
            gat_state = {k.replace("module.", "", 1): v for k, v in gat_state.items()}
        self.gat_model.load_state_dict(gat_state)

        # 分类头
        cls_in = self.content_model.config.hidden_size + 512 + gat_cfg["out_channels"]
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, content_inputs, pixel_values, graph_batch):
        txt = self.content_model(**content_inputs).pooler_output
        vis = self.vision_model(pixel_values=pixel_values).pooler_output
        vis = self.visual_projection(vis)
        gat = self.gat_model(graph_batch)
        feat = torch.cat([txt, vis, gat], dim=1)
        return self.classifier(feat)


# ----------------- 数据 -----------------
class FinalDataset(Dataset):
    """
    读取 CSV + 预处理图数据(data_{idx}.pt) + 加载图片（缺失用白图）
    CSV 列：0=label, 1=title, 2=content, 3=image_path
    """
    def __init__(self, csv_path, preprocessed_dir, project_root):
        self.project_root = project_root
        self.preprocessed_dir = preprocessed_dir
        self.df = pd.read_csv(csv_path, header=None, names=['label','title','content','image_path'])

        self.valid_indices = [
            i for i in range(len(self.df))
            if os.path.exists(os.path.join(self.preprocessed_dir, f"data_{i}.pt"))
        ]

    def __len__(self): return len(self.valid_indices)

    def __getitem__(self, index):
        idx = self.valid_indices[index]
        row = self.df.iloc[idx]
        label, content, path = int(row['label']), str(row['content']), str(row['image_path'])

        # 图像（缺失=白图）
        img = Image.new("RGB", (224,224), color="white")
        if path and path.lower() != "nan":
            full = os.path.join(self.project_root, path)
            try:
                if os.path.exists(full):
                    img = Image.open(full).convert("RGB")
            except Exception:
                pass

        # 载入预处理的图数据（title图）
        gpath = os.path.join(self.preprocessed_dir, f"data_{idx}.pt")
        graph_data = torch.load(gpath, map_location="cpu")
        if hasattr(graph_data, "x") and graph_data.x is not None:
            graph_data.x = graph_data.x.detach().to(torch.float32)
        if hasattr(graph_data, "edge_index") and graph_data.edge_index is not None:
            graph_data.edge_index = graph_data.edge_index.to(torch.long).contiguous()

        return content, img, label, graph_data


def collate_fn(batch):
    contents, images, labels, graphs = zip(*batch)
    return list(contents), list(images), torch.tensor(labels, dtype=torch.long), Batch.from_data_list(list(graphs))


# ----------------- 训练/评估 -----------------
def main():
    ap = argparse.ArgumentParser()
    # 路径
    ap.add_argument("--csv", type=str, default="./datasets/TextClassification/toutiao/toutiao_622.csv")
    ap.add_argument("--project_root", type=str, default="/workspace")
    ap.add_argument("--pre_dir", type=str, default="./processed_data")
    # 三个学生
    ap.add_argument("--content_path", type=str, default="./lightweight_content_model_distilled")
    ap.add_argument("--vision_path",  type=str, default="./lightweight_vision_model_distilled")
    ap.add_argument("--gat_path",     type=str, default="./lightweight_gat_model.pth")
    # 教师CLIP（只用处理器）
    ap.add_argument("--teacher_clip", type=str, default="./model/clip-vit-base-patch32")
    # 训练
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch",  type=int, default=32)
    ap.add_argument("--lr",     type=float, default=2e-5)
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--save_path", type=str, default="./new_final_lightweight_model.pth")
    # GAT 结构
    ap.add_argument("--gat_in", type=int, default=100)
    ap.add_argument("--gat_hidden", type=int, default=64)
    ap.add_argument("--gat_layers", type=int, default=1)
    ap.add_argument("--gat_out", type=int, default=64)
    ap.add_argument("--gat_heads", type=int, default=2)
    # 任务
    ap.add_argument("--num_classes", type=int, default=3)
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ 指定了 --device=cuda 但此环境无 GPU，用 CPU 继续。")
        device = "cpu"

    print(f"--- 开始最终模型微调 (设备: {device}) ---")

    # Tokenizer & 视觉处理器（务必用教师 CLIP 处理器）
    tokenizer = BertTokenizer.from_pretrained(args.content_path)
    vision_proc = CLIPProcessor.from_pretrained(args.teacher_clip)

    # 数据
    dataset = FinalDataset(args.csv, args.pre_dir, args.project_root)
    train_loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=(device=="cuda")
    )
    eval_loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=(device=="cuda")
    )

    # 模型
    gat_cfg = dict(in_channels=args.gat_in, hidden_channels=args.gat_hidden,
                   num_layers=args.gat_layers, out_channels=args.gat_out, heads=args.gat_heads)
    model = FinalLightweightModel(args.content_path, args.vision_path, args.gat_path, gat_cfg, args.num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 类别加权
    valid_labels = pd.read_csv(args.csv, header=None, usecols=[0]).iloc[dataset.valid_indices, 0]
    vc = valid_labels.value_counts().sort_index()
    if not vc.empty:
        weights = vc.sum() / (len(vc) * vc)
        class_weights = torch.tensor(weights.values, dtype=torch.float32, device=device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        print("已启用加权损失函数。")
    else:
        loss_fn = nn.CrossEntropyLoss()

    use_amp = (device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 训练 + 评估
    for epoch in range(1, args.epochs+1):
        # 训练
        model.train()
        pbar = tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}", ncols=120)
        for contents, images, labels, graph_batch in pbar:
            # 文本
            content_inputs = tokenizer(contents, return_tensors='pt', max_length=256,
                                       padding='max_length', truncation=True)
            content_inputs = {k: v.to(device, non_blocking=True) for k, v in content_inputs.items()}
            # 图像 → CLIPProcessor
            pixel_values = vision_proc(images=images, return_tensors='pt')['pixel_values'].to(device, non_blocking=True)
            graph_batch = graph_batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(content_inputs, pixel_values, graph_batch)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=f"{float(loss):.4f}")

        # 评估
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            for contents, images, labels, graph_batch in tqdm(eval_loader, desc=f"Eval  {epoch}/{args.epochs}", ncols=120):
                content_inputs = tokenizer(contents, return_tensors='pt', max_length=256,
                                           padding='max_length', truncation=True)
                content_inputs = {k: v.to(device, non_blocking=True) for k, v in content_inputs.items()}
                pixel_values = vision_proc(images=images, return_tensors='pt')['pixel_values'].to(device, non_blocking=True)
                graph_batch = graph_batch.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(content_inputs, pixel_values, graph_batch)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        acc = accuracy_score(all_labels, all_preds)
        f1m = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        prec, rec, f1c, sup = precision_recall_fscore_support(
            all_labels, all_preds, labels=list(range(args.num_classes)), zero_division=0
        )
        print(f"\nEpoch {epoch}/{args.epochs} | Acc {acc:.4f} | Macro-F1 {f1m:.4f}")
        for i in range(args.num_classes):
            print(f"  Class {i}: P={prec[i]:.4f} R={rec[i]:.4f} F1={f1c[i]:.4f} n={sup[i]}")

    torch.save(model.state_dict(), args.save_path)
    print(f"\n✅ 微调完成！模型已保存到: {args.save_path}")


if __name__ == "__main__":
    main()
