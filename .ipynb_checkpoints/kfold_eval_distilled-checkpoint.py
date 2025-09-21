#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, argparse, time, json, random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, BertTokenizer, CLIPVisionModel, CLIPProcessor
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, global_mean_pool
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
try: torch.set_float32_matmul_precision("high")
except: pass

# --------- 模型/数据，与 finetune_final_model 保持一致 ----------
class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, heads):
        super().__init__()
        self.convs = nn.ModuleList([GATConv(in_channels, hidden_channels, heads=heads)])
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
        self.fc = nn.Linear(hidden_channels * heads, out_channels)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs: x = F.elu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)

class FinalLightweightModel(nn.Module):
    def __init__(self, content_path, vision_path, gat_path, gat_cfg, num_classes):
        super().__init__()
        self.content_model = BertModel.from_pretrained(content_path)
        self.vision_model  = CLIPVisionModel.from_pretrained(vision_path)
        self.visual_projection = nn.Linear(self.vision_model.config.hidden_size, 512, bias=False)
        vp_path = os.path.join(vision_path, "visual_projection.pt")
        self.visual_projection.load_state_dict(torch.load(vp_path, map_location="cpu"))
        self.gat_model = GATModel(**gat_cfg)
        state = torch.load(gat_path, map_location="cpu")
        if isinstance(state, dict) and any(k.startswith("module.") for k in state):  # 兼容DP
            state = {k.replace("module.","",1): v for k,v in state.items()}
        self.gat_model.load_state_dict(state)
        cls_in = self.content_model.config.hidden_size + 512 + gat_cfg["out_channels"]
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    def forward(self, content_inputs, pixel_values, graph_batch):
        txt = self.content_model(**content_inputs).pooler_output
        vis = self.vision_model(pixel_values=pixel_values).pooler_output
        vis = self.visual_projection(vis)
        gat = self.gat_model(graph_batch)
        feat = torch.cat([txt, vis, gat], dim=1)
        return self.classifier(feat)

class FinalDataset(Dataset):
    def __init__(self, csv_path, pre_dir, project_root, vision_processor, allowed_indices=None):
        self.project_root = project_root
        self.pre_dir = pre_dir
        self.df = pd.read_csv(csv_path, header=None, names=['label','title','content','image_path'])
        self.proc = vision_processor
        # 只保留存在图结构文件的索引
        base_valid = [i for i in range(len(self.df)) if os.path.exists(os.path.join(self.pre_dir, f"data_{i}.pt"))]
        if allowed_indices is not None:
            allow = set(allowed_indices)
            self.valid_indices = [i for i in base_valid if i in allow]
        else:
            self.valid_indices = base_valid

    def __len__(self): return len(self.valid_indices)

    def __getitem__(self, k):
        idx = self.valid_indices[k]
        row = self.df.iloc[idx]
        label = int(row['label'])
        content = str(row['content'])
        path = str(row['image_path'])
        img = Image.new("RGB",(224,224),"white")
        if path and path.lower() != "nan":
            full = os.path.join(self.project_root, path)
            try:
                if os.path.exists(full): img = Image.open(full).convert("RGB")
            except: pass
        pixel_values = self.proc(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        graph = torch.load(os.path.join(self.pre_dir, f"data_{idx}.pt"), map_location="cpu")
        if hasattr(graph,"x") and graph.x is not None: graph.x = graph.x.detach().to(torch.float32)
        if hasattr(graph,"edge_index") and graph.edge_index is not None: graph.edge_index = graph.edge_index.to(torch.long).contiguous()
        return content, pixel_values, label, graph

def collate_fn(batch):
    contents, pix, labels, graphs = zip(*batch)
    return (list(contents),
        torch.stack(pix, dim=0),
        torch.tensor(labels, dtype=torch.long),
        Batch.from_data_list(list(graphs)))


# -------------------- 10-fold 主流程 --------------------
def run_fold(args, train_idx, val_idx, seed, fold_id):
    device = "cuda" if (args.device=="cuda" and torch.cuda.is_available()) else "cpu"
    use_amp = (device=="cuda")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    tokenizer = BertTokenizer.from_pretrained(args.content_path)
    vision_proc = CLIPProcessor.from_pretrained(args.teacher_clip)

    train_set = FinalDataset(args.csv, args.pre_dir, args.project_root, vision_proc, allowed_indices=train_idx)
    val_set   = FinalDataset(args.csv, args.pre_dir, args.project_root, vision_proc, allowed_indices=val_idx)

    loader_kw = dict(batch_size=args.batch, pin_memory=(device=="cuda"),
                     collate_fn=collate_fn, drop_last=False, num_workers=args.num_workers)
    train_loader = DataLoader(train_set, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_set,   shuffle=False, **loader_kw)

    gat_cfg = dict(in_channels=args.gat_in, hidden_channels=args.gat_hidden,
                   num_layers=args.gat_layers, out_channels=args.gat_out, heads=args.gat_heads)
    model = FinalLightweightModel(args.content_path, args.vision_path, args.gat_path, gat_cfg, args.num_classes).to(device)

    # 加载模型权重
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # 轻量稳妥超参（可按需改大）
    enc_lr, cls_lr = args.lr_enc, args.lr_cls
    optimizer = torch.optim.AdamW([
        {"params": [p for n,p in model.named_parameters() if not n.startswith("classifier.")], "lr": enc_lr},
        {"params": [p for n,p in model.named_parameters() if     n.startswith("classifier.")], "lr": cls_lr},
    ])
    loss_fn = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(1, args.epochs+1):
        model.train()
        for contents, pixel_values, labels, graph_batch in tqdm(train_loader, desc=f"[Fold {fold_id}] Train {epoch}/{args.epochs}", ncols=120):
            content_inputs = tokenizer(contents, return_tensors="pt", max_length=256, padding="max_length", truncation=True)
            content_inputs = {k:v.to(device, non_blocking=True) for k,v in content_inputs.items()}
            pixel_values = pixel_values.to(device, non_blocking=True)
            graph_batch = graph_batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(content_inputs, pixel_values, graph_batch)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()

    # 验证
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        for contents, pixel_values, labels, graph_batch in tqdm(val_loader, desc=f"[Fold {fold_id}] Eval", ncols=120):
            content_inputs = tokenizer(contents, return_tensors="pt", max_length=256, padding="max_length", truncation=True)
            content_inputs = {k:v.to(device, non_blocking=True) for k,v in content_inputs.items()}
            pixel_values = pixel_values.to(device, non_blocking=True)
            graph_batch = graph_batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(content_inputs, pixel_values, graph_batch)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1m = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--content_path", required=True)   # 学生文本目录（with/without coop 二选一跑）
    ap.add_argument("--vision_path", required=True)    # 学生视觉目录
    ap.add_argument("--gat_path", required=True)       # 学生 GAT ckpt
    ap.add_argument("--teacher_clip", default="./model/clip-vit-base-patch32")
    ap.add_argument("--pre_dir", default="./processed_data")
    ap.add_argument("--num_classes", type=int, default=3)
    ap.add_argument("--model_path", required=True)  # 模型路径
    # GAT
    ap.add_argument("--gat_in", type=int, default=100)
    ap.add_argument("--gat_hidden", type=int, default=64)
    ap.add_argument("--gat_layers", type=int, default=1)
    ap.add_argument("--gat_out", type=int, default=64)
    ap.add_argument("--gat_heads", type=int, default=2)
    # 训练
    ap.add_argument("--epochs", type=int, default=3)     # 先小点，拿 std；要严谨可改大
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr_enc", type=float, default=1e-5)
    ap.add_argument("--lr_cls", type=float, default=3e-4)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", default="./kfold_results.csv")
    args = ap.parse_args()

    # 读取标签做分层
    labels = pd.read_csv(args.csv, header=None, usecols=[0]).iloc[:,0].astype(int).values
    # 仅保留存在 graph 的索引，防止某些样本无预处理图
    valid = np.array([i for i in range(len(labels)) if os.path.exists(os.path.join(args.pre_dir, f"data_{i}.pt"))], dtype=int)
    y_valid = labels[valid]

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
    fold_acc, fold_f1 = [], []
    for fold_id, (tr, va) in enumerate(skf.split(valid, y_valid), 1):
        train_idx = valid[tr].tolist()
        val_idx   = valid[va].tolist()
        acc, f1m = run_fold(args, train_idx, val_idx, seed=args.seed+fold_id, fold_id=fold_id)
        print(f"[Fold {fold_id}] Acc={acc:.4f} | Macro-F1={f1m:.4f}")
        fold_acc.append(acc); fold_f1.append(f1m)

    # 统计 + 保存
    acc_mean, acc_std = float(np.mean(fold_acc)), float(np.std(fold_acc, ddof=1))
    f1_mean,  f1_std  = float(np.mean(fold_f1)),  float(np.std(fold_f1,  ddof=1))
    print("\n===== 10-Fold 汇总 =====")
    print(f"Accuracy : {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"Macro-F1 : {f1_mean:.4f} ± {f1_std:.4f}")

    pd.DataFrame({"fold": list(range(1,11)), "acc": fold_acc, "macro_f1": fold_f1}).to_csv(args.out_csv, index=False)
    print(f"明细已保存到: {args.out_csv}")

if __name__ == "__main__":
    main()
