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
from tqdm import tqdm

from transformers import BertModel, BertTokenizer, CLIPModel, CLIPProcessor

from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, global_mean_pool

# 加速
torch.backends.cudnn.benchmark = True
try: torch.set_float32_matmul_precision("high")
except: pass

# LTP
from ltp import LTP

# ---------------- 与 finetune_distilled_models.py 完全对齐的配置 ----------------
GAT_CONFIG_DEFAULT = dict(in_channels=100, hidden_channels=128, num_layers=2, out_channels=128, heads=4)
LTP_POS_LABELS = [
    'a','b','c','d','e','g','h','i','j','k','m','n','nd','nh','ni','nl','ns','nt','nz','o',
    'p','q','r','u','v','wp','ws','x','z'
]

# ---------------- 模型定义（与微调脚本一致） ----------------
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
            x = F.elu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)

class DistilledMultiModalModel(nn.Module):
    """
    BERT(蒸馏目录) + CLIPModel.vision_model(原结构, 载入蒸馏权重) + GAT(载入蒸馏权重)
    分类头：concat[content, vision, gat] -> 512 -> 256 -> num_classes
    """
    def __init__(self, content_path, vision_weights_path, gat_path, gat_cfg, num_classes,
                 original_content_path, teacher_clip_path):
        super().__init__()
        # 文本
        try:
            self.content_model = BertModel.from_pretrained(content_path)
        except Exception:
            self.content_model = BertModel.from_pretrained(original_content_path)

        # 视觉结构：用 CLIPModel 取 vision_model
        clip_model = CLIPModel.from_pretrained(teacher_clip_path)
        self.vision_model = clip_model.vision_model
        if vision_weights_path and os.path.exists(vision_weights_path):
            state_v = torch.load(vision_weights_path, map_location="cpu")
            if isinstance(state_v, dict) and any(k.startswith("module.") for k in state_v):
                state_v = {k.replace("module.","",1): v for k,v in state_v.items()}
            self.vision_model.load_state_dict(state_v, strict=False)

        # GAT
        self.gat_model = GATModel(**gat_cfg)
        if gat_path and os.path.exists(gat_path):
            state_g = torch.load(gat_path, map_location="cpu")
            if isinstance(state_g, dict) and any(k.startswith("module.") for k in state_g):
                state_g = {k.replace("module.","",1): v for k,v in state_g.items()}
            self.gat_model.load_state_dict(state_g, strict=False)

        # 分类头
        txt_dim = self.content_model.config.hidden_size
        vis_dim = self.vision_model.config.hidden_size
        gat_dim = gat_cfg["out_channels"]
        cls_in = txt_dim + vis_dim + gat_dim
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),   nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, content_inputs, pixel_values, graph_batch):
        txt = self.content_model(**content_inputs).pooler_output
        vis = self.vision_model(pixel_values=pixel_values).pooler_output
        gat = self.gat_model(graph_batch)
        feat = torch.cat([txt, vis, gat], dim=1)
        return self.classifier(feat)

# ---------------- 数据集（LTP 在线构图，与微调脚本一致） ----------------
class LTPCollateDataset(Dataset):
    def __init__(self, csv_path, project_root):
        self.project_root = project_root
        self.df = pd.read_csv(csv_path, header=None, names=['label','title','content','image_path'])
        # LTP
        self.ltp = LTP("LTP/small")
        self.pos_vocab = {tag: i for i, tag in enumerate(LTP_POS_LABELS)}
        self.pos_embeddings = torch.randn(len(self.pos_vocab), GAT_CONFIG_DEFAULT["in_channels"])

    def __len__(self): return len(self.df)

    def _title_to_graph(self, title):
        if not title or pd.isna(title): return None
        try:
            out = self.ltp.pipeline([str(title)], tasks=["cws","pos","dep"])
        except Exception:
            return None
        pos_tags = out.pos[0]
        deps = out.dep[0]
        heads = deps['head'] if deps and 'head' in deps else []
        if not pos_tags or not heads:
            return None
        pos_ids = [self.pos_vocab.get(t, 0) for t in pos_tags]
        node_x = torch.stack([self.pos_embeddings[i] for i in pos_ids]).detach()
        edge_src, edge_tgt = [], []
        for i, h in enumerate(heads):
            hi = h - 1
            if hi >= 0:
                edge_src.append(hi); edge_tgt.append(i)
        edge_index = torch.tensor([[0],[0]], dtype=torch.long) if not edge_src \
                     else torch.tensor([edge_src, edge_tgt], dtype=torch.long)
        return Data(x=node_x, edge_index=edge_index)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row['label'])
        title = str(row['title']) if pd.notna(row['title']) else ""
        content = str(row['content']) if pd.notna(row['content']) else ""
        path = str(row['image_path']) if pd.notna(row['image_path']) else ""
        img = Image.new("RGB",(224,224),"white")
        if path and path.lower() != "nan":
            full = os.path.join(self.project_root, path)
            try:
                if os.path.exists(full): img = Image.open(full).convert("RGB")
            except: pass

        g = self._title_to_graph(title)
        if g is None:
            node_x = torch.zeros(1, GAT_CONFIG_DEFAULT["in_channels"]).detach()
            edge_index = torch.tensor([[0],[0]], dtype=torch.long)
            g = Data(x=node_x, edge_index=edge_index)
        if hasattr(g,"x") and g.x is not None: g.x = g.x.detach().to(torch.float32)
        if hasattr(g,"edge_index") and g.edge_index is not None: g.edge_index = g.edge_index.to(torch.long).contiguous()
        return content, img, label, g

def collate_fn(processor):
    def _fn(batch):
        contents, images, labels, graphs = zip(*batch)
        pixel_values = processor(images=list(images), return_tensors="pt")["pixel_values"]
        batched_graph = Batch.from_data_list(list(graphs))
        return list(contents), pixel_values, torch.tensor(labels, dtype=torch.long), batched_graph
    return _fn

# -------------------- 单折流程（支持 epochs=0 纯评测） --------------------
def run_fold(args, train_idx, val_idx, seed, fold_id):
    device = "cuda" if (args.device=="cuda" and torch.cuda.is_available()) else "cpu"
    use_amp = (device=="cuda")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    # tokenizer 与 processor
    try:
        tokenizer = BertTokenizer.from_pretrained(args.original_content_path)
    except Exception:
        tokenizer = BertTokenizer.from_pretrained(args.content_path)

    # 优先用学生目录的 preprocessor（如果有），否则用 teacher_clip
    if os.path.exists(os.path.join(args.vision_dir_or_distilled, "preprocessor_config.json")):
        vision_proc = CLIPProcessor.from_pretrained(args.vision_dir_or_distilled)
    else:
        vision_proc = CLIPProcessor.from_pretrained(args.teacher_clip)

    full_ds = LTPCollateDataset(args.csv, args.project_root)
    from torch.utils.data import Subset
    train_set = Subset(full_ds, train_idx.tolist())
    val_set   = Subset(full_ds, val_idx.tolist())

    loader_kw = dict(batch_size=args.batch, pin_memory=(device=="cuda"),
                     collate_fn=collate_fn(vision_proc), drop_last=False, num_workers=args.num_workers)
    train_loader = DataLoader(train_set, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_set,   shuffle=False, **loader_kw)

    # GAT 配置对齐
    gat_cfg = dict(
        in_channels=args.gat_in, hidden_channels=args.gat_hidden,
        num_layers=args.gat_layers, out_channels=args.gat_out, heads=args.gat_heads
    )

    # 视觉学生权重：优先显式参数，其次尝试 vision_dir_or_distilled/student_vision_model.pth
    vision_weights_path = args.vision_weights_path
    if (vision_weights_path is None or not os.path.exists(vision_weights_path)):
        maybe = os.path.join(args.vision_dir_or_distilled, "student_vision_model.pth")
        if os.path.exists(maybe):
            vision_weights_path = maybe

    model = DistilledMultiModalModel(
        content_path=args.content_path,
        vision_weights_path=vision_weights_path,
        gat_path=args.gat_path,
        gat_cfg=gat_cfg,
        num_classes=args.num_classes,
        original_content_path=args.original_content_path,
        teacher_clip_path=args.teacher_clip
    ).to(device)

    # 加载微调总权重（即你要评测的 pth）
    if args.model_path and os.path.exists(args.model_path):
        state_all = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_all, strict=False)
        print(f"[Fold {fold_id}] 已加载微调/最终权重: {args.model_path}")
    else:
        print(f"[Fold {fold_id}] ⚠ 未找到 --model_path，使用当前权重直接评估。")

    # 优化器与损失（仅当 epochs > 0 时用于微调）
    enc_lr, cls_lr = args.lr_enc, args.lr_cls
    optimizer = torch.optim.AdamW([
        {"params": [p for n,p in model.named_parameters() if not n.startswith("classifier.")], "lr": enc_lr},
        {"params": [p for n,p in model.named_parameters() if     n.startswith("classifier.")], "lr": cls_lr},
    ])
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 可选微调：默认 epochs=0 直接评测 pth
    if args.epochs > 0:
        for epoch in range(1, args.epochs+1):
            model.train()
            for contents, pixel_values, labels, graph_batch in tqdm(
                train_loader, desc=f"[Fold {fold_id}] Train {epoch}/{args.epochs}", ncols=120
            ):
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

    # 学生（蒸馏后）BERT 目录；里边应有 save_pretrained 的 tokenizer/model（或至少 model）
    ap.add_argument("--content_path", required=True)

    # 视觉：可传“蒸馏目录或任一目录”（用于找 preprocessor_config.json），不再从这里加载模型
    ap.add_argument("--vision_dir_or_distilled", default="./syc_lightweight_models")

    # 学生视觉权重（通常是 syc_lightweight_models/student_vision_model.pth）
    ap.add_argument("--vision_weights_path", default=None)

    # 学生 GAT ckpt
    ap.add_argument("--gat_path", required=True)

    # 教师 CLIP（用于拿 vision_model 结构 & 作为 processor 回退）
    ap.add_argument("--teacher_clip", default="./model/clip-vit-base-patch32")

    # 原始 BERT（tokenizer 优先用它）
    ap.add_argument("--original_content_path", default="./model/chinese-roberta-wwm-ext")

    # 要评测的微调/最终权重（整网 pth）
    ap.add_argument("--model_path", required=True)

    ap.add_argument("--num_classes", type=int, default=3)

    # GAT（默认与微调脚本一致）
    ap.add_argument("--gat_in", type=int, default=100)
    ap.add_argument("--gat_hidden", type=int, default=128)
    ap.add_argument("--gat_layers", type=int, default=2)
    ap.add_argument("--gat_out", type=int, default=128)
    ap.add_argument("--gat_heads", type=int, default=4)

    # 训练（默认 0：纯评测）
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr_enc", type=float, default=1e-5)
    ap.add_argument("--lr_cls", type=float, default=3e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", default="./kfold_results.csv")
    args = ap.parse_args()

    # 分层标签
    labels = pd.read_csv(args.csv, header=None, usecols=[0]).iloc[:,0].astype(int).values
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)

    fold_acc, fold_f1 = [], []
    for fold_id, (tr, va) in enumerate(skf.split(np.arange(len(labels)), labels), 1):
        acc, f1m = run_fold(args, tr, va, seed=args.seed+fold_id, fold_id=fold_id)
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
