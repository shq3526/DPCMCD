#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, argparse, time, gc
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from transformers import BertModel, BertTokenizer, CLIPVisionModel, CLIPProcessor
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from tqdm import tqdm

# PyG
try:
    from torch_geometric.data import Batch
    from torch_geometric.nn import GATConv, global_mean_pool
except ImportError:
    print("❌ 未安装 torch_geometric，请先安装再运行。")
    raise

torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# --------------------------- 模型定义 ---------------------------
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
    def __init__(self, content_path, vision_path, gat_path, gat_cfg, num_classes):
        super().__init__()
        # 文本
        self.content_model = BertModel.from_pretrained(content_path)
        # 视觉
        self.vision_model = CLIPVisionModel.from_pretrained(vision_path)
        self.visual_projection = nn.Linear(self.vision_model.config.hidden_size, 512, bias=False)
        vp_path = os.path.join(vision_path, "visual_projection.pt")
        self.visual_projection.load_state_dict(torch.load(vp_path, map_location="cpu"))
        # GAT
        self.gat_model = GATModel(**gat_cfg)
        state = torch.load(gat_path, map_location="cpu")
        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        self.gat_model.load_state_dict(state)
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


# --------------------------- 数据集 ---------------------------
class FinalDataset(Dataset):
    def __init__(self, csv_path, preprocessed_dir, project_root,
                 vision_processor: CLIPProcessor, cache_graphs=False):
        self.project_root = project_root
        self.pre_dir = preprocessed_dir
        self.df = pd.read_csv(csv_path, header=None, names=['label','title','content','image_path'])
        self.proc = vision_processor
        self.cache_graphs = cache_graphs

        # 只保留有对应预处理图(graph)的样本
        self.valid_indices = [
            i for i in range(len(self.df))
            if os.path.exists(os.path.join(self.pre_dir, f"data_{i}.pt"))
        ]

        # 可选：一次性把图结构读入内存
        self.graph_cache = {}
        if self.cache_graphs:
            for i in self.valid_indices:
                p = os.path.join(self.pre_dir, f"data_{i}.pt")
                self.graph_cache[i] = torch.load(p, map_location="cpu")

        # 训练采样需要的标签序列（与 valid_indices 对齐）
        self.labels_valid = self.df.iloc[self.valid_indices, 0].astype(int).tolist()

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        idx = self.valid_indices[index]
        row = self.df.iloc[idx]
        label = int(row['label'])
        content = str(row['content'])
        path = str(row['image_path'])

        # 图像（缺失则白图）
        img = Image.new("RGB", (224,224), color="white")
        if path and path.lower() != "nan":
            full = os.path.join(self.project_root, path)
            try:
                if os.path.exists(full):
                    img = Image.open(full).convert("RGB")
            except Exception:
                pass

        pixel_values = self.proc(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

        if self.cache_graphs:
            graph_data = self.graph_cache[idx]
        else:
            graph_data = torch.load(os.path.join(self.pre_dir, f"data_{idx}.pt"), map_location="cpu")
        if hasattr(graph_data, "x") and graph_data.x is not None:
            graph_data.x = graph_data.x.detach().to(torch.float32)
        if hasattr(graph_data, "edge_index") and graph_data.edge_index is not None:
            graph_data.edge_index = graph_data.edge_index.to(torch.long).contiguous()

        return content, pixel_values, label, graph_data


def collate_fn(batch):
    contents, pix_list, labels, graphs = zip(*batch)
    pixel_values = torch.stack(pix_list, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    batched_graph = Batch.from_data_list(list(graphs))
    return list(contents), pixel_values, labels, batched_graph


# --------------------------- 采样器/权重/初始化 ---------------------------
def build_sampler(labels_valid, mode="none", beta=0.9995):
    """
    mode = none | inv | cb
      none: 不使用采样器
      inv : 1/n_c 采样
      cb  : Class-Balanced (effective number), beta∈[0.9,0.9999]
    """
    import numpy as np
    if mode == "none":
        return None
    labels = np.array(labels_valid, dtype=int)
    classes, counts = np.unique(labels, return_counts=True)
    count_map = {c: int(n) for c, n in zip(classes, counts)}

    if mode == "inv":
        class_w = {c: 1.0 / count_map[c] for c in count_map}
    elif mode == "cb":
        eff = {c: (1.0 - beta**count_map[c]) for c in count_map}
        class_w = {c: (1.0 - beta) / max(eff[c], 1e-8) for c in count_map}
    else:
        raise ValueError("sampler_mode 必须是 none/inv/cb")

    weights = torch.tensor([class_w[int(y)] for y in labels], dtype=torch.float)
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


def init_classifier_bias_with_priors(model: nn.Module, priors, to="dataset"):
    """
    to = dataset | uniform | none
    - dataset: 用数据先验 log π_i
    - uniform: 全 1/K（当使用强力采样均衡时可选）
    - none: 不改
    """
    if to == "none":
        return
    K = len(priors)
    if to == "uniform":
        priors = [1.0 / K] * K
    with torch.no_grad():
        last = model.classifier[-1]
        bias = torch.log(torch.clamp(torch.tensor(priors, dtype=torch.float32), min=1e-6))
        if last.bias is None:
            last.bias = nn.Parameter(bias.clone())
        else:
            last.bias.copy_(bias)


# --------------------------- 训练/评估 ---------------------------
def run(args):
    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    use_amp = (device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"--- 开始最终模型微调 (设备: {device}) ---")

    tokenizer = BertTokenizer.from_pretrained(args.content_path)
    vision_proc = CLIPProcessor.from_pretrained(args.teacher_clip)

    dataset = FinalDataset(args.csv, args.pre_dir, args.project_root,
                           vision_processor=vision_proc, cache_graphs=args.cache_graphs)

    # 类别统计（基于有效样本）
    valid_labels_series = pd.Series(dataset.labels_valid)
    counts = valid_labels_series.value_counts().sort_index()
    total = counts.sum()
    priors = [counts.get(i, 0) / total for i in range(args.num_classes)]

    # —— 采样器（默认 none，避免过度矫正）——
    sampler = build_sampler(dataset.labels_valid, mode=args.sampler_mode, beta=args.cb_beta)

    # DataLoader
    loader_kw = dict(batch_size=args.batch, pin_memory=(device == "cuda"),
                     collate_fn=collate_fn, drop_last=False)
    if args.num_workers > 0:
        loader_kw.update(num_workers=args.num_workers, persistent_workers=True, prefetch_factor=args.prefetch)

    if sampler is None:
        train_loader = DataLoader(dataset, shuffle=True, **loader_kw)
    else:
        train_loader = DataLoader(dataset, sampler=sampler, **loader_kw)
    eval_loader = DataLoader(dataset, shuffle=False, **loader_kw)

    # 模型
    gat_cfg = dict(in_channels=args.gat_in, hidden_channels=args.gat_hidden,
                   num_layers=args.gat_layers, out_channels=args.gat_out, heads=args.gat_heads)
    model = FinalLightweightModel(args.content_path, args.vision_path, args.gat_path, gat_cfg, args.num_classes).to(device)

    # 分类器 bias 初始化（默认对齐真实数据先验）
    init_classifier_bias_with_priors(model, priors, to=args.bias_to)

    # 可选消融
    if args.disable_vision:
        print("👉 已禁用视觉分支（Text + GAT）")
        model.vision_model.requires_grad_(False)
        model.visual_projection.requires_grad_(False)
        def zero_vision(x): return torch.zeros(x.size(0), 512, device=x.device)
        model.forward_vision = zero_vision
    if args.disable_gat:
        print("👉 已禁用 GAT 分支（Text + Vision）")
        model.gat_model.requires_grad_(False)
        def zero_gat(b): return torch.zeros(b.num_graphs, gat_cfg["out_channels"], device=b.x.device)
        model.forward_gat = zero_gat

    # 损失：是否使用类别权重（默认 True；若用采样器建议关掉或降低）
    if args.use_class_weights:
        class_w = torch.tensor([counts.get(i, 0) for i in range(args.num_classes)], dtype=torch.float32)
        if class_w.min() > 0:
            class_w = class_w.sum() / (len(class_w) * class_w)
            class_w = class_w.to(device)
        else:
            class_w = None
    else:
        class_w = None
    loss_fn = nn.CrossEntropyLoss(weight=class_w, label_smoothing=args.label_smoothing)

    # 优化器（编码器/分类头分组学习率）
    enc_params, cls_params = [], []
    for n, p in model.named_parameters():
        (cls_params if n.startswith("classifier.") else enc_params).append(p)
    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": args.lr_enc},
        {"params": cls_params, "lr": args.lr_cls},
    ])
    grad_clip = 1.0

    # 冻结策略
    if args.freeze_first > 0:
        for n, p in model.named_parameters():
            if not n.startswith("classifier"):
                p.requires_grad_(False)
        print(f"前 {args.freeze_first} 个 epoch 冻结编码器，只训练分类头。")

    # 训练/评估
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_first > 0 and epoch == args.freeze_first + 1:
            for p in model.parameters(): p.requires_grad_(True)
            print("🔓 解冻全部编码器，联合微调。")

        ep_t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}", ncols=120)
        running, seen = 0.0, 0

        for contents, pixel_values, labels, graph_batch in pbar:
            content_inputs = tokenizer(contents, return_tensors="pt", max_length=256,
                                       padding="max_length", truncation=True)
            content_inputs = {k: v.to(device, non_blocking=True) for k, v in content_inputs.items()}
            pixel_values = pixel_values.to(device, non_blocking=True)
            graph_batch = graph_batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                if hasattr(model, "forward_vision"):
                    vis_feat = model.forward_vision(pixel_values)
                    txt_feat = model.content_model(**content_inputs).pooler_output
                    gat_feat = model.forward_gat(graph_batch) if hasattr(model, "forward_gat") else model.gat_model(graph_batch)
                    logits = model.classifier(torch.cat([txt_feat, vis_feat, gat_feat], dim=1))
                elif hasattr(model, "forward_gat"):
                    txt_feat = model.content_model(**content_inputs).pooler_output
                    vis_feat = model.visual_projection(model.vision_model(pixel_values=pixel_values).pooler_output)
                    gat_feat = model.forward_gat(graph_batch)
                    logits = model.classifier(torch.cat([txt_feat, vis_feat, gat_feat], dim=1))
                else:
                    logits = model(content_inputs, pixel_values, graph_batch)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()

            # 进度条
            lv = float(loss.detach())
            running = 0.9 * running + 0.1 * lv if seen > 0 else lv
            seen += labels.size(0)
            ips = seen / max(time.time() - ep_t0, 1e-6)
            pbar.set_postfix(loss=f"{lv:.4f}", avg=f"{running:.4f}", ips=f"{ips:.1f}/s")

        # 评估
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            for contents, pixel_values, labels, graph_batch in tqdm(eval_loader, desc=f"Eval  {epoch}/{args.epochs}", ncols=120):
                content_inputs = tokenizer(contents, return_tensors="pt", max_length=256,
                                           padding="max_length", truncation=True)
                content_inputs = {k: v.to(device, non_blocking=True) for k, v in content_inputs.items()}
                pixel_values = pixel_values.to(device, non_blocking=True)
                graph_batch = graph_batch.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                if hasattr(model, "forward_vision"):
                    vis_feat = model.forward_vision(pixel_values)
                    txt_feat = model.content_model(**content_inputs).pooler_output
                    gat_feat = model.forward_gat(graph_batch) if hasattr(model, "forward_gat") else model.gat_model(graph_batch)
                    logits = model.classifier(torch.cat([txt_feat, vis_feat, gat_feat], dim=1))
                elif hasattr(model, "forward_gat"):
                    txt_feat = model.content_model(**content_inputs).pooler_output
                    vis_feat = model.visual_projection(model.vision_model(pixel_values=pixel_values).pooler_output)
                    gat_feat = model.forward_gat(graph_batch)
                    logits = model.classifier(torch.cat([txt_feat, vis_feat, gat_feat], dim=1))
                else:
                    logits = model(content_inputs, pixel_values, graph_batch)

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        acc = accuracy_score(all_labels, all_preds)
        f1m = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        prec, rec, f1c, sup = precision_recall_fscore_support(
            all_labels, all_preds, labels=list(range(args.num_classes)), zero_division=0
        )
        dt = time.time() - ep_t0
        print(f"\nEpoch {epoch}/{args.epochs} | time {dt:.1f}s | Acc {acc:.4f} | Macro-F1 {f1m:.4f}")
        for i in range(args.num_classes):
            print(f"  Class {i}: P={prec[i]:.4f} R={rec[i]:.4f} F1={f1c[i]:.4f} n={sup[i]}")

    torch.save(model.state_dict(), args.save_path)
    print(f"\n✅ 微调完成，模型已保存到: {args.save_path}")
    try:
        del train_loader, eval_loader, dataset, model, optimizer, loss_fn, tokenizer, vision_proc
    except Exception:
        pass
    gc.collect()
    if device == "cuda":
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        try: torch.cuda.ipc_collect()
        except Exception: pass


# --------------------------- CLI ---------------------------
def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--project_root", type=str, required=True)
    ap.add_argument("--content_path", type=str, required=True)
    ap.add_argument("--vision_path", type=str, required=True)
    ap.add_argument("--gat_path", type=str, required=True)
    ap.add_argument("--teacher_clip", type=str, default="./model/clip-vit-base-patch32")
    ap.add_argument("--pre_dir", type=str, default="./processed_data")
    ap.add_argument("--num_classes", type=int, default=3)

    # GAT 结构
    ap.add_argument("--gat_in", type=int, default=100)
    ap.add_argument("--gat_hidden", type=int, default=64)
    ap.add_argument("--gat_layers", type=int, default=1)
    ap.add_argument("--gat_out", type=int, default=64)
    ap.add_argument("--gat_heads", type=int, default=2)

    # 训练
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr_cls", type=float, default=3e-4)
    ap.add_argument("--lr_enc", type=float, default=1e-5)
    ap.add_argument("--freeze_first", type=int, default=0)
    ap.add_argument("--label_smoothing", type=float, default=0.02)
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))

    # DataLoader
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--cache_graphs", action="store_true")

    # 消融
    ap.add_argument("--disable_vision", action="store_true")
    ap.add_argument("--disable_gat", action="store_true")

    # 采样器/权重/偏置
    ap.add_argument("--sampler_mode", type=str, default="none", choices=["none","inv","cb"])
    ap.add_argument("--cb_beta", type=float, default=0.9995)
    ap.add_argument("--use_class_weights", action="store_true")  # 打开=用权重；默认关闭更稳
    ap.add_argument("--bias_to", type=str, default="dataset", choices=["dataset","uniform","none"])

    ap.add_argument("--save_path", type=str, default="./final_lightweight_model.pth")
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)
