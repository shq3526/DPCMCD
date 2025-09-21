#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估蒸馏/微调后的轻量多模态模型（与 finetune_distilled_models.py 完全对齐）:
- 指标: Accuracy, Macro-F1, Weighted-F1, 每类F1, 混淆矩阵
- 额外: 类别分布、推理速度、模型大小
- 重要: 采用 LTP 动态构图，与微调脚本一致；模型结构也与微调脚本一致（无 visual_projection）
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
from transformers import BertModel, CLIPModel, CLIPProcessor, BertTokenizer

# --------- GAT / PyG ----------
try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import GATConv, global_mean_pool
except Exception as e:
    raise ImportError("需要安装 torch_geometric 及其依赖。") from e

# --------- LTP ----------
from ltp import LTP

# ======== 默认参数（与你上面脚本对齐的路径和超参） ========
DEF_CSV                  = './datasets/TextClassification/toutiao/toutiao_622.csv'
DEF_PROJECT_ROOT         = '/workspace'
DEF_DISTILLED_DIR        = './syc_lightweight_models'  # 存有: student_vision_model.pth / lightweight_gat_model.pth / (BERT save_pretrained)
DEF_ORIG_CONTENT_PATH    = './model/chinese-roberta-wwm-ext'
DEF_ORIG_VISION_PATH     = './model/clip-vit-base-patch32'
DEF_NUM_CLASSES          = 3
DEF_DEVICE               = 'cuda' if torch.cuda.is_available() else 'cpu'
DEF_BATCH                = 32
DEF_TEST_SIZE            = 0.2
DEF_SEED                 = 42
DEF_FINETUNED_CKPT       = './final_distilled_multimodal_model.pth'  # 上面微调脚本保存的权重

# ======== 与微调脚本一致的 GAT 配置与 POS 标签 ========
GAT_CONFIG = {"in_channels": 100, "hidden_channels": 128, "num_layers": 2, "out_channels": 128, "heads": 4}
LTP_POS_LABELS = [
    'a','b','c','d','e','g','h','i','j','k','m','n','nd','nh','ni','nl','ns','nt','nz','o',
    'p','q','r','u','v','wp','ws','x','z'
]

# ================== 模型定义（与微调脚本一致） ==================
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

class DistilledMultiModalModel(nn.Module):
    """BERT(蒸馏目录) + CLIPModel.vision_model(原结构, 载入蒸馏权重) + GAT(载入蒸馏权重)"""
    def __init__(self, content_model_path, vision_weights_path, gat_weights_path, num_classes,
                 original_content_model_path, original_vision_model_path):
        super().__init__()
        # 文本模型：优先从蒸馏目录加载，失败则回退到原始路径（与微调脚本一致）
        try:
            self.content_model = BertModel.from_pretrained(content_model_path)
        except Exception:
            self.content_model = BertModel.from_pretrained(original_content_model_path)

        # 视觉模型结构：用 CLIPModel 取其 vision_model
        clip_model = CLIPModel.from_pretrained(original_vision_model_path)
        self.vision_model = clip_model.vision_model
        if os.path.exists(vision_weights_path):
            self.vision_model.load_state_dict(torch.load(vision_weights_path, map_location='cpu'))

        # GAT
        self.gat_model = GATModel(**GAT_CONFIG)
        if os.path.exists(gat_weights_path):
            self.gat_model.load_state_dict(torch.load(gat_weights_path, map_location='cpu'))

        # 分类头（与微调脚本一致：concat[content, vision, gat] -> [512] -> [256] -> num_classes）
        content_dim = self.content_model.config.hidden_size
        vision_dim  = self.vision_model.config.hidden_size
        gat_dim     = GAT_CONFIG['out_channels']
        in_dim = content_dim + vision_dim + gat_dim
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, content_inputs, pixel_values, graph_batch):
        content_feature = self.content_model(**content_inputs).pooler_output
        vision_outputs  = self.vision_model(pixel_values=pixel_values)
        vision_feature  = vision_outputs.pooler_output
        gat_feature     = self.gat_model(graph_batch)
        feat = torch.cat([content_feature, vision_feature, gat_feature], dim=1)
        return self.classifier(feat)

# ================== 数据集（与微调脚本一致的 LTP 动态构图） ==================
class MultiModalDataset(Dataset):
    def __init__(self, csv_path, project_root):
        self.project_root = project_root
        self.df = pd.read_csv(csv_path, header=None, names=['label','title','content','image_path'])

        # LTP
        print("加载 LTP（small）用于分词/POS/依存 ...")
        self.ltp = LTP("LTP/small")
        self.pos_vocab = {tag: i for i, tag in enumerate(LTP_POS_LABELS)}
        # 随机初始化 POS 向量（与微调脚本一致，作为固定节点特征）
        self.pos_embeddings = torch.randn(len(self.pos_vocab), GAT_CONFIG["in_channels"])

    def __len__(self): return len(self.df)

    def _title_to_graph(self, title):
        if not title or pd.isna(title):
            return None
        try:
            out = self.ltp.pipeline([str(title)], tasks=["cws","pos","dep"])
        except Exception:
            return None

        pos_tags = out.pos[0]
        deps     = out.dep[0]
        heads    = deps['head'] if deps and 'head' in deps else []
        if not pos_tags or not heads:
            return None

        pos_ids = [self.pos_vocab.get(t, 0) for t in pos_tags]
        node_x  = torch.stack([self.pos_embeddings[i] for i in pos_ids]).detach()

        edge_src, edge_tgt = [], []
        for i, h in enumerate(heads):
            hi = h - 1
            if hi >= 0:
                edge_src.append(hi); edge_tgt.append(i)
        if not edge_src:
            edge_index = torch.tensor([[0],[0]], dtype=torch.long)
        else:
            edge_index = torch.tensor([edge_src, edge_tgt], dtype=torch.long)
        return Data(x=node_x, edge_index=edge_index)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row['label'])
        title = str(row['title']) if pd.notna(row['title']) else ""
        content = str(row['content']) if pd.notna(row['content']) else ""
        img_rel = str(row['image_path']) if pd.notna(row['image_path']) else ""

        # image
        image = Image.new('RGB', (224, 224), color='white')
        if img_rel and img_rel.lower() != 'nan':
            try:
                full = os.path.join(self.project_root, img_rel)
                if os.path.exists(full):
                    image = Image.open(full).convert('RGB')
            except Exception:
                pass

        # graph
        g = self._title_to_graph(title)
        if g is None:
            node_x = torch.zeros(1, GAT_CONFIG["in_channels"]).detach()
            edge_index = torch.tensor([[0],[0]], dtype=torch.long)
            g = Data(x=node_x, edge_index=edge_index)

        # detach to be safe
        if g.x is not None: g.x = g.x.detach()
        if g.edge_index is not None: g.edge_index = g.edge_index.detach()

        return content, image, label, g

def collate_fn(batch):
    contents, images, labels, graphs = zip(*batch)
    graphs = [Data(x=g.x.detach() if g.x is not None else None,
                   edge_index=g.edge_index.detach() if g.edge_index is not None else None)
              for g in graphs]
    batched = Batch.from_data_list(graphs)
    return list(contents), list(images), torch.tensor(labels, dtype=torch.long), batched

# ================== 评估流程 ==================
def evaluate(args):
    device = args.device
    print(f"设备: {device}")

    # Tokenizer / Processor：按微调脚本策略
    try:
        tokenizer = BertTokenizer.from_pretrained(args.original_content_model_path)
        print("✓ 使用原始 BERT tokenizer")
    except Exception as e:
        print(f"加载原始 tokenizer 失败: {e}")
        tokenizer = BertTokenizer.from_pretrained(args.distilled_dir)
        print("→ 回退到蒸馏目录 tokenizer")

    if os.path.exists(os.path.join(args.distilled_dir, 'preprocessor_config.json')):
        processor = CLIPProcessor.from_pretrained(args.distilled_dir)
        print("✓ 使用蒸馏目录 CLIP processor")
    else:
        processor = CLIPProcessor.from_pretrained(args.original_vision_model_path)
        print("✓ 使用原始 CLIP processor")

    # 数据
    dataset = MultiModalDataset(args.csv, args.project_root)
    if len(dataset) == 0:
        raise RuntimeError("数据集为空，请检查 CSV。")

    # 分层划分
    y_all = dataset.df['label'].astype(int).values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    _, val_idx = next(sss.split(np.arange(len(dataset)), y_all))

    from torch.utils.data import Subset
    val_ds = Subset(dataset, val_idx.tolist())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    # 显示验证集分布
    val_labels_series = pd.Series(y_all[val_idx]).value_counts().sort_index()
    print("\n[验证集类别分布]")
    for k, v in val_labels_series.items():
        print(f"  类 {k}: {v}")

    # 模型（与微调脚本一致）
    content_path = args.distilled_dir  # BERT 从蒸馏目录加载
    vision_w     = os.path.join(args.distilled_dir, 'student_vision_model.pth')
    gat_w        = os.path.join(args.distilled_dir, 'lightweight_gat_model.pth')
    model = DistilledMultiModalModel(
        content_model_path=content_path,
        vision_weights_path=vision_w,
        gat_weights_path=gat_w,
        num_classes=args.num_classes,
        original_content_model_path=args.original_content_model_path,
        original_vision_model_path=args.original_vision_model_path
    ).to(device)

    # 加载微调权重
    if args.finetuned_ckpt and os.path.exists(args.finetuned_ckpt):
        ckpt = torch.load(args.finetuned_ckpt, map_location=device)
        model.load_state_dict(ckpt, strict=False)
        print(f"\n已加载微调权重: {args.finetuned_ckpt}")
    else:
        print("\n未提供微调权重（或路径不存在），将以当前权重评估。")

    # 模型参数量估计
    n_params = sum(p.numel() for p in model.parameters())
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

    print("\n========== 指标 ==========")
    print(f"Accuracy        : {acc:.4f}")
    print(f"Macro F1        : {f1_macro:.4f}")
    print(f"Weighted F1     : {f1_weighted:.4f}")
    print("\n[每类指标]")
    print(classification_report(all_labels, all_preds, digits=4, zero_division=0))
    cm = confusion_matrix(all_labels, all_preds)
    print("[混淆矩阵]")
    print(cm)

    # 吞吐
    if total_time > 0:
        ips = total_samples / total_time
        print(f"\n[推理速度] {total_samples} 个样本 / {total_time:.2f}s  =>  {ips:.2f} samples/s")

    # 保存报告
    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, 'eval_new_report.json')
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
    ap.add_argument('--project_root', type=str, default=DEF_PROJECT_ROOT)
    ap.add_argument('--distilled_dir', type=str, default=DEF_DISTILLED_DIR, help="蒸馏目录：含BERT(save_pretrained)、student_vision_model.pth、lightweight_gat_model.pth")
    ap.add_argument('--original_content_model_path', type=str, default=DEF_ORIG_CONTENT_PATH)
    ap.add_argument('--original_vision_model_path', type=str, default=DEF_ORIG_VISION_PATH)
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
