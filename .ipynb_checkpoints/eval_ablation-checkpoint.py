#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_new_ablation.py
对与你的 eval_new.py 一致的蒸馏/微调多模态模型做消融实验（零向量遮蔽，严格加载同一权重）：
- 模式：Full / w/o Visual / w/o Syntactic / Only Content
- 指标：Accuracy / Macro Precision / Macro Recall / Macro F1-Score
- 其余流程（LTP 动态构图、tokenizer/processor 选择、数据划分等）与 eval_new.py 保持一致
"""

import os, time, json, argparse
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, CLIPModel, CLIPProcessor, BertTokenizer

# --------- GAT / PyG ----------
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, global_mean_pool

# --------- LTP ----------
from ltp import LTP

# ======== 默认参数（与原评估脚本保持一致） ========
DEF_CSV                  = './datasets/TextClassification/toutiao/toutiao_622.csv'
DEF_PROJECT_ROOT         = '/workspace'
DEF_DISTILLED_DIR        = './syc_lightweight_models'  # student_vision_model.pth / lightweight_gat_model.pth / BERT(save_pretrained)
DEF_ORIG_CONTENT_PATH    = './model/chinese-roberta-wwm-ext'
DEF_ORIG_VISION_PATH     = './model/clip-vit-base-patch32'
DEF_NUM_CLASSES          = 3
DEF_DEVICE               = 'cuda' if torch.cuda.is_available() else 'cpu'
DEF_BATCH                = 32
DEF_TEST_SIZE            = 0.2
DEF_SEED                 = 42
DEF_FINETUNED_CKPT       = './final_distilled_multimodal_model.pth'
DEF_OUT_DIR              = './eval_outputs'
DEF_OUT_CSV              = 'eval_new_ablation_results.csv'

# ======== 与微调脚本一致的 GAT 配置与 POS 标签 ========
GAT_CONFIG = {"in_channels": 100, "hidden_channels": 128, "num_layers": 2, "out_channels": 128, "heads": 4}
LTP_POS_LABELS = [
    'a','b','c','d','e','g','h','i','j','k','m','n','nd','nh','ni','nl','ns','nt','nz','o',
    'p','q','r','u','v','wp','ws','x','z'
]

# ================== 模型定义（与原结构一致，加入 ablation 模式） ==================
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
    BERT(蒸馏目录) + CLIPModel.vision_model(载入student权重) + GAT(载入student权重)
    ablation 通过零向量遮蔽，不改分类器维度，从而可 strict 加载同一份微调权重。
    """
    def __init__(self, content_model_path, vision_weights_path, gat_weights_path, num_classes,
                 original_content_model_path, original_vision_model_path, mode: str = "full"):
        super().__init__()
        assert mode in ["full", "no_visual", "no_syntactic", "only_content"]
        self.mode = mode

        # 文本
        try:
            self.content_model = BertModel.from_pretrained(content_model_path)
        except Exception:
            self.content_model = BertModel.from_pretrained(original_content_model_path)

        # 视觉（取 CLIP 的 vision_model）
        clip_model = CLIPModel.from_pretrained(original_vision_model_path)
        self.vision_model = clip_model.vision_model
        if os.path.exists(vision_weights_path):
            self.vision_model.load_state_dict(torch.load(vision_weights_path, map_location='cpu'))

        # GAT
        self.gat_model = GATModel(**GAT_CONFIG)
        if os.path.exists(gat_weights_path):
            self.gat_model.load_state_dict(torch.load(gat_weights_path, map_location='cpu'))

        # 分类头：与原评估/微调脚本一致（content + vision + gat）
        self.content_dim = self.content_model.config.hidden_size
        self.vision_dim  = self.vision_model.config.hidden_size
        self.gat_dim     = GAT_CONFIG['out_channels']

        in_dim = self.content_dim + self.vision_dim + self.gat_dim
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    @torch.no_grad()
    def _zeros(self, bsz, dim, device):
        return torch.zeros(bsz, dim, device=device, dtype=torch.float32)

    def forward(self, content_inputs, pixel_values, graph_batch):
        content_feature = self.content_model(**content_inputs).pooler_output
        bsz = content_feature.size(0); device = content_feature.device

        # 视觉：full / no_syntactic 使用；否则置零
        if self.mode in ["full", "no_syntactic"]:
            vision_out = self.vision_model(pixel_values=pixel_values)
            vision_feature = vision_out.pooler_output
        else:
            vision_feature = self._zeros(bsz, self.vision_dim, device)

        # GAT：full / no_visual 使用；否则置零
        if self.mode in ["full", "no_visual"]:
            gat_feature = self.gat_model(graph_batch)
        else:
            gat_feature = self._zeros(bsz, self.gat_dim, device)

        feat = torch.cat([content_feature, vision_feature, gat_feature], dim=1)
        return self.classifier(feat)

# ================== 数据集（LTP 动态构图，与原评估一致） ==================
class MultiModalDataset(Dataset):
    def __init__(self, csv_path, project_root):
        self.project_root = project_root
        self.df = pd.read_csv(csv_path, header=None, names=['label','title','content','image_path'])

        print("加载 LTP（small）用于分词/POS/依存 ...")
        self.ltp = LTP("LTP/small")
        self.pos_vocab = {tag: i for i, tag in enumerate(LTP_POS_LABELS)}
        self.pos_embeddings = torch.randn(len(self.pos_vocab), GAT_CONFIG["in_channels"])

    def __len__(self): return len(self.df)

    def _title_to_graph(self, title):
        if not title or pd.isna(title): return None
        try:
            out = self.ltp.pipeline([str(title)], tasks=["cws","pos","dep"])
        except Exception:
            return None

        pos_tags = out.pos[0]
        deps     = out.dep[0]
        heads    = deps['head'] if deps and 'head' in deps else []
        if not pos_tags or not heads: return None

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
        label   = int(row['label'])
        title   = str(row['title']) if pd.notna(row['title']) else ""
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

# ================== 统一评估（返回四项指标） ==================
def run_eval_one_mode(args, mode_key, tokenizer, processor, val_loader):
    device = args.device

    content_path = args.distilled_dir
    vision_w     = os.path.join(args.distilled_dir, 'student_vision_model.pth')
    gat_w        = os.path.join(args.distilled_dir, 'lightweight_gat_model.pth')

    model = DistilledMultiModalModel(
        content_model_path=content_path,
        vision_weights_path=vision_w,
        gat_weights_path=gat_w,
        num_classes=args.num_classes,
        original_content_model_path=args.original_content_model_path,
        original_vision_model_path=args.original_vision_model_path,
        mode=mode_key
    ).to(device)

    if args.finetuned_ckpt and os.path.exists(args.finetuned_ckpt):
        ckpt = torch.load(args.finetuned_ckpt, map_location=device)
        # 分类头维度未改，严格加载
        model.load_state_dict(ckpt, strict=True)
        print(f"✓ Loaded finetuned checkpoint: {args.finetuned_ckpt}")
    else:
        print("⚠️ 未提供微调权重（或不存在），以当前权重评估。")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for contents, images, labels, graph_batch in tqdm(val_loader, desc=f"评估中 ({mode_key})"):
            inputs = tokenizer(contents, return_tensors='pt', max_length=256,
                               padding='max_length', truncation=True).to(device)

            # 只有在需要视觉分支时才编码图像；其它模式下传 dummy 张量仅为占位（不会被用到）
            if mode_key in ('full', 'no_syntactic'):
                pixel_values = processor(images=images, return_tensors='pt')['pixel_values'].to(device)
            else:
                # 传入一个与 batch 一致的零张量，避免 processor 调用开销
                bsz = len(images)
                pixel_values = torch.zeros(bsz, 3, 224, 224, device=device, dtype=torch.float32)

            graph_batch = graph_batch.to(device)
            labels = labels.to(device)

            logits = model(inputs, pixel_values, graph_batch)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    mp, mr, mf1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    return acc, mp, mr, mf1

# ================== 主流程：准备数据 + 循环四种模式 ==================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, default=DEF_CSV)
    ap.add_argument('--project_root', type=str, default=DEF_PROJECT_ROOT)
    ap.add_argument('--distilled_dir', type=str, default=DEF_DISTILLED_DIR)
    ap.add_argument('--original_content_model_path', type=str, default=DEF_ORIG_CONTENT_PATH)
    ap.add_argument('--original_vision_model_path', type=str, default=DEF_ORIG_VISION_PATH)
    ap.add_argument('--finetuned_ckpt', type=str, default=DEF_FINETUNED_CKPT)
    ap.add_argument('--num_classes', type=int, default=DEF_NUM_CLASSES)
    ap.add_argument('--device', type=str, default=DEF_DEVICE)
    ap.add_argument('--batch_size', type=int, default=DEF_BATCH)
    ap.add_argument('--test_size', type=float, default=DEF_TEST_SIZE)
    ap.add_argument('--seed', type=int, default=DEF_SEED)
    ap.add_argument('--out_dir', type=str, default=DEF_OUT_DIR)
    ap.add_argument('--out_csv', type=str, default=DEF_OUT_CSV)
    args = ap.parse_args()

    device = args.device
    print(f"设备: {device}")

    # Tokenizer / Processor 与原评估脚本策略一致
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

    # 数据集与分层划分
    dataset = MultiModalDataset(args.csv, args.project_root)
    y_all = dataset.df['label'].astype(int).values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    _, val_idx = next(sss.split(np.arange(len(dataset)), y_all))
    from torch.utils.data import Subset
    val_ds = Subset(dataset, val_idx.tolist())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    # 循环四种模式
    modes = [
        ('full',         'Full'),
        ('no_visual',    'w/o Visual'),
        ('no_syntactic', 'w/o Syntactic'),
        ('only_content', 'Only Content'),
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for key, label in modes:
        print(f"\n=== 评估模式: {label} ===")
        acc, mp, mr, mf1 = run_eval_one_mode(args, key, tokenizer, processor, val_loader)
        print(f"Accuracy={acc:.4f} | Macro Precision={mp:.4f} | Macro Recall={mr:.4f} | Macro F1={mf1:.4f}")
        rows.append({
            "Setting": label,
            "Accuracy": acc,
            "Macro Precision": mp,
            "Macro Recall": mr,
            "Macro F1-Score": mf1
        })

    # 保存 CSV（可直接用于论文表格）
    df = pd.DataFrame(rows, columns=["Setting","Accuracy","Macro Precision","Macro Recall","Macro F1-Score"])
    out_csv_path = os.path.join(args.out_dir, args.out_csv)
    df.to_csv(out_csv_path, index=False)
    print(f"\n📄 消融结果已保存: {out_csv_path}")
    print(df.to_string(index=False, formatters={
        'Accuracy': lambda x: f"{x:.4f}",
        'Macro Precision': lambda x: f"{x:.4f}",
        'Macro Recall': lambda x: f"{x:.4f}",
        'Macro F1-Score': lambda x: f"{x:.4f}",
    }))

if __name__ == "__main__":
    main()
