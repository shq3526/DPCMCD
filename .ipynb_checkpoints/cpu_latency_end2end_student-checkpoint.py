#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cpu_latency_end2end_student.py
- 在 CPU 上对“蒸馏后轻量多模态学生模型（文本+BERT, 视觉+CLIPVision, 语法图+GAT）”做端到端评测
- 计时：Tokenizer / ImageProc / ModelFwd / End-to-End
- 数据来自 CSV（含文本/图片路径）+ 预处理的图（processed_data/*.pt）
- 产出：屏幕 + JSON + CSV（与 teacher 脚本一致）
"""

import os, time, json, argparse, random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

try:
    import psutil
    PSUTIL_OK = True
except Exception:
    PSUTIL_OK = False

from transformers import BertTokenizer, BertModel, CLIPVisionModel, CLIPProcessor
from tqdm import tqdm

from torch_geometric.data import Data as GeoData, Batch
from torch_geometric.nn import GATConv, global_mean_pool


# ----------------- Dataset -----------------
class FinalDataset(Dataset):
    def __init__(self, csv_path, pre_dir, proj_root, images_root=None):
        """
        csv: 4列 [label, title, content, image_path]
        pre_dir: 预处理图数据目录 processed_data/ 下的 data_{i}.pt
        images_root: 图片根目录（若 csv 的 image_path 已含完整相对路径，可留空）
        """
        self.root = proj_root
        self.pre_dir = pre_dir
        self.df = pd.read_csv(csv_path, header=None, names=['label','title','content','image_path'])
        self.images_root = images_root
        self.valid_idx = [i for i in range(len(self.df))
                          if os.path.exists(os.path.join(self.pre_dir, f'data_{i}.pt'))]

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, i):
        idx = self.valid_idx[i]
        r = self.df.iloc[idx]
        label = int(r['label'])
        title = str(r['title']) if not pd.isna(r['title']) else ""
        content = str(r['content']) if not pd.isna(r['content']) else ""
        text = title + " " + content

        # image
        img = Image.new('RGB', (224,224), 'white')
        p = str(r['image_path']) if not pd.isna(r['image_path']) else ""
        if p and p.lower() != 'nan':
            cands = []
            if self.images_root:
                cands.append(os.path.join(self.images_root, p))
            cands.append(p)  # 原始路径也试一次
            found = None
            for c in cands:
                if os.path.exists(c):
                    found = c; break
            if found:
                try:
                    img = Image.open(found).convert('RGB')
                except Exception:
                    pass

        # graph
        g = torch.load(os.path.join(self.pre_dir, f'data_{idx}.pt'), map_location='cpu')
        if hasattr(g,'x') and g.x is not None:
            g.x = g.x.detach().requires_grad_(False).to(torch.float32)
        if hasattr(g,'edge_index') and g.edge_index is not None:
            g.edge_index = g.edge_index.to(torch.long).contiguous()

        return text, img, label, g


def collate_fn(batch):
    texts, images, labels, graphs = zip(*batch)
    return list(texts), list(images), torch.tensor(labels, dtype=torch.long), Batch.from_data_list(list(graphs))


# ----------------- Model -----------------
class GATModel(nn.Module):
    def __init__(self, in_channels=100, hidden_channels=64, num_layers=1, out_channels=64, heads=2):
        super().__init__()
        self.convs = nn.ModuleList([GATConv(in_channels, hidden_channels, heads=heads)])
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels*heads, hidden_channels, heads=heads))
        self.fc = nn.Linear(hidden_channels*heads, out_channels)

    def forward(self, data: Batch):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
        x = global_mean_pool(x, batch)
        return self.fc(x)


class FinalLightweightModel(nn.Module):
    def __init__(self, content_path, vision_path, gat_path, num_classes=3, vision_proj_file='visual_projection.pt'):
        super().__init__()
        self.content_model = BertModel.from_pretrained(content_path)
        self.vision_model  = CLIPVisionModel.from_pretrained(vision_path)
        self.gat_model     = GATModel(100, 64, 1, 64, 2)
        if gat_path and os.path.exists(gat_path):
            self.gat_model.load_state_dict(torch.load(gat_path, map_location='cpu'), strict=False)

        # 视觉投影
        proj = nn.Linear(self.vision_model.config.hidden_size, 512, bias=False)
        vp = os.path.join(vision_path, vision_proj_file)
        if os.path.exists(vp):
            proj.load_state_dict(torch.load(vp, map_location='cpu'))
        self.visual_projection = proj

        cls_in = self.content_model.config.hidden_size + 512 + 64
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, content_inputs, pixel_values, graph_batch):
        content_feature = self.content_model(**content_inputs).pooler_output
        vision_outputs  = self.vision_model(pixel_values=pixel_values)
        image_feature   = self.visual_projection(vision_outputs.pooler_output)
        gat_feature     = self.gat_model(graph_batch)
        combined        = torch.cat([content_feature, image_feature, gat_feature], dim=1)
        return self.classifier(combined)


# ----------------- helpers -----------------
def get_peak_rss_mb():
    if PSUTIL_OK:
        return float(psutil.Process(os.getpid()).memory_info().rss / (1024**2))
    try:
        hwm = None; rss = None
        with open("/proc/self/status","r") as f:
            for line in f:
                if line.startswith("VmHWM:"): hwm = float(line.split()[1]) / 1024.0
                if line.startswith("VmRSS:"): rss = float(line.split()[1]) / 1024.0
        return float(hwm or rss or 0.0)
    except Exception:
        return 0.0


def append_csv_row(csv_path, row_dict, header_order):
    import csv
    need_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path)==0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header_order)
        if need_header: w.writeheader()
        w.writerow({k: row_dict.get(k, "") for k in header_order})


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    # 数据与模型路径
    ap.add_argument("--csv", type=str, default="./datasets/TextClassification/toutiao/toutiao_622.csv")
    ap.add_argument("--processed_dir", type=str, default="./processed_data")
    ap.add_argument("--images_root", type=str, default="./datasets/TextClassification/toutiao/images")
    ap.add_argument("--content_model", type=str, required=True, help="学生文本BERT本地目录")
    ap.add_argument("--vision_model", type=str, required=True, help="学生视觉ViT本地目录（CLIPVision兼容）")
    ap.add_argument("--gat_ckpt", type=str, default="./lightweight_gat_model.pth")
    ap.add_argument("--student_ckpt", type=str, default="./final_lightweight_model.pth", help="整体学生模型权重（可选）")
    # 运行参数
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--measure", type=int, default=100000)
    ap.add_argument("--subset", type=int, default=-1, help="-1=全量；>0=抽样条数")
    ap.add_argument("--seed", type=int, default=144)
    ap.add_argument("--out", type=str, default="./ckpts/cpu_student_latency.json")
    args = ap.parse_args()

    # 固定线程
    torch.set_num_threads(max(1, args.threads))
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 数据
    ds = FinalDataset(args.csv, args.processed_dir, os.path.abspath("."), images_root=args.images_root)
    if args.subset is not None and args.subset > 0 and args.subset < len(ds):
        idx = random.sample(range(len(ds)), args.subset)
        from torch.utils.data import Subset
        ds = Subset(ds, idx)
    dl = TorchDataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                         pin_memory=False, drop_last=False, collate_fn=collate_fn)

    # 处理器
    tok  = BertTokenizer.from_pretrained(args.content_model)
    proc = CLIPProcessor.from_pretrained(args.vision_model)

    # 模型
    model = FinalLightweightModel(args.content_model, args.vision_model, args.gat_ckpt, num_classes=3).to("cpu")
    if args.student_ckpt and os.path.exists(args.student_ckpt):
        try:
            sd = torch.load(args.student_ckpt, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
            model.load_state_dict(sd, strict=False)
            print(f"[INFO] Loaded student ckpt: {args.student_ckpt}")
        except Exception as e:
            print(f"[WARN] load student ckpt failed: {e}")
    model.eval()

    # 记录
    t_e2e, t_tok, t_img, t_fwd = [], [], [], []
    total_samples, seen_batches = 0, 0

    with torch.no_grad():
        pbar = tqdm(dl, total=len(dl), desc="CPU Student Benchmark")
        for step, (texts, images, labels, gb) in enumerate(pbar):
            t0 = time.perf_counter()

            # Tokenizer
            t1 = time.perf_counter()
            inputs = tok(texts, return_tensors='pt', padding='max_length', truncation=True, max_length=args.seq_len)
            t2 = time.perf_counter()

            # Image proc
            pixels = proc(images=images, return_tensors='pt')['pixel_values']
            t3 = time.perf_counter()

            # Forward（CPU）
            t4 = time.perf_counter()
            logits = model(inputs, pixels, gb)
            t5 = time.perf_counter()

            # 统计
            end2end  = (t5 - t0)
            token_t  = (t2 - t1)
            image_t  = (t3 - t2)
            forward_t= (t5 - t4)

            seen_batches += 1
            if seen_batches > args.warmup:  # 预热后计入
                t_e2e.append(end2end)
                t_tok.append(token_t)
                t_img.append(image_t)
                t_fwd.append(forward_t)
                total_samples += labels.size(0)

            if (seen_batches - args.warmup) >= args.measure:
                break

    bs = args.batch_size

    def summarize(name, lst):
        if not lst:
            return {"batch_avg_ms":0.0,"per_sample_avg_ms":0.0,"p50_ms":0.0,"p90_ms":0.0,"p95_ms":0.0,"p99_ms":0.0}
        arr = np.array(lst, dtype=np.float64)
        per = arr / bs
        return {
            "batch_avg_ms": float(arr.mean()*1000.0),
            "per_sample_avg_ms": float(per.mean()*1000.0),
            "p50_ms": float(np.percentile(per,50)*1000.0),
            "p90_ms": float(np.percentile(per,90)*1000.0),
            "p95_ms": float(np.percentile(per,95)*1000.0),
            "p99_ms": float(np.percentile(per,99)*1000.0),
        }

    res_e2e = summarize("E2E", t_e2e)
    res_tok = summarize("Tok", t_tok)
    res_img = summarize("Img", t_img)
    res_fwd = summarize("Fwd", t_fwd)

    thr = float(total_samples / sum(t_e2e)) if t_e2e else 0.0
    peak_rss = get_peak_rss_mb()

    def pline(n, r):
        print(f"{n}  batch_avg={r['batch_avg_ms']:.2f}ms  p50={r['p50_ms']:.2f}ms  "
              f"p90={r['p90_ms']:.2f}ms  p95={r['p95_ms']:.2f}ms  p99={r['p99_ms']:.2f}ms  "
              f"per_sample_avg={r['per_sample_avg_ms']:.2f}ms")

    print("\n=== CPU Student Latency (lower is better) ===")
    pline("End-to-End", res_e2e)
    pline("Tokenizer", res_tok)
    pline("ImageProc", res_img)
    pline("ModelFwd", res_fwd)
    print(f"\nThroughput: {thr:.2f} samples/s (over {total_samples} samples, measured {max(0, seen_batches - args.warmup)} batches)")
    print(f"CPU Peak RSS: {peak_rss:.1f} MB")

    # 保存
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {
        "device": "cpu",
        "variant": "student-fp32",
        "threads": args.threads,
        "subset": (len(ds) if hasattr(ds,'__len__') else -1),
        "batch_size": bs,
        "warmup_batches": args.warmup,
        "measured_batches": max(0, seen_batches - args.warmup),
        "samples_measured": total_samples,
        "latency_ms": {
            "end2end": res_e2e,
            "tokenizer": res_tok,
            "imageproc": res_img,
            "model_fwd": res_fwd
        },
        "throughput_samples_per_s": thr,
        "cpu_peak_rss_mb": peak_rss
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] JSON -> {args.out}")

    csv_path = args.out.replace(".json", ".csv")
    headers = ["variant","threads","e2e_ms","token_ms","img_ms","fwd_ms","throughput_sps","cpu_peak_rss_mb","batch_size","subset"]
    row = {
        "variant": "student-fp32",
        "threads": args.threads,
        "e2e_ms": round(res_e2e["per_sample_avg_ms"], 3),
        "token_ms": round(res_tok["per_sample_avg_ms"], 3),
        "img_ms": round(res_img["per_sample_avg_ms"], 3),
        "fwd_ms": round(res_fwd["per_sample_avg_ms"], 3),
        "throughput_sps": round(thr, 2),
        "cpu_peak_rss_mb": round(peak_rss, 1),
        "batch_size": bs,
        "subset": payload["subset"]
    }
    append_csv_row(csv_path, row, headers)
    print(f"[SAVED] CSV  -> {csv_path}")


if __name__ == "__main__":
    main()
