#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import torch
import numpy as np
import torch.nn as nn  # <-- 确保导入 nn
import pandas as pd  # <-- 添加这行导入 pandas
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from transformers import BertModel, CLIPVisionModel, CLIPProcessor, BertTokenizer
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
MODEL_PATH         = './final_lightweight_model.pth'  # 要测试的 .pth

# 设备与批量大小
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
WARMUP_STEPS = 10        # 预热批次数（跳过统计）
MEASURE_STEPS = 100      # 统计批次数（可小于数据总批次）
USE_FP16 = True          # GPU 上推荐开启

# 模型配置
GAT_CONFIG = {"in_channels":100,"hidden_channels":64,"num_layers":1,"out_channels":64,"heads":2}
NUM_CLASSES = 3

# 定义模型
class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, heads):
        super().__init__()
        self.convs = nn.ModuleList([GATConv(in_channels, hidden_channels, heads=heads)])
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels*heads, hidden_channels, heads=heads))
        self.fc = nn.Linear(hidden_channels*heads, out_channels)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)  # 使用 F.elu()
        x = global_mean_pool(x, batch)
        return self.fc(x)

class FinalDataset(Dataset):
    def __init__(self, csv_path, pre_dir, root):
        self.root = root
        self.pre_dir = pre_dir
        self.df = pd.read_csv(csv_path, header=None, names=['label','title','content','image_path'])
        self.valid_idx = [i for i in range(len(self.df))
                          if os.path.exists(os.path.join(self.pre_dir, f'data_{i}.pt'))]
    def __len__(self): return len(self.valid_idx)
    def __getitem__(self, i):
        idx = self.valid_idx[i]
        r = self.df.iloc[idx]
        label, content, path = int(r['label']), str(r['content']), str(r['image_path'])
        img = Image.new('RGB', (224,224), 'white')
        if path and path.lower()!='nan':
            p = os.path.join(self.root, path)
            if os.path.exists(p):
                try: img = Image.open(p).convert('RGB')
                except: pass
        g = torch.load(os.path.join(self.pre_dir, f'data_{idx}.pt'), map_location='cpu')
        if hasattr(g,'x') and g.x is not None:
            g.x = g.x.detach().requires_grad_(False).to(torch.float32)
        if hasattr(g,'edge_index') and g.edge_index is not None:
            g.edge_index = g.edge_index.to(torch.long).contiguous()
        return content, img, label, g

def collate_fn(batch):
    contents, images, labels, graphs = zip(*batch)
    return list(contents), list(images), torch.tensor(labels, dtype=torch.long), Batch.from_data_list(list(graphs))

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

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def main():
    print(f"Device: {DEVICE} | BS={BATCH_SIZE} | warmup={WARMUP_STEPS} | measure={MEASURE_STEPS} | fp16={USE_FP16}")
    torch.backends.cudnn.benchmark = True  # 允许选择最优卷积算法

    # 数据 & 处理器
    ds = FinalDataset(CSV_PATH, PREPROCESSED_DATA_DIR, PROJECT_ROOT)
    dl = TorchDataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=(DEVICE=='cuda'),
                         collate_fn=collate_fn, drop_last=False)
    tok = BertTokenizer.from_pretrained(CONTENT_MODEL_PATH)
    proc = CLIPProcessor.from_pretrained(VISION_MODEL_PATH)

    # 模型
    model = FinalLightweightModel(CONTENT_MODEL_PATH, VISION_MODEL_PATH, GAT_MODEL_PATH, NUM_CLASSES).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
        print(f"✅ Loaded: {MODEL_PATH}")
    model.eval()
    if DEVICE=='cuda':
        torch.cuda.reset_peak_memory_stats()

    # 记录数组
    t_end2end, t_token, t_image, t_forward = [], [], [], []

    steps_done = 0
    total_samples = 0

    # 预热 + 统计
    with torch.no_grad():
        pbar = tqdm(dl, total=min(len(dl), WARMUP_STEPS+MEASURE_STEPS), desc="Benchmark")
        for step, (contents, images, labels, gb) in enumerate(pbar):
            t0 = time.perf_counter()

            # Tokenization
            t1 = time.perf_counter()
            inputs = tok(contents, return_tensors='pt', padding='max_length', truncation=True, max_length=256)
            t2 = time.perf_counter()

            # Image processing
            pixels = proc(images=images, return_tensors='pt')['pixel_values']
            t3 = time.perf_counter()

            # Move to device
            inputs = {k:v.to(DEVICE, non_blocking=True) for k,v in inputs.items()}
            pixels = pixels.to(DEVICE, non_blocking=True)
            gb = gb.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward
            cuda_sync()
            t4 = time.perf_counter()
            if USE_FP16 and DEVICE=='cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(inputs, pixels, gb)
            else:
                logits = model(inputs, pixels, gb)
            cuda_sync()
            t5 = time.perf_counter()

            # timings
            end2end = (t5 - t0)
            token_t = (t2 - t1)
            image_t = (t3 - t2)
            forward_t = (t5 - t4)

            # 预热后才计入统计
            if step >= WARMUP_STEPS:
                t_end2end.append(end2end)
                t_token.append(token_t)
                t_image.append(image_t)
                t_forward.append(forward_t)
                total_samples += labels.size(0)
                steps_done += 1

            if steps_done >= MEASURE_STEPS:
                break

    # 统计函数
    def summarize(name, vals, bs):
        if len(vals)==0:
            return f"{name}: n/a"
        arr = np.array(vals)
        per_sample = arr / bs
        return (
            f"{name}  batch_avg={arr.mean()*1000:.2f}ms  "
            f"p50={np.percentile(per_sample,50)*1000:.2f}ms  "
            f"p90={np.percentile(per_sample,90)*1000:.2f}ms  "
            f"p95={np.percentile(per_sample,95)*1000:.2f}ms  "
            f"p99={np.percentile(per_sample,99)*1000:.2f}ms  "
            f"per_sample_avg={(per_sample.mean())*1000:.2f}ms"
        )

    print("\n=== Latency (lower is better) ===")
    print(summarize("End-to-End", t_end2end, BATCH_SIZE))
    print(summarize("Tokenizer", t_token, BATCH_SIZE))
    print(summarize("ImageProc", t_image, BATCH_SIZE))
    print(summarize("ModelFwd", t_forward, BATCH_SIZE))

    if steps_done > 0:
        total_time = sum(t_end2end)
        thr = total_samples / total_time
        print(f"\nThroughput: {thr:.2f} samples/s (over {total_samples} samples, {steps_done} measured batches)")

    if DEVICE=='cuda':
        peak = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"GPU Peak Memory: {peak:.1f} MB")

if __name__ == "__main__":
    main()
