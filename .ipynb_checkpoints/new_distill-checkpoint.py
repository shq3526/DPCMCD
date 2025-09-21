#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在原始脚本基础上最小改动：
- 新增 --run joint：联合蒸馏（支持 InfoNCE 协同损失 + 可选关系蒸馏）
- JointDataset：自动识别 TSV/CSV，正确处理引号内多行文本；0-based 第2/3列=content/image_path
- 本地图片路径：优先 project_root；找不到回退 CSV 目录
其它 text/vision/gat 部分保持不变
"""

import os, json, argparse, csv
from typing import Optional
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
from PIL import Image, UnidentifiedImageError

from transformers import BertModel, BertConfig, BertTokenizer, CLIPModel
from transformers import CLIPVisionModel, CLIPVisionConfig, CLIPProcessor

# ------------------------ 通用 ------------------------
def ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def get_device(name: Optional[str] = None) -> str:
    if name: return name
    return "cuda" if torch.cuda.is_available() else "cpu"

# ======================== 文本蒸馏（原样保留） ========================
class ContentDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer: BertTokenizer, max_len: int = 256):
        self.tokenizer = tokenizer
        # 这里仍按你原来的读取方式（仅用第2列 content）
        df = pd.read_csv(csv_path, header=None, usecols=[2], names=['content'], encoding='utf-8')
        self.texts = df['content'].dropna().astype(str).tolist()
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        return self.tokenizer(self.texts[idx], return_tensors='pt', max_length=self.max_len,
                              padding='max_length', truncation=True)

class TextEncoderWithProjection(nn.Module):
    def __init__(self, text_model: BertModel, projection: nn.Linear):
        super().__init__()
        self.text_model = text_model
        self.text_projection = projection
    def forward(self, input_ids, attention_mask):
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        return self.text_projection(outputs.pooler_output)

def distill_text(csv_path: str, teacher_text_path: str, base_clip_path: str, out_dir: str,
                 batch_size: int, epochs: int, lr: float, device: str):
        print(f"\n==== 正文(Text) 蒸馏开始 @ {device} ====")
        ensure_dir(out_dir)

        tokenizer = BertTokenizer.from_pretrained(teacher_text_path)
        teacher_bert = BertModel.from_pretrained(teacher_text_path)
        base_clip = CLIPModel.from_pretrained(base_clip_path)
        teacher_proj = nn.Linear(teacher_bert.config.hidden_size, base_clip.config.projection_dim, bias=False)
        teacher = TextEncoderWithProjection(teacher_bert, teacher_proj).to(device)
        teacher.eval(); [setattr(p, 'requires_grad', False) for p in teacher.parameters()]

        student_cfg = BertConfig.from_pretrained(teacher_text_path)
        student_cfg.num_hidden_layers = min(getattr(student_cfg, 'num_hidden_layers', 12), 4)
        student_cfg.hidden_size = 384; student_cfg.intermediate_size = 1536; student_cfg.num_attention_heads = 6
        student_bert = BertModel(config=student_cfg)
        student_proj = nn.Linear(student_cfg.hidden_size, base_clip.config.projection_dim, bias=False)
        student = TextEncoderWithProjection(student_bert, student_proj).to(device)

        ds = ContentDataset(csv_path, tokenizer)
        dl = DataLoader(ds, batch_size=batch_size)
        opt = torch.optim.AdamW(student.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler(enabled=device.startswith('cuda'))

        student.train()
        for ep in range(epochs):
            pbar = tqdm(dl, desc=f"[Text] Epoch {ep+1}/{epochs}")
            for batch in pbar:
                ids = batch['input_ids'].squeeze(1).to(device)
                attn = batch['attention_mask'].squeeze(1).to(device)
                with torch.no_grad():
                    t_emb = teacher(ids, attn)
                with torch.cuda.amp.autocast(enabled=device.startswith('cuda')):
                    s_emb = student(ids, attn)
                    loss = loss_fn(s_emb, t_emb)
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        student.text_model.save_pretrained(out_dir)
        torch.save(student.text_projection.state_dict(), os.path.join(out_dir, 'text_projection.pt'))
        tokenizer.save_pretrained(out_dir)
        print(f"[Text] 蒸馏完成 -> {out_dir}")

# ======================== 视觉蒸馏（原样保留） ========================
class VisionDataset(Dataset):
    def __init__(self, csv_path: str, project_root: str):
        self.project_root = project_root
        try:
            df = pd.read_csv(csv_path, header=None, usecols=[3], names=['image_path'], encoding='utf-8', on_bad_lines='skip')
            self.image_paths = df['image_path'].dropna().astype(str).tolist()
        except Exception as e:
            print(f"读取CSV出错: {e}"); self.image_paths = []
        print(f"[Vision] 加载到 {len(self.image_paths)} 条图片路径")
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        p = self.image_paths[idx].strip()
        full_path = os.path.join(self.project_root, p) if not os.path.isabs(p) else p
        try:
            if os.path.exists(full_path):
                img = Image.open(full_path).convert('RGB')
                if img.mode != 'RGB': return None
                return img
        except (FileNotFoundError, OSError, UnidentifiedImageError):
            pass
        return None

def collate_img(batch):
    batch = [x for x in batch if x is not None]
    return batch if batch else None

class VisionEncoderWithProjection(nn.Module):
    def __init__(self, vision_model: CLIPVisionModel, projection: nn.Linear):
        super().__init__()
        self.vision_model = vision_model
        self.visual_projection = projection
    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        return self.visual_projection(outputs.pooler_output)

def distill_vision(csv_path: str, base_clip_path: str, out_dir: str,
                   batch_size: int, epochs: int, lr: float, device: str, project_root: str):
        print(f"\n==== 视觉(Vision) 蒸馏开始 @ {device} ====")
        ensure_dir(out_dir)

        processor = CLIPProcessor.from_pretrained(base_clip_path)
        base_clip = CLIPModel.from_pretrained(base_clip_path)
        teacher = VisionEncoderWithProjection(base_clip.vision_model, base_clip.visual_projection).to(device)
        teacher.eval(); [setattr(p, 'requires_grad', False) for p in teacher.parameters()]

        student_cfg = CLIPVisionConfig.from_pretrained(base_clip_path)
        student_cfg.num_hidden_layers = 6; student_cfg.hidden_size = 256
        student_cfg.intermediate_size = 1024; student_cfg.num_attention_heads = 4
        student_vision = CLIPVisionModel(config=student_cfg)
        student_proj = nn.Linear(student_cfg.hidden_size, base_clip.config.projection_dim, bias=False)
        student = VisionEncoderWithProjection(student_vision, student_proj).to(device)

        ds = VisionDataset(csv_path, project_root)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_img)

        opt = torch.optim.AdamW(student.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler(enabled=device.startswith('cuda'))

        student.train()
        for ep in range(epochs):
            pbar = tqdm(dl, desc=f"[Vision] Epoch {ep+1}/{epochs}")
            for imgs in pbar:
                if imgs is None: continue
                enc = processor(images=imgs, return_tensors='pt')
                pixel_values = enc['pixel_values'].to(device)
                with torch.no_grad():
                    t_emb = teacher(pixel_values)
                with torch.cuda.amp.autocast(enabled=device.startswith('cuda')):
                    s_emb = student(pixel_values)
                    loss = loss_fn(s_emb, t_emb)
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        vision_dir = os.path.join(out_dir, 'vision')
        ensure_dir(vision_dir)
        student.vision_model.save_pretrained(vision_dir)
        torch.save(student.visual_projection.state_dict(), os.path.join(vision_dir, 'visual_projection.pt'))
        print(f"[Vision] 蒸馏完成 -> {vision_dir}")

# ======================== GAT（原样保留） ========================
try:
    from ltp import LTP
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as GeoDataLoader
    from torch_geometric.nn import GATConv, global_mean_pool
except Exception as e:
    LTP = None
    print("⚠️ GAT 相关依赖未就绪：", e)

LTP_POS_LABELS = ['a','b','c','d','e','g','h','i','j','k','m','n','nd','nh','ni','nl','ns','nt','nz','o','p','q','r','u','v','wp','ws','x','z']

class TitleGraphDataset(Dataset):
    def __init__(self, csv_path: str, ltp_model_name: str, in_channels: int):
        assert LTP is not None, "未安装 ltp / torch-geometric"
        try:
            df = pd.read_csv(csv_path, header=None, usecols=[1], names=['title'], on_bad_lines='warn', encoding='utf-8')
            self.titles = df['title'].dropna().astype(str).tolist()
        except Exception as e:
            print("读取CSV失败：", e); self.titles = []
        self.ltp = LTP(ltp_model_name)
        self.pos_vocab = {tag: i for i, tag in enumerate(LTP_POS_LABELS)}
        self.pos_embed = nn.Embedding(len(self.pos_vocab), in_channels)
    def __len__(self): return len(self.titles)
    def __getitem__(self, idx):
        t = self.titles[idx]
        try:
            out = self.ltp.pipeline([t], tasks=["cws","pos","dep"])
        except Exception:
            return None
        pos_tags = out.pos[0]; deps = out.dep[0]; heads = deps['head']
        if not pos_tags or not heads: return None
        pos_ids = torch.tensor([self.pos_vocab.get(tag, 0) for tag in pos_tags], dtype=torch.long)
        node_x = self.pos_embed(pos_ids)
        src, dst = [], []
        for i, h in enumerate(heads):
            h = h - 1
            if h >= 0: src.append(h); dst.append(i)
        edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.tensor([[0],[0]], dtype=torch.long)
        return Data(x=node_x, edge_index=edge_index)

def collate_graph(batch): batch = [b for b in batch if b is not None]; return batch

class GATEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, out_channels: int, heads: int):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
        self.fc = nn.Linear(hidden_channels * heads, out_channels)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs: x = F.elu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)

def distill_gat(csv_path: str, ltp_model_name: str, out_path: str,
                batch_size: int, epochs: int, lr: float, device: str,
                teacher_cfg: dict, student_cfg: dict, teacher_ckpt: Optional[str] = None):
        assert LTP is not None, "GAT 依赖未安装"
        print(f"\n==== GAT 蒸馏开始 @ {device} ====")
        teacher = GATEncoder(**teacher_cfg).to(device)
        if teacher_ckpt and os.path.isfile(teacher_ckpt):
            print("[GAT] 加载教师权重：", teacher_ckpt)
            teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
        teacher.eval(); [setattr(p, 'requires_grad', False) for p in teacher.parameters()]
        student = GATEncoder(**student_cfg).to(device)
        proj = nn.Linear(student_cfg['out_channels'], teacher_cfg['out_channels']).to(device)

        ds = TitleGraphDataset(csv_path, ltp_model_name, in_channels=teacher_cfg['in_channels'])
        if len(ds) == 0: print("[GAT] 无有效标题，跳过。"); return
        dl = GeoDataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_graph)

        opt = torch.optim.AdamW(list(student.parameters()) + list(proj.parameters()), lr=lr)
        loss_fn = nn.MSELoss()

        student.train(); proj.train()
        for ep in range(epochs):
            pbar = tqdm(dl, desc=f"[GAT] Epoch {ep+1}/{epochs}")
            for batch_list in pbar:
                if not batch_list: continue
                batch_graph = next(iter(GeoDataLoader(batch_list, batch_size=len(batch_list)))).to(device)
                with torch.no_grad(): t_emb = teacher(batch_graph)
                s_emb = student(batch_graph); s_proj = proj(s_emb)
                loss = loss_fn(s_proj, t_emb)
                opt.zero_grad(); loss.backward(); opt.step()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        torch.save(student.state_dict(), out_path)
        print(f"[GAT] 蒸馏完成 -> {out_path}")

# ======================== 新增：联合蒸馏（T+V） ========================
def _read_toutiao_robust(csv_path: str):
    """优先按 TSV 读，失败再自动探测分隔符；支持引号内多行。"""
    for sep in ['\t', None, ',']:
        try:
            df = pd.read_csv(
                csv_path, header=None, sep=sep, engine='python',
                quoting=csv.QUOTE_MINIMAL, quotechar='"', on_bad_lines='skip'
            )
            if df.shape[1] >= 4:
                return df
        except Exception:
            pass
    raise ValueError("无法解析 CSV/TSV，请检查文件与分隔符。")

class JointDataset(Dataset):
    """0-based: 第2列 content，第3列 image_path；本地图片"""
    def __init__(self, csv_path: str, tokenizer: BertTokenizer, processor: CLIPProcessor,
                 project_root: str, max_len: int = 256):
        self.tok = tokenizer
        self.proc = processor
        self.max_len = max_len
        self.rows = []

        df = _read_toutiao_robust(csv_path)
        texts = df.iloc[:, 2].astype(str)
        images = df.iloc[:, 3].astype(str)

        csv_dir = os.path.dirname(os.path.abspath(csv_path))
        for t, p in zip(texts, images):
            if not p or p.lower().startswith(('http://', 'https://', 'www.')):  # 仅本地
                continue
            p = p.strip().replace('\\', '/')
            cand1 = p if os.path.isabs(p) else os.path.normpath(os.path.join(project_root, p))
            cand2 = p if os.path.isabs(p) else os.path.normpath(os.path.join(csv_dir, p))
            full = cand1 if os.path.exists(cand1) else (cand2 if os.path.exists(cand2) else None)
            if full:
                self.rows.append((t, full))
        print(f"[Joint] 可用文本-图片对：{len(self.rows)}")

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        text, img_path = self.rows[idx]
        enc_t = self.tok(text, return_tensors='pt', max_length=self.max_len,
                         padding='max_length', truncation=True)
        img = Image.open(img_path).convert('RGB')
        enc_v = self.proc(images=img, return_tensors='pt')
        return {
            "input_ids": enc_t['input_ids'].squeeze(0),
            "attention_mask": enc_t['attention_mask'].squeeze(0),
            "pixel_values": enc_v['pixel_values'].squeeze(0),
        }

def _cos_sim(a, b, t):
    a = F.normalize(a, dim=-1); b = F.normalize(b, dim=-1)
    return (a @ b.t()) / t

def _info_nce(z_t, z_v, t):
    sim = _cos_sim(z_t, z_v, t)
    labels = torch.arange(sim.size(0), device=sim.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))

def distill_joint(csv_path: str, teacher_text_path: str, base_clip_path: str, out_root: str,
                  project_root: str, device: str,
                  student_text_dir: Optional[str], student_vision_dir: Optional[str],
                  joint_epochs: int, joint_lr: float,
                  lambda_align: float, lambda_rel: float, temperature: float):
    print(f"\n==== 联合蒸馏(含协同损失) 开始 @ {device} ====")
    ensure_dir(out_root)

    # 教师
    tok = BertTokenizer.from_pretrained(teacher_text_path)
    clip_proc = CLIPProcessor.from_pretrained(base_clip_path)
    clip = CLIPModel.from_pretrained(base_clip_path).to(device).eval()
    teacher_bert = BertModel.from_pretrained(teacher_text_path).to(device).eval()
    teacher_proj = nn.Linear(teacher_bert.config.hidden_size, clip.config.projection_dim, bias=False).to(device)
    [setattr(p, 'requires_grad', False) for p in list(clip.parameters()) + list(teacher_bert.parameters()) + list(teacher_proj.parameters())]

    # 学生 text
    if student_text_dir and os.path.isdir(student_text_dir):
        s_text = BertModel.from_pretrained(student_text_dir)
        text_proj = nn.Linear(s_text.config.hidden_size, clip.config.projection_dim, bias=False)
        tp = os.path.join(student_text_dir, 'text_projection.pt')
        if os.path.isfile(tp): text_proj.load_state_dict(torch.load(tp, map_location='cpu'))
    else:
        cfg = BertConfig.from_pretrained(teacher_text_path)
        cfg.num_hidden_layers = min(getattr(cfg, 'num_hidden_layers', 12), 4)
        cfg.hidden_size = 384; cfg.intermediate_size = 1536; cfg.num_attention_heads = 6
        s_text = BertModel(config=cfg); text_proj = nn.Linear(cfg.hidden_size, clip.config.projection_dim, bias=False)
    s_text, text_proj = s_text.to(device), text_proj.to(device)

    # 学生 vision
    if student_vision_dir and os.path.isdir(student_vision_dir):
        s_vision = CLIPVisionModel.from_pretrained(student_vision_dir)
        vis_proj = nn.Linear(s_vision.config.hidden_size, clip.config.projection_dim, bias=False)
        vp = os.path.join(student_vision_dir, 'visual_projection.pt')
        if os.path.isfile(vp): vis_proj.load_state_dict(torch.load(vp, map_location='cpu'))
    else:
        vcfg = CLIPVisionConfig.from_pretrained(base_clip_path)
        vcfg.num_hidden_layers = 6; vcfg.hidden_size = 256; vcfg.intermediate_size = 1024; vcfg.num_attention_heads = 4
        s_vision = CLIPVisionModel(config=vcfg); vis_proj = nn.Linear(vcfg.hidden_size, clip.config.projection_dim, bias=False)
    s_vision, vis_proj = s_vision.to(device), vis_proj.to(device)

    # 数据
    ds = JointDataset(csv_path, tok, clip_proc, project_root)
    if len(ds) == 0:
        raise ValueError("没有找到可用的文本-图片对。请检查第2/3列与图片路径（相对 project_root 或 CSV 目录）。")
    dl = DataLoader(ds, batch_size=32, shuffle=True, drop_last=False)

    # 优化
    params = list(s_text.parameters()) + list(text_proj.parameters()) \
           + list(s_vision.parameters()) + list(vis_proj.parameters())
    opt = torch.optim.AdamW(params, lr=joint_lr)
    mse = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith('cuda'))
    s_text.train(); s_vision.train(); text_proj.train(); vis_proj.train()

    for ep in range(joint_epochs):
        pbar = tqdm(dl, desc=f"[Joint] Epoch {ep+1}/{joint_epochs}")
        for batch in pbar:
            ids  = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            pix  = batch['pixel_values'].to(device)

            with torch.no_grad():
                t_txt_feat = teacher_bert(input_ids=ids, attention_mask=attn).pooler_output
                t_text = teacher_proj(t_txt_feat)                          # [B, d]
                v_out  = clip.vision_model(pixel_values=pix)
                t_vision = clip.visual_projection(v_out.pooler_output)     # [B, d]

            with torch.cuda.amp.autocast(enabled=device.startswith('cuda')):
                s_text_feat = s_text(input_ids=ids, attention_mask=attn).pooler_output
                s_text_proj = text_proj(s_text_feat)
                s_vis_out   = s_vision(pixel_values=pix)
                s_vis_proj  = vis_proj(s_vis_out.pooler_output)

                # 逐模态蒸馏
                loss_text   = mse(s_text_proj, t_text.detach())
                loss_vision = mse(s_vis_proj, t_vision.detach())

                # 协同损失：InfoNCE（学生内部对齐）
                loss_align = _info_nce(s_text_proj, s_vis_proj, temperature) if lambda_align > 0 else s_text_proj.sum()*0

                # 关系蒸馏（可选）：学生跨模态相似度 ≈ 教师
                if lambda_rel > 0:
                    S_s = _cos_sim(s_text_proj, s_vis_proj, temperature)
                    with torch.no_grad(): S_t = _cos_sim(t_text, t_vision, temperature)
                    loss_rel = F.mse_loss(F.softmax(S_s, dim=-1), F.softmax(S_t, dim=-1))
                else:
                    loss_rel = s_text_proj.sum()*0

                loss = loss_text + loss_vision + lambda_align*loss_align + lambda_rel*loss_rel

            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            pbar.set_postfix({
                "L_txt": f"{loss_text.item():.3f}",
                "L_vis": f"{loss_vision.item():.3f}",
                "L_align": f"{(loss_align.item() if lambda_align>0 else 0):.3f}",
                "L_rel": f"{(loss_rel.item() if lambda_rel>0 else 0):.3f}"
            })

    # 保存
    if not student_text_dir:   student_text_dir = os.path.join(out_root, 'content_text')
    if not student_vision_dir: student_vision_dir = os.path.join(out_root, 'vision')
    ensure_dir(student_text_dir); ensure_dir(student_vision_dir)
    s_text.save_pretrained(student_text_dir)
    torch.save(text_proj.state_dict(), os.path.join(student_text_dir, 'text_projection.pt'))
    s_vision.save_pretrained(student_vision_dir)
    torch.save(vis_proj.state_dict(), os.path.join(student_vision_dir, 'visual_projection.pt'))
    print(f"[Joint] 完成！学生(Text) -> {student_text_dir}")
    print(f"[Joint] 完成！学生(Vision) -> {student_vision_dir}")

# ======================== 主入口 ========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, required=True)
    ap.add_argument('--teacher_text', type=str, default='./model/chinese-roberta-wwm-ext')
    ap.add_argument('--teacher_clip', type=str, default='./model/clip-vit-base-patch32')
    ap.add_argument('--ltp_model', type=str, default='LTP/small')
    ap.add_argument('--out_root', type=str, default='./lightweight_outputs')
    ap.add_argument('--project_root', type=str, default='.')
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--run', type=str, default='all', choices=['all','text','vision','gat','joint'])

    ap.add_argument('--epochs_text', type=int, default=3)
    ap.add_argument('--epochs_vision', type=int, default=3)
    ap.add_argument('--epochs_gat', type=int, default=5)
    ap.add_argument('--batch_text', type=int, default=16)
    ap.add_argument('--batch_vision', type=int, default=32)
    ap.add_argument('--batch_gat', type=int, default=64)
    ap.add_argument('--lr_text', type=float, default=5e-5)
    ap.add_argument('--lr_vision', type=float, default=5e-5)
    ap.add_argument('--lr_gat', type=float, default=1e-4)
    ap.add_argument('--teacher_gat_ckpt', type=str, default='')

    # 联合蒸馏新增参数（与你的命令一致）
    ap.add_argument('--student_text_dir', type=str, default='')
    ap.add_argument('--student_vision_dir', type=str, default='')
    ap.add_argument('--joint_epochs', type=int, default=2)
    ap.add_argument('--joint_lr', type=float, default=3e-5)
    ap.add_argument('--lambda_align', type=float, default=0.0)
    ap.add_argument('--lambda_rel', type=float, default=0.0)
    ap.add_argument('--temperature', type=float, default=0.07)

    args = ap.parse_args()
    device = get_device(args.device)
    ensure_dir(args.out_root)

    if args.run in ('all','text'):
        out_text = os.path.join(args.out_root, 'content_text')
        distill_text(args.csv, args.teacher_text, args.teacher_clip, out_text,
                     args.batch_text, args.epochs_text, args.lr_text, device)

    if args.run in ('all','vision'):
        out_vision = os.path.join(args.out_root, 'vision_root')
        distill_vision(args.csv, args.teacher_clip, out_vision,
                       args.batch_vision, args.epochs_vision, args.lr_vision, device, args.project_root)

    if args.run in ('all','gat'):
        out_gat = os.path.join(args.out_root, 'lightweight_gat_student.pth')
        teacher_cfg = json.loads(json.dumps({"in_channels":100,"hidden_channels":128,"num_layers":2,"out_channels":128,"heads":4}))
        student_cfg = json.loads(json.dumps({"in_channels":100,"hidden_channels":64,"num_layers":1,"out_channels":64,"heads":2}))
        distill_gat(args.csv, args.ltp_model, out_gat, args.batch_gat, args.epochs_gat, args.lr_gat, device,
                    teacher_cfg, student_cfg, args.teacher_gat_ckpt or None)

    if args.run == 'joint':
        distill_joint(args.csv, args.teacher_text, args.teacher_clip, args.out_root, args.project_root, device,
                      args.student_text_dir or None, args.student_vision_dir or None,
                      args.joint_epochs, args.joint_lr, args.lambda_align, args.lambda_rel, args.temperature)

    print("\n✅ 全流程完成。输出目录：", args.out_root)

if __name__ == '__main__':
    main()
