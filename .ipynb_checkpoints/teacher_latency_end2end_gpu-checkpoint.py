#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
teacher_latency_end2end_gpu.py
在 GPU 上对“教师模型（PromptForClassification，文本+可选图像）”做端到端延迟测试。
- 逐批统计：Tokenizer / ImageProc(CLIP) / ModelFwd / End-to-End
- 支持全量或子集；支持 AMP；支持本地 CLIP 离线
"""

import os, time, json, argparse, random
from typing import List, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader

from openprompt.data_utils.text_classification_dataset import CnClickbaitProcessor
from openprompt.data_utils.utils import InputExample, InputFeatures
from openprompt.prompts import ManualVerbalizer, PtuningTemplate
from openprompt.plms import load_plm
from openprompt.prompt_base import Template
from openprompt.plms.utils import TokenizerWrapper
from openprompt.utils import signature
from openprompt.utils.reproduciblity import set_seed
from openprompt import PromptForClassification

from transformers import CLIPProcessor, CLIPModel


# -------------------- helpers --------------------
def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    class C:
        def __init__(self, **d): self.__dict__.update({k: v for k, v in d.items() if not str(k).startswith("//")})
    return C(**cfg)

def build_teacher(cfg, scripts_path: str, device: str):
    plm, tokenizer, model_config, WrapperClass = load_plm(cfg.model_type, cfg.model_name_or_path)
    with open(os.path.join(scripts_path, "ptuning_template.txt"), "r", encoding="utf-8") as f:
        template_text = f.readlines()[cfg.template_id].rstrip()
    template = PtuningTemplate(model=plm, tokenizer=tokenizer, text=template_text)
    with open(os.path.join(scripts_path, cfg.verbalizer_file_name), "r", encoding="utf-8") as f:
        vd = json.load(f)
    classes = list(vd["label_words"].keys())
    verbalizer = ManualVerbalizer(tokenizer, classes=classes)
    verbalizer.label_words = vd["label_words"]
    model = PromptForClassification(plm=plm, template=template, verbalizer=verbalizer, freeze_plm=False)
    model.to(device).eval()
    return model, tokenizer, template, WrapperClass

def try_load_ckpt(model, ckpt: Optional[str]) -> bool:
    if not ckpt:
        print("[INFO] 未提供 --ckpt，使用初始化权重"); return False
    if not os.path.exists(ckpt):
        print(f"[WARN] ckpt 不存在：{ckpt}，跳过"); return False
    try:
        sd = torch.load(ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)
        print(f"[INFO] 已加载权重：{ckpt}")
        return True
    except Exception as e:
        print(f"[WARN] 加载失败：{e}，跳过使用初始化权重"); return False

def build_clip_local(clip_path: str, device: str, on_gpu: bool = True):
    if not os.path.isdir(clip_path):
        raise FileNotFoundError(f"未找到本地 CLIP 目录：{clip_path}")
    print(f"[INFO] 从本地加载 CLIP：{clip_path}")
    proc = CLIPProcessor.from_pretrained(clip_path, local_files_only=True)
    model = CLIPModel.from_pretrained(clip_path, local_files_only=True)
    model = model.to(device if on_gpu else "cpu").eval()
    return proc, model

# ---- 图片路径解析（新增：稳健规范化）----
def get_meta_path(meta) -> Optional[str]:
    if isinstance(meta, str): return meta
    if isinstance(meta, dict): return meta.get("img_path")
    return None

def resolve_full_path(meta_path: Optional[str], proj_root: str, images_dir: str) -> Optional[str]:
    """把 meta 里的路径规范化为‘可读的绝对路径’；支持绝对路径、datasets/.../images/...、纯文件名。"""
    if not meta_path:
        return None
    p = meta_path.strip()

    # 1) 绝对路径
    if os.path.isabs(p):
        cand = p
    # 2) 项目内相对路径（以 datasets/ 开头）
    elif p.startswith("datasets/"):
        cand = os.path.join(proj_root, p)
    # 3) 认为是纯文件名，拼到 images_dir
    else:
        cand = os.path.join(images_dir, p)

    cand = os.path.normpath(cand)
    if os.path.exists(cand):
        return cand

    # 4) 兜底：basename 放到 images_dir
    base = os.path.basename(p)
    cand2 = os.path.normpath(os.path.join(images_dir, base))
    return cand2 if os.path.exists(cand2) else None

# ---- 使用“已解析好的绝对路径”提取特征 ----
def extract_clip_features_for_batch(full_paths: List[Optional[str]],
                                    proc: CLIPProcessor, clip: CLIPModel,
                                    device: str, on_gpu: bool):
    imgs, idx_map = [], []
    for i, full in enumerate(full_paths):
        if not full or not os.path.exists(full): 
            continue
        try:
            imgs.append(Image.open(full).convert("RGB")); idx_map.append(i)
        except Exception:
            pass
    if not imgs:
        return [None] * len(full_paths)

    inputs = proc(images=imgs, return_tensors="pt")
    if on_gpu:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        feats = clip.get_image_features(**inputs)  # (n, 512)
    feats = feats.detach().to(device if on_gpu else "cpu")
    out = [None] * len(full_paths)
    for j, i in enumerate(idx_map):
        out[i] = feats[j]
    return out

# ---- 复用 TokenizerWrapper（性能优化）----
def build_token_wrapper(WrapperClass, tokenizer, max_len: int):
    wrap_keys = signature(WrapperClass.__init__).args
    kwargs = {"max_seq_length": max_len, "truncate_method": "tail", "tokenizer": tokenizer}
    to_pass = {k: kwargs[k] for k in kwargs if k in wrap_keys}
    return WrapperClass(**to_pass)

def tokenize_batch(examples: List[InputExample], template: Template, token_wrapper,
                   image_features: Optional[List[Optional[torch.Tensor]]] = None):
    """将一批 InputExample -> List[InputFeaturesTensor]；可附带 image_features。"""
    feats = []
    for idx, ex in enumerate(examples):
        wrapped = template.wrap_one_example(ex)
        tokenized = token_wrapper.tokenize_one_example(wrapped, teacher_forcing=False)
        fd = {**tokenized, **wrapped[1]}
        if image_features is not None and "image_features" in signature(InputFeatures.__init__).args:
            feat = image_features[idx]
            if feat is not None:
                fd["image_features"] = feat
        feats.append(InputFeatures(**fd).to_tensor())
    return feats

def print_line(name, batch_secs: List[float], bs: int):
    if not batch_secs:
        print(f"{name}: n/a"); return
    arr = np.array(batch_secs, dtype=np.float64)
    per_sample = arr / bs
    print(f"{name}  batch_avg={arr.mean()*1000:.2f}ms  "
          f"p50={np.percentile(per_sample,50)*1000:.2f}ms  "
          f"p90={np.percentile(per_sample,90)*1000:.2f}ms  "
          f"p95={np.percentile(per_sample,95)*1000:.2f}ms  "
          f"p99={np.percentile(per_sample,99)*1000:.2f}ms  "
          f"per_sample_avg={per_sample.mean()*1000:.2f}ms")

# -------------------- main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="", help="可选：教师 state_dict（二进制）")
    parser.add_argument("--subset", type=int, default=-1, help="-1=全量；>0=抽样条数")
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measure", type=int, default=100)
    parser.add_argument("--cuda", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=144)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use-images", action="store_true")
    parser.add_argument("--clip-path", type=str, default="./model/clip-vit-base-patch32")
    parser.add_argument("--clip-on-gpu", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA 环境")
    device = args.cuda
    torch.cuda.set_device(int(device.split(":")[-1]))
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # 路径
    cfg = load_config(args.config_file)
    proj_root  = os.path.abspath(".")
    data_root  = os.path.join(proj_root, "datasets", "TextClassification", cfg.dataset_name)
    scripts_path = os.path.join(proj_root, "scripts", "TextClassification", cfg.prompt_name)
    images_dir = os.path.join(data_root, "images")

    # 数据
    processor = CnClickbaitProcessor()
    test_set: List[InputExample] = processor.get_test_examples(data_root)
    total = len(test_set)
    if total == 0:
        raise RuntimeError(f"测试集为空，检查路径：{data_root}")
    if args.subset is None or args.subset < 0 or args.subset >= total:
        picked = test_set
    else:
        random.seed(args.seed)
        picked = random.sample(test_set, args.subset)

    # 模型 &（可选）权重
    model, tokenizer, template, WrapperClass = build_teacher(cfg, scripts_path, device)
    used_ckpt = try_load_ckpt(model, args.ckpt)

    # CLIP（可选）
    clip_proc, clip_model = (None, None)
    if args.use_images:
        clip_proc, clip_model = build_clip_local(args.clip_path, device, on_gpu=args.clip_on_gpu)

    # DataLoader：让 collate_fn 原样返回列表（否则会报 InputExample 类型不支持）
    dl = DataLoader(
        picked,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=lambda batch: batch
    )

    # 复用 TokenizerWrapper（初始化一次）
    token_wrapper = build_token_wrapper(WrapperClass, tokenizer, args.seq_len)

    # 记录
    times_e2e, times_tok, times_img, times_fwd = [], [], [], []
    total_samples = 0
    seen_batches = 0

    # 统计图片命中数（可选）
    total_imgs_ok = 0

    for step, batch_examples in enumerate(dl):
        if len(batch_examples) == 0:
            continue

        # 计时开始
        t0 = time.perf_counter()

        # 图像特征（可选）
        img_feats = None
        if args.use_images:
            raw_paths = [get_meta_path(ex.meta) for ex in batch_examples]
            abs_paths = [resolve_full_path(p, proj_root, images_dir) for p in raw_paths]
            t_img0 = time.perf_counter()
            img_feats = extract_clip_features_for_batch(abs_paths, clip_proc, clip_model, device, args.clip_on_gpu)
            t_img1 = time.perf_counter()
            times_img.append(t_img1 - t_img0)
            total_imgs_ok += sum(1 for fp in abs_paths if fp and os.path.exists(fp))

        # Tokenize（含 openprompt 包装）
        t_tok0 = time.perf_counter()
        feats_list = tokenize_batch(batch_examples, template, token_wrapper, image_features=img_feats)
        t_tok1 = time.perf_counter()
        times_tok.append(t_tok1 - t_tok0)

        # collate & to device
        batch = InputFeatures.collate_fct(feats_list)
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)

        # 前向（AMP 可选）
        torch.cuda.synchronize()
        t_fwd0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
            _ = model(batch)
        torch.cuda.synchronize()
        t_fwd1 = time.perf_counter()
        times_fwd.append(t_fwd1 - t_fwd0)

        # End-to-End
        t1 = time.perf_counter()
        times_e2e.append(t1 - t0)

        seen_batches += 1
        total_samples += len(batch["label"])

        # 预热：丢弃前 warmup 批
        if seen_batches <= args.warmup:
            times_e2e.pop(); times_fwd.pop(); times_tok.pop()
            if args.use_images and len(times_img) > 0: times_img.pop()
            total_samples -= len(batch["label"])
            continue

        # 达到测量批次数就停
        if seen_batches - args.warmup >= args.measure:
            break

    bs = args.batch_size

    def print_line(name, batch_secs: List[float], bs: int):
        if not batch_secs:
            print(f"{name}: n/a"); return
        arr = np.array(batch_secs, dtype=np.float64)
        per_sample = arr / bs
        print(f"{name}  batch_avg={arr.mean()*1000:.2f}ms  "
              f"p50={np.percentile(per_sample,50)*1000:.2f}ms  "
              f"p90={np.percentile(per_sample,90)*1000:.2f}ms  "
              f"p95={np.percentile(per_sample,95)*1000:.2f}ms  "
              f"p99={np.percentile(per_sample,99)*1000:.2f}ms  "
              f"per_sample_avg={per_sample.mean()*1000:.2f}ms")

    print("\n=== Latency (lower is better) ===")
    print_line("End-to-End", times_e2e, bs)
    print_line("Tokenizer", times_tok, bs)
    if args.use_images:
        print_line("ImageProc", times_img, bs)
    print_line("ModelFwd", times_fwd, bs)

    measured_batches = max(0, seen_batches - args.warmup)
    if measured_batches > 0 and sum(times_e2e) > 0:
        thr = total_samples / sum(times_e2e)
        print(f"\nThroughput: {thr:.2f} samples/s (over {total_samples} samples, {measured_batches} measured batches)")
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"GPU Peak Memory: {peak:.1f} MB")
    print(f"\n[INFO] Device={device} AMP={args.amp} UseImages={args.use_images} CKPT={used_ckpt}  "
          f"Evaluated={total_samples} (warmup {args.warmup} batches, measure {args.measure} batches)")
    if args.use_images:
        print(f"[INFO] Images resolved & existing: {total_imgs_ok}")

if __name__ == "__main__":
    main()
