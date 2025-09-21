#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cpu_latency_end2end_teacher.py
- 在 CPU 上对“教师模型（PromptForClassification）”做端到端评测
- 支持 FP32 与 QAT-INT8 两种模式（INT8优先加载QAT权重，失败则回退动态量化）
- 计时：Tokenizer / ImageProc / ModelFwd / End-to-End
- 产出：屏幕 + JSON + CSV
"""

import os, time, json, argparse, random, sys
import numpy as np

# ---- 尝试引入 psutil 获取峰值内存；没有就用 /proc/self/status ----
try:
    import psutil
    PSUTIL_OK = True
except Exception:
    PSUTIL_OK = False

import torch
from torch.utils.data import DataLoader
from PIL import Image

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


# --------------- Utils ---------------

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    class C:
        def __init__(self, **k):
            self.__dict__.update({kk: vv for kk, vv in k.items() if not str(kk).startswith("//")})
    return C(**d)

def build_teacher(cfg, scripts_path: str, device: str="cpu"):
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

def try_load_qat_int8(model: torch.nn.Module, ckpt_path: str) -> bool:
    """
    加载QAT convert后的INT8权重；如失败返回False
    注意：必须是 convert 后保存的 state_dict；否则无法直接load到量化模块上。
    """
    if not ckpt_path:
        return False
    if not os.path.exists(ckpt_path):
        print(f"[WARN] INT8 ckpt 不存在: {ckpt_path}")
        return False
    try:
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)
        print(f"[INFO] 已加载 QAT-INT8 权重: {ckpt_path}")
        return True
    except Exception as e:
        print(f"[WARN] 加载 QAT-INT8 失败: {e}")
        return False

def apply_dynamic_int8(model: torch.nn.Module) -> torch.nn.Module:
    """
    保底方案：对 text encoder 的 Linear 做动态量化（仅CPU有效）
    """
    print("[INFO] 回退到动态量化（quantize_dynamic）以便CPU评测...")
    import torch.ao.quantization as tq
    # 仅对 plm 的 Linear 做量化；Prompt/Verbalizer头保留FP32稳定性
    # 兼容BERT：model.plm.bert 或 model.plm.base_model
    text_module = None
    if hasattr(model.plm, "bert"):
        text_module = model.plm.bert
    elif hasattr(model.plm, "base_model"):
        text_module = model.plm.base_model
    if text_module is None:
        print("[WARN] 未找到文本子模块，跳过动态量化。")
        return model
    qtext = tq.quantize_dynamic(
        text_module, {torch.nn.Linear}, dtype=torch.qint8
    )
    # 把量化后的子模块挂回去
    if hasattr(model.plm, "bert"):
        model.plm.bert = qtext
    else:
        model.plm.base_model = qtext
    return model

def build_clip_cpu(clip_path: str):
    if not os.path.isdir(clip_path):
        raise FileNotFoundError(f"未找到本地 CLIP 目录: {clip_path}")
    print(f"[INFO] 从本地加载 CLIP（CPU）: {clip_path}")
    proc = CLIPProcessor.from_pretrained(clip_path, local_files_only=True)
    model = CLIPModel.from_pretrained(clip_path, local_files_only=True).to("cpu").eval()
    return proc, model

def get_img_path_from_meta(ex: InputExample):
    if isinstance(ex.meta, str):
        return ex.meta
    if isinstance(ex.meta, dict):
        return ex.meta.get("img_path")
    return None

def extract_clip_features_cpu(img_paths, data_root, proc, clip):
    imgs, idx_map = [], []
    for i, p in enumerate(img_paths):
        if not p: continue
        # 你的数据里 meta 可能已经带 "datasets/TextClassification/toutiao/images/xxx"
        # 我们优先尝试绝对/相对路径，若不存在再拼接 data_root/images/
        cand = [p, os.path.join(data_root, "images", p)]
        full = None
        for c in cand:
            if os.path.exists(c):
                full = c; break
        if full is None: continue
        try:
            imgs.append(Image.open(full).convert("RGB")); idx_map.append(i)
        except Exception:
            pass
    if not imgs:
        return [None] * len(img_paths)
    inputs = proc(images=imgs, return_tensors="pt")
    with torch.inference_mode():
        feats = clip.get_image_features(**inputs)  # (n, 512)
    feats = feats.detach().cpu()
    out = [None] * len(img_paths)
    for j, i in enumerate(idx_map):
        out[i] = feats[j]
    return out

def tokenize_batch(examples, template: Template, tokenizer, WrapperClass, max_len: int, image_features=None):
    wrap_keys = signature(WrapperClass.__init__).args
    kwargs = {"max_seq_length": max_len, "truncate_method": "tail", "tokenizer": tokenizer}
    to_pass = {k: kwargs[k] for k in kwargs if k in wrap_keys}
    token_wrapper = WrapperClass(**to_pass)

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

def percentile(arr, ps=(50,90,95,99)):
    arr = np.array(arr, dtype=np.float64)
    out = {}
    for p in ps:
        out[f"p{p}"] = float(np.percentile(arr, p)) if arr.size else 0.0
    return out

def get_peak_rss_mb():
    if PSUTIL_OK:
        process = psutil.Process(os.getpid())
        # max / recent? psutil 无法直接给出历史峰值，这里用当前值近似
        rss = process.memory_info().rss / (1024**2)
        return float(rss)
    # 退化：读 /proc/self/status 的 VmHWM（峰值）或 VmRSS（当前）
    try:
        hwm = None; rss = None
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    hwm = float(line.split()[1]) / 1024.0
                if line.startswith("VmRSS:"):
                    rss = float(line.split()[1]) / 1024.0
        return float(hwm or rss or 0.0)
    except Exception:
        return 0.0

def append_csv_row(csv_path, row_dict, header_order):
    import csv
    need_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path)==0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header_order)
        if need_header:
            writer.writeheader()
        writer.writerow({k: row_dict.get(k, "") for k in header_order})


# --------------- Main ---------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_file", type=str, required=True)
    ap.add_argument("--variant", type=str, choices=["teacher-fp32","teacher-int8"], required=True,
                    help="teacher-fp32 或 teacher-int8（QAT优先，失败回退动态量化）")
    ap.add_argument("--ckpt", type=str, default="", help="QAT-INT8 的 state_dict 路径（convert 后保存的）")
    ap.add_argument("--subset", type=int, default=-1, help="-1=全量；>0=抽样条数")
    ap.add_argument("--seq-len", type=int, default=192)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--measure", type=int, default=100000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=144)
    ap.add_argument("--use-images", action="store_true")
    ap.add_argument("--clip-path", type=str, default="./model/clip-vit-base-patch32")
    ap.add_argument("--out", type=str, default="./ckpts/cpu_latency_report.json")
    args = ap.parse_args()

    # 固定 CPU 线程
    torch.set_num_threads(max(1, args.threads))
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)

    set_seed(args.seed)

    # 路径
    cfg = load_config(args.config_file)
    proj_root = os.path.abspath(".")
    data_root = os.path.join(proj_root, "datasets", "TextClassification", cfg.dataset_name)
    scripts_path = os.path.join(proj_root, "scripts", "TextClassification", cfg.prompt_name)

    # 数据
    processor = CnClickbaitProcessor()
    test_set = processor.get_test_examples(data_root)
    if args.subset is None or args.subset < 0 or args.subset >= len(test_set):
        picked = test_set
    else:
        random.seed(args.seed)
        picked = random.sample(test_set, args.subset)

    # 构建 Teacher
    device = "cpu"
    model, tokenizer, template, WrapperClass = build_teacher(cfg, scripts_path, device="cpu")

    used_qat_int8 = False
    if args.variant == "teacher-int8":
        # 先尝试加载 QAT INT8
        if args.ckpt:
            used_qat_int8 = try_load_qat_int8(model, args.ckpt)
        elif hasattr(cfg, "quantized_model_path"):
            used_qat_int8 = try_load_qat_int8(model, cfg.quantized_model_path)
        # 回退动态量化
        if not used_qat_int8:
            model = apply_dynamic_int8(model)

    # CLIP（可选）
    clip_proc, clip_model = (None, None)
    if args.use_images:
        clip_proc, clip_model = build_clip_cpu(args.clip_path)

    # dataloader：直通 InputExample
    dl = DataLoader(
        picked,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=lambda batch: batch
    )

    # 计时容器
    t_e2e, t_tok, t_img, t_fwd = [], [], [], []
    total_samples = 0
    seen_batches = 0

    # 预热 + 测量
    for batch_examples in dl:
        if len(batch_examples) == 0:
            continue

        t0 = time.perf_counter()

        # Image features（可选，CPU）
        img_feats = None
        if args.use_images:
            img_paths = [get_img_path_from_meta(ex) for ex in batch_examples]
            t_img0 = time.perf_counter()
            img_feats = extract_clip_features_cpu(img_paths, data_root, clip_proc, clip_model)
            t_img1 = time.perf_counter()
            t_img.append(t_img1 - t_img0)

        # Tokenize + openprompt 包装
        t_tok0 = time.perf_counter()
        feats_list = tokenize_batch(batch_examples, template, tokenizer, WrapperClass, args.seq_len, image_features=img_feats)
        t_tok1 = time.perf_counter()
        t_tok.append(t_tok1 - t_tok0)

        # collate
        batch = InputFeatures.collate_fct(feats_list)
        # CPU上不需要 .to(device)（张量默认CPU）；为了统一，检查一次
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v  # already on CPU

        # 前向
        t_f0 = time.perf_counter()
        with torch.inference_mode():
            _ = model(batch)
        t_f1 = time.perf_counter()
        t_fwd.append(t_f1 - t_f0)

        # End-to-End
        t1 = time.perf_counter()
        t_e2e.append(t1 - t0)

        seen_batches += 1
        total_samples += len(batch["label"])

        # 丢弃预热批
        if seen_batches <= args.warmup:
            t_e2e.pop(); t_fwd.pop(); t_tok.pop()
            if args.use_images and len(t_img) > 0: t_img.pop()
            total_samples -= len(batch["label"])
            continue

        # 达到测量批次数
        if seen_batches - args.warmup >= args.measure:
            break

    bs = args.batch_size

    def summarize(name, arr):
        if not arr:
            return {"avg": 0.0}
        arr = np.array(arr, dtype=np.float64)
        per_sample = arr / bs
        out = {
            "batch_avg_ms": float(arr.mean()*1000.0),
            "per_sample_avg_ms": float(per_sample.mean()*1000.0),
        }
        out.update({
            "p50_ms": float(np.percentile(per_sample,50)*1000.0),
            "p90_ms": float(np.percentile(per_sample,90)*1000.0),
            "p95_ms": float(np.percentile(per_sample,95)*1000.0),
            "p99_ms": float(np.percentile(per_sample,99)*1000.0),
        })
        return out

    res_tok = summarize("Tokenizer", t_tok)
    res_img = summarize("ImageProc", t_img) if args.use_images else None
    res_fwd = summarize("ModelFwd", t_fwd)
    res_e2e = summarize("End2End", t_e2e)

    measured_batches = max(0, seen_batches - args.warmup)
    thr = float(total_samples / sum(t_e2e)) if t_e2e else 0.0
    peak_rss = get_peak_rss_mb()

    # 打印
    def pline(name, r):
        if r is None: return
        print(f"{name}  batch_avg={r['batch_avg_ms']:.2f}ms  "
              f"p50={r['p50_ms']:.2f}ms  p90={r['p90_ms']:.2f}ms  "
              f"p95={r['p95_ms']:.2f}ms  p99={r['p99_ms']:.2f}ms  "
              f"per_sample_avg={r['per_sample_avg_ms']:.2f}ms")

    print("\n=== CPU Latency (lower is better) ===")
    pline("End-to-End", res_e2e)
    pline("Tokenizer", res_tok)
    if args.use_images:
        pline("ImageProc", res_img)
    pline("ModelFwd", res_fwd)

    if measured_batches > 0:
        print(f"\nThroughput: {thr:.2f} samples/s (over {total_samples} samples, {measured_batches} measured batches)")
    print(f"CPU Peak RSS: {peak_rss:.1f} MB")
    print(f"\n[INFO] Variant={args.variant}  Threads={args.threads}  UseImages={args.use_images}  "
          f"QAT_INT8_Loaded={used_qat_int8}  Evaluated={total_samples}")

    # 保存 JSON
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {
        "device": "cpu",
        "variant": args.variant,
        "threads": args.threads,
        "use_images": bool(args.use_images),
        "subset": len(picked),
        "batch_size": bs,
        "warmup_batches": args.warmup,
        "measured_batches": measured_batches,
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

    # 追加 CSV（方便做论文表）
    csv_path = args.out.replace(".json", ".csv")
    csv_headers = [
        "variant","threads","use_images",
        "e2e_ms","token_ms","img_ms","fwd_ms",
        "throughput_sps","cpu_peak_rss_mb","batch_size","subset"
    ]
    row = {
        "variant": args.variant,
        "threads": args.threads,
        "use_images": args.use_images,
        "e2e_ms": round(res_e2e.get("per_sample_avg_ms", 0.0), 3),
        "token_ms": round(res_tok.get("per_sample_avg_ms", 0.0), 3),
        "img_ms": round((res_img or {}).get("per_sample_avg_ms", 0.0), 3) if args.use_images else "",
        "fwd_ms": round(res_fwd.get("per_sample_avg_ms", 0.0), 3),
        "throughput_sps": round(thr, 2),
        "cpu_peak_rss_mb": round(peak_rss, 1),
        "batch_size": bs,
        "subset": len(picked)
    }
    append_csv_row(csv_path, row, csv_headers)
    print(f"[SAVED] CSV  -> {csv_path}")


if __name__ == "__main__":
    main()
