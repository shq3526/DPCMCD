#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cpu_latency_eval_teacher.py
在 CPU 上对“原始教师模型（未量化）”做小子集延迟测试的独立脚本。
- 读取你的 config.json/txt（与 train_evaluate_unified_full.py 同口径）
- 仅加载模型 + 测试集，抽取一小段（前 N 条或随机 N 条）
- 逐样本（batch=1）计时：Tokenizer（可选）、ModelFwd、End-to-End
- 固定 CPU 线程数，提供 p50/p90/p95/p99 分位
- 输出到屏幕，同时保存 JSON 报告与 TXT 概要

用法示例：
python cpu_latency_eval_teacher.py \
  --config_file ./config/config.txt \
  --subset 200 \
  --seq-len 192 \
  --threads 1 \
  --include-tokenizer \
  --seed 144

（如果你想禁用 tokenizer 计时，只测“输入已准备好后的前向”，就去掉 --include-tokenizer）
"""
import os, sys, time, json, argparse, random
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from openprompt.utils.logging import logger
from openprompt.data_utils.text_classification_dataset import CnClickbaitProcessor
from openprompt.data_utils.utils import InputExample, InputFeatures
from openprompt.prompts import ManualVerbalizer, PtuningTemplate
from openprompt.plms import load_plm
from openprompt.prompt_base import Template
from openprompt.plms.utils import TokenizerWrapper
from openprompt.utils import signature
from openprompt.utils.reproduciblity import set_seed


# ---------- helper: load config ----------
def load_config_from_file(config_path: str):
    print(f"[INFO] 加载配置: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    class C:
        def __init__(self, **d): self.__dict__.update({k: v for k, v in d.items() if not str(k).startswith('//')})
    return C(**cfg)


# ---------- tokenizer wrapper ----------
def build_prompt_model_and_wrappers(config, scripts_path: str, device: str):
    # teacher: 无量化包装
    plm, tokenizer, model_config, WrapperClass = load_plm(config.model_type, config.model_name_or_path)
    template_path = os.path.join(scripts_path, "ptuning_template.txt")
    with open(template_path, "r", encoding="utf-8") as f:
        template_text = f.readlines()[config.template_id].rstrip()
    template = PtuningTemplate(model=plm, tokenizer=tokenizer, text=template_text)
    verbalizer_path = os.path.join(scripts_path, config.verbalizer_file_name)
    with open(verbalizer_path, "r", encoding="utf-8") as f:
        verbalizer_dict = json.load(f)
    classes = list(verbalizer_dict["label_words"].keys())
    verbalizer = ManualVerbalizer(tokenizer, classes=classes)
    verbalizer.label_words = verbalizer_dict["label_words"]
    from openprompt import PromptForClassification
    prompt_model = PromptForClassification(plm=plm, template=template, verbalizer=verbalizer, freeze_plm=False)
    prompt_model.to(device).eval()
    return prompt_model, tokenizer, template, WrapperClass


def tokenize_one(example: InputExample, template: Template, tokenizer, WrapperClass, max_seq_len: int):
    tokenizer_wrapper_init_keys = signature(WrapperClass.__init__).args
    prepare_kwargs = {"max_seq_length": max_seq_len, "truncate_method": "tail", "tokenizer": tokenizer}
    to_pass_kwargs = {k: prepare_kwargs[k] for k in prepare_kwargs if k in tokenizer_wrapper_init_keys}
    tokenizer_wrapper = WrapperClass(**to_pass_kwargs)
    wrapped = template.wrap_one_example(example)
    tokenized = tokenizer_wrapper.tokenize_one_example(wrapped, teacher_forcing=False)
    feats = {**tokenized, **wrapped[1]}
    return InputFeatures(**feats).to_tensor()


# ---------- timing helpers ----------
def percentile(arr, ps=(50, 90, 95, 99)):
    arr = np.array(arr, dtype=np.float64)
    out = {}
    for p in ps:
        out[f"p{p}"] = float(np.percentile(arr, p)) if arr.size else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_file", type=str, required=True)
    ap.add_argument("--subset", type=int, default=200, help="抽取用于延迟评测的样本数（来自测试集）")
    ap.add_argument("--seq-len", type=int, default=192, help="最大序列长度，与训练一致")
    ap.add_argument("--threads", type=int, default=1, help="CPU 线程数（推荐 1 或 4）")
    ap.add_argument("--include-tokenizer", action="store_true", help="是否单独计时 tokenizer（更接近端到端）")
    ap.add_argument("--seed", type=int, default=144)
    ap.add_argument("--out", type=str, default="./ckpts/20250911-153849-toutiao622-Acc0.9677-F1s0.9357.ckpt")
    args = ap.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)

    cfg = load_config_from_file(args.config_file)
    device = "cpu"
    print(f"[INFO] 强制在 CPU 上评测，threads={args.threads}")

    project_root = os.path.abspath(".")
    data_path = os.path.join(project_root, "datasets", "TextClassification", cfg.dataset_name)
    scripts_path = os.path.join(project_root, "scripts", "TextClassification", cfg.prompt_name)

    # dataset
    processor = CnClickbaitProcessor()
    test_set: List[InputExample] = processor.get_test_examples(data_path)
    if len(test_set) == 0:
        print("[ERROR] 测试集为空，请检查数据路径。"); return

    # subset
    if args.subset < len(test_set):
        random.seed(args.seed)
        idx = random.sample(range(len(test_set)), args.subset)
        picked = [test_set[i] for i in idx]
    else:
        picked = test_set
    print(f"[INFO] 取样 {len(picked)} / {len(test_set)} 用于 CPU 延迟测试。")

    # model
    model, tokenizer, template, WrapperClass = build_prompt_model_and_wrappers(cfg, scripts_path, device)

    # tokenizer 计时（可选）
    tokenizer_ms = []
    features_list = []
    if args.include_tokenizer:
        print("[INFO] 启用 tokenizer 计时...")
        for ex in picked:
            t0 = time.perf_counter()
            feats = tokenize_one(ex, template, tokenizer, WrapperClass, args.seq_len)
            t1 = time.perf_counter()
            tokenizer_ms.append((t1 - t0) * 1000.0)
            features_list.append(feats)
    else:
        # 仅一次性 tokenize（不计时），避免把 tokenize 纳入 E2E
        for ex in picked:
            feats = tokenize_one(ex, template, tokenizer, WrapperClass, args.seq_len)
            features_list.append(feats)

    # DataLoader (bs=1)
    dl = DataLoader(features_list, batch_size=1, collate_fn=InputFeatures.collate_fct)

    # warmup model
    with torch.inference_mode():
        for i, batch in enumerate(dl):
            if i >= 10: break
            batch.to("cpu")
            _ = model(batch)

    # timing (strict CPU, batch=1)
    e2e_ms, fwd_ms = [], []
    with torch.inference_mode():
        for batch in dl:
            t0 = time.perf_counter()
            batch.to("cpu")
            t1 = time.perf_counter()
            _ = model(batch)
            t2 = time.perf_counter()
            e2e_ms.append((t2 - t0) * 1000.0)
            fwd_ms.append((t2 - t1) * 1000.0)

    # aggregate
    report = {
        "device": "cpu",
        "threads": args.threads,
        "dataset": cfg.dataset_name,
        "prompt": cfg.prompt_name,
        "seq_len": args.seq_len,
        "subset": len(picked),
        "include_tokenizer": bool(args.include_tokenizer),
        "latency_ms": {
            "tokenizer": {
                "avg": float(np.mean(tokenizer_ms)) if tokenizer_ms else None,
                **(percentile(tokenizer_ms) if tokenizer_ms else {})
            },
            "model_fwd": {
                "avg": float(np.mean(fwd_ms)) if fwd_ms else 0.0,
                **percentile(fwd_ms)
            },
            "end_to_end": {
                "avg": float(np.mean(e2e_ms)) if e2e_ms else 0.0,
                **percentile(e2e_ms)
            }
        }
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # also write a short txt summary
    txt_path = args.out.replace(".json", ".txt")
    lines = []
    lines.append("===== Teacher CPU Latency Report =====")
    lines.append(f"Device: CPU, Threads={args.threads}")
    lines.append(f"Subset: {report['subset']} samples, SeqLen={args.seq_len}, IncludeTokenizer={report['include_tokenizer']}")
    lines.append(f"ModelFwd avg: {report['latency_ms']['model_fwd']['avg']:.3f} ms  "
                 f"(p50={report['latency_ms']['model_fwd']['p50']:.3f}, p95={report['latency_ms']['model_fwd']['p95']:.3f})")
    lines.append(f"End2End  avg: {report['latency_ms']['end_to_end']['avg']:.3f} ms  "
                 f"(p50={report['latency_ms']['end_to_end']['p50']:.3f}, p95={report['latency_ms']['end_to_end']['p95']:.3f})")
    if report['latency_ms']['tokenizer']['avg'] is not None:
        lines.append(f"Tokenizer avg: {report['latency_ms']['tokenizer']['avg']:.3f} ms  "
                     f"(p50={report['latency_ms']['tokenizer']['p50']:.3f}, p95={report['latency_ms']['tokenizer']['p95']:.3f})")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"[SAVED] JSON -> {args.out}")
    print(f"[SAVED] TXT  -> {txt_path}")


if __name__ == "__main__":
    main()
