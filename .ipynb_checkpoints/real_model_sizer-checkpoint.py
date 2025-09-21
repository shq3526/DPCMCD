#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
real_model_sizer.py
精确统计“真实”参数量与模型大小（从本地 checkpoint 读取），支持 .pt/.bin/.safetensors。
同时导出 JSON / CSV / LaTeX 表格，便于论文直接使用。

功能要点：
1) 逐 tensor 统计参数量与字节数（按 dtype 精确：fp32/fp16/bf16/int8 等）
2) 支持把一个模型分成多个部件输入（text/vision/gat...），自动汇总
3) 可以一次比较 Teacher / Student(FP32) / Student(INT8) / 自定义分组
4) 输出：
   - model_sizes.json：包含每个分组的总参、总大小(MB)以及各文件的细分
   - model_sizes.csv：单行扁平表（适合粘进 Excel）
   - model_sizes.tex：LaTeX 表格片段（直接粘进论文）

用法示例：
python real_model_sizer.py \
  --group teacher ./model/chinese-roberta-wwm-ext/pytorch_model.bin ./model/clip-vit-base-patch32/pytorch_model.bin \
  --group student_fp32 ./new_final_lightweight_model.pth \
  --group student_int8 ./model_quantized_native/quantized_roberta_native.pth

# 也可显式标注部件名（可重复多次）：
python real_model_sizer.py \
  --group teacher:text=./model/chinese-roberta-wwm-ext/pytorch_model.bin,vision=./model/clip-vit-base-patch32/pytorch_model.bin \
  --group student_fp32:all=./new_final_lightweight_model.pth \
  --group student_int8:all=./model_quantized_native/quantized_roberta_native.pth

备注：
- 对于“合并权重 all”与“分部件 text/vision/gat”的输入，脚本会分别统计并给出两种口径的总计；
  若两者同时给出，优先以“all”的数为总计，并在 JSON 中保留“parts_sum”为参考。
"""
import os, sys, argparse, json, csv
from typing import Dict, Any, List, Tuple

# Optional deps
try:
    import torch
except Exception as e:
        print("❌ 需要安装 PyTorch 才能解析 .pt/.bin。", file=sys.stderr)
        raise

def human_mb(num_bytes: int) -> float:
    return round(num_bytes / (1024**2), 2)

def dtype_bytes(dtype) -> int:
    # Robust mapping for common dtypes
    import torch
    if dtype == torch.float32: return 4
    if dtype == torch.float:   return 4
    if dtype == torch.float16: return 2
    if dtype == torch.bfloat16:return 2
    if dtype == torch.float64: return 8
    if dtype == torch.int64:   return 8
    if dtype == torch.int32:   return 4
    if dtype == torch.int16:   return 2
    if dtype == torch.int8:    return 1
    if dtype == torch.uint8:   return 1
    if dtype == torch.bool:    return 1
    # quantized types often report element_size() correctly, but add a fallback:
    try:
        return torch.tensor([], dtype=dtype).element_size()
    except Exception:
        return 4

def load_state_dict_any(path: str) -> Dict[str, Any]:
    """支持 .pt/.bin (torch.save) 与 .safetensors，返回一个字典 {tensor_name: torch.Tensor}"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file as load_safetensors
        except Exception:
            raise RuntimeError("需要安装 safetensors 才能读取 .safetensors")
        sd = load_safetensors(path, device="cpu")
        return sd

    # torch .pt/.bin
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]

    if not isinstance(sd, dict):
        raise RuntimeError("不支持的 checkpoint 格式：不是 dict/state_dict。")
    return sd

def stats_from_state_dict(sd: Dict[str, Any]) -> Tuple[int, int, Dict[str, Dict[str, float]]]:
    """返回： (total_params:int, total_bytes:int, per_dtype:dict)"""
    import torch
    total_p = 0
    total_b = 0
    per_dtype = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        n = v.numel()
        db = dtype_bytes(v.dtype)
        b = n * db
        total_p += n
        total_b += b
        key = str(v.dtype).replace("torch.", "")
        if key not in per_dtype:
            per_dtype[key] = {"params": 0, "bytes": 0}
        per_dtype[key]["params"] += n
        per_dtype[key]["bytes"] += b
    # to MB for readability
    for key, rec in per_dtype.items():
        rec["size_mb"] = human_mb(rec["bytes"])
    return int(total_p), int(total_b), per_dtype

def parse_group_arg(group_args: List[str]) -> Dict[str, Dict[str, str]]:
    """解析 --group teacher a.pt b.bin 或 name:part=path[,part=path...]"""
    groups = {}
    for token in group_args:
        if ":" in token and ("=" in token):
            name, tail = token.split(":", 1)
            parts = {}
            for pair in tail.split("," ):
                if "=" not in pair:
                    continue
                k, v = pair.split("=", 1)
                parts[k.strip()] = v.strip()
            groups.setdefault(name, {"parts": {}, "files": []})
            groups[name]["parts"].update(parts)
            groups[name]["files"].extend(list(parts.values()))
        else:
            if os.path.sep in token or token.endswith((".pt", ".bin", ".safetensors")):
                groups.setdefault("_pending", {"files": []})
                groups["_pending"]["files"].append(token.strip())
            else:
                groups.setdefault(token, {"parts": {}, "files": []})
    if "_pending" in groups:
        names = [k for k in groups.keys() if k != "_pending"]
        if not names:
            groups["default"] = groups.pop("_pending")
        else:
            last = names[-1]
            groups[last]["files"].extend(groups["_pending"]["files"])
            groups.pop("_pending")
    return groups

def main():
    ap = argparse.ArgumentParser(description="精确统计模型参数与大小（支持多分组）。")
    ap.add_argument("--group", nargs="+", action="append", help="定义一个分组及其文件。可多次使用 --group。")
    ap.add_argument("--out-json", type=str, default="model_sizes.json")
    ap.add_argument("--out-csv", type=str, default="model_sizes.csv")
    ap.add_argument("--out-tex", type=str, default="model_sizes.tex")
    args = ap.parse_args()

    if not args.group:
        print("请至少提供一个 --group。示例：\n  --group teacher:text=./.../pytorch_model.bin,vision=./.../pytorch_model.bin\n  --group student_fp32:all=./new_final_lightweight_model.pth\n  --group student_int8:all=./model_quantized_native/quantized_roberta_native.pth", file=sys.stderr)
        sys.exit(1)

    merged = {}
    for g in args.group:
        parsed = parse_group_arg(g)
        for name, rec in parsed.items():
            if name not in merged:
                merged[name] = {"parts": {}, "files": []}
            merged[name]["parts"].update(rec.get("parts", {}))
            merged[name]["files"].extend(rec.get("files", []))

    summary = {}
    for name, rec in merged.items():
        parts = rec.get("parts", {})
        files = rec.get("files", [])

        group_info = {"parts": {}, "files": [], "sum_parts_params": 0, "sum_parts_bytes": 0, "per_file": []}
        # parts
        for p_name, p_path in parts.items():
            sd = load_state_dict_any(p_path)
            p_cnt, p_bytes, per_dtype = stats_from_state_dict(sd)
            group_info["parts"][p_name] = {
                "params": p_cnt,
                "size_mb": human_mb(p_bytes),
                "per_dtype": per_dtype,
                "path": p_path
            }
            group_info["sum_parts_params"] += p_cnt
            group_info["sum_parts_bytes"] += p_bytes

        # files
        all_params = 0
        all_bytes = 0
        for p in files:
            sd = load_state_dict_any(p)
            f_cnt, f_bytes, f_dtype = stats_from_state_dict(sd)
            group_info["per_file"].append({
                "path": p, "params": f_cnt, "size_mb": human_mb(f_bytes), "per_dtype": f_dtype
            })
            all_params += f_cnt
            all_bytes += f_bytes

        # final total
        if "all" in parts:
            final_params = group_info["parts"]["all"]["params"]
            final_bytes  = int(round(group_info["parts"]["all"]["size_mb"] * (1024**2)))
        elif files:
            final_params = all_params
            final_bytes  = all_bytes
        else:
            final_params = group_info["sum_parts_params"]
            final_bytes  = group_info["sum_parts_bytes"]

        group_info["total_params"] = int(final_params)
        group_info["total_size_mb"] = human_mb(final_bytes)
        group_info["parts_sum_params"] = int(group_info["sum_parts_params"])
        group_info["parts_sum_size_mb"] = human_mb(group_info["sum_parts_bytes"])

        summary[name] = group_info

    print("===== MODEL SIZE SUMMARY =====")
    for name, info in summary.items():
        print(f"[{name}] params={info['total_params']:,}  size={info['total_size_mb']} MB")

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON -> {args.out_json}")

    import csv
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "total_params", "total_size_mb", "parts_sum_params", "parts_sum_size_mb"])
        for name, info in summary.items():
            writer.writerow([name, info["total_params"], info["total_size_mb"], info["parts_sum_params"], info["parts_sum_size_mb"]])
    print(f"Saved CSV  -> {args.out_csv}")

    def fmt_m(p): 
        return round(p / 1e6, 2)
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Model} & \\textbf{Params (M)} & \\textbf{Size (MB)}\\\\")
    lines.append("\\midrule")
    for name, info in summary.items():
        row_name = name.replace("_","\\_")
        lines.append(f"{row_name} & {fmt_m(info['total_params'])} & {info['total_size_mb']}\\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Parameter count and checkpoint size.}")
    lines.append("\\end{table}")
    tex = "\n".join(lines)
    with open(args.out_tex, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"Saved LaTeX -> {args.out_tex}")

if __name__ == "__main__":
    main()
