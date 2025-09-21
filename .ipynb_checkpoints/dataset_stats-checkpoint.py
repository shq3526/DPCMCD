#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, argparse, pandas as pd, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV 路径：label, title, content, image_path 四列（无表头）")
    ap.add_argument("--project_root", default=".", help="图片相对根目录，用于校验文件是否存在")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, header=None, names=["label","title","content","image_path"])
    n = len(df)

    # 字符长度统计
    title_len = df["title"].astype(str).str.len()
    content_len = df["content"].astype(str).str.len()
    
    # --- 修正逻辑开始 ---

    # 函数：计算单个字符串中的图片路径数量
    def count_paths(path_string, delimiter='|'):
        # 修正：增加对 'nan' 字符串的判断，并更稳健地处理空值
        if pd.isna(path_string) or not isinstance(path_string, str) or path_string.strip().lower() in ['', 'nan']:
            return 0
        
        path_string = path_string.strip("[]")
        return len([p for p in path_string.split(delimiter) if p.strip()])

    # 函数：计算单个字符串中本地存在的图片数量
    def count_local_paths(path_string, project_root, delimiter='|'):
        # 修正：增加对 'nan' 字符串的判断，并更稳健地处理空值
        if pd.isna(path_string) or not isinstance(path_string, str) or path_string.strip().lower() in ['', 'nan']:
            return 0
        
        path_string = path_string.strip("[]")
        paths = [p.strip() for p in path_string.split(delimiter) if p.strip()]
        
        count = 0
        for p in paths:
            if p.startswith(("http://","https://","www.")):
                continue
            full_path = p if os.path.isabs(p) else os.path.normpath(os.path.join(project_root, p))
            if os.path.exists(full_path):
                count += 1
        return count

    # 应用修正后的函数进行统计
    # 直接在原始列上应用，fillna('')不再是必须的，因为函数内部处理了NaN
    img_counts_per_row = df["image_path"].apply(count_paths)
    local_img_counts_per_row = df["image_path"].apply(lambda p: count_local_paths(p, args.project_root))

    total_img_paths = img_counts_per_row.sum()
    total_local_img_paths = local_img_counts_per_row.sum()
    
    # 重新计算包含图片的新闻条数
    has_img_nonempty = (img_counts_per_row > 0)
    
    # --- 修正逻辑结束 ---

    print("===== 数据集统计 (已修正) =====")
    print(f"样本总数 (新闻条数) : {n}")
    print(f"标题平均长度(字符)   : {title_len.mean():.2f} | 中位数: {title_len.median():.0f}")
    print(f"正文平均长度(字符)   : {content_len.mean():.2f} | 中位数: {content_len.median():.0f}")
    print("-" * 25)
    print(f"包含图片的新闻条数   : {has_img_nonempty.sum()} / {n} ({100*has_img_nonempty.mean():.2f}%)")
    print(f"总图片路径数量       : {total_img_paths}")
    print(f"本地存在的总图片数量 : {total_local_img_paths}")
    print(f"平均图片数/新闻      : {total_img_paths / n:.2f}")
    print("\n（说明）“总图片路径数量”是基于'|'分隔符统计的。")
    print("      “平均图片数/新闻”为总图片路径数除以新闻总条数。")

if __name__ == "__main__":
    main()