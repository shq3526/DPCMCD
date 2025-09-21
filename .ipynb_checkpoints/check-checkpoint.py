import os, json
from PIL import Image
from openprompt.data_utils.text_classification_dataset import CnClickbaitProcessor

CONFIG = "./config/config.txt"
with open(CONFIG, "r", encoding="utf-8") as f:
    cfg = json.load(f)

proj_root  = os.path.abspath(".")
dataset    = cfg["dataset_name"]
data_root  = os.path.join(proj_root, "datasets", "TextClassification", dataset)
images_dir = os.path.join(data_root, "images")

def get_meta_path(meta):
    if isinstance(meta, str): return meta
    if isinstance(meta, dict): return meta.get("img_path")
    return None

def resolve_full_path(meta_path: str) -> str | None:
    if not meta_path: 
        return None
    p = meta_path.strip()
    # 1) 如果已经是绝对路径，或以项目内 data_root 开头，直接用
    if os.path.isabs(p):
        cand = p
    elif p.startswith("datasets/"):  # 你的数据正是这种
        cand = os.path.join(proj_root, p)
    else:
        # 认为是纯文件名
        cand = os.path.join(images_dir, p)
    cand = os.path.normpath(cand)
    if os.path.exists(cand):
        return cand
    # 2) 兜底：再用 basename 到 images_dir 试一次
    base = os.path.basename(p)
    cand2 = os.path.join(images_dir, base)
    cand2 = os.path.normpath(cand2)
    return cand2 if os.path.exists(cand2) else None

proc = CnClickbaitProcessor()
test_set = proc.get_test_examples(data_root)

total = len(test_set)
has_meta, has_file, readable = 0, 0, 0
for ex in test_set:
    mp = get_meta_path(ex.meta)
    if mp:
        has_meta += 1
        full = resolve_full_path(mp)
        if full and os.path.exists(full):
            has_file += 1
            try:
                Image.open(full).convert("RGB")
                readable += 1
            except Exception:
                pass

print(f"Test samples: {total}")
print(f"With img meta: {has_meta} ({has_meta/total:.1%})")
print(f"File exists:   {has_file} ({(has_file or 0)/total:.1%})")
print(f"Readable:      {readable} ({(readable or 0)/total:.1%})")
