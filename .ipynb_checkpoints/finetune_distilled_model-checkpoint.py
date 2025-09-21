# 文件名: finetune_final_model.py (修改版)
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

from transformers import BertModel, CLIPVisionModel, CLIPProcessor, BertTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# from ltp import LTP # 不需要
try:
    from torch_geometric.data import Batch  # 用于批处理图
    from torch_geometric.nn import GATConv, global_mean_pool
except ImportError:
    print("❌ 错误: torch_geometric 库未找到或未正确安装。")
    exit()

# ================== 配置 ==================
CSV_PATH = './datasets/TextClassification/toutiao/toutiao_622.csv'
PREPROCESSED_DATA_DIR = './processed_data'          # 预处理数据目录，内含 data_{idx}.pt
PROJECT_ROOT = '/workspace'                         # 与 CSV 的 image_path 拼接
CONTENT_MODEL_PATH = './lightweight_content_model_distilled'
VISION_MODEL_PATH  = './lightweight_vision_model_distilled'   # 目录下需包含 visual_projection.pt
GAT_MODEL_PATH     = './lightweight_gat_model.pth'

GAT_CONFIG = {"in_channels": 100, "hidden_channels": 64, "num_layers": 1, "out_channels": 64, "heads": 2}
NUM_CLASSES = 3
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FINETUNED_MODEL_SAVE_PATH = "./final_lightweight_model.pth"
# ==========================================


class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, heads):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
        self.fc = nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
        x = global_mean_pool(x, batch)
        return self.fc(x)


class FinalDataset(Dataset):
    """读取 CSV + 预处理图数据(data_{idx}.pt) + 加载图片"""
    def __init__(self, csv_path, preprocessed_dir, project_root):
        self.project_root = project_root
        self.preprocessed_dir = preprocessed_dir
        self.df = pd.read_csv(csv_path, header=None, names=['label', 'title', 'content', 'image_path'])

        # 仅保留存在对应预处理文件的样本
        self.valid_indices = [
            idx for idx in range(len(self.df))
            if os.path.exists(os.path.join(self.preprocessed_dir, f'data_{idx}.pt'))
        ]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        # 映射到原始 DataFrame 的行号
        idx = self.valid_indices[index]
        row = self.df.iloc[idx]
        label, content, path = int(row['label']), str(row['content']), str(row['image_path'])

        # 图片（缺失则白图占位）
        image = Image.new('RGB', (224, 224), color='white')
        if path and path.lower() != 'nan':
            try:
                full_path = os.path.join(self.project_root, path)
                if os.path.exists(full_path):
                    image = Image.open(full_path).convert("RGB")
            except Exception:
                pass

        # 载入预处理的图数据，并去掉梯度，规范 dtype
        graph_data_path = os.path.join(self.preprocessed_dir, f'data_{idx}.pt')
        graph_data = torch.load(graph_data_path, map_location='cpu')
        if hasattr(graph_data, "x") and graph_data.x is not None:
            graph_data.x = graph_data.x.detach().requires_grad_(False).to(torch.float32)
        if hasattr(graph_data, "edge_index") and graph_data.edge_index is not None:
            graph_data.edge_index = graph_data.edge_index.to(torch.long).contiguous()

        return content, image, label, graph_data


def collate_fn_final(batch):
    contents, images, labels, graph_data_list = zip(*batch)
    batched_graph = Batch.from_data_list(list(graph_data_list))
    return list(contents), list(images), torch.tensor(list(labels), dtype=torch.long), batched_graph


class FinalLightweightModel(nn.Module):
    """文本BERT + 视觉CLIP-Vision(+projection) + GAT → MLP 分类"""
    def __init__(self, content_path, vision_path, gat_path, num_classes):
        super().__init__()
        # 文本
        self.content_model = BertModel.from_pretrained(content_path)

        # 视觉 + 投影（投影权重从 vision_path/visual_projection.pt 加载）
        self.vision_model = CLIPVisionModel.from_pretrained(vision_path)
        vision_proj = nn.Linear(self.vision_model.config.hidden_size, 512, bias=False)
        vision_proj.load_state_dict(torch.load(os.path.join(vision_path, 'visual_projection.pt'), map_location='cpu'))
        self.visual_projection = vision_proj

        # GAT
        self.gat_model = GATModel(**GAT_CONFIG)
        self.gat_model.load_state_dict(torch.load(gat_path, map_location='cpu'))

        # 分类头
        classifier_input_dim = self.content_model.config.hidden_size + 512 + GAT_CONFIG["out_channels"]
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


def main():
    print(f"--- 开始最终模型微调 (设备: {DEVICE}) ---")
    content_tokenizer = BertTokenizer.from_pretrained(CONTENT_MODEL_PATH)
    vision_processor = CLIPProcessor.from_pretrained(VISION_MODEL_PATH)

    # 数据
    dataset = FinalDataset(CSV_PATH, PREPROCESSED_DATA_DIR, PROJECT_ROOT)
    train_loader = TorchDataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn_final, num_workers=4
    )

    # 模型
    model = FinalLightweightModel(CONTENT_MODEL_PATH, VISION_MODEL_PATH, GAT_MODEL_PATH, NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 类别不平衡加权（基于有效样本）
    valid_labels = dataset.df.iloc[dataset.valid_indices]['label']
    class_counts = valid_labels.value_counts().sort_index()
    if not class_counts.empty:
        weights = class_counts.sum() / (len(class_counts) * class_counts)
        class_weights = torch.tensor(weights.values, dtype=torch.float).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        print("已启用加权损失函数。\n")
    else:
        loss_fn = nn.CrossEntropyLoss()
        print("警告：无法计算类别权重，使用标准损失函数。\n")

    # 训练 + 每个 epoch 评估（整体 + 分类 P/R/F1）
    for epoch in range(NUM_EPOCHS):
        model.train()
        pbar_train = tqdm(train_loader, desc=f"微调 Epoch {epoch + 1}/{NUM_EPOCHS}")
        for batch in pbar_train:
            contents, images, labels, graph_batch = batch

            content_inputs = content_tokenizer(
                contents, return_tensors='pt', max_length=256,
                padding='max_length', truncation=True
            ).to(DEVICE)
            pixel_values = vision_processor(images=images, return_tensors='pt')['pixel_values'].to(DEVICE)
            graph_batch = graph_batch.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(content_inputs, pixel_values, graph_batch)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar_train.set_postfix({"loss": loss.item()})

        # ====== 评估 ======
        model.eval()
        all_preds, all_labels = [], []
        eval_loader = TorchDataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False,
            collate_fn=collate_fn_final, num_workers=4
        )
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="评估中"):
                contents, images, labels, graph_batch = batch
                content_inputs = content_tokenizer(
                    contents, return_tensors='pt', max_length=256,
                    padding='max_length', truncation=True
                ).to(DEVICE)
                pixel_values = vision_processor(images=images, return_tensors='pt')['pixel_values'].to(DEVICE)
                graph_batch = graph_batch.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(content_inputs, pixel_values, graph_batch)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        acc = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        # —— 按类输出 Prec/Rec/F1（类别顺序固定为 0/1/2）
        prec, rec, f1_per_class, support = precision_recall_fscore_support(
            all_labels, all_preds, labels=[0, 1, 2], zero_division=0
        )

        print(f"Epoch {epoch+1}")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Macro F1 : {f1_macro:.4f}")
        print("  —— 每类指标 ——")
        for i in range(3):
            print(f"    Class {i} | P={prec[i]:.4f} R={rec[i]:.4f} F1={f1_per_class[i]:.4f} | n={support[i]}")

    # 保存最终权重
    torch.save(model.state_dict(), FINETUNED_MODEL_SAVE_PATH)
    print(f"\n--- 微调完成！最终模型已保存至: '{FINETUNED_MODEL_SAVE_PATH}' ---")


if __name__ == "__main__":
    main()
