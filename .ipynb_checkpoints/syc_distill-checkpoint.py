import torch
import os
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor, BertModel, BertTokenizer
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.data import DataLoader as GATDataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from ltp import LTP
import pandas as pd
from PIL import Image

# ================== 配置 ==================
CSV_PATH = './datasets/TextClassification/toutiao/toutiao_622.csv'
PROJECT_ROOT = '/workspace'
VISION_TEACHER_MODEL_PATH = './model/clip-vit-base-patch32'
CONTENT_TEACHER_MODEL_PATH = './model/chinese-roberta-wwm-ext'
GAT_TEACHER_CONFIG = {"in_channels": 100, "hidden_channels": 128, "num_layers": 2, "out_channels": 128, "heads": 4}
BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "./syc_new_lightweight_models"

# ================== 视觉模型部分 ==================

class VisionDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, project_root):
        self.project_root = project_root
        df = pd.read_csv(csv_path, header=None, usecols=[3], names=['image_path'], encoding='utf-8')
        self.image_paths = df['image_path'].dropna().astype(str).tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path_from_csv = self.image_paths[idx].strip()
        full_path = os.path.join(self.project_root, path_from_csv)
        try:
            image = Image.open(full_path).convert("RGB")
            return image
        except Exception:
            # 如果图像读取失败，返回 None
            return None

def collate_fn_vision(batch):
    # 过滤掉 None 项
    batch = [img for img in batch if img is not None]
    if not batch:  # 如果批次为空，返回一个空的Tensor
        return None
    return torch.stack(batch)  # 假设所有图像尺寸一致，可以通过 torch.stack 处理

def distill_vision():
    print(f"--- 视觉模型蒸馏开始 (设备: {DEVICE}) ---")
    processor = CLIPProcessor.from_pretrained(VISION_TEACHER_MODEL_PATH)
    teacher_model = CLIPModel.from_pretrained(VISION_TEACHER_MODEL_PATH).vision_model.to(DEVICE)
    teacher_model.eval()

    # 创建学生模型
    student_model = CLIPModel.from_pretrained(VISION_TEACHER_MODEL_PATH).vision_model.to(DEVICE)
    student_model.train()

    # 加载数据集
    dataset = VisionDataset(CSV_PATH, PROJECT_ROOT)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_vision)

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    for epoch in range(NUM_EPOCHS):
        pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        for images in pbar:
            if images is None: continue
            inputs = processor(images=images, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                teacher_embeds = teacher_model(inputs['pixel_values'])
            student_embeds = student_model(inputs['pixel_values'])
            loss = loss_fn(student_embeds, teacher_embeds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})

    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # 手动保存学生模型权重
    torch.save(student_model.state_dict(), os.path.join(OUTPUT_DIR, "student_vision_model.pth"))

    # 保存配置文件
    processor.save_pretrained(OUTPUT_DIR)
    print(f"视觉模型蒸馏完成，模型已保存至: {OUTPUT_DIR}")

# ================== 内容模型部分 ==================

class ContentDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, tokenizer):
        self.tokenizer = tokenizer
        df = pd.read_csv(csv_path, header=None, usecols=[2], names=['content'], encoding='utf-8')
        self.texts = df['content'].dropna().astype(str).tolist()

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        return self.tokenizer(self.texts[idx], return_tensors='pt', max_length=256, padding='max_length', truncation=True)

def distill_content():
    print(f"--- 正文模型蒸馏开始 (设备: {DEVICE}) ---")
    tokenizer = BertTokenizer.from_pretrained(CONTENT_TEACHER_MODEL_PATH)
    teacher_model = BertModel.from_pretrained(CONTENT_TEACHER_MODEL_PATH).to(DEVICE)
    teacher_model.eval()

    student_model = BertModel.from_pretrained(CONTENT_TEACHER_MODEL_PATH).to(DEVICE)
    student_model.train()

    dataset = ContentDataset(CSV_PATH, tokenizer)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    for epoch in range(NUM_EPOCHS):
        pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        for batch in pbar:
            input_ids = batch['input_ids'].squeeze(1).to(DEVICE)
            attention_mask = batch['attention_mask'].squeeze(1).to(DEVICE)
            with torch.no_grad():
                teacher_embeds = teacher_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
            student_embeds = student_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
            loss = loss_fn(student_embeds, teacher_embeds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})

    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    student_model.save_pretrained(OUTPUT_DIR)
    print(f"正文模型蒸馏完成，模型已保存至: {OUTPUT_DIR}")

# ================== GAT模型部分 ==================


LTP_POS_LABELS = [
    'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j', 'k', 'm', 'n', 'nd',
    'nh', 'ni', 'nl', 'ns', 'nt', 'nz', 'o', 'p', 'q', 'r', 'u', 'v',
    'wp', 'ws', 'x', 'z'
]

class GATDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path):
        print("--- 正在初始化GAT数据集 ---")
        try:
            df = pd.read_csv(csv_path, header=None, usecols=[1], names=['title'], on_bad_lines='warn', encoding='utf-8')
            self.titles = df['title'].dropna().astype(str).tolist()
        except Exception as e:
            print(f"读取CSV出错: {e}")
            self.titles = []
        
        if not self.titles:
            print("数据集中没有有效的标题，无法继续。")
            return

        print("正在加载LTP模型...")
        self.ltp = LTP("LTP/small")
        print("LTP模型加载完毕。")

        self.pos_vocab = {tag: i for i, tag in enumerate(LTP_POS_LABELS)}
        self.pos_embedding = nn.Embedding(len(self.pos_vocab), GAT_TEACHER_CONFIG["in_channels"])

    def __len__(self):
        return len(self.titles) if hasattr(self, 'titles') else 0

    def __getitem__(self, idx):
        title = self.titles[idx]
        if not title: return None
        try:
            output = self.ltp.pipeline([title], tasks=["cws", "pos", "dep"])
        except Exception:
            return None
            
        pos_tags = output.pos[0]
        deps_dict = output.dep[0]
        heads = deps_dict['head']   # 这是一个包含所有head索引的列表
        labels = deps_dict['label'] # 这是一个包含所有label字符串的列表
        
        # 确保返回结果不为空
        if not pos_tags or not heads:
             return None

        pos_ids = torch.tensor([self.pos_vocab.get(tag, 0) for tag in pos_tags], dtype=torch.long)
        node_features = self.pos_embedding(pos_ids)
        
        edge_sources, edge_targets = [], []
        # 根据单词数量遍历heads列表
        for i in range(len(heads)):
            head_idx = heads[i] - 1
            if head_idx >= 0:
                edge_sources.append(head_idx)
                edge_targets.append(i)

        if not edge_sources:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        else:
            edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
            
        return Data(x=node_features, edge_index=edge_index)

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


def distill_gat():
    print(f"--- GAT模型蒸馏开始 (设备: {DEVICE}) ---")
    teacher_model = GATModel(**GAT_TEACHER_CONFIG).to(DEVICE)
    student_model = GATModel(**GAT_TEACHER_CONFIG).to(DEVICE)
    teacher_model.eval()

    dataset = GATDataset(CSV_PATH)
    data_loader = GATDataLoader(dataset, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    for epoch in range(NUM_EPOCHS):
        pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        for batch_data in pbar:
            if batch_data is None: continue
            with torch.no_grad():
                teacher_embeds = teacher_model(batch_data)
            student_embeds = student_model(batch_data)
            loss = loss_fn(student_embeds, teacher_embeds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})

    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    torch.save(student_model.state_dict(), OUTPUT_DIR + "/lightweight_gat_model.pth")
    print(f"GAT模型蒸馏完成，模型已保存至: {OUTPUT_DIR}")

# ================== 执行协同蒸馏 ==================

if __name__ == "__main__":
    distill_vision()
    distill_content()
    distill_gat()
