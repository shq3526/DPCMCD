# =====================================================================================
# Independent Knowledge Distillation for Multi-Modal Components
#
# This script performs knowledge distillation on three separate modalities:
# 1. GAT Model (Syntactic Features from Titles)
# 2. Vision Model (Image Features via CLIP)
# 3. Content Model (Textual Features via BERT)
#
# Each component is distilled independently from a larger teacher model to a
# smaller, more efficient student model. This is part of the 'nosyc' (non-synergistic)
# workflow for model compression.
#
# Author: [Haoqian Song, Haoran Yin, Fuwen Zhao]
# Date: September 18, 2025
# =====================================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
from ltp import LTP
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from transformers import BertModel, BertConfig, BertTokenizer, CLIPModel, CLIPVisionModel, CLIPVisionConfig
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from torch.utils.data import Dataset

# ================== Configuration ==================
CSV_PATH = './datasets/TextClassification/toutiao/toutiao_622.csv'
# Teacher model is larger, more complex
TEACHER_CONFIG_GAT = {"in_channels": 100, "hidden_channels": 128, "num_layers": 2, "out_channels": 128, "heads": 4}
# Student model is smaller and shallower for efficiency
STUDENT_CONFIG_GAT = {"in_channels": 100, "hidden_channels": 64, "num_layers": 1, "out_channels": 64, "heads": 2}
STUDENT_CONFIG_CONTENT = {"num_hidden_layers": 4, "hidden_size": 384, "intermediate_size": 1536, "num_attention_heads": 6}
STUDENT_CONFIG_VISION = {"num_hidden_layers": 6, "hidden_size": 256, "intermediate_size": 1024, "num_attention_heads": 4}
BATCH_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_FILE_GAT = "./lightweight_gat_model.pth"
OUTPUT_DIR_CONTENT = "./lightweight_content_model_distilled"
OUTPUT_DIR_VISION = "./lightweight_vision_model_distilled"
BASE_CLIP_MODEL_NAME = './model/clip-vit-base-patch32'
TEACHER_MODEL_PATH_CONTENT = './model/chinese-roberta-wwm-ext'
PROJECT_ROOT = '/workspace'
# =====================================================

# -------------------------- GAT Model Section --------------------------
# Part-of-speech tags used by the LTP toolkit for Chinese.
LTP_POS_LABELS = [
    'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j', 'k', 'm', 'n', 'nd',
    'nh', 'ni', 'nl', 'ns', 'nt', 'nz', 'o', 'p', 'q', 'r', 'u', 'v',
    'wp', 'ws', 'x', 'z'
]

class GATDataset(torch.utils.data.Dataset):
    """Dataset for GAT that converts text titles into graph structures."""
    def __init__(self, csv_path):
        print("--- Initializing GAT Dataset ---")
        df = pd.read_csv(csv_path, header=None, usecols=[1], names=['title'], on_bad_lines='warn', encoding='utf-8')
        self.titles = df['title'].dropna().astype(str).tolist()
        self.ltp = LTP("LTP/small")
        self.pos_vocab = {tag: i for i, tag in enumerate(LTP_POS_LABELS)}
        self.pos_embedding = nn.Embedding(len(self.pos_vocab), TEACHER_CONFIG_GAT["in_channels"])

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        title = self.titles[idx]
        # Use LTP to get part-of-speech tags and dependency parsing
        output = self.ltp.pipeline([title], tasks=["cws", "pos", "dep"])
        pos_tags = output.pos[0]
        deps_dict = output.dep[0]
        heads = deps_dict['head']
        # Create node features from POS tag embeddings
        pos_ids = torch.tensor([self.pos_vocab.get(tag, 0) for tag in pos_tags], dtype=torch.long)
        node_features = self.pos_embedding(pos_ids)
        # Create edge index from dependency tree
        edge_sources, edge_targets = [], []
        for i in range(len(heads)):
            head_idx = heads[i] - 1
            if head_idx >= 0:
                edge_sources.append(head_idx)
                edge_targets.append(i)
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long) if edge_sources else torch.tensor([[0], [0]], dtype=torch.long)
        return Data(x=node_features, edge_index=edge_index)

def collate_fn_gat(batch):
    """Custom collate function to filter out None items from a batch."""
    return [item for item in batch if item is not None]

class GATModel(nn.Module):
    """Graph Attention Network model for processing syntactic dependency graphs."""
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, heads):
        super().__init__()
        self.convs = nn.ModuleList([GATConv(in_channels, hidden_channels, heads=heads)])
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
    """Performs knowledge distillation for the GAT model."""
    print(f"--- Starting GAT Distillation (Device: {DEVICE}) ---")
    teacher_model = GATModel(**TEACHER_CONFIG_GAT).to(DEVICE)
    student_model = GATModel(**STUDENT_CONFIG_GAT).to(DEVICE)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False # Freeze teacher model
    dataset = GATDataset(csv_path=CSV_PATH)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_gat)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss() # Use MSE to match student and teacher outputs
    # Projection layer to match student's output dimension to the teacher's
    projection = nn.Linear(STUDENT_CONFIG_GAT['out_channels'], TEACHER_CONFIG_GAT['out_channels']).to(DEVICE)
    optimizer.add_param_group({'params': projection.parameters(), 'lr': LEARNING_RATE})
    student_model.train()
    projection.train()
    for epoch in range(NUM_EPOCHS):
        pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        for batch_data in pbar:
            if not batch_data: continue
            batch_graph = next(iter(DataLoader(batch_data, batch_size=len(batch_data)))).to(DEVICE)
            with torch.no_grad():
                teacher_embeds = teacher_model(batch_graph)
            student_embeds = student_model(batch_graph)
            student_embeds_projected = projection(student_embeds)
            loss = loss_fn(student_embeds_projected, teacher_embeds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})
    torch.save(student_model.state_dict(), OUTPUT_FILE_GAT)
    print(f"\n--- Lightweight GAT model saved to: '{OUTPUT_FILE_GAT}' ---")


# -------------------------- Vision Model Section --------------------------
class VisionDataset(Dataset):
    """Dataset for loading images specified in the CSV file."""
    def __init__(self, csv_path, project_root):
        self.project_root = project_root
        df = pd.read_csv(csv_path, header=None, usecols=[3], names=['image_path'], encoding='utf-8', on_bad_lines='skip')
        self.image_paths = df['image_path'].dropna().astype(str).tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path_from_csv = self.image_paths[idx].strip()
        full_path = os.path.join(self.project_root, path_from_csv)
        try:
            if os.path.exists(full_path):
                image = Image.open(full_path).convert("RGB")
                return image
        except (FileNotFoundError, OSError, UnidentifiedImageError):
            # Gracefully handle corrupted or missing image files
            pass
        return None

def collate_fn_vision(batch):
    """Custom collate function to filter out None (failed to load) images."""
    batch = [img for img in batch if img is not None]
    if not batch: return None
    return batch

class VisionEncoderWithProjection(nn.Module):
    """Wrapper for the CLIP vision model and its projection layer."""
    def __init__(self, vision_model, projection):
        super().__init__()
        self.vision_model = vision_model
        self.visual_projection = projection

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        return self.visual_projection(outputs.pooler_output)

def distill_vision():
    """Performs knowledge distillation for the Vision (CLIP) model."""
    print(f"--- Starting Vision Model Distillation (Device: {DEVICE}) ---")
    processor = CLIPProcessor.from_pretrained(TEACHER_MODEL_PATH_CONTENT)
    base_clip = CLIPModel.from_pretrained(TEACHER_MODEL_PATH_CONTENT)
    teacher_model = VisionEncoderWithProjection(base_clip.vision_model, base_clip.visual_projection).to(DEVICE)
    teacher_model.eval()
    for param in teacher_model.parameters(): param.requires_grad = False
    
    # Create a smaller student model from a modified config
    student_config = CLIPVisionConfig.from_pretrained(TEACHER_MODEL_PATH_CONTENT)
    for key, value in STUDENT_CONFIG_VISION.items(): setattr(student_config, key, value)
    student_vision = CLIPVisionModel(config=student_config)
    student_projection = nn.Linear(student_config.hidden_size, base_clip.config.projection_dim, bias=False)
    student_model = VisionEncoderWithProjection(student_vision, student_projection).to(DEVICE)
    
    dataset = VisionDataset(csv_path=CSV_PATH, project_root=PROJECT_ROOT)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_vision)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()
    student_model.train()
    for epoch in range(NUM_EPOCHS):
        pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        for images in pbar:
            if images is None: continue
            inputs = processor(images=images, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                teacher_embeds = teacher_model(pixel_values=inputs['pixel_values'])
            student_embeds = student_model(pixel_values=inputs['pixel_values'])
            loss = loss_fn(student_embeds, teacher_embeds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})
            
    # Save the student model's components for later finetuning
    if not os.path.exists(OUTPUT_DIR_VISION): os.makedirs(OUTPUT_DIR_VISION)
    student_model.vision_model.save_pretrained(OUTPUT_DIR_VISION)
    torch.save(student_model.visual_projection.state_dict(), os.path.join(OUTPUT_DIR_VISION, 'visual_projection.pt'))
    processor.save_pretrained(OUTPUT_DIR_VISION)
    print(f"--- Lightweight vision model components saved to: '{OUTPUT_DIR_VISION}' ---")


# -------------------------- Content (Text) Model Section --------------------------
class ContentDataset(Dataset):
    """Dataset for loading text content from the CSV file."""
    def __init__(self, csv_path, tokenizer):
        self.tokenizer = tokenizer
        df = pd.read_csv(csv_path, header=None, usecols=[2], names=['content'], encoding='utf-8')
        self.texts = df['content'].dropna().astype(str).tolist()

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        return self.tokenizer(self.texts[idx], return_tensors='pt', max_length=256, padding='max_length', truncation=True)

class TextEncoderWithProjection(nn.Module):
    """Wrapper for the BERT text model and a projection layer."""
    def __init__(self, text_model, projection):
        super().__init__()
        self.text_model = text_model
        self.text_projection = projection

    def forward(self, input_ids, attention_mask):
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        return self.text_projection(outputs.pooler_output)

def distill_content():
    """Performs knowledge distillation for the Content (BERT) model."""
    print(f"--- Starting Content Model Distillation (Device: {DEVICE}) ---")
    tokenizer = BertTokenizer.from_pretrained(TEACHER_MODEL_PATH_CONTENT)
    teacher_bert = BertModel.from_pretrained(TEACHER_MODEL_PATH_CONTENT)
    base_clip = CLIPModel.from_pretrained(BASE_CLIP_MODEL_NAME)
    # The teacher projects BERT embeddings into the same space as CLIP embeddings
    teacher_projection = nn.Linear(teacher_bert.config.hidden_size, base_clip.config.projection_dim, bias=False)
    teacher_model = TextEncoderWithProjection(teacher_bert, teacher_projection).to(DEVICE)
    teacher_model.eval()
    for param in teacher_model.parameters(): param.requires_grad = False

    # Create a smaller student BERT model from a modified config
    student_config = BertConfig.from_pretrained(TEACHER_MODEL_PATH_CONTENT)
    for key, value in STUDENT_CONFIG_CONTENT.items(): setattr(student_config, key, value)
    student_bert = BertModel(config=student_config)
    student_projection = nn.Linear(student_config.hidden_size, base_clip.config.projection_dim, bias=False)
    student_model = TextEncoderWithProjection(student_bert, student_projection).to(DEVICE)

    dataset = ContentDataset(csv_path=CSV_PATH, tokenizer=tokenizer)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    student_model.train()
    for epoch in range(NUM_EPOCHS):
        pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        for batch in pbar:
            input_ids = batch['input_ids'].squeeze(1).to(DEVICE)
            attention_mask = batch['attention_mask'].squeeze(1).to(DEVICE)
            with torch.no_grad():
                teacher_embeds = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            student_embeds = student_model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(student_embeds, teacher_embeds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})

    # Save the student model's components for later finetuning
    if not os.path.exists(OUTPUT_DIR_CONTENT): os.makedirs(OUTPUT_DIR_CONTENT)
    student_model.text_model.save_pretrained(OUTPUT_DIR_CONTENT)
    torch.save(student_model.text_projection.state_dict(), os.path.join(OUTPUT_DIR_CONTENT, 'text_projection.pt'))
    tokenizer.save_pretrained(OUTPUT_DIR_CONTENT)
    print(f"--- Lightweight content model saved to: '{OUTPUT_DIR_CONTENT}' ---")


if __name__ == "__main__":
    # Execute the distillation process for each modality sequentially.
    distill_gat()
    distill_vision()
    distill_content()