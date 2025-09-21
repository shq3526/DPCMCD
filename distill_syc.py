# =====================================================================================
# Synergistic Knowledge Distillation for Multi-Modal Components
#
# This script performs knowledge distillation for three modalities: Vision, Content, and GAT.
# Unlike the 'nosyc' script, this 'syc' (synergistic) approach uses student models
# with the *same architecture* as the teacher models. This is a form of self-distillation
# aimed at improving the robustness of the original architecture.
#
# Author: [Haoqian Song, Haoran Yin, Fuwen Zhao]
# Date: September 18, 2025
# =====================================================================================

import torch
import os
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor, BertModel, BertTokenizer
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.data import DataLoader as GATDataLoader
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from ltp import LTP
import pandas as pd
from PIL import Image
from torch_geometric.data import Data

# ================== Configuration ==================
CSV_PATH = './datasets/TextClassification/toutiao/toutiao_622.csv'
PROJECT_ROOT = '/workspace'
VISION_TEACHER_MODEL_PATH = './model/clip-vit-base-patch32'
CONTENT_TEACHER_MODEL_PATH = './model/chinese-roberta-wwm-ext'
# Note: For GAT, student and teacher models share the same configuration in this script.
GAT_TEACHER_CONFIG = {"in_channels": 100, "hidden_channels": 128, "num_layers": 2, "out_channels": 128, "heads": 4}
BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "./syc_new_lightweight_models"
# ===================================================

# ================== Vision Model Section ==================

class VisionDataset(torch.utils.data.Dataset):
    """Dataset to load images for the vision model."""
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
            # Return None if image loading fails
            return None

def collate_fn_vision(batch):
    """Filters out None items from the batch before processing."""
    batch = [img for img in batch if img is not None]
    if not batch:  # If the batch is empty after filtering
        return None
    # This assumes all images can be resized to the same dimensions by the processor
    return batch

def distill_vision():
    """Performs self-distillation for the vision model."""
    print(f"--- Starting Vision Model Distillation (Device: {DEVICE}) ---")
    processor = CLIPProcessor.from_pretrained(VISION_TEACHER_MODEL_PATH)
    teacher_model = CLIPModel.from_pretrained(VISION_TEACHER_MODEL_PATH).vision_model.to(DEVICE)
    teacher_model.eval()

    # Create a student model with the same architecture as the teacher
    student_model = CLIPModel.from_pretrained(VISION_TEACHER_MODEL_PATH).vision_model.to(DEVICE)
    student_model.train()

    dataset = VisionDataset(CSV_PATH, PROJECT_ROOT)
    # Note: A standard Torch DataLoader is used here, not a PyG one.
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_vision)

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    for epoch in range(NUM_EPOCHS):
        pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        for images in pbar:
            if images is None: continue
            inputs = processor(images=images, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                # Get the pooler_output from the teacher
                teacher_embeds = teacher_model(**inputs).pooler_output
            # Get the pooler_output from the student
            student_embeds = student_model(**inputs).pooler_output
            loss = loss_fn(student_embeds, teacher_embeds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})

    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # Save the student model's state dictionary manually
    torch.save(student_model.state_dict(), os.path.join(OUTPUT_DIR, "student_vision_model.pth"))
    
    # Save the processor configuration for easy reloading
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Vision model distillation complete. Model saved to: {OUTPUT_DIR}")

# ================== Content Model Section ==================

class ContentDataset(torch.utils.data.Dataset):
    """Dataset to load text content for the BERT model."""
    def __init__(self, csv_path, tokenizer):
        self.tokenizer = tokenizer
        df = pd.read_csv(csv_path, header=None, usecols=[2], names=['content'], encoding='utf-8')
        self.texts = df['content'].dropna().astype(str).tolist()

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        return self.tokenizer(self.texts[idx], return_tensors='pt', max_length=256, padding='max_length', truncation=True)

def distill_content():
    """Performs self-distillation for the content (BERT) model."""
    print(f"--- Starting Content Model Distillation (Device: {DEVICE}) ---")
    tokenizer = BertTokenizer.from_pretrained(CONTENT_TEACHER_MODEL_PATH)
    teacher_model = BertModel.from_pretrained(CONTENT_TEACHER_MODEL_PATH).to(DEVICE)
    teacher_model.eval()

    # Create a student model with the same architecture
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
    # Use save_pretrained for Hugging Face models to save config and weights
    student_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR) # Also save the tokenizer
    print(f"Content model distillation complete. Model saved to: {OUTPUT_DIR}")

# ================== GAT Model Section ==================

LTP_POS_LABELS = [
    'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j', 'k', 'm', 'n', 'nd',
    'nh', 'ni', 'nl', 'ns', 'nt', 'nz', 'o', 'p', 'q', 'r', 'u', 'v',
    'wp', 'ws', 'x', 'z'
]

class GATDataset(torch.utils.data.Dataset):
    """Dataset that converts titles to graph data on-the-fly."""
    def __init__(self, csv_path):
        print("--- Initializing GAT Dataset ---")
        try:
            df = pd.read_csv(csv_path, header=None, usecols=[1], names=['title'], on_bad_lines='warn', encoding='utf-8')
            self.titles = df['title'].dropna().astype(str).tolist()
        except Exception as e:
            print(f"Error reading CSV: {e}")
            self.titles = []
        
        if not self.titles:
            print("No valid titles found in the dataset. Cannot proceed.")
            return

        print("Loading LTP model...")
        self.ltp = LTP("LTP/small")
        print("LTP model loaded.")

        self.pos_vocab = {tag: i for i, tag in enumerate(LTP_POS_LABELS)}
        self.pos_embedding = nn.Embedding(len(self.pos_vocab), GAT_TEACHER_CONFIG["in_channels"])

    def __len__(self):
        return len(self.titles) if hasattr(self, 'titles') else 0

    def __getitem__(self, idx):
        title = self.titles[idx]
        if not title: return None
        try:
            output = self.ltp.pipeline([str(title)], tasks=["cws", "pos", "dep"])
        except Exception:
            return None
            
        pos_tags = output.pos[0]
        deps_dict = output.dep[0]
        heads = deps_dict['head']
        
        if not pos_tags or not heads:
             return None

        pos_ids = torch.tensor([self.pos_vocab.get(tag, 0) for tag in pos_tags], dtype=torch.long)
        node_features = self.pos_embedding(pos_ids)
        
        edge_sources, edge_targets = [], []
        for i in range(len(heads)):
            head_idx = heads[i] - 1
            if head_idx >= 0:
                edge_sources.append(head_idx)
                edge_targets.append(i)

        # Ensure a valid graph structure even if no edges are found
        if not edge_sources:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        else:
            edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
            
        return Data(x=node_features, edge_index=edge_index)

def collate_fn_gat(batch):
    """Custom collate function to filter out None items from a batch."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return Batch.from_data_list(batch)

class GATModel(nn.Module):
    """Graph Attention Network model."""
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
    """Performs self-distillation for the GAT model."""
    print(f"--- Starting GAT Model Distillation (Device: {DEVICE}) ---")
    teacher_model = GATModel(**GAT_TEACHER_CONFIG).to(DEVICE)
    # Student model has the same architecture
    student_model = GATModel(**GAT_TEACHER_CONFIG).to(DEVICE)
    teacher_model.eval()

    dataset = GATDataset(CSV_PATH)
    data_loader = GATDataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_gat)

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    for epoch in range(NUM_EPOCHS):
        pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        for batch_data in pbar:
            if batch_data is None: continue
            batch_data = batch_data.to(DEVICE)
            with torch.no_grad():
                teacher_embeds = teacher_model(batch_data)
            student_embeds = student_model(batch_data)
            loss = loss_fn(student_embeds, teacher_embeds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})

    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    torch.save(student_model.state_dict(), os.path.join(OUTPUT_DIR, "lightweight_gat_model.pth"))
    print(f"GAT model distillation complete. Model saved to: {OUTPUT_DIR}")


# ================== Execute Synergistic Distillation ==================

if __name__ == "__main__":
    distill_vision()
    distill_content()
    distill_gat()