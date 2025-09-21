# =====================================================================================
# Finetuning Script for Synergistically Distilled Models
#
# This script loads the multi-modal components distilled by 'distill_syc.py'
# and finetunes them end-to-end on the downstream classification task.
#
# A key feature of this workflow is the on-the-fly processing of titles into
# graph structures within the Dataset class, removing the need for a separate
# preprocessing step.
#
# Author: [Haoqian Song, Haoran Yin, Fuwen Zhao]
# Date: September 18, 2025
# =====================================================================================

import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

from transformers import BertModel, CLIPModel, CLIPProcessor, BertTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

try:
    from torch_geometric.data import Batch
    from torch_geometric.nn import GATConv, global_mean_pool
    from torch_geometric.data import Data
except ImportError:
    print("❌ Error: torch_geometric library not found or not installed correctly.")
    exit()

from ltp import LTP

# ================== Configuration ==================
CSV_PATH = './datasets/TextClassification/toutiao/toutiao_622.csv'
PROJECT_ROOT = '/workspace'
DISTILLED_MODELS_DIR = './syc_lightweight_models'      # Output directory from the distillation script

# Original model paths (for loading tokenizers and model architecture)
ORIGINAL_CONTENT_MODEL_PATH = './model/chinese-roberta-wwm-ext'
ORIGINAL_VISION_MODEL_PATH = './model/clip-vit-base-patch32'

# Paths to the distilled models to be loaded
CONTENT_MODEL_PATH = DISTILLED_MODELS_DIR
VISION_MODEL_WEIGHTS_PATH = os.path.join(DISTILLED_MODELS_DIR, 'student_vision_model.pth')
GAT_MODEL_WEIGHTS_PATH = os.path.join(DISTILLED_MODELS_DIR, 'lightweight_gat_model.pth')

# Model and Training Configuration
GAT_CONFIG = {"in_channels": 100, "hidden_channels": 128, "num_layers": 2, "out_channels": 128, "heads": 4}
NUM_CLASSES = 3
BATCH_SIZE = 16   # Reduced batch size to prevent potential memory issues
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FINETUNED_MODEL_SAVE_PATH = "./final_distilled_model_syc.pth"

# GAT-related configuration
LTP_POS_LABELS = [
    'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j', 'k', 'm', 'n', 'nd',
    'nh', 'ni', 'nl', 'ns', 'nt', 'nz', 'o', 'p', 'q', 'r', 'u', 'v',
    'wp', 'ws', 'x', 'z'
]
# ===================================================

# ================== Model Definitions ==================

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

class MultiModalDataset(Dataset):
    """Multi-modal dataset: Text + Image + On-the-fly Graph Creation"""
    def __init__(self, csv_path, project_root):
        self.project_root = project_root
        self.df = pd.read_csv(csv_path, header=None, names=['label', 'title', 'content', 'image_path'])
        
        print("Loading LTP model for on-the-fly graph processing...")
        self.ltp = LTP("LTP/small")
        print("LTP model loaded.")
        
        self.pos_vocab = {tag: i for i, tag in enumerate(LTP_POS_LABELS)}
        # Create a static embedding matrix to avoid gradient issues during data loading
        self.pos_embeddings = torch.randn(len(self.pos_vocab), GAT_CONFIG["in_channels"])

    def __len__(self):
        return len(self.df)

    def _process_text_to_graph(self, title):
        """Converts a title string into a graph Data object."""
        if not title or pd.isna(title): return None
        try:
            output = self.ltp.pipeline([str(title)], tasks=["cws", "pos", "dep"])
        except Exception:
            return None
            
        pos_tags = output.pos[0]
        deps_dict = output.dep[0]
        heads = deps_dict['head']
        
        if not pos_tags or not heads: return None

        # Generate node features using the pre-computed embedding matrix
        pos_ids = [self.pos_vocab.get(tag, 0) for tag in pos_tags]
        node_features = torch.stack([self.pos_embeddings[pid] for pid in pos_ids]).detach()
        
        edge_sources, edge_targets = [], []
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

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row['label'])
        title = str(row['title']) if pd.notna(row['title']) else ""
        content = str(row['content']) if pd.notna(row['content']) else ""
        image_path = str(row['image_path']) if pd.notna(row['image_path']) else ""

        # Process image (default to a white image on failure)
        image = Image.new('RGB', (224, 224), color='white')
        if image_path and image_path.lower() != 'nan':
            try:
                full_path = os.path.join(self.project_root, image_path)
                if os.path.exists(full_path):
                    image = Image.open(full_path).convert("RGB")
            except Exception:
                pass

        # Process graph structure
        graph_data = self._process_text_to_graph(title)
        if graph_data is None:
            # Create a fallback single-node graph
            node_features = torch.zeros(1, GAT_CONFIG["in_channels"]).detach()
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            graph_data = Data(x=node_features, edge_index=edge_index)

        # Ensure graph data tensors are detached from the computation graph
        if hasattr(graph_data, 'x') and graph_data.x is not None:
            graph_data.x = graph_data.x.detach()
        if hasattr(graph_data, 'edge_index') and graph_data.edge_index is not None:
            graph_data.edge_index = graph_data.edge_index.detach()

        return content, image, label, graph_data

def collate_fn_multimodal(batch):
    """Custom collate function for the multi-modal dataset."""
    contents, images, labels, graph_data_list = zip(*batch)
    
    # Detach graph data tensors again as a safeguard
    clean_graph_list = []
    for graph_data in graph_data_list:
        clean_graph = Data(
            x=graph_data.x.detach() if graph_data.x is not None else None,
            edge_index=graph_data.edge_index.detach() if graph_data.edge_index is not None else None
        )
        clean_graph_list.append(clean_graph)
    
    batched_graph = Batch.from_data_list(clean_graph_list)
    return list(contents), list(images), torch.tensor(list(labels), dtype=torch.long), batched_graph

class DistilledMultiModalModel(nn.Module):
    """The final multi-modal classification model built from distilled components."""
    def __init__(self, content_model_path, vision_weights_path, gat_weights_path, num_classes):
        super().__init__()
        
        print("Loading distilled content model...")
        try:
            self.content_model = BertModel.from_pretrained(content_model_path)
            print("✓ Content model loaded successfully from distilled directory.")
        except Exception as e:
            print(f"Failed to load from distilled directory: {e}. Attempting to load from original path...")
            self.content_model = BertModel.from_pretrained(ORIGINAL_CONTENT_MODEL_PATH)
            print("✓ Content model loaded successfully from original path.")
        
        print("Loading distilled vision model...")
        clip_model = CLIPModel.from_pretrained(ORIGINAL_VISION_MODEL_PATH)
        self.vision_model = clip_model.vision_model
        if os.path.exists(vision_weights_path):
            self.vision_model.load_state_dict(torch.load(vision_weights_path, map_location='cpu'))
            print("✓ Distilled vision model weights loaded successfully.")
        else:
            print(f"⚠️ Warning: Vision model weights not found at: {vision_weights_path}. Using original pre-trained weights.")
        
        print("Loading distilled GAT model...")
        self.gat_model = GATModel(**GAT_CONFIG)
        if os.path.exists(gat_weights_path):
            self.gat_model.load_state_dict(torch.load(gat_weights_path, map_location='cpu'))
            print("✓ Distilled GAT model weights loaded successfully.")
        else:
            print(f"⚠️ Warning: GAT model weights not found at: {gat_weights_path}. Using randomly initialized weights.")
        
        # Fusion and Classifier Layers
        content_dim = self.content_model.config.hidden_size
        vision_dim = self.vision_model.config.hidden_size
        gat_dim = GAT_CONFIG["out_channels"]
        combined_dim = content_dim + vision_dim + gat_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        print(f"Model initialized. Feature dimensions: Content({content_dim}) + Vision({vision_dim}) + GAT({gat_dim}) = {combined_dim}")

    def forward(self, content_inputs, pixel_values, graph_batch):
        content_features = self.content_model(**content_inputs).pooler_output
        vision_features = self.vision_model(pixel_values=pixel_values).pooler_output
        gat_features = self.gat_model(graph_batch)
        combined_features = torch.cat([content_features, vision_features, gat_features], dim=1)
        logits = self.classifier(combined_features)
        return logits

def main():
    print(f"--- Starting Distilled Model Finetuning (Device: {DEVICE}) ---")
    
    print("Checking for model files...")
    if not os.path.exists(DISTILLED_MODELS_DIR):
        print(f"❌ Error: Distilled models directory not found: {DISTILLED_MODELS_DIR}")
        return
    print(f"Contents of distilled models directory: {os.listdir(DISTILLED_MODELS_DIR)}")
    
    print("Loading tokenizers and processors...")
    try:
        content_tokenizer = BertTokenizer.from_pretrained(ORIGINAL_CONTENT_MODEL_PATH)
        print("✓ BERT tokenizer loaded successfully.")
    except Exception as e:
        print(f"Failed to load tokenizer: {e}"); return
    
    try:
        if os.path.exists(os.path.join(DISTILLED_MODELS_DIR, 'preprocessor_config.json')):
            vision_processor = CLIPProcessor.from_pretrained(DISTILLED_MODELS_DIR)
            print("✓ Vision processor loaded from distilled directory.")
        else:
            vision_processor = CLIPProcessor.from_pretrained(ORIGINAL_VISION_MODEL_PATH)
            print("✓ Vision processor loaded from original path.")
    except Exception as e:
        print(f"Failed to load vision processor: {e}. Falling back to original CLIP processor...")
        vision_processor = CLIPProcessor.from_pretrained(ORIGINAL_VISION_MODEL_PATH)
    
    print("Creating dataset...")
    dataset = MultiModalDataset(CSV_PATH, PROJECT_ROOT)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = TorchDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_multimodal, num_workers=0)
    val_loader = TorchDataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_multimodal, num_workers=0)
    print(f"Dataset sizes: Training={len(train_dataset)}, Validation={len(val_dataset)}")
    
    print("Creating model...")
    model = DistilledMultiModalModel(CONTENT_MODEL_PATH, VISION_MODEL_WEIGHTS_PATH, GAT_MODEL_WEIGHTS_PATH, NUM_CLASSES).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Calculate class weights to handle class imbalance
    labels = dataset.df['label'].values
    class_counts = pd.Series(labels).value_counts().sort_index()
    if len(class_counts) > 1:
        weights = len(labels) / (len(class_counts) * class_counts.values)
        class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        print("Weighted loss function enabled to handle class imbalance.")
    else:
        loss_fn = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{NUM_EPOCHS}")
        
        for batch in train_pbar:
            try:
                contents, images, labels, graph_batch = batch
                content_inputs = content_tokenizer(contents, return_tensors='pt', max_length=256, padding='max_length', truncation=True).to(DEVICE)
                pixel_values = vision_processor(images=images, return_tensors='pt')['pixel_values'].to(DEVICE)
                graph_batch = graph_batch.to(DEVICE)
                labels = labels.to(DEVICE)
                
                logits = model(content_inputs, pixel_values, graph_batch)
                loss = loss_fn(logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix({"loss": loss.item()})
            except Exception as e:
                print(f"Error during training batch: {e}"); continue
        
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        model.eval()
        val_preds, val_labels, val_loss = [], [], 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validating")
            for batch in val_pbar:
                try:
                    contents, images, labels, graph_batch = batch
                    content_inputs = content_tokenizer(contents, return_tensors='pt', max_length=256, padding='max_length', truncation=True).to(DEVICE)
                    pixel_values = vision_processor(images=images, return_tensors='pt')['pixel_values'].to(DEVICE)
                    graph_batch = graph_batch.to(DEVICE)
                    labels = labels.to(DEVICE)
                    
                    logits = model(content_inputs, pixel_values, graph_batch)
                    loss = loss_fn(logits, labels)
                    val_loss += loss.item()
                    
                    preds = torch.argmax(logits, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
                except Exception as e:
                    print(f"Error during validation batch: {e}"); continue
        
        if len(val_preds) > 0:
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
            
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
            print(f"  Training Loss: {avg_train_loss:.4f}")
            print(f"  Validation Loss: {avg_val_loss:.4f}")
            print(f"  Validation Accuracy: {val_acc:.4f}")
            print(f"  Validation F1 (macro): {val_f1:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), FINETUNED_MODEL_SAVE_PATH)
                print(f"  ✓ Best model saved (Validation Accuracy: {best_val_acc:.4f})")
        else:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation failed, no valid predictions.")
        print("-" * 50)
    
    print(f"\n--- Finetuning Complete! Best Validation Accuracy: {best_val_acc:.4f} ---")
    print(f"Final model saved to: {FINETUNED_MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()