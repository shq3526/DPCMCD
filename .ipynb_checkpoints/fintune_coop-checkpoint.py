# 文件名: finetune_distilled_models.py
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
    print("❌ 错误: torch_geometric 库未找到或未正确安装。")
    exit()

from ltp import LTP

# ================== 配置 ==================
CSV_PATH = './datasets/TextClassification/toutiao/toutiao_622.csv'
PROJECT_ROOT = '/workspace'
DISTILLED_MODELS_DIR = './syc_lightweight_models'      # 蒸馏程序的输出目录

# 原始模型路径（用于加载tokenizer和模型结构）
ORIGINAL_CONTENT_MODEL_PATH = './model/chinese-roberta-wwm-ext'  # 原始BERT模型路径
ORIGINAL_VISION_MODEL_PATH = './model/clip-vit-base-patch32'     # 原始CLIP模型路径

# 蒸馏模型路径
CONTENT_MODEL_PATH = DISTILLED_MODELS_DIR                  # BERT模型通过save_pretrained保存
VISION_MODEL_WEIGHTS_PATH = os.path.join(DISTILLED_MODELS_DIR, 'student_vision_model.pth')
GAT_MODEL_WEIGHTS_PATH = os.path.join(DISTILLED_MODELS_DIR, 'lightweight_gat_model.pth')

# 模型配置
GAT_CONFIG = {"in_channels": 100, "hidden_channels": 128, "num_layers": 2, "out_channels": 128, "heads": 4}
NUM_CLASSES = 3  # 根据你的数据集调整类别数
BATCH_SIZE = 16   # 减小batch size避免内存问题
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FINETUNED_MODEL_SAVE_PATH = "./final_distilled_multimodal_model.pth"

# GAT相关配置
LTP_POS_LABELS = [
    'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j', 'k', 'm', 'n', 'nd',
    'nh', 'ni', 'nl', 'ns', 'nt', 'nz', 'o', 'p', 'q', 'r', 'u', 'v',
    'wp', 'ws', 'x', 'z'
]

# ================== 模型定义 ==================

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

class MultiModalDataset(Dataset):
    """多模态数据集：文本+图像+图结构"""
    def __init__(self, csv_path, project_root):
        self.project_root = project_root
        self.df = pd.read_csv(csv_path, header=None, names=['label', 'title', 'content', 'image_path'])
        
        # 初始化LTP和词性标注
        print("正在加载LTP模型...")
        self.ltp = LTP("LTP/small")
        print("LTP模型加载完毕。")
        
        self.pos_vocab = {tag: i for i, tag in enumerate(LTP_POS_LABELS)}
        
        # 创建位置编码矩阵（不使用nn.Embedding避免梯度问题）
        self.pos_embeddings = torch.randn(len(self.pos_vocab), GAT_CONFIG["in_channels"])

    def __len__(self):
        return len(self.df)

    def _process_text_to_graph(self, title):
        """将标题文本转换为图数据"""
        if not title or pd.isna(title):
            return None
            
        try:
            output = self.ltp.pipeline([str(title)], tasks=["cws", "pos", "dep"])
        except Exception:
            return None
            
        pos_tags = output.pos[0]
        deps_dict = output.dep[0]
        heads = deps_dict['head']
        
        if not pos_tags or not heads:
            return None

        # 生成节点特征（直接使用预计算的embedding，避免梯度）
        pos_ids = [self.pos_vocab.get(tag, 0) for tag in pos_tags]
        node_features = torch.stack([self.pos_embeddings[pid] for pid in pos_ids]).detach()
        
        # 生成边
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

        # 处理图像
        image = Image.new('RGB', (224, 224), color='white')  # 默认白色图像
        if image_path and image_path.lower() != 'nan':
            try:
                full_path = os.path.join(self.project_root, image_path)
                if os.path.exists(full_path):
                    image = Image.open(full_path).convert("RGB")
            except Exception:
                pass

        # 处理图结构
        graph_data = self._process_text_to_graph(title)
        if graph_data is None:
            # 创建一个单节点图作为后备
            node_features = torch.zeros(1, GAT_CONFIG["in_channels"]).detach()
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            graph_data = Data(x=node_features, edge_index=edge_index)

        # 确保图数据没有梯度
        if hasattr(graph_data, 'x') and graph_data.x is not None:
            graph_data.x = graph_data.x.detach()
        if hasattr(graph_data, 'edge_index') and graph_data.edge_index is not None:
            graph_data.edge_index = graph_data.edge_index.detach()

        return content, image, label, graph_data

def collate_fn_multimodal(batch):
    """多模态数据的批处理函数"""
    contents, images, labels, graph_data_list = zip(*batch)
    
    # 确保所有图数据都没有梯度
    clean_graph_list = []
    for graph_data in graph_data_list:
        clean_graph = Data(
            x=graph_data.x.detach() if graph_data.x is not None else None,
            edge_index=graph_data.edge_index.detach() if graph_data.edge_index is not None else None
        )
        clean_graph_list.append(clean_graph)
    
    # 批处理图数据
    batched_graph = Batch.from_data_list(clean_graph_list)
    
    return list(contents), list(images), torch.tensor(list(labels), dtype=torch.long), batched_graph

class DistilledMultiModalModel(nn.Module):
    """基于蒸馏模型的多模态分类模型"""
    def __init__(self, content_model_path, vision_weights_path, gat_weights_path, num_classes):
        super().__init__()
        
        # 加载蒸馏后的内容模型
        print("正在加载蒸馏后的内容模型...")
        try:
            self.content_model = BertModel.from_pretrained(content_model_path)
            print("✓ 从蒸馏目录加载内容模型成功")
        except Exception as e:
            print(f"从蒸馏目录加载失败: {e}")
            print("尝试从原始模型路径加载...")
            self.content_model = BertModel.from_pretrained(ORIGINAL_CONTENT_MODEL_PATH)
            print("✓ 从原始路径加载内容模型成功")
        
        # 加载蒸馏后的视觉模型
        print("正在加载蒸馏后的视觉模型...")
        # 首先加载原始CLIP模型结构
        clip_model = CLIPModel.from_pretrained(ORIGINAL_VISION_MODEL_PATH)
        self.vision_model = clip_model.vision_model
        # 然后加载蒸馏后的权重
        if os.path.exists(vision_weights_path):
            self.vision_model.load_state_dict(torch.load(vision_weights_path, map_location='cpu'))
            print("✓ 加载蒸馏后的视觉模型权重成功")
        else:
            print(f"⚠️ 警告: 视觉模型权重文件不存在: {vision_weights_path}")
            print("使用原始视觉模型权重")
        
        # 加载蒸馏后的GAT模型
        print("正在加载蒸馏后的GAT模型...")
        self.gat_model = GATModel(**GAT_CONFIG)
        if os.path.exists(gat_weights_path):
            self.gat_model.load_state_dict(torch.load(gat_weights_path, map_location='cpu'))
            print("✓ 加载蒸馏后的GAT模型权重成功")
        else:
            print(f"⚠️ 警告: GAT模型权重文件不存在: {gat_weights_path}")
            print("使用随机初始化的GAT模型权重")
        
        # 特征融合和分类层
        content_dim = self.content_model.config.hidden_size  # 768 for BERT
        vision_dim = self.vision_model.config.hidden_size    # CLIP vision hidden size
        gat_dim = GAT_CONFIG["out_channels"]                 # 128
        
        combined_dim = content_dim + vision_dim + gat_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        print(f"模型初始化完成。特征维度: Content({content_dim}) + Vision({vision_dim}) + GAT({gat_dim}) = {combined_dim}")

    def forward(self, content_inputs, pixel_values, graph_batch):
        # 提取内容特征
        content_outputs = self.content_model(**content_inputs)
        content_features = content_outputs.pooler_output  # [batch_size, hidden_size]
        
        # 提取视觉特征
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        # vision_model返回的是BaseModelOutputWithPooling，取last_hidden_state的全局平均池化
        vision_features = vision_outputs.pooler_output    # [batch_size, hidden_size]
        
        # 提取图特征
        gat_features = self.gat_model(graph_batch)        # [batch_size, out_channels]
        
        # 特征融合
        combined_features = torch.cat([content_features, vision_features, gat_features], dim=1)
        
        # 分类
        logits = self.classifier(combined_features)
        return logits

def main():
    print(f"--- 开始蒸馏模型微调 (设备: {DEVICE}) ---")
    
    # 检查蒸馏模型文件是否存在
    print("检查模型文件...")
    if not os.path.exists(DISTILLED_MODELS_DIR):
        print(f"❌ 错误: 蒸馏模型目录不存在: {DISTILLED_MODELS_DIR}")
        return
        
    print(f"蒸馏模型目录内容: {os.listdir(DISTILLED_MODELS_DIR)}")
    
    # 初始化tokenizer和processor
    print("正在加载tokenizer和processor...")
    try:
        # 直接使用原始BERT tokenizer避免兼容性问题
        content_tokenizer = BertTokenizer.from_pretrained(ORIGINAL_CONTENT_MODEL_PATH)
        print("✓ 加载BERT tokenizer成功")
    except Exception as e:
        print(f"加载tokenizer失败: {e}")
        return
    
    try:
        # 视觉processor从蒸馏目录加载（蒸馏程序保存了processor）
        if os.path.exists(os.path.join(DISTILLED_MODELS_DIR, 'preprocessor_config.json')):
            vision_processor = CLIPProcessor.from_pretrained(DISTILLED_MODELS_DIR)
            print("✓ 从蒸馏目录加载vision processor成功")
        else:
            vision_processor = CLIPProcessor.from_pretrained(ORIGINAL_VISION_MODEL_PATH)
            print("✓ 从原始路径加载vision processor成功")
    except Exception as e:
        print(f"加载vision processor失败: {e}")
        print("使用原始CLIP processor...")
        vision_processor = CLIPProcessor.from_pretrained(ORIGINAL_VISION_MODEL_PATH)
    
    # 创建数据集和数据加载器
    print("正在创建数据集...")
    dataset = MultiModalDataset(CSV_PATH, PROJECT_ROOT)
    
    # 数据集划分（简单的训练/验证划分）
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = TorchDataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn_multimodal, num_workers=0  # 设为0避免多进程问题
    )
    val_loader = TorchDataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn_multimodal, num_workers=0
    )
    
    print(f"数据集大小: 训练集 {len(train_dataset)}, 验证集 {len(val_dataset)}")
    
    # 创建模型
    print("正在创建模型...")
    model = DistilledMultiModalModel(
        CONTENT_MODEL_PATH, 
        VISION_MODEL_WEIGHTS_PATH, 
        GAT_MODEL_WEIGHTS_PATH, 
        NUM_CLASSES
    ).to(DEVICE)
    
    # 优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 计算类别权重（处理类别不平衡）
    labels = dataset.df['label'].values
    class_counts = pd.Series(labels).value_counts().sort_index()
    if len(class_counts) > 1:
        weights = len(labels) / (len(class_counts) * class_counts.values)
        class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        print("已启用加权损失函数处理类别不平衡。")
    else:
        loss_fn = nn.CrossEntropyLoss()
    
    # 训练循环
    best_val_acc = 0
    for epoch in range(NUM_EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"训练 Epoch {epoch + 1}/{NUM_EPOCHS}")
        
        for batch in train_pbar:
            try:
                contents, images, labels, graph_batch = batch
                
                # 准备输入
                content_inputs = content_tokenizer(
                    contents, return_tensors='pt', max_length=256,
                    padding='max_length', truncation=True
                ).to(DEVICE)
                pixel_values = vision_processor(images=images, return_tensors='pt')['pixel_values'].to(DEVICE)
                graph_batch = graph_batch.to(DEVICE)
                labels = labels.to(DEVICE)
                
                # 前向传播
                logits = model(content_inputs, pixel_values, graph_batch)
                loss = loss_fn(logits, labels)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix({"loss": loss.item()})
            except Exception as e:
                print(f"训练批次出错: {e}")
                continue
        
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        # 验证阶段
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="验证中")
            for batch in val_pbar:
                try:
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
                    val_loss += loss.item()
                    
                    preds = torch.argmax(logits, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
                except Exception as e:
                    print(f"验证批次出错: {e}")
                    continue
        
        # 计算指标
        if len(val_preds) > 0:
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
            
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
            print(f"  训练损失: {avg_train_loss:.4f}")
            print(f"  验证损失: {avg_val_loss:.4f}")
            print(f"  验证准确率: {val_acc:.4f}")
            print(f"  验证F1 (macro): {val_f1:.4f}")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), FINETUNED_MODEL_SAVE_PATH)
                print(f"  ✓ 保存最佳模型 (验证准确率: {best_val_acc:.4f})")
        else:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - 验证失败，没有有效预测")
        
        print("-" * 50)
    
    print(f"\n--- 微调完成！最佳验证准确率: {best_val_acc:.4f} ---")
    print(f"最终模型已保存至: {FINETUNED_MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()