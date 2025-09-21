# 文件名: preprocess_data.py
import os
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from ltp import LTP
from tqdm import tqdm

# ================== 配置 ==================
# 原始数据路径
CSV_PATH = './datasets/TextClassification/toutiao/toutiao_622.csv'
# 预处理后数据的保存目录
SAVE_DIR = './processed_data'
# GAT模型输入维度，需要和主脚本GAT_CONFIG["in_channels"]一致
EMBEDDING_DIM = 100 
# LTP 词性标签列表，需要和主脚本一致
LTP_POS_LABELS = ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j', 'k', 'm', 'n', 'nd', 'nh', 'ni', 'nl', 'ns', 'nt', 'nz', 'o', 'p', 'q', 'r', 'u', 'v', 'wp', 'ws', 'x', 'z']
# ==========================================

def preprocess():
    print("--- 开始数据预处理 ---")
    
    # 1. 创建保存目录
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"创建目录: {SAVE_DIR}")

    # 2. 加载LTP模型和数据
    print("正在加载LTP模型...")
    ltp = LTP("LTP/small")
    print("LTP模型加载完成。")
    
    df = pd.read_csv(CSV_PATH, header=None, names=['label', 'title', 'content', 'image_path'])
    
    # 3. 初始化词性嵌入层和词汇表
    pos_vocab = {tag: i for i, tag in enumerate(LTP_POS_LABELS)}
    pos_embedding = nn.Embedding(len(pos_vocab), EMBEDDING_DIM)

    # 4. 遍历数据并处理
    processed_count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="正在处理数据"):
        title = str(row['title'])
        
        try:
            # 运行LTP pipeline
            output = ltp.pipeline([title], tasks=["cws", "pos", "dep"])
            pos_tags = output.pos[0]
            deps_dict = output.dep[0]

            if not pos_tags or not deps_dict['head']:
                print(f"警告: 第 {idx} 行标题 '{title}' 无法处理，跳过。")
                continue
            
            # 构建图数据
            pos_ids = torch.tensor([pos_vocab.get(tag, 0) for tag in pos_tags], dtype=torch.long)
            node_features = pos_embedding(pos_ids).detach().to(torch.float32) # 直接生成特征向量

            edge_sources, edge_targets = [], []
            for i in range(len(deps_dict['head'])):
                head_idx = deps_dict['head'][i] - 1
                if head_idx >= 0:
                    edge_sources.append(head_idx)
                    edge_targets.append(i)
            
            # 确保即使没有边，也有一个有效的图结构
            if not edge_sources:
                 edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)

            graph_data = Data(x=node_features, edge_index=edge_index)

            # 保存图数据
            save_path = os.path.join(SAVE_DIR, f'data_{idx}.pt')
            torch.save(graph_data, save_path)
            processed_count += 1

        except Exception as e:
            print(f"错误: 处理第 {idx} 行时发生异常: {e}，跳过。")
            continue
            
    print(f"\n--- 预处理完成 ---")
    print(f"总计 {len(df)} 条数据，成功处理并保存了 {processed_count} 条。")
    print(f"预处理文件保存在: '{SAVE_DIR}'")

if __name__ == "__main__":
    preprocess()