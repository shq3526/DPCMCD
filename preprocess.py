# =====================================================================================
# Data Preprocessing Script for GAT
#
# This script reads a source CSV file, processes the 'title' column for each row
# to generate a syntactic dependency graph using the LTP toolkit, and saves each
# graph as a separate PyTorch Geometric 'Data' object (`.pt` file).
#
# These preprocessed files are required by the 'nosyc' finetuning workflow.
#
# Author: [Haoqian Song, Haoran Yin, Fuwen Zhao]
# Date: September 18, 2025
# =====================================================================================

import os
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from ltp import LTP
from tqdm import tqdm

# ================== Configuration ==================
CSV_PATH = './datasets/TextClassification/toutiao/toutiao_622.csv'
SAVE_DIR = './processed_data'
EMBEDDING_DIM = 100
LTP_POS_LABELS = ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j', 'k', 'm', 'n', 'nd', 'nh', 'ni', 'nl', 'ns', 'nt', 'nz', 'o', 'p', 'q', 'r', 'u', 'v', 'wp', 'ws', 'x', 'z']
# ===================================================

def preprocess():
    """Main function to perform the data preprocessing."""
    print("--- Starting Data Preprocessing ---")

    # 1. Create the save directory if it doesn't exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created directory: {SAVE_DIR}")

    # 2. Load LTP model and the dataset
    print("Loading LTP model...")
    ltp = LTP("LTP/small")
    print("LTP model loaded successfully.")

    df = pd.read_csv(CSV_PATH, header=None, names=['label', 'title', 'content', 'image_path'])

    # 3. Initialize the part-of-speech embedding layer and vocabulary
    pos_vocab = {tag: i for i, tag in enumerate(LTP_POS_LABELS)}
    pos_embedding = nn.Embedding(len(pos_vocab), EMBEDDING_DIM)

    # 4. Iterate through the data, process each entry, and save
    processed_count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing data"):
        title = str(row['title'])

        try:
            # Run the LTP pipeline to get POS and dependency info
            output = ltp.pipeline([title], tasks=["cws", "pos", "dep"])
            pos_tags = output.pos[0]
            deps_dict = output.dep[0]

            if not pos_tags or not deps_dict['head']:
                print(f"Warning: Title for row {idx} '{title}' could not be processed, skipping.")
                continue

            # Build the graph data object
            pos_ids = torch.tensor([pos_vocab.get(tag, 0) for tag in pos_tags], dtype=torch.long)
            node_features = pos_embedding(pos_ids).detach().to(torch.float32) # Generate feature vectors

            edge_sources, edge_targets = [], []
            for i in range(len(deps_dict['head'])):
                head_idx = deps_dict['head'][i] - 1
                if head_idx >= 0:
                    edge_sources.append(head_idx)
                    edge_targets.append(i)
            
            # Ensure a valid graph structure, even with no edges
            if not edge_sources:
                 edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)

            graph_data = Data(x=node_features, edge_index=edge_index)

            # Save the graph data object to a file
            save_path = os.path.join(SAVE_DIR, f'data_{idx}.pt')
            torch.save(graph_data, save_path)
            processed_count += 1

        except Exception as e:
            print(f"Error: An exception occurred while processing row {idx}: {e}, skipping.")
            continue
            
    print(f"\n--- Preprocessing Complete ---")
    print(f"A total of {len(df)} records were found. {processed_count} were successfully processed and saved.")
    print(f"Preprocessed files are saved in: '{SAVE_DIR}'")

if __name__ == "__main__":
    preprocess()
