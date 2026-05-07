# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from torch_geometric.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class GraphTransformerLayer(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim=0, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.q_proj = nn.Linear(in_channels, out_channels * heads)
        self.k_proj = nn.Linear(in_channels, out_channels * heads)
        self.v_proj = nn.Linear(in_channels, out_channels * heads)
        self.edge_proj = nn.Linear(edge_dim, heads) if edge_dim > 0 else None
        self.out_proj = nn.Linear(out_channels * heads, out_channels * heads)
        self.dropout_layer = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_channels * heads)

    def forward(self, x, edge_index, edge_attr=None):
        H, C = self.heads, self.out_channels
        Q = self.q_proj(x).view(-1, H, C)
        K = self.k_proj(x).view(-1, H, C)
        V = self.v_proj(x).view(-1, H, C)

        row, col = edge_index
        attn_scores = (Q[row] * K[col]).sum(dim=-1) / np.sqrt(C)
        if edge_attr is not None and self.edge_proj is not None:
            attn_scores += self.edge_proj(edge_attr)

        attn_weights = F.softmax(attn_scores, dim=0)
        attn_weights = self.dropout_layer(attn_weights)

        out_messages = attn_weights.unsqueeze(-1) * V[col]
        out = torch.zeros_like(Q).scatter_add_(
            0,
            row.unsqueeze(-1).unsqueeze(-1).expand_as(out_messages),
            out_messages
        )

        out = out.view(-1, H * C)
        x_proj = x if x.shape[-1] == out.shape[-1] else self.out_proj(x)
        out = self.out_proj(out)
        out = self.norm(out + x_proj)
        return out, attn_weights


class TransformerGraphClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=2, heads=4, dropout=0.3, edge_dim=0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim * heads)
        self.edge_encoder = nn.Linear(edge_dim, edge_dim) if edge_dim > 0 else None
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim * heads, hidden_dim, edge_dim=edge_dim, heads=heads, dropout=dropout)
            for _ in range(4)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim * heads) for _ in range(4)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * heads, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        if self.edge_encoder is not None and edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
        x = self.input_proj(x)
        attn_weights_all_layers = []
        
        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x, attn_weights = layer(x, edge_index, edge_attr)
            x = self.dropout(F.relu(norm(x)))
            x = x + residual
            attn_weights_all_layers.append(attn_weights.cpu())
            
        x = global_mean_pool(x, batch)
        out = self.classifier(x)
        return out, attn_weights_all_layers, edge_index


class ProteinGraphDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.files = sorted([f for f in os.listdir(root_dir) if f.endswith('.pt')])
        print(f"Found {len(self.files)} protein graphs in {root_dir}")

    def len(self):
        return len(self.files)

    def get(self, idx):
        file_path = os.path.join(self.root_dir, self.files[idx])
        data = torch.load(file_path)
        
        # Remove '.pt' extension to ensure consistency with prediction filenames
        data.filename = os.path.splitext(self.files[idx])[0]
        data.node_labels = list(data.seq) if hasattr(data, 'seq') else [str(i) for i in range(data.x.size(0))]
        return data

    def __len__(self):
        return len(self.files)


def visualize_residue_attention(seq, row, col, attn_weights, save_path=None, summary_path=None):
    row = row.cpu().numpy() if hasattr(row, 'cpu') else row
    col = col.cpu().numpy() if hasattr(col, 'cpu') else col
    attn_weights = attn_weights.cpu().numpy() if hasattr(attn_weights, 'cpu') else attn_weights

    if attn_weights.ndim == 2:
        attn_weights = attn_weights.mean(axis=1)

    num_nodes = len(seq)
    node_attention = np.zeros(num_nodes)

    # Aggregate incoming attention weights for each target residue (node)
    for i in range(len(attn_weights)):
        target = col[i]
        if 0 <= target < num_nodes:
            node_attention[target] += attn_weights[i]

    norm_attn = (node_attention - node_attention.min()) / (node_attention.max() - node_attention.min() + 1e-6)

    plt.figure(figsize=(12, 1.2))
    plt.imshow([norm_attn], cmap='viridis', aspect='auto')
    xticks = list(range(0, num_nodes, 50))
    if (num_nodes - 1) not in xticks:
        xticks.append(num_nodes - 1)
    plt.xticks(ticks=xticks, labels=[str(x) for x in xticks], fontsize=8)
    plt.yticks([])
    plt.colorbar(label="Attention")
    plt.title("Residue-level Attention Heatmap")

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved heatmap to {save_path}")
    else:
        plt.show()

    topk = 5
    top_indices = np.argsort(node_attention)[-topk:][::-1]

    summary_lines = []
    summary_lines.append(f"\nAttention summary for {os.path.basename(save_path)}:")
    for idx in top_indices:
        aa = seq[idx]
        attn = node_attention[idx]
        summary_lines.append(f"Position {idx}: Amino Acid {aa}, Attention {attn:.4f}")

    if summary_path:
        with open(summary_path, 'a') as f:
            f.write("\n".join(summary_lines) + "\n")


def parse_cd_hit_clusters(clstr_file):
    cluster_map = {}
    cluster_id = -1
    with open(clstr_file, 'r') as f:
        for line in f:
            if line.startswith('>Cluster'):
                cluster_id += 1
            else:
                parts = line.strip().split('>')
                if len(parts) > 1:
                    seq_name = parts[1].split('...')[0]
                    cluster_map[seq_name] = cluster_id
    return cluster_map


def cluster_select_top2_per_score_interval_all(pred_csv_path, clstr_file):
    pred_df = pd.read_csv(pred_csv_path)
    cluster_map = parse_cd_hit_clusters(clstr_file)
    pred_df['filename'] = pred_df['filename'].str.replace('.pt', '', regex=False)
    pred_df['cluster'] = pred_df['filename'].map(cluster_map)
    pred_df = pred_df.dropna(subset=['cluster'])
    pred_df['cluster'] = pred_df['cluster'].astype(int)

    intervals = [
        (0.9, 1.0),
        (0.8, 0.9),
        (0.7, 0.8),
        (0.6, 0.7),
        (0.5, 0.6),
        (0.0, 0.5)
    ]

    results = []
    for low, high in intervals:
        df_interval = pred_df[
            (pred_df['prob_class1'] > low) &
            (pred_df['prob_class1'] <= high) &
            (pred_df['pred_class'] == pred_df['true_class'])  # Filter for correct predictions only
        ]
        if df_interval.empty:
            continue

        if low == 0.0 and high == 0.5:
            # Special case for <0.5: directly select the two globally lowest-scoring samples (ignore clusters)
            top2 = df_interval.sort_values('prob_class1', ascending=True).head(2)
            print(f"Selected top 2 samples with lowest prob_class1 in interval {low}-{high}:")
            print(top2[['filename', 'prob_class1', 'pred_class', 'true_class']])
        else:
            # Standard logic: Select the highest score within each cluster, then the top 2 globally
            top_in_clusters = df_interval.loc[df_interval.groupby('cluster')['prob_class1'].idxmax()]
            top2 = top_in_clusters.sort_values('prob_class1', ascending=False).head(2)

        top2['score_interval'] = f"{low}-{high}"
        results.append(top2)

    final_top2 = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    return final_top2


def find_top_positive_proteins(model_path, dataset_path, seq_csv_path, clstr_path, hidden_dim=256, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset = ProteinGraphDataset(dataset_path)
    if len(dataset) == 0:
        print("Dataset empty.")
        return

    sample = dataset[0]
    input_dim = sample.x.shape[1]
    edge_dim = sample.edge_attr.shape[1] if sample.edge_attr is not None else 0

    model = TransformerGraphClassifier(input_dim, hidden_dim, edge_dim=edge_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    results = []
    attn_dict = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            batch = batch.to(device)
            out, attn_all, edge_index = model(batch)
            probs = F.softmax(out, dim=1)
            class1_probs = probs[:, 1].cpu().numpy()
            preds = probs.argmax(dim=1).cpu().numpy()

            for i in range(len(batch)):
                fname = batch.filename[i]
                if fname.endswith('.pt'):
                    fname = fname[:-3]

                results.append({
                    "filename": fname,
                    "prob_class1": class1_probs[i],
                    "pred_class": preds[i],
                    "true_class": batch.y[i].item()
                })

                # Cache attention info for downstream visualization
                if fname not in attn_dict:
                    attn_dict[fname] = {
                        "attn_weights": attn_all,
                        "row": edge_index[0],
                        "col": edge_index[1],
                        "node_labels": batch.node_labels[i]
                    }

    df = pd.DataFrame(results)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/full_predictions.csv", index=False)

    top2_per_interval = cluster_select_top2_per_score_interval_all("outputs/full_predictions.csv", clstr_path)
    top2_csv_path = "outputs/top2_per_score_interval.csv"
    
    if not top2_per_interval.empty:
        top2_per_interval.to_csv(top2_csv_path, index=False)
        print(f"Saved Top2 per score interval CSV to {top2_csv_path}")
    else:
        print("No samples found in defined score intervals, skipping CSV save.")
        print("No samples found in defined score intervals, skipping visualization.")
        return

    vis_dir = "outputs/residue_attention_heatmaps_all_intervals"
    os.makedirs(vis_dir, exist_ok=True)
    summary_path = os.path.join(vis_dir, "residue_attention_summary_all_intervals.txt")
    
    if os.path.exists(summary_path):
        os.remove(summary_path)

    os.makedirs("outputs/attention_data", exist_ok=True) 

    for _, row in top2_per_interval.iterrows():
        fname = row['filename']
        if fname not in attn_dict:
            print(f"Warning: Attention info missing for {fname}, skipping visualization.")
            continue
            
        info = attn_dict[fname]
        save_path = os.path.join(vis_dir, f"{fname}_heatmap_{row['score_interval'].replace('.','p')}.png")
        
        # Generate heatmap using the first layer's attention weights by default
        visualize_residue_attention(
            seq=info['node_labels'],
            row=info['row'],
            col=info['col'],
            attn_weights=info['attn_weights'][0], 
            save_path=save_path,
            summary_path=summary_path
        )
        
        # Export complete attention tensors for all layers
        attn_save = {
            "attn_weights_all_layers": [a.cpu().numpy() for a in info['attn_weights']], 
            "row": info['row'].cpu().numpy(),
            "col": info['col'].cpu().numpy(),
            "node_labels": info['node_labels']
        }
        np.save(f"outputs/attention_data/{fname}.npy", attn_save)

if __name__ == "__main__":
    MODEL_PATH = "best_model.pt"
    DATASET_PATH = "/root/data/graphs"
    SEQ_CSV_PATH = "/root/data/dataset.csv"
    CLSTR_PATH = "/root/data/dataset_250_faa90.clstr"
    find_top_positive_proteins(MODEL_PATH, DATASET_PATH, SEQ_CSV_PATH, CLSTR_PATH)