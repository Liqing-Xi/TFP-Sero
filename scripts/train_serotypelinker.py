# -*- coding: utf-8 -*-
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, DataLoader
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report
)
import numpy as np
import itertools
from tqdm import tqdm
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
import seaborn as sns

os.makedirs("outputs", exist_ok=True)

# ========= 自定义 GraphTransformer Layer =========
class GraphTransformerLayer(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim=0, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.dropout = dropout

        self.q_proj = nn.Linear(in_channels, out_channels * heads)
        self.k_proj = nn.Linear(in_channels, out_channels * heads)
        self.v_proj = nn.Linear(in_channels, out_channels * heads)

        if edge_dim > 0:
            self.edge_proj = nn.Linear(edge_dim, heads)
        else:
            self.edge_proj = None

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
            edge_emb = self.edge_proj(edge_attr)
            attn_scores += edge_emb

        attn_weights = F.softmax(attn_scores, dim=0)
        attn_weights = self.dropout_layer(attn_weights)

        out_messages = attn_weights.unsqueeze(-1) * V[col]
        out = torch.zeros_like(Q)
        out = out.scatter_add_(0, row.unsqueeze(-1).unsqueeze(-1).expand_as(out_messages), out_messages)

        out = out.view(-1, H * C)
        x_proj = x if x.shape[-1] == out.shape[-1] else self.out_proj(x)
        out = self.out_proj(out)
        out = self.norm(out + x_proj)
        return out

# ========= 数据集 =========
class ProteinGraphDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.files = sorted([f for f in os.listdir(root_dir) if f.endswith('.pt')])

    def len(self):
        return len(self.files)

    def get(self, idx):
        return torch.load(os.path.join(self.root_dir, self.files[idx]))

    def __len__(self):
        return len(self.files)

# ========= 模型定义 =========
class TransformerGraphClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=2, heads=4, dropout=0.3, edge_dim=0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim * heads)
        self.edge_dim = edge_dim
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
        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = layer(x, edge_index, edge_attr)
            x = self.dropout(F.relu(norm(x)))
            x = x + residual

        x = global_mean_pool(x, batch)
        return self.classifier(x)

# ========= 训练函数 =========
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in tqdm(loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

# ========= 图像与报告保存 =========
def save_confusion_matrix(y_true, y_pred, filename):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = np.trace(cm) / float(np.sum(cm))
    accuracy_percentage = accuracy * 100

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy_percentage:.2f}%')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(filename)
    plt.close()

def save_roc_curve(y_true, y_scores, filename):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(filename)
    plt.close()

def save_pr_auc(y_true, y_scores, filename):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, color='b', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig(filename)
    plt.close()

# ========= 验证函数 =========
def evaluate_with_metrics(model, loader, device, criterion, save_plots=False, prefix="best_model"):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0
    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating"):
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs

            probs = torch.softmax(out, dim=1)[:, 1]
            preds = out.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    acc = (y_true == y_pred).mean()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ap = average_precision_score(y_true, y_prob)

    if save_plots:
        save_confusion_matrix(y_true, y_pred, f"outputs/{prefix}_confusion_matrix.png")
        save_roc_curve(y_true, y_prob, f"outputs/{prefix}_roc_curve.png")
        save_pr_auc(y_true, y_prob, f"outputs/{prefix}_pr_curve.png")

        report = classification_report(y_true, y_pred, target_names=['class_0', 'class_1'])
        with open(f"outputs/{prefix}_classification_report.txt", "w") as f:
            f.write(report)

    print(f"Val Loss: {total_loss / len(loader.dataset):.4f}, Acc: {acc:.4f}, AUC: {roc_auc:.4f}, AP: {ap:.4f}")
    return total_loss / len(loader.dataset), acc, roc_auc

# ========= Early Stopping =========
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def step(self, score):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ========= 主函数 =========
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ProteinGraphDataset("/root/data/graphs_10")
    input_dim = dataset[0].x.shape[1]
    edge_dim = dataset[0].edge_attr.shape[1] if dataset[0].edge_attr is not None else 0
    print(f"Input dim: {input_dim}, Edge dim: {edge_dim}")

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    param_grid = {
        "batch_size": [32],
        "hidden_dim": [256],
        "learning_rate": [1e-4],
    }

    best_auc, best_params = 0, None
    criterion = nn.CrossEntropyLoss()

    for batch_size, hidden_dim, lr in itertools.product(*param_grid.values()):
        print(f"Trying batch_size={batch_size}, hidden_dim={hidden_dim}, lr={lr}")
        model = TransformerGraphClassifier(input_dim, hidden_dim, edge_dim=edge_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        early_stopper = EarlyStopping()

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        for epoch in range(1, 201):
            print(f"\nEpoch {epoch}")
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_auc = evaluate_with_metrics(model, val_loader, device, criterion)

            if val_auc > best_auc:
                best_auc = val_auc
                best_params = (batch_size, hidden_dim, lr)
                torch.save(model.state_dict(), "best_model.pt")
                evaluate_with_metrics(model, val_loader, device, criterion, save_plots=True)
                print("Saved best model.")

            early_stopper.step(val_auc)
            if early_stopper.early_stop:
                print("Early stopping.")
                break

    print(f"Best AUC: {best_auc:.4f} with params: {best_params}")

if __name__ == "__main__":
    main()
