# -*- coding: utf-8 -*-
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, accuracy_score, classification_report
)

from train_transformerconv import TransformerGraphClassifier, ProteinGraphDataset
from torch_geometric.data import DataLoader

def extract_group_id(name):
    group = name.rsplit('_', 1)[0]
    if "_YP" in group:
        group = group.split("_YP")[0]
    return group

def save_confusion_matrix(y_true, y_pred, filename):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = np.trace(cm) / float(np.sum(cm))
    accuracy_percentage = accuracy * 100

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'],
                yticklabels=['0', '1'])
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
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
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

def save_classification_report(y_true, y_pred, filename):
    report = classification_report(y_true, y_pred, target_names=['0', '1'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(filename)
    print(f"[✓] Classification report saved to {filename}")

def plot_group_metrics(y_true, y_pred, y_prob, prefix):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ap = average_precision_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)

    save_confusion_matrix(y_true, y_pred, f"outputs/{prefix}_confusion_matrix.png")
    save_roc_curve(y_true, y_prob, f"outputs/{prefix}_roc_curve.png")
    save_pr_auc(y_true, y_prob, f"outputs/{prefix}_pr_curve.png")
    save_classification_report(y_true, y_pred, f"outputs/{prefix}_classification_report.csv")

    return roc_auc, ap

def main():
    # === 路径配置 ===
    csv_path = "/root/data/test_dataset_250.csv"
    pt_folder = "/root/data/test_graphs_10"
    model_path = "best_model.pt"
    output_csv = "outputs/test_predictions.csv"
    prefix = "test"

    os.makedirs("outputs", exist_ok=True)

    # === 读取 CSV 文件 ===
    df_meta = pd.read_csv(csv_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pt_files = {
        f: torch.load(os.path.join(pt_folder, f), map_location=device)
        for f in os.listdir(pt_folder) if f.endswith(".pt")
    }

    # === 初始化模型结构 ===
    example_data = next(iter(pt_files.values()))
    input_dim = example_data.x.shape[1]
    edge_dim = example_data.edge_attr.shape[1] if example_data.edge_attr is not None else 0
    model = TransformerGraphClassifier(input_dim=input_dim, hidden_dim=256, edge_dim=edge_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # === 单样本预测 ===
    all_results = []
    with torch.no_grad():
        for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Testing"):
            seq_name, label = row["seq_name"], row["label"]
            pt_file = f"{seq_name}.pt"
            if pt_file not in pt_files:
                print(f"Warning: {pt_file} not found.")
                continue
            data = pt_files[pt_file].to(device)
            out = model(data)
            prob = torch.softmax(out, dim=1)[0, 1].item()
            pred = out.argmax(dim=1).item()
            group = extract_group_id(seq_name)
            all_results.append({
                "seq_name": seq_name,
                "group": group,
                "true_label": int(label),
                "pred_label": int(pred),
                "prob": round(prob, 4)
            })

    # === 保存每个样本的预测结果 ===
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(output_csv, index=False)
    print(f"[✓] Saved individual predictions to {output_csv}")

    # === 基于组进行聚合与评估 ===
    group_results = df_results.groupby("group").agg({
        "true_label": "max",       # 任一为 1，则组为 1
        "pred_label": "max",       # 任一预测为 1，则组为 1
        "prob": "max"              # 概率取最大
    }).reset_index()

    # === 绘制并评估 group 级别的性能 ===
    roc_auc, ap = plot_group_metrics(
        group_results["true_label"],
        group_results["pred_label"],
        group_results["prob"],
        prefix
    )
    print(f"[✓] Group-level ROC AUC: {roc_auc:.4f}, PR AUC: {ap:.4f}")

if __name__ == "__main__":
    main()
