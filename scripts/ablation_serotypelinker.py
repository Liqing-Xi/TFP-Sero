# -*- coding: utf-8 -*-
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score

from train_transformerconv import TransformerGraphClassifier


def extract_group_id(name):
    group = name.rsplit('_', 1)[0]
    if "_YP" in group:
        group = group.split("_YP")[0]
    return group


def evaluate_group(df):
    """Aggregates predictions at the group level and calculates metrics."""
    group_results = df.groupby("group").agg({
        "true_label": "max",
        "pred_label": "max",
        "prob": "max"
    }).reset_index()

    y_true = group_results["true_label"]
    y_pred = group_results["pred_label"]
    y_prob = group_results["prob"]

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    acc = accuracy_score(y_true, y_pred)

    return roc_auc, pr_auc, acc


def apply_ablation(data, mode):
    """
    Applies specific ablation strategies to the graph.
    Note: 'data' must be a cloned object to prevent in-place mutation of the original dataset.
    """
    if mode == "no_node":
        # Zero out node features; node count and structural topology remain intact
        data.x = torch.zeros_like(data.x)

    elif mode == "no_edge":
        # Zero out edge features; topology remains intact (message passing still occurs)
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.edge_attr = torch.zeros_like(data.edge_attr)
            
    elif mode == "no_graph":
        # Structural ablation: Remove all edges, resulting in a set of isolated nodes
        data.edge_index = torch.empty((2, 0), dtype=torch.long, device=data.x.device)
        
        # Safely clear edge attributes to prevent dimension mismatches in downstream GNN layers
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.edge_attr = torch.empty((0, data.edge_attr.size(1)), 
                                         dtype=data.edge_attr.dtype, 
                                         device=data.edge_attr.device)

    return data


def run_ablation(mode, df_meta, pt_files, model, device, output_dir):
    print(f"\n===== Running Ablation: {mode} =====")

    all_results = []

    with torch.no_grad():
        for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc=f"{mode} Testing"):
            seq_name = row["seq_name"]
            label = row["label"]

            pt_file = f"{seq_name}.pt"
            if pt_file not in pt_files:
                continue

            # Clone data to prevent in-place modifications from contaminating the cached dataset
            data = pt_files[pt_file].clone().to(device)

            # Inject a batch dimension for single-graph inference
            if not hasattr(data, 'batch') or data.batch is None:
                data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

            if mode != "full":
                data = apply_ablation(data, mode)

            out = model(data)

            prob = torch.softmax(out, dim=1)[0, 1].item()
            pred = out.argmax(dim=1).item()
            group = extract_group_id(seq_name)

            all_results.append({
                "seq_name": seq_name,
                "group": group,
                "true_label": int(label),
                "pred_label": int(pred),
                "prob": round(prob, 6)
            })

    # Save instance-level predictions
    df_results = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, f"predictions_{mode}.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"[✓] Saved: {csv_path}")

    # Calculate and return group-level evaluation metrics
    roc_auc, pr_auc, acc = evaluate_group(df_results)
    print(f"[{mode}] ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}, ACC: {acc:.4f}")

    return roc_auc, pr_auc, acc


def main():
    # Path configuration
    csv_path = "/root/data/test_dataset_250.csv"
    pt_folder = "/root/data/test_graphs_10"
    model_path = "best_model.pt"
    output_dir = "outputs_ablation"

    os.makedirs(output_dir, exist_ok=True)

    ablation_modes = ["full", "no_node", "no_edge", "no_graph"]

    df_meta = pd.read_csv(csv_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cache graph objects in memory
    pt_files = {
        f: torch.load(os.path.join(pt_folder, f), map_location=device)
        for f in os.listdir(pt_folder) if f.endswith(".pt")
    }

    # Initialize model dimensions based on a sample graph
    example_data = next(iter(pt_files.values()))
    input_dim = example_data.x.shape[1]
    edge_dim = example_data.edge_attr.shape[1] if example_data.edge_attr is not None else 0

    model = TransformerGraphClassifier(
        input_dim=input_dim,
        hidden_dim=256,
        edge_dim=edge_dim
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Execute ablation study across all defined modes
    results_summary = []

    for mode in ablation_modes:
        roc_auc, pr_auc, acc = run_ablation(
            mode, df_meta, pt_files, model, device, output_dir
        )

        results_summary.append({
            "mode": mode,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "accuracy": acc
        })

    # Export aggregated summary
    df_summary = pd.DataFrame(results_summary)
    summary_path = os.path.join(output_dir, "ablation_summary.csv")
    df_summary.to_csv(summary_path, index=False)

    print("\n===== Ablation Summary =====")
    print(df_summary)
    print(f"[✓] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()