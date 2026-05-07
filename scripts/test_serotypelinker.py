# -*- coding: utf-8 -*-
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    classification_report
)

from train_transformerconv import TransformerGraphClassifier, ProteinGraphDataset
from torch_geometric.data import DataLoader


def extract_group_id(name):
    group = name.rsplit('_', 1)[0]
    if "_YP" in group:
        group = group.split("_YP")[0]
    return group


def save_classification_report(y_true, y_pred, filename):
    report = classification_report(
        y_true,
        y_pred,
        target_names=['0', '1'],
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(filename)

    print(f"[✓] Classification report saved to {filename}")


def plot_group_metrics(y_true, y_pred, y_prob, prefix):
    ap = average_precision_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)

    save_classification_report(
        y_true,
        y_pred,
        f"outputs/{prefix}_classification_report.csv"
    )

    return ap, acc


def main():
    # Path configuration
    csv_path = "/root/data/test_dataset.csv"
    pt_folder = "/root/data/test_graphs"
    model_path = "serotypelinker_model.pt"

    output_csv = "outputs/test_predictions.csv"
    prefix = "test"

    os.makedirs("outputs", exist_ok=True)

    df_meta = pd.read_csv(csv_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load all graph objects into memory
    pt_files = {
        f: torch.load(os.path.join(pt_folder, f), map_location=device)
        for f in os.listdir(pt_folder)
        if f.endswith(".pt")
    }

    # Initialize model architecture based on a sample graph
    example_data = next(iter(pt_files.values()))
    input_dim = example_data.x.shape[1]
    edge_dim = (
        example_data.edge_attr.shape[1]
        if example_data.edge_attr is not None
        else 0
    )

    model = TransformerGraphClassifier(
        input_dim=input_dim,
        hidden_dim=256,
        edge_dim=edge_dim
    ).to(device)

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.eval()

    # Instance-level prediction loop
    all_results = []

    with torch.no_grad():
        for _, row in tqdm(
            df_meta.iterrows(),
            total=len(df_meta),
            desc="Testing"
        ):
            seq_name = row["seq_name"]
            label = row["label"]
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

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(output_csv, index=False)
    print(f"[✓] Saved individual predictions to {output_csv}")

    # Group-level aggregation and evaluation
    group_results = df_results.groupby("group").agg({
        "true_label": "max",
        "pred_label": "max",
        "prob": "max"
    }).reset_index()

    ap, acc = plot_group_metrics(
        group_results["true_label"],
        group_results["pred_label"],
        group_results["prob"],
        prefix
    )

    print(f"[✓] Group-level AP: {ap:.4f}, ACC: {acc:.4f}")


if __name__ == "__main__":
    main()