# -*- coding: utf-8 -*-
import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm

from train_transformerconv import TransformerGraphClassifier


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to metadata CSV file"
    )

    parser.add_argument(
        "--graph_dir",
        required=True,
        help="Directory containing graph .pt files"
    )

    parser.add_argument(
        "--model",
        default="model/serotypelinker_model.pt",
        help="Path to trained model checkpoint"
    )

    parser.add_argument(
        "--output",
        default="prediction_serotype.csv",
        help="Output prediction CSV file"
    )

    args = parser.parse_args()

    csv_path = args.csv
    pt_folder = args.graph_dir
    model_path = args.model
    output_csv = args.output

    df_meta = pd.read_csv(csv_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Info] Using device: {device}")

    pt_files = {
        f: torch.load(os.path.join(pt_folder, f), map_location=device)
        for f in os.listdir(pt_folder)
        if f.endswith(".pt")
    }

    if not pt_files:
        print("[Error] No graph files found.")
        return

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

    results = []

    with torch.no_grad():

        for _, row in tqdm(
            df_meta.iterrows(),
            total=len(df_meta),
            desc="Predicting"
        ):

            seq_name = row["seq_name"]

            pt_file = f"{seq_name}.pt"

            if pt_file not in pt_files:
                print(f"[Warning] Missing graph file: {pt_file}")
                continue

            data = pt_files[pt_file].to(device)

            out = model(data)

            prob = torch.softmax(out, dim=1)[0, 1].item()

            pred = out.argmax(dim=1).item()

            results.append({
                "seq_name": seq_name,
                "pred_label": int(pred),
                "prob": round(prob, 6)
            })

    df_out = pd.DataFrame(results)

    df_out.to_csv(output_csv, index=False)

    print(f"[✓] Predictions saved to: {output_csv}")


if __name__ == "__main__":
    main()