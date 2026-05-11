# -*- coding: utf-8 -*-

import os
import argparse
import torch
import pandas as pd

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from peft import LoraConfig
from peft import get_peft_model


def load_faa_file(file_path):

    sequences = []
    names = []

    with open(file_path, 'r') as file:

        sequence = ''

        for line in file:

            line = line.strip()

            if line.startswith('>'):

                if sequence:
                    sequences.append(sequence)

                sequence = ''

                names.append(line[1:])

            else:
                sequence += line

        if sequence:
            sequences.append(sequence)

    return names, sequences


class ProteinDataset(Dataset):

    def __init__(
        self,
        sequences,
        names,
        tokenizer,
        max_length=1024
    ):

        self.sequences = sequences
        self.names = names
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):

        return len(self.sequences)

    def __getitem__(self, idx):

        encoding = self.tokenizer(
            self.sequences[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'name': self.names[idx]
        }


def init_lora_model(model):

    config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=[
            "query",
            "key",
            "value",
            "dense"
        ]
    )

    return get_peft_model(model, config)


def load_model(
    esm_model,
    checkpoint,
    device
):

    model = AutoModelForSequenceClassification.from_pretrained(
        esm_model,
        num_labels=2
    )

    model = init_lora_model(model)

    state_dict = torch.load(
        checkpoint,
        map_location=device
    )

    state_dict = {
        k.replace(
            "base_model.model.",
            "base_model."
        ): v
        for k, v in state_dict.items()
    }

    model.load_state_dict(state_dict)

    model.to(device)

    model.eval()

    return model


def predict(
    input_faa,
    esm_model,
    checkpoint,
    output_csv,
    batch_size
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"[Info] Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        esm_model
    )

    model = load_model(
        esm_model,
        checkpoint,
        device
    )

    names, sequences = load_faa_file(
        input_faa
    )

    dataset = ProteinDataset(
        sequences,
        names,
        tokenizer
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    results = []

    with torch.no_grad():

        for batch in tqdm(
            dataloader,
            desc="Predicting"
        ):

            input_ids = batch[
                'input_ids'
            ].to(device)

            attention_mask = batch[
                'attention_mask'
            ].to(device)

            outputs = model(
                input_ids,
                attention_mask=attention_mask
            )

            probs = torch.softmax(
                outputs.logits,
                dim=-1
            )

            preds = torch.argmax(
                probs,
                dim=-1
            )

            scores = probs.max(dim=-1).values

            for name, pred, score in zip(
                batch['name'],
                preds.tolist(),
                scores.tolist()
            ):

                results.append({
                    "protein_name": name,
                    "preds": pred,
                    "score": round(score, 6)
                })

    pd.DataFrame(results).to_csv(
        output_csv,
        index=False
    )

    print(f"[?] Results saved to {output_csv}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Input .faa file"
    )

    parser.add_argument(
        "--model",
        default="model/tfpfinder_model.pth"
    )

    parser.add_argument(
        "--esm_model",
        default="data/esm2_t33_650M_UR50D"
    )

    parser.add_argument(
        "--output",
        default="tfpfinder_results.csv"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64
    )

    args = parser.parse_args()

    predict(
        input_faa=args.input,
        esm_model=args.esm_model,
        checkpoint=args.model,
        output_csv=args.output,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()