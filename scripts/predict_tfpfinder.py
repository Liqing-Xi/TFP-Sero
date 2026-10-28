# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

# Compatibility across PyTorch versions for mixed precision
try:
    from torch.cuda.amp import autocast
except ImportError:
    from torch.amp import autocast


def load_faa_file(file_path):
    """
    Parses a FASTA/FAA file into protein identifiers and sequences.
    """
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
                # Retain the full header or strip leading '>'
                names.append(line[1:].split()[0])
            else:
                sequence += line
        if sequence:
            sequences.append(sequence)

    return names, sequences


def chunk_protein_sequence(sequence, chunk_size=1022, stride=512):
    """
    Splits long sequences into overlapping windows, reserving 2 tokens for <cls> and <eos>.
    """
    seq_len = len(sequence)
    if seq_len <= chunk_size:
        return [sequence]
    chunks = []
    start = 0
    while start < seq_len:
        end = min(start + chunk_size, seq_len)
        chunks.append(sequence[start:end])
        if end == seq_len:
            break
        start += stride
    return chunks


class ProteinChunkInferenceDataset(Dataset):
    """
    Inference dataset matching the training chunking logic.
    """
    def __init__(self, names, sequences, tokenizer, max_length=1024, stride=512):
        self.names = names
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.chunk_size = max_length - tokenizer.num_special_tokens_to_add(pair=False)
        self.stride = stride

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        name = self.names[idx]
        seq = str(self.sequences[idx]).strip()
        chunks = chunk_protein_sequence(seq, chunk_size=self.chunk_size, stride=self.stride)
        encodings = [
            self.tokenizer(
                c,
                truncation=True,
                max_length=self.chunk_size + self.tokenizer.num_special_tokens_to_add(pair=False),
                padding=False,
                add_special_tokens=True
            )
            for c in chunks
        ]
        return {
            'name': name,
            'encodings': encodings
        }


class ProteinChunkInferenceCollator:
    """
    Flattens sequence chunks across batch items and tracks protein indices.
    """
    def __init__(self, tokenizer):
        self.base_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    def __call__(self, batch):
        flat_features = []
        protein_indices = []
        names = []
        for p_idx, item in enumerate(batch):
            names.append(item['name'])
            for enc in item['encodings']:
                flat_features.append({
                    'input_ids': enc['input_ids'],
                    'attention_mask': enc['attention_mask']
                })
                protein_indices.append(p_idx)

        padded_batch = self.base_collator(flat_features)
        return {
            'names': names,
            'input_ids': padded_batch['input_ids'],
            'attention_mask': padded_batch['attention_mask'],
            'protein_indices': torch.tensor(protein_indices, dtype=torch.long),
            'batch_size': len(batch)
        }


def forward_chunk_pooling(model, batch_dict, device, max_sub_batch=4):
    """
    Forward pass with micro-batching and sequence-level max-pooling.
    """
    input_ids = batch_dict['input_ids'].to(device)
    attention_mask = batch_dict['attention_mask'].to(device)
    protein_indices = batch_dict['protein_indices'].to(device)
    batch_size = batch_dict['batch_size']
    total_chunks = input_ids.size(0)

    if total_chunks <= max_sub_batch:
        chunk_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    else:
        chunk_logits_list = []
        for i in range(0, total_chunks, max_sub_batch):
            sub_ids = input_ids[i:i + max_sub_batch]
            sub_mask = attention_mask[i:i + max_sub_batch]
            sub_out = model(input_ids=sub_ids, attention_mask=sub_mask)
            chunk_logits_list.append(sub_out.logits)
        chunk_logits = torch.cat(chunk_logits_list, dim=0)

    # FP16-safe minimum value
    min_val = -1e4 if chunk_logits.dtype == torch.float16 else -1e9
    protein_logits = torch.full((batch_size, 2), min_val, device=device, dtype=chunk_logits.dtype)

    for p_idx in range(batch_size):
        mask = (protein_indices == p_idx)
        p_chunk_logits = chunk_logits[mask]
        protein_logits[p_idx] = torch.max(p_chunk_logits, dim=0).values

    return protein_logits


def predict(input_faa, model_dir, output_csv, batch_size=16, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Running on device: {device}")

    # Load standalone model and tokenizer exported from training pipeline
    print(f"[Info] Loading model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    names, sequences = load_faa_file(input_faa)
    print(f"[Info] Loaded {len(sequences)} sequences from {input_faa}")

    dataset = ProteinChunkInferenceDataset(names, sequences, tokenizer)
    collator = ProteinChunkInferenceCollator(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collator
    )

    all_names = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            with autocast(enabled=torch.cuda.is_available()):
                protein_logits = forward_chunk_pooling(model, batch, device, max_sub_batch=4)
                # Probability for the target positive class (Label 1)
                probs = torch.softmax(protein_logits, dim=1)[:, 1]

            all_names.extend(batch['names'])
            all_probs.extend(probs.float().cpu().numpy())

    all_probs = np.array(all_probs)
    preds = (all_probs >= threshold).astype(int)

    results_df = pd.DataFrame({
        "protein_name": all_names,
        "pred_label": preds,
        "prob_target": np.round(all_probs, 6)
    })

    results_df.to_csv(output_csv, index=False)
    print(f"[Done] Predictions saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Protein Sequence Classification Inference")
    parser.add_argument("--input", required=True, help="Path to input .faa/.fasta file")
    parser.add_argument("--model", default="/root/SPTFPs_test/results/esm2_final_merged", help="Path to exported Hugging Face model directory")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification decision threshold")

    args = parser.parse_args()

    predict(
        input_faa=args.input,
        model_dir=args.model,
        output_csv=args.output,
        batch_size=args.batch_size,
        threshold=args.threshold
    )


if __name__ == "__main__":
    main()
