import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType, 
    PeftModel
)
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_auc_score,
    matthews_corrcoef,
    classification_report
)

# Compatibility across PyTorch versions for mixed precision
try:
    from torch.cuda.amp import autocast, GradScaler
except ImportError:
    from torch.amp import autocast, GradScaler

# -----------------------------
# 1. Cluster-Aware Balancing and Leak-Free Splitting
# -----------------------------
def build_balanced_cluster_dataset(df_raw, ratio=10, random_state=42):
    """
    Cluster-level sampling:
    1. Retain 100% of positive clusters.
    2. Sample negative clusters to match sequence ratio of ~1:ratio.
    """
    cluster_stats = df_raw.groupby('cluster_id').agg(
        seq_count=('label', 'count'),
        cluster_label=('label', 'first')
    ).reset_index()

    pos_clusters = cluster_stats[cluster_stats['cluster_label'] == 1]
    neg_clusters = cluster_stats[cluster_stats['cluster_label'] == 0].sample(frac=1.0, random_state=random_state)

    total_pos = pos_clusters['seq_count'].sum()
    target_neg_count = total_pos * ratio

    selected_neg_cluster_ids = []
    accum_neg = 0
    for _, row in neg_clusters.iterrows():
        selected_neg_cluster_ids.append(row['cluster_id'])
        accum_neg += row['seq_count']
        if accum_neg >= target_neg_count:
            break

    selected_cluster_ids = set(pos_clusters['cluster_id']).union(set(selected_neg_cluster_ids))
    balanced_df = df_raw[df_raw['cluster_id'].isin(selected_cluster_ids)].copy().reset_index(drop=True)

    pos_cnt = (balanced_df['label'] == 1).sum()
    neg_cnt = (balanced_df['label'] == 0).sum()
    print(f">>> [Balanced Dataset] Positive: {pos_cnt} | Negative: {neg_cnt} | Ratio: 1 : {neg_cnt / pos_cnt:.2f}")
    return balanced_df


def split_by_cluster(data, n_splits=5, test_ratio=0.2, random_state=42):
    """
    Cluster-based partitioning:
    1. Stratified split of positive and negative clusters for an independent test set.
    2. Greedy load-balanced fold assignment for cross-validation.
    3. Folds indexed from 1 to 5 to match output directories.
    """
    cluster_stats = data.groupby('cluster_id').agg(
        seq_count=('label', 'count'),
        cluster_label=('label', 'first')
    ).reset_index()

    pos_c = cluster_stats[cluster_stats['cluster_label'] == 1]
    neg_c = cluster_stats[cluster_stats['cluster_label'] == 0]

    pos_tr_val_c, pos_test_c = train_test_split(pos_c, test_size=test_ratio, random_state=random_state)
    neg_tr_val_c, neg_test_c = train_test_split(neg_c, test_size=test_ratio, random_state=random_state)

    test_cluster_ids = set(pos_test_c['cluster_id']).union(set(neg_test_c['cluster_id']))
    test_data = data[data['cluster_id'].isin(test_cluster_ids)].copy().reset_index(drop=True)
    train_val_data = data[~data['cluster_id'].isin(test_cluster_ids)].copy().reset_index(drop=True)

    def assign_balanced_folds(df_group, n_folds, seed):
        df_sorted = df_group.sample(frac=1.0, random_state=seed).sort_values(by='seq_count', ascending=False)
        fold_counts = [0] * n_folds
        cluster_fold_map = {}
        for _, row in df_sorted.iterrows():
            min_fold = int(np.argmin(fold_counts))
            cluster_fold_map[row['cluster_id']] = min_fold
            fold_counts[min_fold] += row['seq_count']
        return cluster_fold_map

    pos_map = assign_balanced_folds(pos_tr_val_c, n_splits, seed=random_state)
    neg_map = assign_balanced_folds(neg_tr_val_c, n_splits, seed=random_state)
    full_cluster_to_fold = {**pos_map, **neg_map}

    train_val_data['fold'] = train_val_data['cluster_id'].map(full_cluster_to_fold) + 1
    return train_val_data, test_data


# -----------------------------
# 2. Overlapping Sliding Window Dataset and Collator
# -----------------------------
def chunk_protein_sequence(sequence, chunk_size=1022, stride=512):
    """
    Splits long sequences into overlapping windows, reserving 2 tokens for special tokens (<cls>, <eos>).
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


class ProteinChunkDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024, stride=512):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.chunk_size = max_length - tokenizer.num_special_tokens_to_add(pair=False)
        self.stride = stride

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        seq = str(row['sequence']).strip()
        label = int(row['label'])
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
        return {'encodings': encodings, 'label': label}


class ProteinChunkCollator:
    def __init__(self, tokenizer):
        self.base_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    def __call__(self, batch):
        flat_features = []
        protein_indices = []
        labels = []
        for p_idx, item in enumerate(batch):
            labels.append(item['label'])
            for enc in item['encodings']:
                flat_features.append({
                    'input_ids': enc['input_ids'],
                    'attention_mask': enc['attention_mask']
                })
                protein_indices.append(p_idx)
        padded_batch = self.base_collator(flat_features)
        return {
            'input_ids': padded_batch['input_ids'],
            'attention_mask': padded_batch['attention_mask'],
            'protein_indices': torch.tensor(protein_indices, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'batch_size': len(batch)
        }


# -----------------------------
# 3. Model Architecture and Micro-Batch Forward Pooling
# -----------------------------
def get_lora_config():
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["query", "key", "value"],
        modules_to_save=["classifier"]
    )


def build_esm_lora(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)

    # Ensure classification head parameters remain trainable
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True

    model.print_trainable_parameters()
    return model


def forward_chunk_pooling(model, batch_dict, device, max_sub_batch=2):
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

    # FP16-safe minimum value to avoid -inf underflow
    min_val = -1e4 if chunk_logits.dtype == torch.float16 else -1e9
    protein_logits = torch.full((batch_size, 2), min_val, device=device, dtype=chunk_logits.dtype)

    for p_idx in range(batch_size):
        mask = (protein_indices == p_idx)
        p_chunk_logits = chunk_logits[mask]
        protein_logits[p_idx] = torch.max(p_chunk_logits, dim=0).values

    return protein_logits


def evaluate(model, loader, device, desc="Evaluating"):
    model.eval()
    y_true, y_probs = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            labels = batch['labels'].to(device)
            with autocast(enabled=torch.cuda.is_available()):
                protein_logits = forward_chunk_pooling(model, batch, device, max_sub_batch=4)
                probs = torch.softmax(protein_logits, dim=1)[:, 1]

            y_true.extend(labels.cpu().numpy())
            y_probs.extend(probs.float().cpu().numpy())

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    try:
        roc_auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        roc_auc = float("nan")

    return pr_auc, roc_auc, y_true, y_probs


# -----------------------------
# 4. Model Soup (Weight Averaging) and Standalone Export
# -----------------------------
def export_averaged_5fold_model(model_path, fold_adapter_dirs, export_dir, tokenizer):
    """
    Full-parameter Model Soup Fusion:
    1. Loads adapter weights for each fold and merges with base model using merge_and_unload().
    2. Performs element-wise arithmetic averaging across all parameters in full space.
    3. Exports as a standalone Hugging Face model without runtime PEFT dependencies.
    """
    print(f"\n{'='*20} Executing 5-Fold Model Soup Fusion {'='*20}")
    num_folds = len(fold_adapter_dirs)
    merged_state_dict = None

    for idx, fold_dir in enumerate(fold_adapter_dirs):
        print(f">>> [{idx+1}/{num_folds}] Merging Fold {idx+1} ({fold_dir})...")
        base = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=2, torch_dtype=torch.float32
        )
        peft_m = PeftModel.from_pretrained(base, fold_dir)

        # Merge adapter into base weights
        merged_single = peft_m.merge_and_unload()
        single_state = merged_single.state_dict()

        if merged_state_dict is None:
            merged_state_dict = {k: v.cpu().clone().float() for k, v in single_state.items()}
        else:
            for k in merged_state_dict:
                merged_state_dict[k] += single_state[k].cpu().float()

        del base, peft_m, merged_single, single_state
        torch.cuda.empty_cache()
        gc.collect()

    print(">>> Averaging weights across all folds...")
    for k in merged_state_dict:
        merged_state_dict[k] /= num_folds

    final_model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=2, torch_dtype=torch.float32
    )
    final_model.load_state_dict(merged_state_dict)

    avg_cls_weight = final_model.classifier.out_proj.weight.data
    print(f"    [Verification] Mean classifier out_proj weight: {avg_cls_weight.mean().item():.6f}")

    final_model.save_pretrained(export_dir)
    tokenizer.save_pretrained(export_dir)
    print(f"[Done] Exported merged standalone model to: {export_dir}")

    return final_model


# -----------------------------
# 5. Training Pipeline
# -----------------------------
def main():
    os.makedirs("results", exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "esm2_t33_650M_UR50D"
    data_path = "dataset_cluster.csv"

    print(">>> Loading raw dataset...")
    raw_data = pd.read_csv(data_path).dropna(subset=['sequence', 'label', 'cluster_id'])
    raw_data['label'] = raw_data['label'].astype(int)

    # 1. Cluster-level sampling (ratio 1:10)
    balanced_data = build_balanced_cluster_dataset(raw_data, ratio=10, random_state=42)

    # 2. Split dataset by cluster (fold: 1-5)
    train_val_data, test_data = split_by_cluster(balanced_data, n_splits=5, test_ratio=0.2, random_state=42)

    test_data.to_csv("results/independent_test_set.csv", index=False)
    train_val_data.to_csv("results/train_val_set_with_folds.csv", index=False)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    collator = ProteinChunkCollator(tokenizer=tokenizer)

    pos_test = (test_data['label'] == 1).sum()
    neg_test = (test_data['label'] == 0).sum()
    print(f">>> Independent Test Set: {len(test_data)} (Pos: {pos_test}, Neg: {neg_test}, Ratio: 1:{neg_test/pos_test:.1f})")

    n_splits = 5
    accum_steps = 4
    num_epochs = 10
    saved_fold_dirs = []
    fold_val_pr_aucs = []

    class_weights = torch.tensor([1.0, 1.5], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Cross-validation training loop
    for fold_num in range(1, n_splits + 1):
        print(f"\n{'='*20} Fold {fold_num}/{n_splits} {'='*20}")
        train_df = train_val_data[train_val_data['fold'] != fold_num].reset_index(drop=True)
        val_df = train_val_data[train_val_data['fold'] == fold_num].reset_index(drop=True)

        pos_tr, neg_tr = (train_df['label'] == 1).sum(), (train_df['label'] == 0).sum()
        pos_va, neg_va = (val_df['label'] == 1).sum(), (val_df['label'] == 0).sum()
        print(f"Fold {fold_num} Train: {len(train_df)} (Pos: {pos_tr}, Neg: {neg_tr}) | Val: {len(val_df)} (Pos: {pos_va}, Neg: {neg_va})")

        train_loader = DataLoader(
            ProteinChunkDataset(train_df, tokenizer),
            batch_size=2, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collator
        )
        val_loader = DataLoader(
            ProteinChunkDataset(val_df, tokenizer),
            batch_size=16, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collator
        )

        model = build_esm_lora(model_path).to(device)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4,
            weight_decay=0.01
        )
        scaler = GradScaler(enabled=torch.cuda.is_available())

        best_pr_auc = -1.0
        fold_save_dir = f"results/best_lora_fold_{fold_num}"

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()

            for step, batch in enumerate(tqdm(train_loader, desc=f"Fold {fold_num} Epoch {epoch+1} Train")):
                labels = batch['labels'].to(device)

                cycle_start = (step // accum_steps) * accum_steps
                actual_accum = min(accum_steps, len(train_loader) - cycle_start)

                with autocast(enabled=torch.cuda.is_available()):
                    protein_logits = forward_chunk_pooling(model, batch, device, max_sub_batch=2)
                    loss = criterion(protein_logits, labels) / actual_accum

                scaler.scale(loss).backward()

                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            val_pr_auc, val_roc_auc, y_true_v, y_probs_v = evaluate(
                model, val_loader, device, desc=f"Fold {fold_num} Epoch {epoch+1} Val"
            )

            preds_05 = (y_probs_v >= 0.5).astype(int)
            pos_recall = np.sum((preds_05 == 1) & (y_true_v == 1)) / (np.sum(y_true_v == 1) + 1e-10)
            print(f"Epoch {epoch+1} - Val PR-AUC: {val_pr_auc:.4f} | ROC-AUC: {val_roc_auc:.4f} | Pos Recall (threshold 0.5): {pos_recall:.4f}")

            # Save best checkpoint according to validation PR-AUC
            if val_pr_auc > best_pr_auc:
                best_pr_auc = val_pr_auc
                model.save_pretrained(fold_save_dir)
                print(f"[*] New best score! Checkpoint saved to: {fold_save_dir} (PR-AUC: {val_pr_auc:.4f})")

        saved_fold_dirs.append(fold_save_dir)
        fold_val_pr_aucs.append(best_pr_auc)

        del model, optimizer, scaler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    # -----------------------------
    # 6. Model Soup Fusion & Final Test Set Evaluation
    # -----------------------------
    export_dir = "results/esm2_final_merged"
    final_merged_model = export_averaged_5fold_model(
        model_path=model_path,
        fold_adapter_dirs=saved_fold_dirs,
        export_dir=export_dir,
        tokenizer=tokenizer
    ).to(device)

    print(f"\n{'='*20} Evaluating Independent Test Set with Merged Model {'='*20}")
    test_loader = DataLoader(
        ProteinChunkDataset(test_data, tokenizer),
        batch_size=16, shuffle=False, num_workers=4, collate_fn=collator
    )

    test_pr_auc, test_roc_auc, y_true_test, test_probs = evaluate(
        final_merged_model, test_loader, device, desc="Evaluating Test Set (Merged Model)"
    )

    final_preds = (test_probs >= 0.5).astype(int)
    try:
        test_mcc = matthews_corrcoef(y_true_test, final_preds)
    except Exception:
        test_mcc = 0.0

    print("\n" + "="*45)
    print("[Final Evaluation on Independent Test Set]")
    print(f"Decision Threshold:               0.5000")
    print(f"PR-AUC (AUPRC):                   {test_pr_auc:.4f}")
    print(f"ROC-AUC (AUROC):                  {test_roc_auc:.4f}")
    print(f"MCC:                              {test_mcc:.4f}")
    print("="*45)

    print("\n[Classification Report]:")
    print(classification_report(y_true_test, final_preds, target_names=['Non-Target (0)', 'Target (1)'], digits=4))

    # Export test set predictions
    test_data_copy = test_data.copy().reset_index(drop=True)
    test_data_copy['true_label'] = y_true_test
    test_data_copy['pred_prob'] = test_probs
    test_data_copy['pred_label'] = final_preds

    save_cols = [c for c in ['seq_name', 'cluster_id', 'group', 'true_label', 'pred_label', 'pred_prob'] if c in test_data_copy.columns]
    test_data_copy[save_cols].to_csv("results/test_predictions_final_merged.csv", index=False)
    print(">>> Test predictions saved to: results/test_predictions_final_merged.csv")
    print(f">>> Ready for release: Artifacts in '{export_dir}' can be uploaded to Hugging Face or GitHub Releases.")


if __name__ == "__main__":
    main()
