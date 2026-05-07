import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle, resample
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from peft import get_peft_model, LoraConfig

def load_csv(file_path):
    """Loads and filters dataset, ensuring required columns exist."""
    data = pd.read_csv(file_path)
    data = data.dropna(subset=['sequence', 'label'])
    data = data[data['label'].isin([0, 1])]
    
    if 'seq_name' not in data.columns:
        data['seq_name'] = [f'seq_{i}' for i in range(len(data))]
    if 'group' not in data.columns:
        data['group'] = 'unknown'
        
    return data

def balance_data(data):
    """Balances the dataset by oversampling the majority class (label 0)."""
    label_1_data = data[data['label'] == 1]
    label_0_data = data[data['label'] == 0]
    
    label_0_sample = resample(label_0_data, n_samples=len(label_1_data) * 10, random_state=42)
    balanced_data = pd.concat([label_1_data, label_0_sample])
    
    return shuffle(balanced_data, random_state=42)

def split_data_with_cv(data, test_ratio=0.2, n_splits=5):
    """Splits data into a holdout test set and K-folds for training."""
    test_data = data.sample(frac=test_ratio, random_state=42)
    train_data = data.drop(test_data.index)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    return train_data, test_data, kf


class ProteinDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence = row['sequence']
        label = row['label']
        seq_name = row['seq_name']
        group = row['group']

        encoding = self.tokenizer(
            sequence, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )

        return { 
            'input_ids': encoding['input_ids'].squeeze(0), 
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
            'seq_name': seq_name,
            'group': group
        }


class ProteinClassifier(nn.Module):
    def __init__(self, model_path, num_labels):
        super(ProteinClassifier, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)


def init_lora_model(model):
    """Initializes the LoRA configuration for parameter-efficient fine-tuning."""
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["query", "key", "value", "dense"]
    )
    return get_peft_model(model, lora_config)


def save_classification_report(y_true, y_pred, filename):
    report = classification_report(y_true, y_pred, target_names=['Non-TFPs','TFPs'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(filename)
    print(f"Classification report saved to {filename}")

def save_predictions_csv(dataset, y_true, y_pred, y_prob, filename):
    seq_names = [dataset[i]['seq_name'] for i in range(len(dataset))]
    groups = [dataset[i]['group'] for i in range(len(dataset))]
    df = pd.DataFrame({
        'seq_name': seq_names,
        'group': groups,
        'true_label': y_true,
        'pred_label': y_pred,
        'prob': y_prob
    })
    df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")

def evaluate_model(model, loader, dataset, device='cpu', save_prefix='eval'):
    model.eval()
    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {save_prefix}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            _, predicted = torch.max(outputs.logits, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())

    save_predictions_csv(dataset, y_true, y_pred, y_scores, f"{save_prefix}_predictions.csv")
    save_classification_report(y_true, y_pred, f"{save_prefix}_classification_report.csv")
    return y_true, y_pred, y_scores


class EarlyStopping:
    """Halts training if the monitored metric stops improving after a set patience."""
    def __init__(self, patience=3, delta=0.01, monitor='loss'):
        self.patience = patience
        self.delta = delta
        self.monitor = monitor
        self.best_metric = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_metric):
        if self.best_metric is None:
            self.best_metric = current_metric
        elif (self.monitor == 'loss' and current_metric > self.best_metric - self.delta) or \
             (self.monitor == 'accuracy' and current_metric < self.best_metric + self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_metric = current_metric
            self.counter = 0


def train_model_with_lora_cv(model, train_data, kf, optimizer, criterion, tokenizer,
                              num_epochs=10, device='cpu', class_weights=None):
    """Trains the model using K-Fold cross-validation and mixed-precision."""
    best_models = []
    best_results = []
    all_model_weights = []
    all_accuracies = []

    scaler = GradScaler()

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
        print(f"\n=== Training fold {fold+1}/{kf.get_n_splits()} ===")
        train_fold_data = train_data.iloc[train_idx]
        val_fold_data = train_data.iloc[val_idx]

        train_dataset = ProteinDataset(train_fold_data, tokenizer)
        val_dataset = ProteinDataset(val_fold_data, tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=8, pin_memory=True)

        model.to(device)
        model.train()
        early_stopping = EarlyStopping(patience=3, delta=0.01, monitor='loss')
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                with autocast():
                    outputs = model(input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)

                if class_weights is not None:
                    weight = torch.tensor([class_weights[label.item()] for label in labels]).to(loss.device)
                    loss = (loss * weight).mean()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = correct / total
            print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.4f}")

            # Validation Phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = model(input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)
                    
                    if class_weights is not None:
                        weight = torch.tensor([class_weights[label.item()] for label in labels]).to(loss.device)
                        loss = (loss * weight).mean()

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.logits, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            print(f"Validation: Loss={val_loss:.4f}, Accuracy={val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()

            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            torch.save(best_model_state, f"best_model_fold_{fold+1}.pth")
            best_models.append(f"best_model_fold_{fold+1}.pth")
            best_results.append({'fold': fold+1, 'best_val_loss': best_val_loss, 'val_accuracy': val_acc})

            all_model_weights.append(best_model_state)
            all_accuracies.append(val_acc)

            evaluate_model(model, val_loader, val_dataset, device=device, save_prefix=f"fold_{fold+1}_val")

    # Create an ensemble model by calculating the weighted average of weights across all folds
    total_acc = sum(all_accuracies)
    weighted_avg_weights = {}
    for key in all_model_weights[0].keys():
        weighted_sum = sum([all_model_weights[i][key] * all_accuracies[i] for i in range(len(all_model_weights))])
        weighted_avg_weights[key] = weighted_sum / total_acc

    torch.save(weighted_avg_weights, "final_weighted_model.pth")
    print("Weighted average model saved.")

    return best_models, best_results

def evaluate_on_test_set(model, test_loader, test_dataset, device='cpu'):
    evaluate_model(model, test_loader, test_dataset, device=device, save_prefix="test")

def main():
    file_path = "dataset.csv"
    data = load_csv(file_path)
    balanced_data = balance_data(data)
    train_data, test_data, kf = split_data_with_cv(balanced_data)

    model_path = "esm2_t33_650M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    num_labels = 2

    train_dataset = ProteinDataset(train_data, tokenizer)
    test_dataset = ProteinDataset(test_data, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=8)

    class_weights = compute_class_weight('balanced', classes=np.array([0,1]), y=train_data['label'])
    class_weights = dict(zip([0,1], class_weights))
    print("Class weights:", class_weights)

    model = ProteinClassifier(model_path, num_labels=num_labels)
    model = init_lora_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    train_model_with_lora_cv(model, train_data, kf, optimizer, criterion, tokenizer,
                              num_epochs=10, device='cuda', class_weights=class_weights)

    model.load_state_dict(torch.load("final_weighted_model.pth"))
    model.to('cuda')

    print("Evaluating final weighted model on test set...")
    evaluate_on_test_set(model, test_loader, test_dataset, device='cuda')

if __name__ == "__main__":
    main()