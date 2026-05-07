import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
import pandas as pd
from tqdm import tqdm
import os


def load_faa_file(file_path):
    """Parses a FASTA (.faa) file to extract protein names and sequences."""
    sequences, names = [], []
    with open(file_path, 'r') as file:
        sequence, name = '', None
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if sequence:
                    sequences.append(sequence)
                sequence = ''
                name = line[1:]
                names.append(name)
            else:
                sequence += line
        if sequence:
            sequences.append(sequence)
    return names, sequences


class ProteinDataset(Dataset):
    def __init__(self, sequences, names, tokenizer, max_length=1024):
        self.sequences = sequences
        self.names = names  
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        name = self.names[idx]  
        
        encoding = self.tokenizer(
            sequence, truncation=True, padding='max_length', 
            max_length=self.max_length, return_tensors='pt'
        )
        
        return { 
            'input_ids': encoding['input_ids'].squeeze(0), 
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'names': name  
        }


def init_lora_model(model):
    """Applies LoRA configuration to the base model."""
    lora_config = LoraConfig(
        r=8,  
        lora_alpha=32,  
        lora_dropout=0.1,  
        bias="none",  
        target_modules=["query", "key", "value", "dense"]
    )
    return get_peft_model(model, lora_config)


def load_lora_model(model_path, final_model_path):
    """Loads the base model, applies LoRA, and restores the fine-tuned weights."""
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model = init_lora_model(model)

    # Explicitly move to primary GPU before loading the state dict
    model = model.to('cuda:0')  

    state_dict = torch.load(final_model_path)
    # Map checkpoint keys back to the current model structure
    new_state_dict = {key.replace("base_model.model.", "base_model."): value for key, value in state_dict.items()}
    model.load_state_dict(new_state_dict)

    # Optimize for inference speed and memory
    model.to(dtype=torch.bfloat16)  
    model = model.to(memory_format=torch.channels_last)  

    # Enable DataParallel if multiple GPUs are detected
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    return model


def save_progress(sequence_idx, checkpoint_file="checkpoint.txt"):
    """Caches the current index to allow resuming interrupted inference runs."""
    with open(checkpoint_file, "w") as f:
        f.write(str(sequence_idx))


def load_progress(checkpoint_file="checkpoint.txt"):
    """Retrieves the last saved index from the checkpoint file."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            return int(f.read())
    return 0  


def save_results_to_csv(results, file_path="prediction_results.csv"):
    """Appends prediction results to a CSV, writing the header only if the file is new."""
    is_file_exists = os.path.exists(file_path)
    
    df = pd.DataFrame(results, columns=["protein_name", "preds", "score"])
    df.to_csv(file_path, mode='a', header=not is_file_exists, index=False)


def predict_sequences(model, tokenizer, dataset, batch_size, checkpoint_file="checkpoint.txt"):
    """Runs batch inference with AMP and periodically flushes results to disk."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    local_predictions = []
    processed_count = 0
    start_idx = load_progress(checkpoint_file)  
    processed = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing", leave=False)):
            
            # Fast-forward through already processed samples
            if processed < start_idx:  
                processed += len(batch['input_ids'])
                continue

            input_ids = batch['input_ids'].to('cuda:0')  
            attention_mask = batch['attention_mask'].to('cuda:0')
            names = batch['names']  

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):  
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                predicted_classes = torch.argmax(probs, dim=-1)
                scores = probs.max(dim=-1).values

            batch_predictions = list(zip(names, predicted_classes.tolist(), scores.tolist()))
            local_predictions.extend(batch_predictions)
            processed_count += len(batch_predictions)

            # Flush results to disk every 10,000 sequences to prevent memory leaks
            if processed_count >= 10000:
                save_results_to_csv(local_predictions, file_path="prediction_results.csv")
                local_predictions = []  
                processed_count = 0  

            processed += len(batch['input_ids'])
            save_progress(processed, checkpoint_file)

    # Save any remaining results that didn't hit the 10,000 threshold
    if local_predictions:
        save_results_to_csv(local_predictions, file_path="prediction_results.csv")


def process_faa_file_parallel(faa_file_path, model_path, final_model_path, batch_size):
    """Pipeline orchestrator for loading data, initializing the model, and running inference."""
    model = load_lora_model(model_path, final_model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    names, sequences = load_faa_file(faa_file_path)
    dataset = ProteinDataset(sequences, names, tokenizer)

    predict_sequences(model, tokenizer, dataset, batch_size)


def main():
    faa_file_path = "final_non-ST_SE_combined.faa"
    model_path = "esm2_t33_650M_UR50D"
    final_model_path = "final_weighted_model.pth"
    
    batch_size = 64  
    
    process_faa_file_parallel(faa_file_path, model_path, final_model_path, batch_size)


if __name__ == "__main__":
    main()