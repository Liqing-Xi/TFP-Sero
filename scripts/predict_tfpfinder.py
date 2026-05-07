import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
import pandas as pd
from tqdm import tqdm
import os

# 1. 解析 .faa 文件
def load_faa_file(file_path):
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

# 2. 构建数据集
class ProteinDataset(Dataset):
    def __init__(self, sequences, names, tokenizer, max_length=1024):
        self.sequences = sequences
        self.names = names  # 蛋白质名称
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        name = self.names[idx]  # 获取蛋白质名称
        encoding = self.tokenizer(sequence, truncation=True, padding='max_length', 
                                  max_length=self.max_length, return_tensors='pt')
        return { 
            'input_ids': encoding['input_ids'].squeeze(0), 
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'names': name  # 返回蛋白质名称
        }

# 3. 初始化 LoRA 模型
def init_lora_model(model):
    lora_config = LoraConfig(
        r=8,  
        lora_alpha=32,  
        lora_dropout=0.1,  
        bias="none",  
        target_modules=["query", "key", "value", "dense"]
    )
    model = get_peft_model(model, lora_config)
    return model

# 4. 载入 LoRA 模型
def load_lora_model(model_path, final_model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model = init_lora_model(model)

    # 确保所有的模型参数都在 CUDA 设备上
    model = model.to('cuda:0')  # 显式地移动到 cuda:0 上

    state_dict = torch.load(final_model_path)
    new_state_dict = {key.replace("base_model.model.", "base_model."): value for key, value in state_dict.items()}
    model.load_state_dict(new_state_dict)

    model.to(dtype=torch.bfloat16)  # 使用 bfloat16 推理
    model = model.to(memory_format=torch.channels_last)  # 启用 channels_last 内存格式

    # 使用 DataParallel 进行多卡推理
    # 如果只使用一张卡，可以去掉 DataParallel
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    return model

# 5. 保存进度到文件
def save_progress(sequence_idx, checkpoint_file="checkpoint.txt"):
    with open(checkpoint_file, "w") as f:
        f.write(str(sequence_idx))

# 6. 从进度文件加载进度
def load_progress(checkpoint_file="checkpoint.txt"):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            return int(f.read())
    return 0  # 如果没有进度文件，则从头开始

# 7. 预测函数
def predict_sequences(model, tokenizer, dataset, batch_size, checkpoint_file="checkpoint.txt"):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    local_predictions = []
    processed_count = 0
    start_idx = load_progress(checkpoint_file)  # 从上次中断的位置恢复
    processed = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing", leave=False)):
            if processed < start_idx:  # 跳过已经处理的部分
                processed += len(batch['input_ids'])
                continue

            input_ids = batch['input_ids'].to('cuda:0')  # 将输入数据移到 GPU 上
            attention_mask = batch['attention_mask'].to('cuda:0')
            names = batch['names']  # 获取当前批次的蛋白质名称

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):  # 强制使用 bfloat16
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                predicted_classes = torch.argmax(probs, dim=-1)
                scores = probs.max(dim=-1).values

            # 保存蛋白质名称、预测类别和分数
            batch_predictions = list(zip(names, predicted_classes.tolist(), scores.tolist()))
            local_predictions.extend(batch_predictions)
            processed_count += len(batch_predictions)

            # 每处理10000个蛋白质序列就保存一次结果
            if processed_count >= 10000:
                save_results_to_csv(local_predictions, file_path="prediction_results.csv")
                local_predictions = []  # 重置local_predictions以便下次保存
                processed_count = 0  # 重置计数器

            # 每次更新进度
            processed += len(batch['input_ids'])
            save_progress(processed, checkpoint_file)

    # 如果还有未保存的结果（即小于10000的剩余部分）
    if local_predictions:
        save_results_to_csv(local_predictions, file_path="prediction_results.csv")

# 8. 保存预测结果
def save_results_to_csv(results, file_path="prediction_results.csv"):
    # 检查文件是否已存在
    is_file_exists = os.path.exists(file_path)
    
    # 使用 DataFrame 将结果保存到 CSV
    df = pd.DataFrame(results, columns=["protein_name", "preds", "score"])
    
    # 如果文件不存在，表示是第一次保存，写入表头
    df.to_csv(file_path, mode='a', header=not is_file_exists, index=False)

# 9. 处理 .faa 文件（使用 DataParallel 版本）
def process_faa_file_parallel(faa_file_path, model_path, final_model_path, batch_size):
    model = load_lora_model(model_path, final_model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    names, sequences = load_faa_file(faa_file_path)
    dataset = ProteinDataset(sequences, names, tokenizer)

    predict_sequences(model, tokenizer, dataset, batch_size)

# 10. 主函数（使用 DataParallel）
def main():
    faa_file_path = "final_non-ST_SE_combined.faa"
    model_path = "esm2_t33_650M_UR50D"
    final_model_path = "final_weighted_model.pth"
    
    batch_size = 64  # 增大 batch_size，提高吞吐量

    process_faa_file_parallel(faa_file_path, model_path, final_model_path, batch_size)

if __name__ == "__main__":
    main()
