from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset

# 设置tokenizers不并行加速，避免报错
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 使用 Huggingface datasets 的惰性加载，避免一次性加载整个数据集到内存
        self.samples = load_dataset("json", data_files=data_path, split="train")
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

    # tokenizer把文本转换成token id
        tokens = self.tokenizer(
            str(sample["text"]),  # 这里假设json文件中每条数据的文本字段是"text"，包含了文本内容
            add_special_tokens=True,
            max_length=self.max_length - 2,  # 留出[BOS]和[EOS]的位置
            truncation=True, # 如果长度超过了max，自动剪切
        ).input_ids
    # 需要加上EOS，BOS，以及PAD填充
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens)) # 填充到max_length 
        input_ids = torch.tensor(input_ids, dtype=torch.long) # 转换成torch tensor
    # 需要自行编写labels，防止PAD参与loss计算
        labels = input_ids.clone() # 直接复制input_ids作为labels
        labels[labels == self.tokenizer.pad_token_id] = -100 # 将PAD token的标签设置为-100，告诉loss函数忽略这些位置
    # 需要编写attention_mask，告诉模型哪些位置是有效的，哪些位置是PAD
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long() # 非PAD位置为1，PAD位置为0
    # 返回input_ids,  attention_mask，labels
        return {
            "input_ids": input_ids,    
            "attention_mask": attention_mask,     
            "labels": labels,
        }