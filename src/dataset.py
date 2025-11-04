import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import json

class ProsQADataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_q_len, max_a_len):
        self.tokenizer = tokenizer
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len
        self.data = [json.loads(line) for line in open(jsonl_path, 'r')]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        answer = item['answer'] + self.tokenizer.eos_token
        
        # Tokenize question
        question_tokenized = self.tokenizer(
            question,
            max_length=self.max_q_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize answer for labels
        labels_tokenized = self.tokenizer(
            answer,
            max_length=self.max_a_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = labels_tokenized.input_ids.squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100 # Mask padding

        return {
            "input_ids": question_tokenized.input_ids.squeeze(0),
            "attention_mask": question_tokenized.attention_mask.squeeze(0),
            "labels": labels
        }