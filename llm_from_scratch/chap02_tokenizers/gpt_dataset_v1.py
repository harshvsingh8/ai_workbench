import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, seq_length, stride=1):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.stride = stride
        
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(text)
            
        for i in range(0, len(token_ids) - seq_length, stride):
            input_chunk = token_ids[i : i + seq_length]
            target_chunk = token_ids[i + 1 : i + seq_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))


    def __len__(self):
        return len(self.input_ids)


    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

import tiktoken

# Creates a DataLoader for the GPTDatasetV1 dataset using the tiktoken tokenizer and DataLoader.
def create_dataloader_v1(text,
                         batch_size = 4,
                         max_seq_length = 256,
                         stride = 128,
                         shuffle = True,
                         drop_last = True,
                         num_workers = 0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(text, tokenizer, max_seq_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers = num_workers
    )
    return dataloader
