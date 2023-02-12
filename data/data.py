from torch.utils.data import Dataset
import numpy as np
import torch


class PaulGrahamEssaysDataset(Dataset):
    def __init__(self, ctx_size, split='train'):
        data_path = split + '.bin'
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.ctx_size = ctx_size

    def __len__(self):
        # we don't want to get any index out of range errors
        return len(self.data) - self.ctx_size + 1

    def __getitem__(self, idx):
        # we want a sequence length of 1 more than the ctx_len
        # seq: "hello", "there", "my", "friend"
        # y = ["there", "my", "friend"]
        # x = ["hello", "there", "my"]
        seq = torch.from_numpy((self.data[idx:idx + self.ctx_size + 1]).astype(np.int64))
        return seq[:-1], seq[1:]