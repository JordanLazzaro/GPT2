import torch

class DataLoader:
    def __init__(self, dataset, batch_size=8, device=None):
        self.data = dataset
        self.batch_size = batch_size

    def _get_batch(self):
        idxs = torch.randint(len(self.data), (self.batch_size,)) 
        Xs, Ys = zip(*[self.data[i] for i in idxs])
        
        return torch.stack(Xs), torch.stack(Ys)

    def __iter__(self):
        return self

    def __next__(self):
        return self._get_batch()