import torch
from torch.utils.data import Dataset, DataLoader

class GPT2Dataset(Dataset):
    def __init__(self, token_ids: list[int], context_length: int):
        self.token_ids = token_ids
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.token_ids) - self.context_length
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pass

def build_dataloader(dataset: GPT2Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    pass