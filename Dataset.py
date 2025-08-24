import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle

class RNADataset(Dataset):
    def __init__(self, n_memories, memory_folder, start=0):
        self.n_memories=n_memories
        self.memory_folder=memory_folder
        self.start=start

    def __len__(self):
        return self.n_memories-self.start

    def __getitem__(self, idx):
        filename=f'{self.memory_folder}/transition{idx+self.start}.p'
        with open(filename,'rb') as f:
            memory=pickle.load(f)
        return memory
