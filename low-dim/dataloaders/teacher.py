import torch as tc
from torch.utils.data import DataLoader , Dataset
import numpy as np
import random
import os
import pickle
import torch.nn as nn
import torch.nn.functional as F

class MyDataset(Dataset):
    def __init__(self, inputs, labels):
        self.data = inputs
        self.targets  = labels
        self.original_targets  = labels
        self.classes  = [0]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def load_data(num_data: int, train_prop: float):

    n = num_data

    save_path = f"./dataloaders/cache/teacher_{n}_{train_prop}.pkl"
    if os.path.exists(save_path):
        with open(save_path, "rb") as fil:
            trainset , testset = pickle.load(fil)
        return trainset , testset
    os.makedirs( os.path.dirname(save_path) , exist_ok = True)


    X = tc.randn( n, 1 ) * 2

    T_hd = 128
    W1 = tc.randn( 1, T_hd )
    W2 = tc.randn( T_hd, 1 ) * 0.6
    Y = F.sigmoid(X @ W1) @ W2
    
    # shuffle data
    idxs = list(range(n))
    random.shuffle( idxs )
    idxs = tc.LongTensor( idxs )
    X = X[idxs]
    Y = Y[idxs]

    trainset = MyDataset(X[: int(n * train_prop)], Y[: int(n * train_prop)])
    testset  = MyDataset(X[int(n * train_prop) :], Y[int(n * train_prop) :])
    
    with open(save_path, "wb") as fil:
        pickle.dump([trainset, testset] , fil)

    return trainset , testset

