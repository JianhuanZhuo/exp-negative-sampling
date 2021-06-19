import os
import random
import pickle
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import numpy as np


class SRNSML1MDataset(Dataset):
    def __init__(self, config):
        self.config = config
        folder = os.path.join(os.path.dirname(__file__), "SRNS_ml1m")
        self.train_data = pickle.load(open(os.path.join(folder, 'train.pkl'), 'rb'))
        self.val_data = pickle.load(open(os.path.join(folder, 'val.pkl'), 'rb'))
        self.test_data = pickle.load(open(os.path.join(folder, 'test.pkl'), 'rb'))
        self.test_data_neg = pickle.load(open(os.path.join(folder, 'test_neg.pkl'), 'rb'))

        self.size = config['sample_group_size']

        self.num_user = max(np.max(self.train_data[:, 0]), np.max(self.test_data[:, 0])) + 1
        self.num_item = max(np.max(self.train_data[:, 1]), np.max(self.test_data[:, 1])) + 1

        self.uis = defaultdict(set)
        for u, i in self.train_data:
            self.uis[u].add(i)

        self.ui_list = {
            u: list(iss)
            for u, iss in self.uis.items()
        }

        self.index_by_user = 'dataset/index_by_user' in config and config['dataset/index_by_user']

    def __len__(self):
        if self.index_by_user:
            return len(self.ui_list)
        else:
            return len(self.train_data)

    def __getitem__(self, index):
        if self.index_by_user:
            user = index
        else:
            user = self.train_data[index][0]

        positives = random.choices(self.ui_list[user], k=self.size)
        negatives = []
        for _ in range(self.size):
            neg = np.random.randint(0, self.num_item)
            while neg in self.uis[user]:
                neg = np.random.randint(0, self.num_item)
            negatives.append(neg)

        return torch.tensor(user), torch.tensor(positives), torch.tensor(negatives)
