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

        self.size_pos = self.size_neg = config['sample_group_size']
        if 'sample_group_size_pos' in config:
            self.size_pos = config['sample_group_size_pos']
        if 'sample_group_size_neg' in config:
            self.size_neg = config['sample_group_size_neg']

        self.num_user = max(np.max(self.train_data[:, 0]), np.max(self.test_data[:, 0])) + 1
        self.num_item = max(np.max(self.train_data[:, 1]), np.max(self.test_data[:, 1])) + 1

        self.uis = defaultdict(set)
        self.all_items = set()
        for u, i in self.train_data:
            self.uis[u].add(i)
            self.all_items.add(i)

        self.u_negs = {
            u: list(self.all_items - iss)
            for u, iss in self.uis.items()
        }

        self.ui_list = {
            u: list(iss)
            for u, iss in self.uis.items()
        }

        self.index_by_user = config.get_or_default('dataset/index_by_user', False)

        self.ui_loss_record = torch.ones([self.num_user, self.num_item]) * 100

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

        if self.config.get_or_default("dataset/noise", False):
            nop = self.config.get_or_default("dataset/noise_p", 0.0)
            neg_n = int(max(1, nop*self.size_pos))
            pos_n = self.size_pos - neg_n
            positives = random.choices(self.ui_list[user], k=pos_n) + random.choices(self.u_negs[user], k=neg_n)
            random.shuffle(positives)
        else:
            positives = random.choices(self.ui_list[user], k=self.size_pos)

        negatives = random.choices(self.u_negs[user], k=self.size_neg)

        return torch.tensor(user), torch.tensor(positives), torch.tensor(negatives)
