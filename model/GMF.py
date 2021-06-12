import torch
from torch.nn import Module, Embedding, Linear


class GMF(Module):
    def __init__(self, user_num, item_num, config):
        super().__init__()
        self.config = config
        self.user_embedding = Embedding(user_num, config['dim'])
        self.item_embedding = Embedding(item_num, config['dim'])
        self.beta = Linear(config['dim'], 1)

    def forward(self, users, items):
        users = self.user_embedding(users)
        items = self.item_embedding(items)
        return self.beta(users * items)

