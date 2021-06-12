import math

import torch
import numpy as np


class Evaluator:

    def __init__(self, config, summary, dataset):
        self.config = config
        self.summary = summary
        self.dataset = dataset

        self.user_input = torch.tensor(dataset.test_data[:, 0]).unsqueeze(dim=1).repeat([1, 101])

        self.item_input = torch.tensor(np.concatenate([
            dataset.test_data[:, 1].reshape([-1, 1]),
            dataset.test_data_neg
        ], axis=1))

        assert self.user_input.shape == self.item_input.shape
        self.stop_delay = config.get_or_default('evaluator_args/stop_delay', 10)
        self.score_cache = []

    def evaluate(self, model, epoch):
        # 失能验证
        if 'eval' in self.config and not self.config['eval']:
            return
        with torch.no_grad():
            model.eval()
            sample_num = self.user_input.shape[0]
            score = model(self.user_input.reshape([-1]).cuda(), self.item_input.reshape([-1]).cuda())
            score = score.reshape([sample_num, 101])
            rank = score.argsort(1, descending=True).argsort(1)[:, 0]
            recall1 = (rank < 1).float().mean()
            recall3 = (rank < 3).float().mean()
            ndcg = math.log(2) / torch.log(rank + 2)
            ndcg1 = torch.mean((rank < 1) * ndcg)
            ndcg3 = torch.mean((rank < 3) * ndcg)

            self.summary.add_scalar('Eval/recall1', recall1, global_step=epoch)
            self.summary.add_scalar('Eval/recall3', recall3, global_step=epoch)
            self.summary.add_scalar('Eval/ndcg', torch.sum(ndcg), global_step=epoch)
            self.summary.add_scalar('Eval/ndcg1', ndcg1, global_step=epoch)
            self.summary.add_scalar('Eval/ndcg3', ndcg3, global_step=epoch)

            print(f"Eval: r1:{recall1.item():0.4} r3:{recall3.item():0.4} n1:{ndcg1.item():0.4} n3:{ndcg3.item():0.4}")

            self.score_cache.append(ndcg.item())
            if len(self.score_cache) > self.stop_delay:
                self.score_cache.pop(0)

    def should_stop(self):
        return len(self.score_cache) == self.stop_delay and np.argmax(self.score_cache) == 0


if __name__ == '__main__':
    pass
