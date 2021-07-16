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
            ndcg = math.log(2) / torch.log(rank + 2)
            for k in self.config.get_or_default("evaluator_args/eval_xs", [1, 3]):
                recall = (rank < k).float().mean()
                self.summary.add_scalar(f'Eval/recall{k}', recall, global_step=epoch)
                if k == 1:
                    continue
                self.summary.add_scalar(f'Eval/ndcg{k}', torch.mean((rank < k) * ndcg), global_step=epoch)

            self.score_cache.append(torch.sum(ndcg).item())
            if len(self.score_cache) > self.stop_delay:
                self.score_cache.pop(0)

    def should_stop(self):
        if self.config.get_or_default("evaluator_args/use_stop", False):
            return len(self.score_cache) == self.stop_delay and np.argmax(self.score_cache) == 0
        return False

    def record_softw(self, softw, epoch):
        self.summary.add_histogram('Analysis/softw', softw, global_step=epoch)

    def record_ig(self, dataset, epoch):
        self.summary.add_histogram('Analysis/ui_user', dataset.ui_loss.mean(dim=1), global_step=epoch)
        self.summary.add_histogram('Analysis/ui_item', dataset.ui_loss.mean(dim=0), global_step=epoch)
        ig = dataset.ui_ig
        self.summary.add_histogram('Analysis/ig_user', ig.mean(dim=1), global_step=epoch)
        self.summary.add_histogram('Analysis/ig_item', ig.mean(dim=0), global_step=epoch)
        self.summary.add_histogram('Analysis/xig_user', ig, global_step=epoch)
        self.summary.add_histogram('Analysis/xig_item', ig, global_step=epoch)
        self.summary.add_scalar('Analysis/ig_pos', (ig != 0.0).float().mean(), global_step=epoch)
        self.summary.add_scalar('Analysis/ig_mean', ig.float().mean(), global_step=epoch)


if __name__ == '__main__':
    pass
